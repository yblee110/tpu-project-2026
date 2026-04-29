# GPT Writes One Token at a Time. This Model Writes Entire Blocks at Once.

## Building a Diffusion Language Model from Scratch with Flax NNX on TPU

GPT has been the undisputed king of text generation for years. But what if there's a fundamentally better way to generate text — not one token at a time, left to right, but **entire blocks in parallel**, like watching noise crystallize into coherent sentences?

That's exactly what **Diffusion Language Models (DLLMs)** do. Inspired by image diffusion models like Stable Diffusion, DLLMs start from a sequence of masked ("noised") tokens and iteratively denoise them until clean text emerges. And the surprising part? It takes only **5 small changes** to turn a GPT into a Diffusion model.

In this post, I'll walk through how I built a character-level DLLM trained on the complete works of **Sherlock Holmes** using **Flax NNX** — Flax's next-generation neural network API — and ran it on **Google Cloud TPUs**. The full implementation fits in a single Jupyter Notebook and demonstrates how minimal the code changes are when moving from a standard GPT to a Diffusion model.

All code is available in the [yblee/tpu-project-2026](https://github.com/yblee/tpu-project-2026) repository, which is heavily inspired by the original [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) and extends it with JAX/Flax NNX implementations optimized for TPU training.

---

## What is a Diffusion Language Model?

If you've used Stable Diffusion or Midjourney, you already understand the core idea. Image diffusion models start from pure noise and iteratively denoise it into a coherent image. **DLLMs apply the same principle to discrete text.**

Instead of predicting the next token given past tokens, a DLLM receives a sequence where some tokens have been replaced with a special `[MASK]` token, and it learns to recover the original tokens.

### DLLM vs. GPT at a Glance

| Feature | GPT (Autoregressive) | DLLM (Diffusion) |
| :--- | :--- | :--- |
| **Generation** | Sequential, one token at a time | Parallel, entire blocks at once |
| **Attention** | Causal (past tokens only) | Bidirectional (all tokens) |
| **Training Objective** | Next-token prediction | Masked token recovery (denoising) |
| **Context** | Unidirectional | Global / Bidirectional |

The bidirectional attention is the key insight: during generation, the model can see the entire sequence — both already-decoded tokens and still-masked positions — to make better predictions about each position simultaneously.

---

## Training Data: The Complete Sherlock Holmes

To keep things accessible and fun, I trained the model on the **complete Sherlock Holmes collection** by Arthur Conan Doyle — sourced from the [Sherlock Books](https://www.kaggle.com/datasets/talesgomes27/sherleck-books) dataset on Kaggle. This includes all four novels and fifty-six short stories, from *A Study in Scarlet* to *The Case-Book of Sherlock Holmes*.

Why Sherlock Holmes?

- **Rich, consistent prose style** — Doyle's Victorian English has a distinctive rhythm and vocabulary that makes it easy to visually evaluate generation quality
- **Right size for experimentation** — The complete corpus is large enough to train meaningful patterns but small enough to iterate quickly (no need for terabytes of storage or multi-GPU setups for the PyTorch version)
- **Character-level modeling** — At the character level, the model learns to spell Victorian-era words, reproduce dialogue formatting (`" Holmes said, "`), and even pick up on recurring phrases like `"elementary"` or `"the game is afoot"`
- **Public domain** — All works are freely available via Project Gutenberg

The data pipeline strips Project Gutenberg headers/footers and encodes every character into a simple integer mapping. A special `_` token is added to the vocabulary to serve as the mask token for diffusion training.

---

## The 5 Changes That Turn GPT into a Diffusion Model

One of the most illuminating aspects of this project is how **little** needs to change. Roughly 80% of the code between `gpt.py` and `diffusion.py` is identical. Here are the 5 surgical modifications:

**1. Add a mask token to the vocabulary**

```python
chars = sorted(list(set(text)))
chars = ["_"] + chars  # Mask token added
mask_token_id = stoi["_"]
```

The underscore `_` serves as the "noise" token, representing positions that need to be predicted.

**2. Switch from causal to bidirectional attention**

```python
# GPT:  y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# DLLM: y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

Removing the causal mask allows every position to attend to every other position — essential for jointly predicting masked tokens.

**3. Change the training objective from next-token to unmasking**

```python
# GPT:  y = x shifted by 1 (predict next token)
# DLLM: y = original x (recover from masked input)

x_orig = jnp.stack([dataset[i : i + block_size] for i in ix])
mask = jax.random.uniform(rng, ...) < mask_probs
x = jnp.where(mask, mask_token_id, x_orig)  # Replace some tokens with mask
```

During training, each sample gets a random masking ratio. The model sees the corrupted `x` and tries to recover `x_orig`.

**4. Only masked tokens contribute to the loss**

```python
loss_all = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
loss = jnp.sum(loss_all * mask) / (jnp.sum(mask) + 1e-6)
```

We don't penalize the model for positions it wasn't asked to predict. Only the masked positions count.

**5. Replace sequential decoding with confidence-based parallel decoding**

This is the most substantial change and deserves its own section.

---

## Parallel Decoding: How DLLMs Generate Text

Unlike GPT's simple "predict next token, append, repeat" loop, DLLMs use an iterative **confidence-based decoding** strategy:

```python
while jnp.any(masked):
    logits, _ = model(x)
    probs = jax.nn.softmax(logits / temp, axis=-1)
    top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
    confidences = jnp.sum(top_k_probs, axis=-1)

    # Only decode positions where the model is confident enough
    decode_mask = (confidences >= confidence_threshold) & masked
    if not jnp.any(decode_mask):
        # Force-decode at least one position per step
        flat_idx = jnp.argmax(jnp.where(masked, confidences, -1.0))
        decode_mask = decode_mask.at[jnp.unravel_index(flat_idx, decode_mask.shape)].set(True)

    x = jnp.where(decode_mask, sampled_tokens, x)
    masked = masked & ~decode_mask
```

The process works like this:

1. Start with a block of masked tokens
2. Run the model on the entire sequence
3. Calculate confidence (top-k probability sum) at each position
4. **Lock in** tokens where confidence exceeds a threshold
5. Repeat until all positions are decoded
6. Move to the next block

The result? Instead of generating 240 tokens in 240 sequential steps (like GPT), the model can decode multiple tokens per step — sometimes resolving the entire block in just 10-20 iterations.

---

## Why Flax NNX?

This is where things get interesting. While the original `tiny-diffusion` is implemented in PyTorch, I rewrote the entire training pipeline using **Flax NNX** — the new object-oriented API for Flax.

### The Old Flax vs. NNX

Traditional Flax (`linen`) used a functional programming model where modules were immutable dataclasses and you had to manage separate parameter dictionaries. NNX changes this by introducing **stateful, object-oriented modules** that feel much closer to PyTorch:

```python
class MultiHeadAttention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.c_q = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.c_k = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.c_v = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x, cos_sin):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, n_head, head_dim)
        k = self.c_k(x).reshape(B, T, n_head, head_dim)
        v = self.c_v(x).reshape(B, T, n_head, head_dim)
        # ... attention computation
        return self.c_proj(y)
```

Notice how parameters live directly on the module (`self.c_q`, `self.c_k`, etc.) — no separate `params` dictionary needed. This makes the code dramatically easier to read and debug.

### Training Loop with NNX

The training loop is equally clean:

```python
rngs = nnx.Rngs(1337)
model = DiffusionModel(rngs)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate))

@nnx.jit
def train_step(model, optimizer, x, y, mask):
    def loss_fn(model):
        _, loss = model(x, y, mask)
        return loss
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss
```

Key observations:
- **`nnx.Optimizer`** wraps both the model and the optimizer state, so `optimizer.update(grads)` updates model parameters in-place
- **`nnx.jit`** replaces the traditional `jax.jit` and automatically handles NNX graph tracing
- **`nnx.value_and_grad`** works just like `jax.value_and_grad` but understands NNX modules
- No need for `train_state` boilerplate or `apply_fn` indirection

### Non-Parameter State with `nnx.Variable`

One subtle but important detail: the precomputed rotary embeddings are stored using `nnx.Variable` rather than regular parameters:

```python
class DiffusionModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        # ...
        cos, sin = self._precompute_rotary_embeddings(block_size * 2)
        self.cos = nnx.Variable(cos)  # Not a trainable parameter
        self.sin = nnx.Variable(sin)
```

`nnx.Variable` registers the tensor as part of the module's state (so it's included in serialization and JIT tracing) but excludes it from gradient computation. This is the NNX equivalent of PyTorch's `register_buffer`.

---

## Scaling Up: TPU Data Parallelism with JAX

One of the main motivations for using JAX/Flax was to leverage **Google Cloud TPUs** for distributed training. JAX makes this almost trivially easy.

### Device Mesh and Sharding

```python
devices = jax.devices()
mesh = Mesh(devices, ('batch',))
data_sharding = NamedSharding(mesh, P('batch'))
replicated_sharding = NamedSharding(mesh, P())
```

With just three lines, we define a device mesh that splits the batch dimension across all available TPU cores. On a v3-8 Pod, this means 8 cores each processing 1/8 of the batch.

### Sharding the Data

```python
def get_batch(split, rng):
    # ... generate x, x_orig, mask ...
    x = jax.device_put(x, data_sharding)       # Shard across TPUs
    x_orig = jax.device_put(x_orig, data_sharding)
    mask = jax.device_put(mask, data_sharding)
    return x, x_orig, mask
```

`jax.device_put` with `NamedSharding` automatically distributes the batch data. The compiler handles the rest — no manual communication primitives needed.

### Scaled-Up Architecture

The TPU version scales up the model significantly compared to the PyTorch baseline:

| Parameter | PyTorch (GPU) | JAX/Flax (TPU) |
| :--- | :--- | :--- |
| `n_embd` | 384 | 768 |
| `n_head` | 6 | 12 |
| `n_layer` | 6 | 12 |
| `block_size` | 256 | 1024 |
| `batch_size` | 64 | 256 |
| Parameters | ~10.7M | ~85M |

This brings the model to **GPT-2 Small/Medium scale** with a 4x larger context window, something that would be painfully slow on a single GPU but trains efficiently on a TPU Pod.

---

## Model Architecture Deep Dive

The architecture itself is a clean Transformer with several modern design choices:

**RMSNorm instead of LayerNorm** — Simplified normalization with no learnable parameters:
```python
def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)
```

**Rotary Positional Embeddings (RoPE)** — Relative positional encoding applied to Q and K:
```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)
```

**QK-Norm** — Normalizing queries and keys before attention (stabilizes training):
```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = rms_norm(q), rms_norm(k)  # QK norm
```

**ReLU-Squared activation** — A simple but effective alternative to GeGLU:
```python
x = jax.nn.relu(x) ** 2
```

**No bias in any linear layer** — Following modern best practices for large-scale training.

The full block follows a standard Pre-Norm Transformer pattern:

```python
class Block(nnx.Module):
    def __call__(self, x, cos_sin):
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x
```

---

## Visualizing the Decoding Process

One of the most compelling aspects of DLLMs is watching them generate text. The notebook includes a real-time visualization that renders the decoding process as a "Cryptographic Decoupling" — masked positions rapidly cycle through random cryptographic characters, then lock into place with a crimson highlight as the model gains confidence.

The visualization works by tracking the `decode_mask` at each iteration:

```python
while jnp.any(masked):
    # ... model forward pass ...
    # Masked positions: rapidly rotating cryptographic characters (blue neon)
    # Decoded positions: confirmed with high confidence (crimson highlight)
    display_crypto_state(temp_tokens, current_mask, block_start)
```

It's mesmerizing to watch: characters appear to "crystallize" out of noise, often resolving common words and patterns first, then filling in the ambiguous positions last.

---

## Lessons Learned

**DLLMs are surprisingly approachable.** The 5 modifications needed to turn GPT into a Diffusion model are each small and individually well-understood. The resulting system is elegant and generates text in a fundamentally different way.

**Flax NNX dramatically lowers the barrier to JAX.** If you've been avoiding JAX because of the functional programming overhead, NNX is the answer. The API is close enough to PyTorch that porting code is straightforward, while you still get all of JAX's compilation and distributed computing benefits.

**TPU scaling with JAX is (almost) too easy.** With `Mesh`, `NamedSharding`, and `device_put`, distributing training across TPU cores requires only a handful of lines. The compiler handles communication and synchronization automatically.

**Parallel decoding is a genuine speedup.** Watching the model decode 20-30 tokens per iteration instead of 1 is a striking demonstration of why non-autoregressive generation is an active area of research.

---

## Get Started

You can explore the full implementation in these files:

- `diffusion_nnx.ipynb` — Interactive TPU notebook (Google Colab compatible)
- `diffusion_nnx.py` — Standalone TPU training script
- `diffusion.py` / `gpt.py` — PyTorch implementations for comparison

**Try it locally:**
```bash
git clone https://github.com/yblee/tpu-project-2026.git
cd tpu-project-2026/tiny-diffusion
uv sync
uv run diffusion.py  # Generate text with pre-trained weights
```

**Try it on TPU:**
Open `diffusion_nnx.ipynb` in a TPU-enabled Google Colab and run all cells. The notebook will automatically install JAX TPU dependencies, download the Sherlock Holmes dataset from Kaggle, and start training.

---

## Acknowledgements

This project is hosted at [yblee/tpu-project-2026](https://github.com/yblee/tpu-project-2026). The JAX/Flax NNX implementations and TPU adaptations were heavily **inspired by** the original [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) by Nathan Barry — a brilliantly minimal educational project that demonstrates how few changes are needed to turn GPT into a Diffusion model. The PyTorch baseline in turn draws from Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) and ["Let's build GPT"](https://github.com/karpathy/ng-video-lecture) implementations.
