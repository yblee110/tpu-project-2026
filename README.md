# GPT Writes One Token at a Time. This Model Writes Entire Blocks at Once.

## Building a Diffusion Language Model from Scratch with Flax NNX on TPU

GPT has been the undisputed king of text generation for years. But what if there's a fundamentally better way to generate text — not one token at a time, left to right, but **entire blocks in parallel**, like watching noise crystallize into coherent sentences?

That's exactly what **Diffusion Language Models (DLLMs)** do. Inspired by image diffusion models like Stable Diffusion, DLLMs start from a sequence of masked ("noised") tokens and iteratively denoise them until clean text emerges. And the surprising part? It takes only **5 small changes** to turn a GPT into a Diffusion model.

All code is available in the [yblee/tpu-project-2026](https://github.com/yblee/tpu-project-2026) repository, which is heavily inspired by the original [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) and extends it with JAX/Flax NNX implementations optimized for TPU training.

---

## What is a Diffusion Language Model?

If you've used Stable Diffusion or Midjourney, you already understand the core idea. Image diffusion models start from pure noise and iteratively denoise it into a coherent image. **DLLMs apply the same principle to discrete text.**

Instead of predicting the next token given past tokens, a DLLM receives a sequence where some tokens have been replaced with a special `MASK` token, and it learns to recover the original tokens.

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

The model is trained on the **complete Sherlock Holmes collection** by Arthur Conan Doyle — sourced from the [Sherlock Books](https://www.kaggle.com/datasets/talesgomes27/sherleck-books) dataset on Kaggle. This includes all four novels and fifty-six short stories, from *A Study in Scarlet* to *The Case-Book of Sherlock Holmes*.

- **Rich, consistent prose style** — Doyle's Victorian English has a distinctive rhythm and vocabulary that makes it easy to visually evaluate generation quality
- **Right size for experimentation** — The complete corpus is large enough to train meaningful patterns but small enough to iterate quickly
- **Character-level modeling** — At the character level, the model learns to spell Victorian-era words, reproduce dialogue formatting (`" Holmes said, "`), and even pick up on recurring phrases like `"elementary"` or `"the game is afoot"`
- **Public domain** — All works are freely available via Project Gutenberg

---

## The 5 Changes That Turn GPT into a Diffusion Model

Roughly 80% of the code between `gpt.py` and `diffusion.py` is identical. Here are the 5 surgical modifications:

**1. Add a mask token to the vocabulary**

```python
chars = sorted(list(set(text)))
chars = ["_"] + chars  # Mask token added
mask_token_id = stoi["_"]
```

**2. Switch from causal to bidirectional attention**

```python
# GPT:  y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# DLLM: y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

**3. Change the training objective from next-token to unmasking**

```python
x_orig = jnp.stack([dataset[i : i + block_size] for i in ix])
mask = jax.random.uniform(rng, ...) < mask_probs
x = jnp.where(mask, mask_token_id, x_orig)  # Replace some tokens with mask
```

**4. Only masked tokens contribute to the loss**

```python
loss_all = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
loss = jnp.sum(loss_all * mask) / (jnp.sum(mask) + 1e-6)
```

**5. Replace sequential decoding with confidence-based parallel decoding**

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
        flat_idx = jnp.argmax(jnp.where(masked, confidences, -1.0))
        decode_mask = decode_mask.at[jnp.unravel_index(flat_idx, decode_mask.shape)].set(True)

    x = jnp.where(decode_mask, sampled_tokens, x)
    masked = masked & ~decode_mask
```

1. Start with a block of masked tokens
2. Run the model on the entire sequence
3. Calculate confidence (top-k probability sum) at each position
4. **Lock in** tokens where confidence exceeds a threshold
5. Repeat until all positions are decoded
6. Move to the next block

Instead of generating 240 tokens in 240 sequential steps (like GPT), the model can decode multiple tokens per step — sometimes resolving the entire block in just 10-20 iterations.

---

## Why Flax NNX?

### The Old Flax vs. NNX

Traditional Flax (`linen`) used a functional programming model where modules were immutable dataclasses and you had to manage separate parameter dictionaries. NNX introduces **stateful, object-oriented modules** that feel much closer to PyTorch:

```python
class MultiHeadAttention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.c_q = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.c_k = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.c_v = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
```

Parameters live directly on the module — no separate `params` dictionary needed.

### Training Loop with NNX

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

### Non-Parameter State with `nnx.Variable`

```python
class DiffusionModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        cos, sin = self._precompute_rotary_embeddings(block_size * 2)
        self.cos = nnx.Variable(cos)  # Not a trainable parameter
        self.sin = nnx.Variable(sin)
```

`nnx.Variable` registers the tensor as part of the module's state but excludes it from gradient computation — the NNX equivalent of PyTorch's `register_buffer`.

---

## Scaling Up: TPU Data Parallelism with JAX

### Device Mesh and Sharding

```python
devices = jax.devices()
mesh = Mesh(devices, ('batch',))
data_sharding = NamedSharding(mesh, P('batch'))
replicated_sharding = NamedSharding(mesh, P())
```

With just three lines, the batch dimension is split across all available TPU cores.

### Scaled-Up Architecture

| Parameter | PyTorch (GPU) | JAX/Flax (TPU) |
| :--- | :--- | :--- |
| `n_embd` | 384 | 768 |
| `n_head` | 6 | 12 |
| `n_layer` | 6 | 12 |
| `block_size` | 256 | 1024 |
| `batch_size` | 64 | 256 |
| Parameters | ~10.7M | ~85M |

This brings the model to **GPT-2 Small/Medium scale** with a 4x larger context window.

---

## Model Architecture Deep Dive

- **RMSNorm instead of LayerNorm** — Simplified normalization with no learnable parameters
- **Rotary Positional Embeddings (RoPE)** — Relative positional encoding applied to Q and K
- **QK-Norm** — Normalizing queries and keys before attention (stabilizes training)
- **ReLU-Squared activation** — A simple but effective alternative to GeGLU
- **No bias in any linear layer** — Following modern best practices for large-scale training

```python
class Block(nnx.Module):
    def __call__(self, x, cos_sin):
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x
```

---

## Get Started

### Files

- `diffusion_nnx.ipynb` — Interactive TPU notebook (Google Colab compatible)
- `diffusion_nnx.py` — Standalone TPU training script
- `diffusion.py` / `gpt.py` — PyTorch implementations for comparison

### Run Locally

```bash
git clone https://github.com/yblee/tpu-project-2026.git
cd tpu-project-2026/tiny-diffusion
uv sync
uv run diffusion.py  # Generate text with pre-trained weights
```

### Run on TPU

Open `diffusion_nnx.ipynb` in a TPU-enabled Google Colab and run all cells. The notebook will automatically install JAX TPU dependencies, download the Sherlock Holmes dataset from Kaggle, and start training.

---

## Acknowledgements

This project is hosted at [yblee/tpu-project-2026](https://github.com/yblee/tpu-project-2026). The JAX/Flax NNX implementations and TPU adaptations were heavily **inspired by** the original [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) by Nathan Barry. The PyTorch baseline draws from Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) and ["Let's build GPT"](https://github.com/karpathy/ng-video-lecture) implementations.
