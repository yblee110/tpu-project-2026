# Flax NNX implementation of Diffusion
import os
import sys
import time
import glob
import re
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
import kagglehub
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Initialize TPU Pod distributed cluster (for multi-host communication)
if jax.process_count() == 1:
    try:
        jax.distributed.initialize()
    except Exception:
        pass

# --- Scaled up hyperparameters ---
global_batch_size = 256  # Global batch size distributed across all TPUs
block_size = 1024        # Expanded context length
max_iters = 20000        # Increased number of training iterations
eval_interval = 1000
learning_rate = 3e-4
eval_iters = 200
n_embd = 768             # Scaled up parameters (GPT-2 Small/Medium scale)
n_head = 12
n_layer = 12
head_dim = n_embd // n_head
# ---------------------------------

# Configure Mesh combining TPU devices (split batch dimension for data parallelism)
devices = jax.devices()
mesh = Mesh(devices, ('batch',))
data_sharding = NamedSharding(mesh, P('batch'))
replicated_sharding = NamedSharding(mesh, P())

# Load data
path = kagglehub.dataset_download("talesgomes27/sherleck-books")
text = ""
for file in glob.glob(os.path.join(path, "*.txt")):
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        # Remove Project Gutenberg header
        match_start = re.search(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", content, re.IGNORECASE)
        if match_start:
            content = content[match_start.end():]
        # Remove Project Gutenberg footer
        match_end = re.search(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK", content, re.IGNORECASE)
        if match_end:
            content = content[:match_end.start()]
        
        text += content.strip() + "\n\n"

chars = sorted(list(set(text)))
chars = ["_"] + chars # Add mask token
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
mask_token_id = stoi["_"]

def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return "".join([itos[int(n)] for n in l])

# Train and test splits
data = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, rng):
    data_split = train_data if split == "train" else val_data
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    ix = jax.random.randint(rng1, (global_batch_size,), 0, len(data_split) - block_size)
    x_orig = jnp.stack([data_split[i : i + block_size] for i in ix])
    y = x_orig

    # Mask tokens with random probability per sample
    mask_probs = jax.random.uniform(rng2, (global_batch_size, 1))
    mask = jax.random.uniform(rng3, (global_batch_size, block_size)) < mask_probs
    x = jnp.where(mask, mask_token_id, x_orig)

    # Distribute (Shard) the generated batch across multiple TPUs
    x = jax.device_put(x, data_sharding)
    y = jax.device_put(y, data_sharding)
    mask = jax.device_put(mask, data_sharding)

    return x, y, mask

def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)

def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)

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

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)

        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

        # No causal mask for bidirectional attention
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        y = jnp.matmul(attn_weights, v)

        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(n_embd, 4 * n_embd, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(4 * n_embd, n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = jax.nn.relu(x)**2
        x = self.c_proj(x)
        return x

class Block(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.attn = MultiHeadAttention(rngs)
        self.mlp = MLP(rngs)

    def __call__(self, x, cos_sin):
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x

class Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(vocab_size, n_embd, rngs=rngs)
        
        self.rotary_seq_len = block_size * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.cos = nnx.Variable(cos)
        self.sin = nnx.Variable(sin)

        self.blocks = [Block(rngs) for _ in range(n_layer)]
        self.lm_head = nnx.Linear(n_embd, vocab_size, use_bias=False, rngs=rngs)

    def _precompute_rotary_embeddings(self, seq_len, base=10000):
        channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        cos, sin = jnp.cos(freqs), jnp.sin(freqs)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def __call__(self, idx, targets=None, mask=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        x = rms_norm(x)

        cos_sin = (self.cos.value[:, :T], self.sin.value[:, :T])

        for block in self.blocks:
            x = block(x, cos_sin)
        x = rms_norm(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            if mask is not None:
                loss_all = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
                loss = jnp.sum(loss_all * mask) / (jnp.sum(mask) + 1e-6)
            else:
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        return logits, loss

def train_step(model, optimizer, idx, targets, mask):
    def loss_fn(model):
        _, loss = model(idx, targets, mask)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

@nnx.jit
def fast_train_step(model, optimizer, idx, targets, mask):
    return train_step(model, optimizer, idx, targets, mask)

def generate(model, max_new_tokens, prompt_len=16, temp=1.0, confidence_threshold=0.95, top_k=3, rng=None):
    if rng is None:
        rng = jax.random.PRNGKey(42)
        
    all_tokens = data[:prompt_len].tolist()
    total_steps = 0

    while len(all_tokens) - prompt_len < max_new_tokens:
        block_len = min(240, prompt_len + max_new_tokens - len(all_tokens))
        
        # Initialize x with mask_token_id
        x = jnp.full((1, block_size), mask_token_id, dtype=jnp.int32)
        x = x.at[0, :prompt_len].set(jnp.array(all_tokens[-prompt_len:], dtype=jnp.int32))
        
        masked = jnp.zeros((1, block_size), dtype=jnp.bool)
        masked = masked.at[0, prompt_len : prompt_len + block_len].set(True)
        
        while jnp.any(masked):
            total_steps += 1
            logits, _ = model(x)
            probs = jax.nn.softmax(logits / temp, axis=-1)
            
            top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
            confidences = jnp.sum(top_k_probs, axis=-1)
            
            decode_mask = (confidences >= confidence_threshold) & masked
            
            # If no position meets threshold, decode the one with highest confidence
            if not jnp.any(decode_mask):
                masked_confidences = jnp.where(masked, confidences, -1.0)
                flat_idx = jnp.argmax(masked_confidences)
                decode_mask = decode_mask.at[jnp.unravel_index(flat_idx, decode_mask.shape)].set(True)
            
            # Sample from top-k
            top_k_probs_norm = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)
            rng, subkey = jax.random.split(rng)
            
            # Categorical expects logits or probabilities
            # Here we have probabilities
            sampled_k = jax.random.categorical(subkey, jnp.log(top_k_probs_norm + 1e-10).reshape(-1, top_k))
            sampled_k = sampled_k.reshape(1, block_size)
            
            sampled_tokens = jnp.take_along_axis(top_k_indices, sampled_k[..., None], axis=-1).squeeze(-1)
            
            x = jnp.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask
            
        all_tokens.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    return decode(all_tokens)

def estimate_loss(model, rng):
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            rng, subkey = jax.random.split(rng)
            xb, yb, mb = get_batch(split, subkey)
            _, loss = model(xb, yb, mb)
            losses.append(loss)
        out[split] = jnp.mean(jnp.array(losses))
    return out, rng

if __name__ == "__main__":
    rngs = nnx.Rngs(1337)
    model = Model(rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate))
    
    train_flag = "--train" in sys.argv
    weights_path = "weights/diffusion_nnx.msgpack"
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    if os.path.exists(weights_path) and not train_flag:
        print(f"Loading weights from {weights_path}")
        pass
    else:
        print("Training from scratch")
        start = time.time()
        rng = jax.random.PRNGKey(1337)
        for i in range(max_iters):
            if i % eval_interval == 0 or i == max_iters - 1:
                losses, rng = estimate_loss(model, rng)
                print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {time.time() - start:.2f}s")
                sample = generate(model, 240, rng=rng)
                print(f"Sample:\n{sample}\n")

            rng, subkey = jax.random.split(rng)
            xb, yb, mb = get_batch("train", subkey)
            loss = fast_train_step(model, optimizer, xb, yb, mb)
        
        print(f"Total training time: {time.time() - start:.2f}s")
