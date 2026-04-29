# tiny-diffusion

A character-level language diffusion model for text generation trained on Sherlock Holmes books, in ~365 lines of code! It is only 10.7 million parameters (for the base PyTorch version), so you can also try it out locally!

![Demo](https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/animation.gif)

This repo also contains a tiny GPT implementation in ~313 lines of code. ~80% of the code between the two files is the exact same, highlighting the core differences between Autoregressive and Diffusion language models.

**🚀 NEW:** We now feature **JAX/Flax (NNX)** implementations (`diffusion_nnx.py`, `gpt_nnx.py`, and `diffusion_nnx.ipynb`) optimized for distributed training on **Google Cloud TPUs**.

> This is `v2` of this project, which simplified the diffusion code from ~1,000 lines to ~400, and slightly altered the architecture. To view the original version, view the `old` branch.


## Quick Start

### Installation
```bash
# Install dependencies (Python 3.10+)
uv sync

# Download the trained model weights (if you don't want to train it from scratch)
mkdir -p weights && wget -P weights https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/{gpt,diffusion}.pt
```

### Generation
Generate text with the trained PyTorch models:
```bash
# Diffusion (parallel decoding)
uv run diffusion.py

# GPT (autoregressive)
uv run gpt.py
```
Both models generate 2,000 characters by default and use the first 16 characters of the dataset as the initial context. These parameters can be modified in the `generate` function.

### Training
To train the base PyTorch models from scratch:

```bash
# Train diffusion model
uv run diffusion.py --train

# Train GPT model
uv run gpt.py --train
```
The `gpt` model trains for 5,000 iterations while the `diffusion` model trains for 10,000, taking ~10 and ~20 minutes respectively on an A100 GPU. The weights are saved to the `weights/` directory.

The diffusion model trains for twice as long because half as many tokens count towards the loss during training (only masked tokens contribute to the loss).

### Visualization
Visualize the generation process step-by-step directly in your terminal using an awesome "Cryptographic Decoupling" theme:

```bash
# Visualize diffusion model only
uv run visualize.py

# Compare diffusion and GPT side-by-side sequentially
uv run visualize.py --compare

# Generate more blocks
uv run visualize.py --blocks 10
```

---

## 🔥 Scaling Up: JAX & Google Cloud TPUs

If you want to experience distributed training on Google Cloud TPUs, this repository provides scaled-up versions of both models using **JAX and Flax's new NNX API**.

The `*_nnx.py` scripts implement a larger 768-dimension model (GPT-2 Small/Medium scale) with a 1024 context length. Using `NamedSharding`, the code automatically parallelizes batch processing across all available TPU cores (e.g., an 8-core v3-8 Pod).

### Running on a TPU VM
1. Provision a Cloud TPU VM (e.g., `v3-8`).
2. SSH into the machine and clone this repo.
3. Install the specific JAX TPU wheel:
   ```bash
   pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
   ```
4. Install the rest of the dependencies:
   ```bash
   uv pip install -r pyproject.toml
   ```
5. Run the TPU-optimized training script:
   ```bash
   python diffusion_nnx.py --train
   ```

You can also interactively train and visualize the model on a TPU-enabled Colab environment using the provided `diffusion_nnx.ipynb` notebook.

---

## Differences Between The Models

### GPT (Autoregressive)
- Predicts the next token given all previous tokens
- Uses **causal attention** (can only look at past tokens)
- Generates text **sequentially** (one token at a time, left-to-right)
- Training: minimize cross-entropy loss on next token prediction

### Diffusion (Non-Autoregressive)
- Predicts original tokens given partially masked sequences
- Uses **bidirectional attention** (can look at all tokens)
- Generates text **in parallel** and in blocks: fills in masked tokens iteratively, then moves to the next block
- Training: minimize cross-entropy loss on denoising masked tokens

### Key Modifications
The diffusion model makes **5 key changes** to the GPT architecture:

1. **Add mask token** to vocabulary (`_`) for representing noised tokens
2. **Change attention** from causal to bidirectional (`is_causal=False`)
3. **Change generation** from sequential to confidence-based parallel decoding
4. **Change training objective** from next token prediction to unmasking
5. **Only masked tokens** contribute to the loss during training


## Acknowledgements
This project is a fork and extension of the original [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) repository created by Nathan Barry. The JAX/Flax implementations and TPU adaptations were built upon this fantastic foundational work.

The code for `gpt.py` and `diffusion.py` take heavy inspiration from the Andrej Karpathy GPT implementations listed below:
- [nanochat GPT implementation](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)
- ["Let's build GPT" video GPT implementation](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)

My GPT implementation, `gpt.py`, aims to strike a balance between simplicity and good generation quality.

The `diffusion.py` file is a modified version of `gpt.py` with as few modifications as possible to get it to do language diffusion.


## License

MIT
