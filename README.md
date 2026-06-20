# Self-Attention from Scratch

A minimal, heavily-commented implementation of GPT-2-style transformers for learning how self-attention works.

---

## What This Is and Why It Exists

This is a learning resource, not a production implementation. Every design decision favors clarity over performance. The code is meant to be read alongside the explanations in `docs/` — not deployed anywhere.

If you have used PyTorch before and know what a matrix multiplication is, you have enough background. Prior transformer experience is not required.

A few things to keep in mind before you dive in:

- Comments explain *why*, not just *what*. When something looks overly verbose, that is intentional.
- No abstractions are hidden inside library calls you cannot inspect. The attention mechanism is written out explicitly.
- Performance trade-offs (batching strategies, fused kernels, memory layout) are deliberately ignored. You will see them called out in comments when they would normally matter.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a tiny model (< 5 min on CPU)
python train.py

# Visualize attention patterns from a trained model
python visualize.py --checkpoint checkpoints/checkpoint_epoch_20.pt

# Just see the positional encoding pattern
python visualize.py --checkpoint checkpoints/checkpoint_epoch_20.pt --pe_only
```

---

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--d_model` | 128 | Embedding dimension (must be divisible by `--n_heads`) |
| `--n_heads` | 4 | Number of attention heads |
| `--n_layers` | 4 | Number of transformer blocks |
| `--seq_len` | 128 | Context length (tokens per training example) |
| `--batch_size` | 32 | Training batch size |
| `--epochs` | 20 | Number of training epochs |
| `--lr` | 3e-4 | Learning rate (Adam optimizer) |
| `--device` | `cpu` | `cpu` or `cuda` |

Example — a slightly larger model on GPU:

```bash
python train.py --d_model 256 --n_heads 8 --n_layers 6 --device cuda
```

---

## Project Structure

```
self-attention/
├── src/
│   ├── attention.py           # Multi-head self-attention
│   ├── positional_encoding.py # Sinusoidal PE
│   ├── transformer.py         # TransformerBlock (attention + FFN + LayerNorm)
│   ├── model.py               # Full GPT model
│   ├── data.py                # Character-level dataset
│   └── visualization.py       # Plotting utilities
├── docs/
│   ├── 01_attention_explained.md
│   ├── 02_positional_encoding.md
│   ├── 03_gpt2_architecture.md
│   └── 04_training_guide.md
├── train.py                   # Training entry point
├── visualize.py               # Visualization entry point
└── requirements.txt
```

---

## Key Concepts Covered

- **Scaled dot-product attention** — the core operation, worked through step by step with a NumPy walkthrough before the PyTorch version
- **Causal masking** — how to prevent a token from attending to future positions during autoregressive generation
- **Multi-head attention** — splitting `d_model` into parallel attention heads, running them simultaneously, and concatenating the results
- **Sinusoidal positional encoding** — why position information needs to be injected and how sine/cosine frequencies encode it
- **Pre-LayerNorm transformer blocks** — the GPT-2 convention of normalizing inputs rather than outputs of each sub-layer
- **Residual connections** — what they do to gradient flow and why every sub-layer wraps its output in `x + sublayer(x)`
- **Weight tying** — sharing weights between the token embedding matrix and the final output projection
- **Character-level language modeling** — training on Shakespeare text so the dataset fits in memory and loss is easy to interpret

---

## Learning Path

Work through these in order. Each step builds on the previous one.

1. Read `docs/01_attention_explained.md` — understand what queries, keys, and values represent before touching any code.
2. Read `src/attention.py` — follow `scaled_dot_product_attention_numpy()` line by line.
3. Read `docs/02_positional_encoding.md` — understand why transformers need explicit position information.
4. Run `python train.py` — watch the loss drop. You do not need to understand everything yet; just observe that it works.
5. Read `docs/03_gpt2_architecture.md` — understand how individual blocks are stacked into a full model.
6. Run `python visualize.py` — inspect the attention heatmaps and compare different heads and layers.
7. Read `docs/04_training_guide.md` — now that you have seen it work, experiment with hyperparameters and see what breaks.

---

## What This Is Not

- **Not optimized.** No Flash Attention, no fused kernels, no mixed precision. A real implementation would look different.
- **Not a library.** There is no `pip install`. Clone it, read it, run it.
- **Not GPT-2 weights.** This trains from scratch on a small text corpus. The architecture follows GPT-2 conventions; the weights have nothing to do with OpenAI's model.
