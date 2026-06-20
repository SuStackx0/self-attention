# Self-Attention from Scratch

Self-attention and GPT-2-style transformer implemented from scratch in Python, for learning purposes.

## Quick Start

```bash
pip install -r requirements.txt
python train.py                  # trains on CPU in ~5 min
python visualize.py --checkpoint checkpoints/checkpoint_epoch_20.pt
```

## Structure

```
src/
  attention.py           # scaled dot-product + multi-head attention
  positional_encoding.py # sinusoidal PE
  transformer.py         # TransformerBlock (pre-LN, attention + FFN)
  model.py               # full GPT model + generate()
  data.py                # char-level dataset (Hamlet excerpt)
  visualization.py       # heatmaps, PE plot, loss curve
train.py                 # training entry point
visualize.py             # visualization entry point
docs/                    # explanation docs (start here)
```

## Training flags

| Flag | Default | |
|---|---|---|
| `--d_model` | 128 | embedding dim |
| `--n_heads` | 4 | attention heads |
| `--n_layers` | 2 | transformer blocks |
| `--seq_len` | 64 | context length |
| `--epochs` | 20 | |
| `--device` | cpu | or `cuda` |

## Read in this order

1. `docs/01_attention_explained.md` — Q, K, V and the attention formula
2. `src/attention.py` — numpy walkthrough first, then torch
3. `docs/02_positional_encoding.md` — why and how sinusoidal PE works
4. `src/transformer.py` + `src/model.py` — how blocks stack into GPT
5. `python train.py` — watch loss drop
6. `python visualize.py` — inspect attention heatmaps
