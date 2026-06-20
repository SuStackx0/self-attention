"""
visualize.py — Visualization entry point for a trained GPT checkpoint.

This script loads a checkpoint produced by train.py and generates:
  - An attention heatmap for a chosen layer and head
  - A positional encoding heatmap

Run examples:
    # Full visualization
    python visualize.py --checkpoint checkpoints/checkpoint_final.pt

    # Specify seed text, layer, and head
    python visualize.py --checkpoint checkpoints/checkpoint_final.pt \
        --text "To be or not" --layer 1 --head 2 --output_dir checkpoints/

    # Only plot positional encoding, skip attention
    python visualize.py --checkpoint checkpoints/checkpoint_final.pt --pe_only
"""

import argparse
import os
import sys

import torch

from src.model import GPT, GPTConfig
from src.visualization import plot_attention_heatmap
from src.positional_encoding import get_encoding


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize attention and positional encoding from a "
                    "GPT checkpoint."
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pt checkpoint file produced by "
                             "train.py.")
    parser.add_argument("--text", type=str, default="To be or not",
                        help='Seed text to visualize attention for '
                             '(default: "To be or not")')
    parser.add_argument("--layer", type=int, default=0,
                        help="Transformer layer index to visualize (0-based, "
                             "default: 0)")
    parser.add_argument("--head", type=int, default=0,
                        help="Attention head index to visualize (0-based, "
                             "default: 0)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help='Directory to save PNG files (default: ".")')
    parser.add_argument("--pe_only", action="store_true",
                        help="If set, only plot positional encoding and exit.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Positional encoding plot helper
# ---------------------------------------------------------------------------

def plot_positional_encoding(d_model: int, seq_len: int,
                              save_path: str) -> None:
    """
    Generate and save a heatmap of the sinusoidal positional encoding matrix.

    Shape: (seq_len, d_model) — rows are positions, columns are dimensions.
    We use matplotlib directly here so this file has no extra dependencies.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pe = get_encoding(seq_len=seq_len, d_model=d_model)  # (seq_len, d_model)

    # get_encoding may return a torch.Tensor or a numpy array — handle both.
    if hasattr(pe, "numpy"):
        pe = pe.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu", origin="lower")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Position")
    ax.set_title("Sinusoidal Positional Encoding")
    fig.colorbar(im, ax=ax, label="Encoding value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate checkpoint path
    # ------------------------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        print(
            f"ERROR: Checkpoint not found: {args.checkpoint}\n"
            "Make sure you have trained the model first with train.py and "
            "that --checkpoint points to a valid .pt file.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    config: GPTConfig = ckpt["config"]
    vocab = ckpt["vocab"]
    chars, stoi, itos = vocab

    print(
        f"  d_model={config.d_model}, n_heads={config.n_heads}, "
        f"n_layers={config.n_layers}, seq_len={config.seq_len}, "
        f"vocab_size={config.vocab_size}"
    )

    # ------------------------------------------------------------------
    # Positional encoding only?
    # ------------------------------------------------------------------
    if args.pe_only:
        pe_path = os.path.join(args.output_dir, "positional_encoding.png")
        print("Plotting positional encoding …")
        plot_positional_encoding(
            d_model=config.d_model,
            seq_len=config.seq_len,
            save_path=pe_path,
        )
        print(f"Saved positional encoding plot to: {pe_path}")
        return

    # ------------------------------------------------------------------
    # Reconstruct and load model
    # ------------------------------------------------------------------
    print("Reconstructing model …")
    model = GPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Validate layer / head indices
    # ------------------------------------------------------------------
    if args.layer >= config.n_layers:
        print(
            f"ERROR: --layer {args.layer} is out of range. "
            f"This model has {config.n_layers} layer(s) (indices 0–{config.n_layers - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.head >= config.n_heads:
        print(
            f"ERROR: --head {args.head} is out of range. "
            f"This model has {config.n_heads} head(s) (indices 0–{config.n_heads - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Encode seed text
    # ------------------------------------------------------------------
    seed_text = args.text
    known_chars = [ch for ch in seed_text if ch in stoi]
    unknown_chars = [ch for ch in seed_text if ch not in stoi]

    if unknown_chars:
        print(
            f"  Warning: the following characters are not in the vocabulary "
            f"and will be skipped: {unknown_chars}"
        )

    if not known_chars:
        print(
            "ERROR: None of the characters in --text are present in the "
            "model's vocabulary. Try a different seed string.",
            file=sys.stderr,
        )
        sys.exit(1)

    ids = [stoi[ch] for ch in known_chars]
    token_labels = list(known_chars)  # one label per token position

    # Truncate to seq_len if the seed is too long
    max_len = config.seq_len
    if len(ids) > max_len:
        print(
            f"  Seed text has {len(ids)} tokens; truncating to seq_len={max_len}."
        )
        ids = ids[:max_len]
        token_labels = token_labels[:max_len]

    idx = torch.tensor([ids], dtype=torch.long)  # (1, T)

    # ------------------------------------------------------------------
    # Forward pass — collect attention weights
    # ------------------------------------------------------------------
    print("Running forward pass …")
    with torch.no_grad():
        _logits, _loss, attn_weights_list = model(idx)

    # attn_weights_list: list of tensors, one per layer
    # Each tensor shape: (batch, n_heads, T, T)
    if not attn_weights_list:
        print(
            "ERROR: The model did not return attention weights. "
            "Check that your GPT.forward() includes them in its return value.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Captured attention weights from {len(attn_weights_list)} layer(s).")

    # ------------------------------------------------------------------
    # Attention heatmap
    # ------------------------------------------------------------------
    attn_path = os.path.join(
        args.output_dir,
        f"attention_layer{args.layer}_head{args.head}.png",
    )

    plot_attention_heatmap(
        attn_weights=attn_weights_list,
        tokens=token_labels,
        layer=args.layer,
        head=args.head,
        save_path=attn_path,
    )
    print(f"Saved attention heatmap to: {attn_path}")

    # ------------------------------------------------------------------
    # Positional encoding plot (bonus — always save alongside attention)
    # ------------------------------------------------------------------
    pe_path = os.path.join(args.output_dir, "positional_encoding.png")
    print("Plotting positional encoding …")
    plot_positional_encoding(
        d_model=config.d_model,
        seq_len=config.seq_len,
        save_path=pe_path,
    )
    print(f"Saved positional encoding plot to: {pe_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
