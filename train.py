"""
train.py — Entry point for training a GPT-2-style transformer.

This script wires together:
  - src/data.py   : tokenization and DataLoaders
  - src/model.py  : GPT model definition and config
  - src/visualization.py : loss-curve plotting

Run example:
    python train.py --epochs 20 --d_model 128 --device auto
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data import get_dataloaders, TINY_TEXT
from src.model import GPT, GPTConfig
from src.visualization import plot_loss_curve


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small GPT-2-style transformer."
    )

    # Model architecture
    parser.add_argument("--d_model", type=int, default=128,
                        help="Embedding / hidden dimension (default: 128)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of transformer layers (default: 2)")

    # Data / training
    parser.add_argument("--seq_len", type=int, default=64,
                        help="Context (sequence) length (default: 64)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate for AdamW (default: 3e-4)")

    # Hardware / I/O
    parser.add_argument("--device", type=str, default="cpu",
                        help='Device: "cpu", "cuda", "mps", or "auto" to '
                             'pick best available (default: cpu)')
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to write checkpoints and plots "
                             "(default: checkpoints)")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save a checkpoint every N epochs (default: 5)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: torch.device) -> float:
    """Return average loss over the full validation set."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


# ---------------------------------------------------------------------------
# Text generation helper
# ---------------------------------------------------------------------------

def generate_sample(model: nn.Module, vocab: tuple, seed_text: str,
                    max_new_tokens: int, device: torch.device) -> str:
    """Encode seed_text, run model.generate, decode and return the result."""
    chars, stoi, itos = vocab

    # Encode — skip unknown characters silently
    ids = [stoi[ch] for ch in seed_text if ch in stoi]
    if not ids:
        return "(seed text contained no known characters)"

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out_idx = model.generate(idx, max_new_tokens=max_new_tokens,
                                 temperature=1.0, top_k=40)
    model.train()

    return "".join(itos[i] for i in out_idx[0].tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 50)
    print("         === GPT Training ===")
    print("=" * 50)
    print(f"  d_model   : {args.d_model}")
    print(f"  n_heads   : {args.n_heads}")
    print(f"  n_layers  : {args.n_layers}")
    print(f"  seq_len   : {args.seq_len}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs    : {args.epochs}")
    print(f"  lr        : {args.lr}")
    print(f"  device    : {device}")
    print(f"  checkpoint: {args.checkpoint_dir}/")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1/4] Building vocabulary and DataLoaders …")
    train_loader, val_loader, vocab = get_dataloaders(
        text=TINY_TEXT,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        split=0.9,
    )
    chars, stoi, itos = vocab
    vocab_size = len(stoi)
    print(f"      vocab size : {vocab_size}")
    print(f"      train batches: {len(train_loader)}, "
          f"val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n[2/4] Building model …")
    config = GPTConfig(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model = GPT(config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n[3/4] Training …\n")

    train_losses: list[float] = []   # one value per training step
    val_losses: list[float] = []     # one value per epoch

    def save_checkpoint(tag: str) -> None:
        path = os.path.join(args.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "vocab": vocab,
                "args": vars(args),
            },
            path,
        )
        print(f"      Checkpoint saved: {path}")

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch:>3}/{args.epochs}",
                unit="batch",
                leave=True,
            )

            for x, y in epoch_bar:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                _, loss, _ = model(x, targets=y)
                loss.backward()

                # Gradient clipping keeps training stable
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                step_loss = loss.item()
                train_losses.append(step_loss)

                perplexity = math.exp(min(step_loss, 20))  # clamp to avoid overflow
                epoch_bar.set_postfix(
                    loss=f"{step_loss:.4f}",
                    ppl=f"{perplexity:.2f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            # Validation at end of each epoch
            val_loss = evaluate(model, val_loader, device)
            val_losses.append(val_loss)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"      [Epoch {epoch}] val_loss={val_loss:.4f}  "
                  f"val_ppl={val_ppl:.2f}")

            # Periodic checkpoint
            if epoch % args.save_every == 0:
                save_checkpoint(f"epoch{epoch:03d}")

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving emergency checkpoint …")
        save_checkpoint("interrupted")
        print("Exiting cleanly.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Post-training
    # ------------------------------------------------------------------
    print("\n[4/4] Post-training steps …")

    # Save final checkpoint
    save_checkpoint("final")

    # Loss curve
    # val_losses has one value per epoch; train_losses has one per step.
    # Interpolate val_losses so it has the same length as train_losses for
    # side-by-side plotting.
    steps_per_epoch = len(train_loader)
    val_interp: list[float] = []
    for epoch_idx, vl in enumerate(val_losses):
        val_interp.extend([vl] * steps_per_epoch)
    # Trim / pad to exact length in case of rounding
    val_interp = val_interp[: len(train_losses)]
    while len(val_interp) < len(train_losses):
        val_interp.append(val_interp[-1] if val_interp else 0.0)

    loss_curve_path = os.path.join(args.checkpoint_dir, "loss_curve.png")
    plot_loss_curve(train_losses, val_interp, save_path=loss_curve_path)
    print(f"      Loss curve saved: {loss_curve_path}")

    # Sample generation
    seed = "To be or not"
    print(f"\n--- Generated text (seed: '{seed}') ---")
    sample = generate_sample(
        model, vocab, seed_text=seed,
        max_new_tokens=200, device=device,
    )
    print(sample)
    print("--------------------------------------\n")

    print("Training complete.")


if __name__ == "__main__":
    main()
