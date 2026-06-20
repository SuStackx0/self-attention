"""Plotting utilities for attention heatmaps, positional encoding, and training curves."""

import numpy as np
import matplotlib.pyplot as plt

from .positional_encoding import get_encoding


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(attn_weights, tokens, layer=0, head=0, save_path=None):
    """
    Visualise what a single attention head looked at.

    Args:
        attn_weights: list of tensors, one per transformer layer.
                      Each tensor has shape (B, n_heads, T, T).
                      attn_weights[layer][batch, head, query, key] = weight.
        tokens:       list of strings — the characters/tokens for axis labels.
                      Length should match T (the sequence length).
        layer:        which transformer layer to visualise (0-indexed).
        head:         which attention head to visualise (0-indexed).
        save_path:    if given, save figure to this path (PNG); otherwise show.

    The heatmap rows = query positions, columns = key positions.
    A bright cell (i, j) means position i attended strongly to position j.
    """
    # --- Extract the weight matrix for the chosen layer and head ---
    # attn_weights[layer] shape: (B, n_heads, T, T)
    weights_tensor = attn_weights[layer]           # (B, n_heads, T, T)
    weights_tensor = weights_tensor[0]             # pick first batch item → (n_heads, T, T)
    weights_matrix = weights_tensor[head]          # pick chosen head → (T, T)

    # Convert to NumPy for matplotlib
    weights_np = weights_matrix.cpu().numpy()      # (T, T)
    T = weights_np.shape[0]                        # sequence length

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(weights_np, cmap='Blues', aspect='auto', vmin=0.0, vmax=1.0)

    # Add colourbar on the right
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Only label axes if the sequence is short enough to be readable.
    # More than 30 tokens makes tick labels overlap badly.
    if T <= 30 and tokens is not None:
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(
            [repr(t)[1:-1] for t in tokens[:T]],   # escape special chars like newline
            rotation=45,
            ha='right',
            fontsize=9,
        )
        ax.set_yticklabels(
            [repr(t)[1:-1] for t in tokens[:T]],
            fontsize=9,
        )
    else:
        # Too many tokens — skip labels to avoid clutter
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Key position", fontsize=11)
        ax.set_ylabel("Query position", fontsize=11)

    ax.set_title(f"Attention Layer {layer}, Head {head}", fontsize=13, pad=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()          # free memory — important in long training runs
    else:
        plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# Positional encoding visualisation
# ---------------------------------------------------------------------------

def plot_positional_encoding(d_model: int = 64, seq_len: int = 50, save_path=None):
    """
    Visualise the sinusoidal positional encoding matrix.

    Each row is a position (0 to seq_len-1).
    Each column is a dimension (0 to d_model-1).
    The alternating sin/cos pattern is clearly visible, with lower dimensions
    oscillating quickly (high frequency) and higher dimensions slowly.

    Args:
        d_model:   embedding dimension for the encoding (default 64)
        seq_len:   number of positions to show (default 50)
        save_path: if given, save figure; otherwise show.
    """
    # Compute the encoding — returns shape (seq_len, d_model)
    pe = get_encoding(seq_len, d_model)

    fig, ax = plt.subplots(figsize=(10, 5))

    # RdBu_r is a diverging colormap centred at zero — good for values in [-1, 1]
    im = ax.imshow(pe, cmap='RdBu_r', aspect='auto', vmin=-1.0, vmax=1.0)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Dimension", fontsize=11)
    ax.set_ylabel("Position", fontsize=11)
    ax.set_title("Sinusoidal Positional Encoding", fontsize=13, pad=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# Training / validation loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses: list, val_losses: list, save_path=None):
    """
    Plot training and validation loss over training steps.

    Args:
        train_losses: list of floats — training loss recorded at each logging step.
        val_losses:   list of floats — validation loss at each logging step.
                      Should have the same length as train_losses.
        save_path:    if given, save figure; otherwise show.

    Tips for reading the plot:
        - Both curves should decrease early in training.
        - If val_loss flattens while train_loss keeps falling → overfitting.
        - If both are flat from the start → learning rate may be too low.
    """
    steps = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, train_losses, color='steelblue',   linestyle='-',
            linewidth=2, label='Train loss')
    ax.plot(steps, val_losses,   color='darkorange',  linestyle='--',
            linewidth=2, label='Val loss')

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training Loss", fontsize=13, pad=12)

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
