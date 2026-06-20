"""Transformer block: pre-LayerNorm style (GPT-2), with multi-head attention and feed-forward network."""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applied independently to each token position. Architecture:
        Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model) → Dropout

    The 4× expansion is standard from the original transformer paper.
    GELU (Gaussian Error Linear Unit) is used in GPT-2 instead of ReLU —
    it has smoother gradients near zero.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: embedding dimension (input and output size)
            dropout: regularisation applied after the second linear layer
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # expand:  (d_model → 4*d_model)
            nn.GELU(),                          # activation
            nn.Linear(4 * d_model, d_model),   # project: (4*d_model → d_model)
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, T, d_model)

        Returns:
            tensor of shape (B, T, d_model) — same shape, each position processed independently
        """
        return self.net(x)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# Transformer Block (GPT-2 / Pre-LN style)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One Transformer block in the GPT-2 (pre-LayerNorm) style.

    Pre-LN vs Post-LN
    -----------------
    The original "Attention Is All You Need" paper used Post-LN:
        x = LayerNorm(x + sublayer(x))

    GPT-2 switched to Pre-LN:
        x = x + sublayer(LayerNorm(x))

    Pre-LN is more stable during training because the residual stream
    (the main path carrying information) is never passed through LayerNorm
    directly. The gradient flows through the residual connection without
    scaling, which avoids vanishing/exploding gradients in deep networks.
    This means Pre-LN usually trains without learning-rate warm-up.

    Architecture:
        x → LayerNorm → MultiHeadAttention → + (residual) → x
        x → LayerNorm → FeedForward        → + (residual) → x
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model:  embedding dimension
            n_heads:  number of attention heads
            dropout:  applied inside attention and feed-forward
        """
        super().__init__()

        # LayerNorm before each sub-layer (Pre-LN style)
        self.ln1 = nn.LayerNorm(d_model)  # normalises before attention
        self.ln2 = nn.LayerNorm(d_model)  # normalises before feed-forward

        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff   = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x:    tensor of shape (B, T, d_model)
            mask: optional causal mask of shape (1, 1, T, T)

        Returns:
            x:            tensor of shape (B, T, d_model)
            attn_weights: tensor of shape (B, n_heads, T, T)
        """
        # --- Sub-layer 1: Multi-Head Attention (with Pre-LN and residual) ---
        # Normalise first, then attend, then add to the original x (residual).
        # The residual connection lets gradients flow unchanged through
        # the block, making very deep networks much easier to train.
        attn_out, attn_weights = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out   # (B, T, d_model) — residual connection

        # --- Sub-layer 2: Feed-Forward (with Pre-LN and residual) ---
        x = x + self.ff(self.ln2(x))  # (B, T, d_model) — residual connection

        return x, attn_weights  # (B, T, d_model), (B, H, T, T)
