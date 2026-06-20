"""Scaled dot-product attention and multi-head attention module."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure NumPy implementation — shows the math with no framework magic
# ---------------------------------------------------------------------------

def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention using NumPy.

    This is the exact formula from "Attention Is All You Need":
        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Args:
        Q:    numpy array of shape (seq_len, d_k) — queries
        K:    numpy array of shape (seq_len, d_k) — keys
        V:    numpy array of shape (seq_len, d_k) — values
        mask: optional boolean array of shape (seq_len, seq_len)
              True means "block this position" (e.g. future tokens)

    Returns:
        output:  numpy array of shape (seq_len, d_k)
        weights: numpy array of shape (seq_len, seq_len) — attention map
    """
    seq_len, d_k = Q.shape  # (seq_len, d_k)

    # Step 1: Dot-product between every query and every key.
    # Q @ K^T → shape (seq_len, seq_len)
    # Entry [i, j] = how much query i attends to key j.
    scores = Q @ K.T                     # (seq_len, seq_len)

    # Step 2: Scale down so the dot products don't grow too large.
    # Large dot products push softmax into regions with tiny gradients.
    scores = scores / math.sqrt(d_k)    # (seq_len, seq_len)

    # Step 3: Apply mask (if provided).
    # Adding a very large negative number makes those positions ≈ 0 after softmax.
    if mask is not None:
        scores = scores + mask * -1e9   # (seq_len, seq_len)

    # Step 4: Softmax along the key dimension (axis=-1).
    # Each row now sums to 1 — a probability distribution over values.
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))  # numerically stable
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)     # (seq_len, seq_len)

    # Step 5: Weighted sum of values.
    # Each output token is a blend of all value vectors, weighted by attention.
    output = weights @ V                # (seq_len, d_k)

    return output, weights


# ---------------------------------------------------------------------------
# PyTorch implementation — same logic, now with batches and multiple heads
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention using PyTorch.

    Args:
        Q:    tensor of shape (batch, heads, seq_len, d_k)
        K:    tensor of shape (batch, heads, seq_len, d_k)
        V:    tensor of shape (batch, heads, seq_len, d_k)
        mask: optional tensor of shape (1, 1, seq_len, seq_len)
              True positions will be masked out (set to -1e9 before softmax)

    Returns:
        output:  tensor of shape (batch, heads, seq_len, d_k)
        weights: tensor of shape (batch, heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)  # last dimension of Q

    # Step 1: Q K^T
    # We use transpose(-2, -1) to swap the last two dims of K.
    # (batch, heads, seq_len, d_k) @ (batch, heads, d_k, seq_len)
    # → (batch, heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, T, T)

    # Step 2: Scale
    scores = scores / math.sqrt(d_k)               # (B, H, T, T)

    # Step 3: Apply causal mask if provided.
    # mask is True where attention should be blocked (future positions).
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))  # (B, H, T, T)

    # Step 4: Softmax along the last dim (over keys).
    weights = F.softmax(scores, dim=-1)             # (B, H, T, T)

    # Step 5: Weighted sum of values.
    # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, d_k)
    # → (batch, heads, seq_len, d_k)
    output = torch.matmul(weights, V)              # (B, H, T, d_k)

    return output, weights


# ---------------------------------------------------------------------------
# Multi-Head Attention module
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in "Attention Is All You Need".

    Instead of one big attention function, we run h smaller attention
    operations in parallel (the "heads"), then concatenate and project
    the results. This lets the model attend to different aspects of the
    input at different positions simultaneously.
    """

    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model:  total embedding dimension (e.g. 128)
            n_heads:  number of parallel attention heads (e.g. 4)
                      d_model must be divisible by n_heads
        """
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head, e.g. 32

        # Four learned linear projections — no bias (common in transformers).
        # Each maps the full d_model to d_model; we then split across heads.
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # value projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # output projection

        # Stored so callers can visualize what the model attended to.
        self.attn_weights = None

    def forward(self, x, mask=None):
        """
        Args:
            x:    input tensor of shape (B, T, d_model)
            mask: optional causal mask of shape (1, 1, T, T)

        Returns:
            output:       tensor of shape (B, T, d_model)
            attn_weights: tensor of shape (B, n_heads, T, T)
        """
        B, T, _ = x.shape  # batch size, sequence length, d_model

        # --- Project inputs into Q, K, V ---
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)  # (B, T, d_model)
        V = self.W_v(x)  # (B, T, d_model)

        # --- Split d_model across n_heads ---
        # We reshape (B, T, d_model) → (B, T, n_heads, d_k)
        # then transpose to (B, n_heads, T, d_k) so each head
        # has its own (T, d_k) slice to work with independently.
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)

        # --- Run scaled dot-product attention in parallel across all heads ---
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        # attn_out:     (B, H, T, d_k)
        # attn_weights: (B, H, T, T)

        # Store for later visualization (detach to avoid keeping computation graph)
        self.attn_weights = attn_weights.detach()

        # --- Reassemble heads ---
        # Reverse the transpose and reshape back to (B, T, d_model).
        # contiguous() is needed before view() when the tensor isn't contiguous in memory.
        attn_out = attn_out.transpose(1, 2).contiguous()  # (B, T, H, d_k)
        attn_out = attn_out.view(B, T, self.d_model)       # (B, T, d_model)

        # --- Final linear projection ---
        output = self.W_o(attn_out)  # (B, T, d_model)

        return output, attn_weights


# ---------------------------------------------------------------------------
# Causal (autoregressive) mask
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int, device):
    """
    Build a boolean upper-triangular mask for autoregressive (causal) attention.

    Position i should only see positions 0 .. i, never future positions.
    We mark future positions as True so that masked_fill sets them to -inf.

    Returns:
        mask: boolean tensor of shape (1, 1, seq_len, seq_len)
              True  → this position is MASKED (blocked)
              False → this position is allowed

    Example for seq_len=4:
        [[False, True,  True,  True ],
         [False, False, True,  True ],
         [False, False, False, True ],
         [False, False, False, False]]
    """
    # torch.ones creates an all-True matrix; triu(..., diagonal=1) keeps
    # only the strict upper triangle (the future positions).
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)   # (seq_len, seq_len)

    # Add batch and head dimensions so it broadcasts over (B, H, T, T).
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    return mask
