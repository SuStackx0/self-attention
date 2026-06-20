"""Sinusoidal positional encoding — classic formula from 'Attention Is All You Need'."""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pure NumPy helper — useful for visualisation and understanding
# ---------------------------------------------------------------------------

def get_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Compute sinusoidal positional encodings as a NumPy array.

    Formula from the paper:
        PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    where pos is the position in the sequence and i is the dimension index.

    Args:
        seq_len: number of positions to encode
        d_model: embedding dimension

    Returns:
        pe: numpy array of shape (seq_len, d_model)
    """
    # Column vector of positions: [[0], [1], ..., [seq_len-1]]
    positions = np.arange(seq_len).reshape(-1, 1)    # (seq_len, 1)

    # Row vector of even dimension indices: [0, 2, 4, ..., d_model-2]
    # We only need half the dimensions because each pair (2i, 2i+1) shares
    # the same denominator.
    dim_indices = np.arange(0, d_model, 2)           # (d_model/2,)

    # Compute the denominator: 10000^(2i / d_model)
    # This creates slowly varying sinusoids for large i (high frequencies for small i).
    denominators = np.power(10000.0, dim_indices / d_model)  # (d_model/2,)

    # Compute the angles: pos / denominator
    # Broadcasting: (seq_len, 1) / (d_model/2,) → (seq_len, d_model/2)
    angles = positions / denominators                # (seq_len, d_model/2)

    # Allocate output array
    pe = np.zeros((seq_len, d_model))               # (seq_len, d_model)

    # Even columns (0, 2, 4, ...) → sine
    pe[:, 0::2] = np.sin(angles)                    # (seq_len, d_model/2)

    # Odd columns (1, 3, 5, ...) → cosine
    # Handle the edge case where d_model is odd (cos needs one fewer column)
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])  # (seq_len, d_model/2)

    return pe  # (seq_len, d_model)


# ---------------------------------------------------------------------------
# PyTorch module — wraps the encoding as a non-trainable buffer
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as a PyTorch module.

    The encoding is NOT learned — it is fixed at construction time.
    We register it as a 'buffer' so it:
      - moves to the correct device with .to(device)
      - is saved and loaded with the model's state_dict
      - is NOT updated by the optimizer

    Usage:
        pe = SinusoidalPositionalEncoding(d_model=128, max_seq_len=512)
        x_with_pos = pe(x)  # x: (B, T, d_model)
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model:     embedding dimension
            max_seq_len: maximum sequence length the model will ever see
            dropout:     applied after adding positional encoding (regularisation)
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Compute the full encoding matrix up to max_seq_len.
        pe_numpy = get_encoding(max_seq_len, d_model)  # (max_seq_len, d_model)

        # Convert to a PyTorch tensor.
        pe_tensor = torch.tensor(pe_numpy, dtype=torch.float32)  # (max_seq_len, d_model)

        # Add a batch dimension so it broadcasts: (1, max_seq_len, d_model)
        pe_tensor = pe_tensor.unsqueeze(0)  # (1, max_seq_len, d_model)

        # Register as a buffer (not a parameter — the optimizer ignores it).
        self.register_buffer('pe', pe_tensor)
        # self.pe is now accessible as self.pe, shape (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.

        Args:
            x: token embeddings, shape (B, T, d_model)

        Returns:
            x + positional_encoding, shape (B, T, d_model), after dropout
        """
        B, T, d_model = x.shape  # (B, T, d_model)

        # Slice only the positions we need (up to T).
        # self.pe shape: (1, max_seq_len, d_model)
        # self.pe[:, :T, :] shape: (1, T, d_model) — broadcasts over batch dim B
        x = x + self.pe[:, :T, :]  # (B, T, d_model)

        return self.dropout(x)     # (B, T, d_model)
