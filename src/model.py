"""Full GPT model: token embedding + positional encoding + N transformer blocks + language model head."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import make_causal_mask
from .positional_encoding import SinusoidalPositionalEncoding
from .transformer import TransformerBlock


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """
    Holds all hyperparameters for the GPT model.

    Using a dataclass makes it easy to pass config around and log it.

    Defaults are intentionally small so training fits on a CPU:
        vocab_size: number of unique tokens (characters, in our case)
        seq_len:    maximum number of tokens the model processes at once
        d_model:    embedding dimension (width of the model)
        n_heads:    number of attention heads (must divide d_model evenly)
        n_layers:   depth of the model (number of stacked transformer blocks)
        dropout:    regularisation strength (0 = no dropout)
    """
    vocab_size: int
    seq_len:    int   = 128
    d_model:    int   = 128
    n_heads:    int   = 4
    n_layers:   int   = 2
    dropout:    float = 0.1


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    A small GPT-style language model.

    Architecture (top to bottom):
        1. Token embedding         — maps each token id to a d_model vector
        2. Positional encoding     — adds position information (sinusoidal)
        3. N × TransformerBlock    — self-attention + feed-forward, stacked
        4. Final LayerNorm         — stabilises activations before the head
        5. LM head (Linear)        — projects d_model → vocab_size (logits)

    Weight tying:
        The LM head and token embedding share the same weight matrix.
        This is a well-known trick from "Using the Output Embedding to Improve
        Language Models" (Press & Wolf, 2017) and is used in GPT-2. It reduces
        the parameter count and often improves performance.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # --- Layers ---
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # token_embedding.weight shape: (vocab_size, d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.seq_len,
            dropout=config.dropout,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Final layer norm applied after all blocks (GPT-2 style)
        self.ln_f = nn.LayerNorm(config.d_model)

        # Language model head: maps each position's embedding to a vocab distribution
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # --- Weight tying ---
        # Share the embedding matrix with the output projection.
        # Intuition: the same geometric space that maps token→vector at the input
        # is used to score vector→token at the output.
        self.lm_head.weight = self.token_embedding.weight
        # shared weight shape: (vocab_size, d_model)

        # --- Initialise weights ---
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialise Linear and Embedding weights with a small normal distribution.

        std=0.02 is the value used in GPT-2. It keeps activations from growing
        too large at the start of training (avoids exploding gradients).
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass through the GPT model.

        Args:
            idx:     token indices, shape (B, T) — integers in [0, vocab_size)
            targets: optional target indices, shape (B, T) — for computing loss
                     Each targets[b, t] is the "next token" after idx[b, t].

        Returns:
            logits:          (B, T, vocab_size) — raw (unnormalised) predictions
            loss:            scalar cross-entropy loss, or None if targets not given
            all_attn_weights: list of length n_layers, each (B, n_heads, T, T)
        """
        B, T = idx.shape  # batch size, sequence length

        # Sanity check: sequence must fit within the positional encoding
        assert T <= self.config.seq_len, (
            f"Sequence length {T} exceeds maximum {self.config.seq_len}"
        )

        # --- Step 1: Token embedding ---
        # Look up each token id in the embedding table.
        x = self.token_embedding(idx)  # (B, T, d_model)

        # --- Step 2: Add positional encoding ---
        # The model needs to know where each token is in the sequence.
        x = self.pos_encoding(x)       # (B, T, d_model)

        # --- Step 3: Build causal mask ---
        # Prevents each position from attending to future positions.
        mask = make_causal_mask(T, device=idx.device)  # (1, 1, T, T)

        # --- Step 4: Pass through all transformer blocks ---
        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask=mask)  # (B, T, d_model), (B, H, T, T)
            all_attn_weights.append(attn_weights)

        # --- Step 5: Final layer norm ---
        x = self.ln_f(x)  # (B, T, d_model)

        # --- Step 6: Project to vocabulary ---
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # --- Step 7: Compute loss if targets are given ---
        loss = None
        if targets is not None:
            # F.cross_entropy expects:
            #   input:  (N, C) — N samples, C classes
            #   target: (N,)   — integer class index for each sample
            # We flatten (B, T) into B*T samples.
            loss = F.cross_entropy(
                logits.view(B * T, self.config.vocab_size),  # (B*T, vocab_size)
                targets.view(B * T),                         # (B*T,)
            )

        return logits, loss, all_attn_weights

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressively generate new tokens given a starting context.

        At each step:
            1. Run a forward pass on the current sequence (cropped to seq_len)
            2. Take the logits at the very last position (the "next token" prediction)
            3. Optionally apply temperature scaling and top-k filtering
            4. Sample from the resulting distribution
            5. Append the sampled token and repeat

        Args:
            idx:            starting token indices, shape (B, T)
            max_new_tokens: how many tokens to generate
            temperature:    > 1 → more random; < 1 → more greedy; 1 = unmodified
            top_k:          if set, only sample from the k most likely tokens

        Returns:
            idx:  extended token indices, shape (B, T + max_new_tokens)
        """
        self.eval()  # turn off dropout during generation

        for _ in range(max_new_tokens):

            # --- Crop context if it's longer than what the model supports ---
            # We can only process seq_len tokens at a time.
            idx_cropped = idx[:, -self.config.seq_len:]  # (B, T_cropped)

            # --- Forward pass (no targets needed, so loss will be None) ---
            logits, _, _ = self(idx_cropped)
            # logits shape: (B, T_cropped, vocab_size)

            # --- Take logits at the last position only ---
            # That's where the model predicts what comes next.
            logits = logits[:, -1, :]  # (B, vocab_size)

            # --- Temperature scaling ---
            # Dividing by temperature < 1 makes the distribution sharper (more confident).
            # Dividing by temperature > 1 flattens it (more random).
            logits = logits / temperature  # (B, vocab_size)

            # --- Top-k filtering (optional) ---
            # Zero out all logits except the k largest, then re-normalise via softmax.
            # This prevents the model from sampling very unlikely tokens.
            if top_k is not None:
                # Find the k-th largest logit value per batch item.
                # topk returns (values, indices); we only need values.
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                                                                        # (B, k)
                # The threshold is the smallest value in the top-k.
                threshold = top_k_values[:, -1].unsqueeze(-1)           # (B, 1)
                # Set all logits below the threshold to -inf so softmax ignores them.
                logits = logits.masked_fill(logits < threshold, float('-inf'))
                # (B, vocab_size)

            # --- Softmax → probability distribution ---
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # --- Sample one token from the distribution ---
            # torch.multinomial draws one sample per row according to probs.
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # --- Append sampled token to the running sequence ---
            idx = torch.cat([idx, next_token], dim=1)  # (B, T+1) → (B, T+max_new_tokens)

        return idx  # (B, T + max_new_tokens)
