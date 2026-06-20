# GPT-2 Architecture

## 1. Overview: How Blocks Stack

The full GPT-2 model is a sequence of stages, each outputting a tensor of the same shape `(B, T, d_model)` — batch size, sequence length, embedding dimension. Because every stage preserves this shape, blocks can be stacked arbitrarily deep.

Here is the data flow from input tokens to output logits:

```
Input token IDs   shape: (B, T)
       |
  Token Embedding      -> (B, T, d_model)
       |
  + Positional Encoding   (added in-place)
       |
  TransformerBlock 1   -> (B, T, d_model)
       |
  TransformerBlock 2   -> (B, T, d_model)
       |
      ...
       |
  TransformerBlock N   -> (B, T, d_model)
       |
  LayerNorm (final)    -> (B, T, d_model)
       |
  LM Head (linear)     -> (B, T, vocab_size)
       |
  Output logits
```

Each transformer block does two things: mixes information across tokens (attention), and processes each token independently with extra capacity (feed-forward). This interleaving is what lets the model build up progressively more abstract representations with each layer.

---

## 2. Inside a Transformer Block

A transformer block has two sub-layers:

1. **Multi-head self-attention** — lets every token look at every previous token.
2. **Feed-forward network (FFN)** — a two-layer MLP applied independently to each token position.

Both sub-layers use **residual connections**: instead of replacing the input with the sub-layer's output, they add the output back to the input.

$$x \leftarrow x + \text{Sublayer}(\cdots)$$

This means the block is always computing a "correction" to add to the current representation, rather than computing the representation from scratch. This turns out to be much easier to train.

Each sub-layer also uses **Layer Normalization** to stabilize the distribution of activations. Whether the normalization goes before or after the sub-layer is a key design choice.

---

## 3. Pre-LayerNorm vs. Post-LayerNorm

The original "Attention Is All You Need" paper (Vaswani et al., 2017) used **Post-LayerNorm**, where normalization is applied after the residual addition:

```
Post-LN:  x = LayerNorm(x + Sublayer(x))
```

GPT-2 switched to **Pre-LayerNorm**, where normalization is applied to the input before the sub-layer:

```
Pre-LN:   x = x + Sublayer(LayerNorm(x))
```

This project follows the GPT-2 Pre-LN convention.

Why does it matter? In Post-LN, the residual path passes through LayerNorm, which modifies the magnitude of the gradients flowing backward. In very deep networks, this can cause gradients to vanish at the earlier layers. Pre-LN keeps a clean residual path — gradients can flow backward through the additions without passing through any normalization. This makes training significantly more stable without requiring carefully tuned learning rate warmup schedules.

A practical consequence: with Post-LN, training GPT-2-scale models often required careful warmup over thousands of steps before the model would learn at all. With Pre-LN, the model trains smoothly from the start.

---

## 4. Feed-Forward Network

After attention mixes information across token positions, the feed-forward network processes each token position independently. It has no cross-token interaction — it is the same two-layer MLP applied at every position separately.

The structure is:

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

where $W_1$ expands from $d_{model}$ to $4 \cdot d_{model}$, and $W_2$ projects back from $4 \cdot d_{model}$ to $d_{model}$. The factor of 4 is a convention from the original paper and has been retained in GPT-2.

**Why GELU?** GELU (Gaussian Error Linear Unit) is defined as:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the CDF of the standard normal distribution. In practice it is approximated. Compared to ReLU, GELU is smooth and non-zero for slightly negative inputs, which tends to give better gradient flow and slightly better performance on language tasks. GPT-2 uses GELU; this project does the same.

**Why a 4x expansion?** After attention compresses all contextual information into a $d_{model}$-dimensional vector, the FFN is an opportunity to do nonlinear processing with more capacity. Widening to $4 \cdot d_{model}$ before projecting back gives the network room to decompose the representation into many features, apply nonlinear transformations to each, and recompose. Empirically this ratio works well; it has been kept as a default across most transformer variants.

---

## 5. Residual Connections: Why They Matter

Without residual connections, a deep network looks like this in the backward pass:

$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_N} \cdot \prod_{i=1}^{N} \frac{\partial x_i}{\partial x_{i-1}}$$

Each factor $\partial x_i / \partial x_{i-1}$ is a Jacobian matrix. If these are consistently smaller than 1 in magnitude, the product shrinks exponentially with depth — **vanishing gradients**. The first layers receive near-zero gradient signal and barely learn.

With residual connections, each layer computes $x_{i+1} = x_i + F_i(x_i)$, so:

$$\frac{\partial x_{i+1}}{\partial x_i} = I + \frac{\partial F_i}{\partial x_i}$$

The identity matrix $I$ ensures there is always a direct gradient path, regardless of what $F_i$'s Jacobian looks like. Even if $F_i$'s gradients are tiny, the identity term carries a gradient of magnitude 1 straight through. This is sometimes called the **gradient highway**.

This is why ResNets (for images) and transformers (for text) can be trained to depths of dozens or even hundreds of layers. The residual connection is one of the most important architectural innovations in deep learning.

---

## 6. Weight Tying

In a language model, two components both deal with the vocabulary:

1. The **token embedding matrix** $E$ of shape `(vocab_size, d_model)`, which maps token IDs to $d_{model}$-dimensional vectors at the input.
2. The **LM head** (linear layer) of shape `(d_model, vocab_size)`, which maps the final hidden state back to logits over the vocabulary at the output.

Weight tying means these two matrices share the same underlying data: `lm_head.weight = token_embedding.weight`. The LM head is literally the transpose of the embedding matrix.

**Why does this work?** The geometric intuition is that the embedding space encodes similarity between tokens — tokens with similar meanings are nearby. The LM head's job is to measure how similar the current hidden state is to each possible next token. Using the same geometry for both encoding and decoding is consistent: if "cat" and "kitten" are nearby in embedding space, then predicting "kitten" should be almost as easy as predicting "cat" when the context is about cats.

**Practical effects:**

- Fewer parameters. The embedding matrix can be large (e.g., `50257 × 768 ≈ 38M` parameters for GPT-2 small). Tying saves those parameters entirely.
- Often improves performance, especially when the model is parameter-constrained. The shared parameters act as a regularizer, preventing the input and output representations from diverging.

---

## 7. What to Look at in the Code

**`TransformerBlock.forward()` in `src/transformer.py`**

This is the clearest place to see the Pre-LN pattern in action. You should find two blocks that look like:

```python
x = x + self.attention(self.ln1(x), mask)
x = x + self.ff(self.ln2(x))
```

The `self.ln1` and `self.ln2` are the LayerNorm layers applied *before* their respective sub-layers. The result of the sub-layer is added back to `x` (the residual). The output shape matches the input shape exactly.

**`FeedForward` in `src/transformer.py`**

This class implements the two-layer MLP. Confirm the expansion factor: the first linear layer maps `d_model -> 4 * d_model`, the activation is GELU, and the second linear layer maps `4 * d_model -> d_model`.

**`GPT.__init__()` in `src/model.py`**

Look for the line that ties the weights. It should read something like:

```python
self.lm_head.weight = self.token_embedding.weight
```

After this assignment, both attributes point to the same tensor in memory. Any gradient update that touches the embedding will also update the LM head, and vice versa.

**`GPT.forward()` in `src/model.py`**

Trace the full forward pass: embedding lookup, add positional encoding, pass through each block in a loop, apply final LayerNorm, project through LM head. When a `targets` argument is provided, the forward pass also computes the cross-entropy loss between the logits and the target token IDs — this is what the training loop minimizes.
