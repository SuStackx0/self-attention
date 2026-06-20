# Positional Encoding

## 1. The Problem: Transformers Are Order-Blind

Consider how scaled dot-product attention works. Each token produces a query, and that query is compared against every key via a dot product. The dot product between two vectors does not depend on where those vectors appear in a list — it depends only on their values. The attention mechanism treats the input as a **set**, not a **sequence**.

This means that if you shuffle the tokens in your input, the attention weights change, but only because different token embeddings now occupy different positions in the matrix — the mechanism itself has no notion of "which token came first."

Here is a concrete example. Suppose your vocabulary has embeddings for "dog," "bites," and "man." The sentences "dog bites man" and "man bites dog" have exactly the same tokens. Without positional information, the model receives the same set of three embeddings in either case and cannot distinguish between them. It cannot know that the subject is different.

For a language model this is a serious problem. Word order carries most of the meaning in English (and nearly all of it in some other languages). A model that cannot see order cannot learn grammar.

---

## 2. The Solution: Inject Position Information

The fix is to add a **position-dependent signal** to each token's embedding before the token enters the transformer. If the embedding for token $t$ at position $pos$ is $e_t$, the model actually receives:

$$x_{pos} = e_t + PE_{pos}$$

where $PE_{pos}$ is a vector that is uniquely determined by $pos$ and has the same dimension $d_{model}$ as the token embedding.

For this to work well, the positional encoding needs a few properties:

1. **Uniqueness.** Each position must produce a distinct vector. Two different positions must not look the same.
2. **Boundedness.** The positional vectors should not be so large that they overwhelm the semantic content of the token embedding.
3. **Generalization.** Ideally, the encoding should work for any sequence length, including lengths longer than those seen during training.
4. **Structure.** Nearby positions should have similar encodings; distant positions should be more different. The model should be able to infer relative distance from the encoding.

Sinusoidal positional encoding satisfies all four of these properties.

---

## 3. Sinusoidal Encoding: The Formula

For a position $pos$ and a dimension index $i$ (where $i$ runs from 0 to $d_{model}/2 - 1$), the encoding is:

$$PE_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Even dimensions get a sine, odd dimensions get a cosine, and each pair $(2i, 2i+1)$ uses the same frequency $\omega_i = 10000^{-2i/d_{model}}$.

**Intuition: a multi-scale clock.**

Think of how a binary number encodes a position. The lowest bit flips every step (fast oscillation, fine-grained). The next bit flips every two steps. The next every four steps. And so on. Each bit tracks position at a different scale, and together they uniquely identify any integer.

Sinusoidal encoding does the same thing but continuously. Dimension pair $i=0$ uses frequency $\omega_0 = 1/10000^0 = 1.0$ — the fastest oscillation. Dimension pair $i = d_{model}/2 - 1$ uses frequency $\omega_{max} = 10000^{-1} = 0.0001$ — the slowest oscillation, almost flat across typical sequence lengths.

Low-index dimensions distinguish fine-grained position (is this the 3rd or 4th token?). High-index dimensions carry coarse position information (is this token in the first half or second half of the sequence?).

**Why sine and cosine together?**

Using a pair (sine, cosine) instead of just one function has a useful algebraic property. For any fixed offset $k$, the encoding at position $pos + k$ can be written as a linear combination of the encodings at position $pos$:

$$\sin(\omega(pos + k)) = \sin(\omega \cdot pos)\cos(\omega k) + \cos(\omega \cdot pos)\sin(\omega k)$$

$$\cos(\omega(pos + k)) = \cos(\omega \cdot pos)\cos(\omega k) - \sin(\omega \cdot pos)\sin(\omega k)$$

In matrix form: $PE_{pos+k} = R_k \cdot PE_{pos}$ for a rotation matrix $R_k$ that depends only on $k$, not on $pos$. This means the model can potentially learn to compute relative positions $k$ using simple linear operations on the positional encodings — a useful inductive bias.

---

## 4. What the Encoding Looks Like

If you visualize the positional encoding matrix as an image — rows are positions (0 to max_seq_len), columns are dimensions (0 to $d_{model}$) — you see alternating vertical stripes.

The leftmost columns oscillate rapidly from positive to negative as you move down the rows (fast sine waves). The rightmost columns change very slowly (nearly constant across most sequence lengths). The overall pattern looks like a stack of sinusoidal waves at decreasing frequencies laid side by side.

To see this directly, run:

```
python visualize.py --pe_only
```

This will plot the encoding matrix as a heatmap. Notice that no two rows are identical — every position has a unique "fingerprint" across the dimension axis.

---

## 5. Why Not Learned Positional Embeddings?

Learned positional embeddings replace the formula with a lookup table: a matrix of shape `(max_seq_len, d_model)` initialized randomly and trained by gradient descent, just like token embeddings.

This is simpler to implement and often performs equally well or better on fixed-length tasks. GPT-2 (the original paper) actually uses learned positional embeddings.

This project uses sinusoidal encoding instead, for three reasons:

1. **Pedagogical clarity.** The formula makes explicit what the encoding is doing. Learned embeddings are a black box.
2. **Length generalization.** Sinusoidal encoding produces a vector for any position, even positions beyond $max\_seq\_len$. Learned embeddings have no entry for positions they have never seen.
3. **No extra parameters.** Sinusoidal encoding is parameter-free. Learned embeddings add $max\_seq\_len \times d_{model}$ parameters (e.g., $1024 \times 512 = 524{,}288$ extra weights for GPT-2 small).

The practical difference for a small educational project is negligible. For production models trained on very long documents, learned absolute positional embeddings have mostly been replaced by relative positional schemes (RoPE, ALiBi) that handle arbitrary lengths more gracefully. Those are beyond the scope of this project.

---

## 6. What to Look at in the Code

**`get_encoding()` in `src/positional_encoding.py`**

This function computes the entire $PE$ matrix from scratch using NumPy or PyTorch. Find the line that computes the frequencies $\omega_i = 10000^{-2i/d_{model}}$. Notice that this is often computed in log space for numerical stability:

$$\omega_i = \exp\!\left(-\frac{2i}{d_{model}} \ln(10000)\right)$$

Then find where `torch.sin` is applied to even indices and `torch.cos` to odd indices. Match these lines to the two-part formula above. The indexing `[:, 0::2]` selects even columns; `[:, 1::2]` selects odd columns.

**`SinusoidalPositionalEncoding.forward()` in `src/positional_encoding.py`**

This is the PyTorch module that wraps `get_encoding()`. Notice that the positional encoding matrix is registered as a **buffer** (`self.register_buffer(...)`) rather than a parameter. A buffer is a tensor that lives on the correct device (CPU or GPU) and is saved/loaded with the model checkpoint, but it is not updated by the optimizer. The encoding is fixed — it is math, not learned weights.

The `forward()` method simply slices `PE[:seq_len, :]` (to handle variable-length inputs shorter than the maximum) and adds it to the incoming token embeddings.
