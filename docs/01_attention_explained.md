# Attention Explained

## 1. The Problem: Why Do We Need Attention?

When you read the sentence "The cat sat on the mat because it was tired," your brain instantly resolves "it" to "the cat," not "the mat." You do this by scanning back through the sentence and recognizing that a cat can be tired, a mat cannot. You are, in a sense, attending to the word "cat" more than any other word when you process "it."

Early sequence models (RNNs and LSTMs) processed tokens one by one, left to right, and compressed the entire history into a single fixed-size vector before passing it forward. By the time the model reached "tired," the signal about "the cat" at the beginning of the sentence had been diluted through many steps of compression. Long-range dependencies were genuinely hard to learn.

Attention is a mechanism that lets every token look directly at every other token in the sequence and decide how much weight to give it — regardless of distance. Instead of a compressed summary, each token has direct access to all previous tokens. This solves the long-range dependency problem cleanly.

---

## 2. Query, Key, Value: The Core Idea

Here is an analogy that makes the machinery concrete.

Imagine a library. You walk in with a search query — say, "books about the French Revolution." The library has a catalog with index cards (keys) for each book. You compare your query against every index card to find the best matches, then you retrieve the actual books (values) for the matches you care about most.

Attention works the same way:

- **Query (Q):** What this token is "looking for." When processing "it," the query encodes something like "find the noun this pronoun refers to."
- **Key (K):** What each token "advertises" about itself. "cat" advertises noun-hood, animacy, etc.
- **Value (V):** The actual information a token contributes once selected.

In practice, each token's embedding is linearly projected into three separate vectors Q, K, and V through learned weight matrices $W^Q$, $W^K$, and $W^V$:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

where $X$ is the matrix of token embeddings, one row per token. The projections are learned during training — the model figures out what "looking for" and "advertising" should mean.

---

## 3. Scaled Dot-Product Attention: The Math

Once you have Q, K, V, the computation has four steps.

**Step 1: Compute compatibility scores.**

For each query token, compute a dot product with every key token:

$$\text{scores} = QK^T$$

The dot product measures how well-aligned two vectors are. If query $q_i$ and key $k_j$ point in similar directions in high-dimensional space, their dot product is large. This gives a raw unnormalized score for how much token $i$ should attend to token $j$.

**Step 2: Scale to prevent saturation.**

Divide every score by $\sqrt{d_k}$, where $d_k$ is the dimension of the key vectors:

$$\text{scaled scores} = \frac{QK^T}{\sqrt{d_k}}$$

Here is why this matters. Each element of the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ is a sum of $d_k$ terms. Even if each term has unit variance, the variance of the sum grows as $d_k$, so the magnitude of the dot products grows like $\sqrt{d_k}$. When scores are very large in magnitude, softmax (the next step) pushes its output toward one-hot — nearly all weight on a single token, near-zero gradients everywhere else. Dividing by $\sqrt{d_k}$ keeps the scores in a reasonable range and maintains useful gradients.

**Step 3: Softmax to get attention weights.**

$$\text{weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

Softmax converts raw scores into a probability distribution over the keys. Each row of the weight matrix sums to 1. The model is saying: "for query token $i$, distribute 100% of my attention budget across all key tokens."

**Step 4: Weighted sum of values.**

$$\text{output} = \text{weights} \cdot V$$

Multiply the attention weights by the value matrix. Each output token is a weighted blend of all value vectors. Tokens with high attention weights contribute more to the output; tokens with near-zero weight contribute almost nothing.

**Putting it all together:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This is the entire core formula. It is elegant because it is differentiable end-to-end — gradients flow through the softmax and back into $W^Q$, $W^K$, $W^V$, letting the model learn what to attend to.

---

## 4. The Causal Mask

In a language model, at inference time, token $i$ has not seen token $i+1$ yet — you generate the sequence left to right. If during training you allowed every token to see all other tokens (including future ones), the model would cheat by just copying the answer from the next position. The model would never learn to actually predict.

The solution is a **causal mask**: before applying softmax, set all positions in the upper triangle of the score matrix to $-\infty$:

$$\text{score}[i][j] = -\infty \quad \text{if } j > i$$

When softmax sees $-\infty$, it outputs exactly 0 (since $e^{-\infty} = 0$). Position $i$ ends up with zero weight on all future positions $j > i$ and can only attend to positions $0$ through $i$. This is called **masked self-attention** or **causal attention**.

Concretely, the mask is a boolean upper-triangular matrix. For a sequence of length 4:

```
Position can attend to:
       0    1    2    3
  0  [ OK  --   --   -- ]
  1  [ OK  OK   --   -- ]
  2  [ OK  OK   OK   -- ]
  3  [ OK  OK   OK   OK ]
```

The `--` positions are masked to $-\infty$ before softmax.

---

## 5. Multi-Head Attention

A single attention operation computes one set of relationships between tokens. But a sentence has many kinds of relationships simultaneously. "The cat sat on the mat because it was tired" involves coreference (it → cat), syntactic structure (sat → subject cat), and spatial relationships (sat → on the mat). A single attention head is unlikely to capture all of these at once.

**Multi-head attention** runs several attention operations in parallel, each on a lower-dimensional subspace of the embeddings:

1. Project Q, K, V into $h$ different subspaces using $h$ different sets of learned weight matrices.
2. Run scaled dot-product attention independently in each subspace.
3. Concatenate the $h$ outputs along the feature dimension.
4. Project the concatenated output back to $d_{model}$ with another learned matrix $W^O$.

The formula:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)$$

If $d_{model} = 512$ and $h = 8$, each head operates in dimension $d_k = 512 / 8 = 64$. The total computation is similar to one large attention at full dimension, but now the model has 8 independent "perspectives" on the token relationships.

In practice, different heads specialize. In trained language models, researchers have found heads that track syntactic subject-verb agreement, heads that track positional offsets (always attend to the previous token), and heads that track coreference. The model discovers these specializations on its own.

---

## 6. What to Look at in the Code

**`scaled_dot_product_attention_numpy()` in `src/attention.py`**

This is the pedagogical version using NumPy — no batching, no heads, no GPU. Trace through it line by line and match each operation to the four steps above. Find where the $\sqrt{d_k}$ scaling happens. Find where the optional mask is applied. This is the simplest possible implementation of the formula.

**`scaled_dot_product_attention()` in `src/attention.py`**

This is the same computation but written in PyTorch with full batching. The input tensors now have shape `(batch_size, n_heads, seq_len, d_k)`. The einsum or matmul that computes $QK^T$ now operates over the last two dimensions simultaneously for all batches and all heads. Confirm that the math is identical to the NumPy version — only the tensor layout changed.

**`MultiHeadAttention.forward()` in `src/attention.py`**

Watch how the input of shape `(B, T, d_model)` is projected and then reshaped into `(B, n_heads, T, d_k)`. This reshape is how the "split into heads" is implemented — it is just a view operation, not a for loop. After attention, the heads are concatenated by reshaping back to `(B, T, d_model)` and passed through $W^O$.

**`make_causal_mask()` in `src/attention.py`**

This function constructs the upper-triangular boolean mask. Look at how `torch.tril` (lower triangle) is used: positions where the lower triangle is 0 are positions that should be masked. The mask is then used to fill score positions with a large negative number before softmax.
