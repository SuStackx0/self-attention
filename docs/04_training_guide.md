# Training Guide

## 1. Quick Start

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the default tiny model (runs in under 5 minutes on CPU):

```bash
python train.py
```

Visualize attention patterns from a saved checkpoint:

```bash
python visualize.py --checkpoint checkpoints/checkpoint_epoch_20.pt
```

To view only the positional encoding heatmap without loading a model:

```bash
python visualize.py --pe_only
```

The training script saves a checkpoint after each epoch to the `checkpoints/` directory. If training is interrupted, you can inspect any saved checkpoint with `visualize.py`.

---

## 2. What the Model is Learning

This project trains a **character-level language model** on the Shakespeare dataset. The task is simple: given the last `seq_len` characters, predict the next character. Every forward pass produces one prediction per position, and the model is trained to maximize the probability of the correct next character at every position simultaneously.

The loss function is **cross-entropy**:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P(\text{next char}_i \mid \text{context}_i)$$

Lower loss means the model is assigning higher probability to the correct next character. The loss is averaged over all positions in the batch.

**Perplexity** is a more interpretable version of the same quantity:

$$\text{Perplexity} = e^{\mathcal{L}}$$

You can think of perplexity as the effective number of characters the model is confused between at each step. If perplexity is 5, the model behaves as if it is picking uniformly among 5 equally likely options.

**The baseline to beat:** A model that knows nothing — just predicts every character with equal probability — achieves perplexity equal to the vocabulary size, which for Shakespeare characters is approximately 60 to 70. Any well-trained model should get well below this.

**What to expect:** After training the default tiny model for 20 epochs on the Shakespeare dataset, you should see train perplexity drop to roughly 3 to 5, and val perplexity to roughly 4 to 7. The model will generate text that looks superficially Shakespearean — correct capitalization, recognizable word patterns, occasional coherent phrases — even though it operates one character at a time.

---

## 3. Interpreting the Loss Curve

A **healthy training run** looks like this:

- Both train and val loss start high (near $\ln(\text{vocab\_size}) \approx 4.2$) and drop sharply in the first few epochs.
- After the initial drop, both curves continue to decrease more gradually.
- Val loss stays close to train loss, or slightly above it.

**Signs of trouble:**

| Symptom | Likely cause | What to try |
|---|---|---|
| Loss stays flat from the start | Learning rate too low, or bug in the training loop | Increase `lr` by 10x; add a print to confirm loss.backward() is being called |
| Loss oscillates wildly | Learning rate too high | Decrease `lr` by 10x |
| Loss decreases then suddenly spikes | Gradient explosion (rare with Pre-LN + AdamW) | Add gradient clipping: `torch.nn.utils.clip_grad_norm_` |
| Val loss much higher than train | Overfitting | Use a larger dataset; add dropout; reduce model size |
| Val loss slightly higher than train | Normal, especially for tiny models | This is expected; do not worry |

**Note on overfitting for this project:** The Shakespeare dataset is small (roughly 1 MB). A model with more than a few hundred thousand parameters will memorize it fairly quickly. Seeing val loss slightly above train loss is completely expected and does not indicate a problem. The goal here is to understand the architecture, not to squeeze out maximum generalization.

---

## 4. Training Hyperparameters: What Each Does

**`d_model` — Embedding dimension**

This is the width of every vector that flows through the model. Wider means more capacity to represent relationships between tokens. However, every weight matrix scales quadratically or better with `d_model`, so doubling `d_model` roughly quadruples the number of parameters in the attention layers and quadruples them in the feed-forward layers. Start small (64 or 128) to keep training fast.

**`n_heads` — Number of attention heads**

Each head gets `d_model / n_heads` dimensions to work with. More heads means more distinct "perspectives" on token relationships, but each head has a narrower view. The constraint is that `d_model` must be divisible by `n_heads`. Values of 2, 4, and 8 all work well for small models. Setting `n_heads = 1` is a useful ablation — you can visualize one attention pattern instead of many.

**`n_layers` — Number of transformer blocks**

Depth is how the model builds abstraction. Shallow models (1-2 layers) learn surface statistics; deeper models (4+ layers) can learn more compositional structure. For character-level Shakespeare, 2 to 4 layers is plenty. Adding more layers beyond this mostly adds parameters without improving the loss on this dataset.

**`seq_len` — Context window (sequence length)**

This is how many characters the model can see before making a prediction. Longer context means the model can use earlier parts of a sentence or paragraph. The computational cost of attention scales as $O(T^2)$ in sequence length — doubling `seq_len` from 64 to 128 quadruples the attention cost per forward pass. For this dataset, 64 to 128 is a reasonable range.

**`lr` — Learning rate for AdamW**

This controls how large each gradient update is. The AdamW optimizer adapts the learning rate per parameter, so this is really a scale factor on those adaptive updates. For small transformer models, `3e-4` (i.e., 0.0003) is a widely used default and a good starting point. If your loss drops very slowly, try `1e-3`. If it oscillates, try `1e-4`.

---

## 5. Reading Attention Heatmaps

The visualizer in `visualize.py` produces heatmaps of the attention weight matrices from one or more transformer blocks. Here is how to read them.

**Axes:**

- **Rows** are **query positions** — the token that is doing the attending. Row $i$ shows which tokens token $i$ is paying attention to.
- **Columns** are **key positions** — the tokens being attended to. Column $j$ shows which query tokens attend to token $j$.

**Color:**

- Brighter or warmer color = higher attention weight = this query-key pair has strong affinity.
- Darker color = near-zero attention weight = this query mostly ignores this key.

**What to look for in a well-trained model:**

- **Diagonal pattern:** Many heads attend strongly to the immediately preceding one or two tokens. This is the model using recent context, similar to an n-gram.
- **Periodic pattern:** If the text has regular structure (newlines, spaces after punctuation), some heads will attend at fixed offsets — for example, always attending to the most recent newline.
- **Long-range pattern:** Some heads spread attention widely, picking up on tokens much earlier in the context. These are the heads doing longer-range modeling.
- **Upper triangle is always zero.** This is the causal mask. If you see any nonzero values in the upper triangle (above the diagonal), something is wrong with the mask implementation.

**Comparing heads:** Run the visualizer with multiple heads visible side by side. Heads that look like identity matrices (strong diagonal) are attending locally. Heads with diffuse, spread-out patterns are attending globally. The diversity of patterns is exactly why multi-head attention works better than single-head attention.

---

## 6. Experimenting

Once you have a baseline run working, try these experiments in order. Each one changes one variable at a time, so you can understand what each component contributes.

**Experiment 1: Increase depth.**

Change `n_layers` from 2 to 4. Does val loss improve? Does it take longer to converge? This tests whether depth helps on this dataset. For Shakespeare at this scale, you will likely see a modest improvement.

**Experiment 2: Compare head counts.**

Run two training sessions side by side:

```bash
python train.py --n_heads 1
python train.py --n_heads 8
```

Then visualize the attention patterns from each. With `n_heads=1`, you get one heatmap. With `n_heads=8`, you get eight. How diverse are the patterns with 8 heads? Does a single head try to do everything at once?

**Experiment 3: Longer context.**

Change `seq_len` from 64 to 128. Does the generated text become more coherent? Does training take noticeably longer? Watch the attention heatmaps: do you see more long-range patterns emerge when the model has access to a longer context?

**Experiment 4: Use a GPU.**

If you have a CUDA-compatible GPU or Apple Silicon:

```bash
python train.py --device auto
```

The `--device auto` flag should select CUDA or MPS automatically. Training on GPU can be 10x to 50x faster, allowing you to run larger models or more epochs in the same wall-clock time.

---

## 7. Generating Text

After training, the `GPT.generate()` method in `src/model.py` produces text by sampling one character at a time:

1. Feed a starting string (prompt) through the model to get logits over the vocabulary.
2. Sample the next character from those logits.
3. Append the sampled character to the input and repeat.

Two parameters control the character of the output:

**Temperature**

The logits are divided by the temperature before softmax:

$$P(\text{next char}) = \text{softmax}\!\left(\frac{\text{logits}}{T}\right)$$

- `T < 1.0`: Sharpens the distribution — the most likely character becomes even more likely. Output is more predictable and repetitive, but rarely nonsensical.
- `T = 1.0`: Samples from the model's learned distribution directly.
- `T > 1.0`: Flattens the distribution — low-probability characters become more competitive. Output is more diverse and surprising, but also more prone to incoherence.

A temperature of 0.8 is a good starting point — slightly conservative, produces readable text with some variety.

**Top-k sampling**

Before sampling, keep only the top $k$ most probable next characters and set all others to zero probability (then renormalize). This prevents the model from ever sampling a character that the model considers very unlikely, which eliminates most of the "garbage" outputs that temperature alone cannot prevent.

Setting `k = 1` is equivalent to greedy decoding — always pick the most probable character. This is maximally deterministic but also maximally repetitive; the model tends to get stuck in loops. Values of `k = 10` to `k = 40` strike a good balance between coherence and variety.

To generate text from the command line after training:

```bash
python train.py --generate --prompt "ROMEO:" --temperature 0.8 --top_k 20
```

Try different prompts, temperatures, and top-k values. Notice how higher temperature makes the output more "creative" at the cost of occasional spelling errors and incoherent words.
