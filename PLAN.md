# Lab 1 — Part 1 Plan: N-Gram Language Model

## Goal
Train add-one-smoothed N-gram models (n = 1…10) on *Julius Caesar* + *Macbeth*,
report perplexity on *Hamlet*, and generate sample text.

## Project layout
Everything lives in **one Jupyter notebook**. Part 2 will be added later as new
sections of the same notebook, so the shared helpers defined in Part 1 can be
called directly.

```
Lab1/
├── requirements.txt      # nltk, numpy, matplotlib, pandas, jupyter (+ scikit-learn, datasets later)
├── lab1.ipynb            # Part 1 now; Part 2 sections appended later
└── results/              # saved perplexity CSV + plot
```

## Notebook structure

### 0. Setup
- Markdown title cell.
- Imports, plus `nltk.download` for `gutenberg`, `punkt`, and `stopwords`.
- A fixed random seed.

### 1. Shared helpers (Part 2 reuses these)
- `normalize_token(tok)`: lowercase, then a regex strips anything that isn't a letter, digit, or apostrophe.
- `preprocess_sentence(text_or_tokens, remove_stopwords=False) -> list[str]`: accepts a raw string (Part 2 sentences) or a token list (NLTK). Drops tokens that end up empty.
- `pad(tokens, n)`: adds `n-1` × `<s>` at the start and one `</s>` at the end.
- `make_ngrams(tokens, n)`: returns n-gram tuples. Part 2.2 reuses it for bi-gram features.

### 2. Data loading & preprocessing
- Load `gutenberg.sents(...)` for Caesar, Macbeth, and Hamlet, so sentence boundaries are kept.
- Show a few sentences before and after preprocessing.
- Corpus stats: sentence and token counts, vocabulary size, and the OOV rate on Hamlet.

### 3. Vocabulary
- Built from the **training data only**.
- Words with count < `min_count` (default 2) become `<UNK>`. Unknown Hamlet words also map to `<UNK>`.
- `V` includes `</s>` and `<UNK>` but not `<s>`, since the model never predicts `<s>`.

### 4. `NGramLM` class
- `fit(sentences)`: counts `(context, word)` pairs and contexts in `Counter`s, plus `next_words[ctx]` for generation.
- `prob(w, ctx)` = `(C(ctx, w) + k) / (C(ctx) + k·V)`.
  With `k=1` this is Laplace smoothing. For a unigram the context is `()`.
- `perplexity(sentences)` = `exp(−Σ log P / N)` over every predicted token, including `</s>`.
- `generate(max_len, seed, strategy="sample" | "top_k")`: a NumPy distribution over the vocabulary, sampled until `</s>`.

### 5. Sanity checks
These are assert cells on a toy corpus:
- Probabilities sum to 1 for both a seen and an unseen context.
- An unseen n-gram gets P > 0.
- Toy perplexity matches a hand-computed value.

### 6. Unigram baseline
- Train the model, report perplexity, and show generated samples.

### 7. Bigram → 10-gram comparison
- Perplexity on Hamlet and on the training set for n = 1…10.
- Show a table, a line plot, and save both to `results/`.

### 8. Text generation
- Samples for n = 1, 2, 3, 5, 10.

### 9. Extra experiments (optional)
- Add-k with k ∈ {1, 0.1, 0.01}.
- Stopwords removed vs. kept.
- The fraction of Hamlet n-grams never seen in training, for each n.

### 10. Discussion (markdown)
- Why perplexity bottoms out at small n and then climbs.
- Why the high-n samples are either memorized or random.
- The role of smoothing.

## Environment
- Create `.venv`, install `requirements.txt`, and register it as a Jupyter kernel.
- `git init` inside `Lab1/` (right now the folder sits inside a repo rooted at the home directory).

## Done when
`lab1.ipynb` runs top-to-bottom with every sanity check passing, and it produces:
- the perplexity table and plot for n = 1…10
- the generated samples
- the written discussion
