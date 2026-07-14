# A Sentence's Journey Through a Transformer 🚂

*What actually happens between typing "I do not love this movie" and reading "NEGATIVE, 91.5%"?*

This repo fine-tunes DistilBERT for sentiment analysis — and then **opens the box**: a companion script traces one sentence through every station of the model and renders an HTML report with attention heatmaps, so you can *see* the machine think.

Companion project to the article **["A Sentence's Journey Through a Transformer"](https://mohammadjebelli.com/a-sentences-journey-through-a-transformer/)**.
Sister project: [cnn-cat-dog-explained](https://github.com/mohammad-jebelli/cnn-cat-dog-explained) — the same idea, for images.

---

## The journey in one picture

```
"I do not love this movie"
        │
        ▼  tokenizer splits text, adds two invisible passengers
[CLS] i do not love this movie [SEP]
        │
        ▼  vocabulary lookup (addresses, no meaning yet)
[101, 1045, 2079, 2025, 2293, 2023, 3185, 102]
        │
        ▼  embedding table + positional stamp (meaning + seat number)
8 raw vectors × 768 numbers
        │
        ▼  6 layers × 12 heads of self-attention
        │  each word asks (Q), advertises (K), hands over content (V)
8 sentence-aware vectors
        │
        ▼  keep only [CLS] — the meeting secretary's summary
1 summary vector
        │
        ▼  dense (2 neurons) → logits [+1.19, −1.19]
        ▼  softmax → [91.5%, 8.5%]
        │
        ▼
NEGATIVE ☹️  (91.5% confident)
```

Every arrow above is explained in the article, computed by hand with toy numbers, and then shown on the real model by `explain_sentence.py`.

## What's in here

| File | What it does |
|---|---|
| `train_model.py` | Fine-tunes `distilbert-base-uncased` on IMDB (3k samples, ~3 min on a laptop GPU, ~90% accuracy). Best-epoch saving, dynamic padding, fp16, history plot. |
| `predict.py` | Quick sentiment check for any sentence, using *your* trained model. |
| `explain_sentence.py` | ⭐ The star. Traces one sentence through every stage and writes `explain_output/report.html`: token table, embedding strips, attention heatmaps (first & last layer), what [CLS] listened to, logits & softmax. |
| `requirements.txt` | Dependencies (PyTorch installed separately — see below). |
| `docs/JOURNEY.md` | The full station-by-station walkthrough with the toy-number calculations from the article. |
| `images/` | The report images for the sample sentence, so you can browse without running anything. |

## Quickstart

```bash
python -m venv venv && source venv/bin/activate

# 1) PyTorch first — pick the line matching your machine:
pip install torch --index-url https://download.pytorch.org/whl/cu121   # NVIDIA GPU
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU only

# 2) The rest:
pip install -r requirements.txt

# 3) Train (downloads DistilBERT + IMDB on first run):
python train_model.py

# 4) Open the box:
python explain_sentence.py "I do not love this movie"
# → open explain_output/report.html in a browser
```

> Behind a restricted network? Route through your proxy:
> `HTTPS_PROXY=... python train_model.py` or `proxychains4 python train_model.py`

## Three things you'll see in the report that surprised us

**1. Two invisible passengers.** You type 6 words; 8 tokens board the train. `[CLS]` is an artificial token whose *learned mission* is to listen to everyone and carry the sentence summary — only its vector reaches the decision layer. `[SEP]` just marks the end… or so it seems.

**2. The attention sink.** In the last layer's heatmap, almost every word dumps 50–67% of its attention onto `[SEP]`. Not a bug: softmax forces every word to spend exactly 100% attention, but by layer 6 most words are *saturated* — nothing left to resolve. The model discovered its own escape valve: pour the mandatory attention into the one token that carries no meaning, so nothing gets contaminated. Nobody designed this; training found it.

**3. The real work happens early.** Layer 1 shows words studying their neighbors (`do → not`, `this → love`); layer 6 is mostly leftovers. Judging a transformer by its last layer is like judging a meeting by its final 30 seconds.

## The two families (why this model isn't ChatGPT)

Same engine, opposite missions:

| | BERT family (this repo) | GPT family (ChatGPT, Claude) |
|---|---|---|
| Mission | **judge** a finished text | **continue** an unfinished text |
| Attention | bidirectional — every word sees all | masked — no peeking at the future |
| Summary token | `[CLS]` (the secretary) | none — last word's vector suffices |
| Final layer | dense, 2 neurons → verdict | dense, ~30k neurons → next-word odds |

*The reader vs. the writer. The judge reads the whole case file; the writer only knows what's written so far.*

## Results

| epoch | train loss | eval accuracy | eval F1 |
|---|---|---|---|
| 1 | 0.455 | 89.2% | 0.896 |
| 2 | 0.214 | 89.5% | 0.899 |
| 3 | 0.124 | **90.2%** | **0.905** |

~163 s total on an RTX 4050 laptop GPU (3000 train / 3000 test IMDB samples, fp16, dynamic padding).

## License

MIT — use it, break it, teach with it.
