# The Journey, Station by Station

This is the condensed version of the learning path behind this repo — every station of a sentence's trip through a transformer, with hand-computable toy numbers. The article tells the story; this file is the technical spine.

Toy convention: vectors are 2-dimensional (real model: 768), machines are tiny dense layers with hand-picked weights. All logic is exact; only the sizes are shrunk.

---

## Station 0 — The problem

Computers only understand numbers. Every language system must walk this path:

```
language → numbers → computation → numbers → language
```

An LLM is a **L**arge **La**nguage **M**odel: a giant next-word guesser. Your phone keyboard and ChatGPT differ in *size*, not in *kind*. This repo's model guesses "positive/negative" instead of the next word — but the engine inside is the same: a transformer.

A dictionary of positive/negative words fails on the first real sentence:

```
"this movie was beautiful, exciting and NOT good"
dictionary: 3 positive vs 1 negative → POSITIVE ❌
```

Meaning lives in the *relations between words*, not in words alone. The transformer is a machine for seeing those relations.

## Station 1 — Why 2017 changed everything

Before: **RNNs** read word-by-word, dragging a memory along.

Two fatal flaws:
1. **Goldfish memory** — distant words fade; by the time "was sick" arrives, "cat" (6 steps back) is nearly gone.
2. **The forced queue** — step *n* cannot start before *n−1*. A thousand GPUs, one working. Training huge models was physically impossible.

**"Attention Is All You Need" (2017):** drop the chain; let every word look at every other word *directly and simultaneously*. Distance becomes free; training becomes parallel; models scale; LLMs are born.

The price: attention cost grows with the *square* of length (10 words → 100 looks; 1000 words → 1,000,000). That square is exactly why every model has a **context length** ceiling.

## Station 2 — Text to numbers (two separate steps)

**Step A — tokenization.** Split text into pieces. Letters are too fine (no meaning), whole words too coarse (huge vocab, OOV blindness, `play`/`playing` unrelated). Winner: **subwords** — keep frequent words whole, split rare ones into familiar pieces:

```
love      → [love]
lovingly  → [loving, ##ly]      ## = "glued to the previous piece"
```

OOV dies: any unseen word decomposes into known pieces.

**Step B — vocabulary lookup.** Each piece gets its row number in a ~30k-entry vocabulary:

```
[CLS]=101  i=1045  do=2079  not=2025  love=2293  this=2023  movie=3185  [SEP]=102
```

These numbers are **seat numbers** — pure addresses, zero meaning. Row 2293 is not "closer" to row 2294 in any meaningful way.

## Station 3 — Embedding: where meaning is born

A single number per word fails (where does "pizza" sit on a positive↔negative axis?). Words need a **list** of numbers — a vector — one number per latent feature:

```
          [positivity, intensity, human-ness, food-ness]   (illustrative!)
love    → [   0.9,        0.8,       0.9,       0.1    ]
hate    → [  -0.9,        0.8,       0.7,       0.1    ]
pizza   → [   0.3,        0.2,       0.1,       0.9    ]
```

Now meaning = **location**: similar words are neighbors, and relations become directions:

```
king − man + woman ≈ queen
Paris − France + Italy ≈ Rome
```

The embedding table (30k × 768) is **learned during training, only read at inference** — a dictionary written once, consulted forever.

**The crack:** one row per token. "bank" gets ONE fixed vector, stuck between 🏦 and 🏞️. The table indexes by token, not by sentence — it *structurally cannot* disambiguate. Enter attention.

## Station 4 — Attention: score, percent, mix

Similarity of two vectors = multiply matching numbers, add up (dot product). Big = related.

Toy sentence: `put money in the bank`, keeping content words:

```
put   = [0, 1]
money = [2, 0]
bank  = [1, 1]     ← the torn-between-two-worlds vector
```

Bank resolves itself in three steps:

```
1. SCORE:    bank·put = 1,  bank·money = 2,  bank·bank = 2
2. PERCENT:  softmax → put 15.6%, money 42.2%, bank 42.2%
3. MIX:      new_bank = Σ share × vector → leans toward money 🏦
```

Same table, sentence with `river` instead of `money` → same three steps push bank toward the shore. **The table says what a word is in general; attention says what it is in this sentence.**

## Station 5 — Q, K, V: relevance ≠ similarity

A verb needs its subject without *resembling* it. Raw-vector scoring can't see that. Fix: before attention, each word makes three specialized versions of itself through three shared dense layers (weights learned in training):

```
Q (query):  what am I looking for?
K (key):    what do I advertise?
V (value):  what do I hand over if attended to?
```

Library analogy: your request (Q) is matched against book spines (K); what you take home is the book's content (V).

Scoring becomes `Q(me) · K(others)`; mixing collects `V(others)`. One Q-maker, one K-maker, one V-maker per head — shared across all words, applied one word at a time.

Full toy pass (Q-maker swaps the two numbers, K-maker is identity, V-maker doubles):

```
        Q       K       V
put    [1,0]   [0,1]   [0,2]
money  [0,2]   [2,0]   [4,0]
bank   [1,1]   [1,1]   [2,2]

bank's turn:
score:   Q(bank)·K(put)=1   Q(bank)·K(money)=2   Q(bank)·K(bank)=2
softmax: 15.6% / 42.2% / 42.2%
mix:     0.156×[0,2] + 0.422×[4,0] + 0.422×[2,2] ≈ [2.5, 1.2]  → money-ish ✅
```

## Station 6 — Softmax, exactly

```
softmax(xᵢ) = e^xᵢ / Σ e^xⱼ        (e ≈ 2.718)
```

Exponentiate (this *exaggerates* gaps), then divide by the sum. Scores 1,2,2:

```
e¹=2.72  e²=7.39  e²=7.39   sum=17.5   →   15.6% / 42.2% / 42.2%
```

Two properties that matter later:
- **Temperature:** divide scores by T before softmax. Big T → flat/indifferent; small T → sharp/decisive. (The same "temperature" knob you see in LLM APIs.)
- **No escape:** outputs always sum to exactly 100%. "I attend to nothing" does not exist. Remember this.

Negative logits are harmless: e^(−2)=0.14 — a small positive share, never zero, never negative.

## Station 7 — Positional encoding: order matters

Attention alone is order-blind: "dog chased cat" and "cat chased dog" contain the same vectors and produce identical scores — a bag of words. (RNNs had order for free; we dropped it with the chain.)

Fix: **input = word vector + position vector.** Each seat (1st, 2nd, 3rd…) has its own learned signature. Same word, different seat → different input → different Q/K/V → different attention. Bonus: Q-makers can now learn *positional* questions ("looking for a noun *before* me" = English subjects).

## Station 8 — Layers: six re-readings

The whole recipe (embed+position → QKV → score÷√d → softmax → mix) runs **6 times in a stack** (DistilBERT; BERT has 12 — a hyperparameter). Layer n+1 takes layer n's *cooked* vectors as input: each pass builds on the previous understanding — like re-reading a difficult text. Probing shows early layers capture grammar/neighbors, late layers capture semantics — exactly like CNN layers going edges → textures → faces.

Each layer also has **12 heads**: twelve parallel, independent attentions with their own Q/K/V makers, each free to specialize (negation, subjects, neighbors…). Their outputs are concatenated. The report's heatmaps are head-averages.

```
6 layers × 12 heads × (8 tokens × 8 targets) — for our sentence: 4,608 attention shares
```

## Station 9 — The two invisible passengers

The tokenizer adds two artificial tokens:

**[SEP]** — end/boundary marker. In single-sentence tasks: nearly jobless. Hold that thought.

**[CLS]** — the *meeting secretary*. A token representing no word, participating in all 6 layers of attention. Because it has no meaning of its own, all it can do is **collect**. Its embedding row stores not a meaning but a **learned mission** — the question it walks in with ("where are the sentiment-bearing words?"). After layer 6, its vector = the sentence summary. All other vectors are discarded; only [CLS] proceeds to the verdict.

## Station 10 — The attention sink (our favorite discovery)

The report's layer-6 heatmap shows every word dumping 50–67% of its attention on [SEP]. Solved with two facts you now own:

1. softmax forbids "attend to nothing" — 100% must be spent;
2. by layer 6, most words are **saturated** — their ambiguity was resolved layers ago.

A saturated word forced to spend attention would *contaminate* its clean vector by borrowing meaningful V's. The escape: pour it into the one token whose V is ≈ nothing. **[SEP] becomes the attention trash bin.** Nobody designed this; training discovered it. (Literature name: *attention sink / no-op attention*.)

Corollary: judge a transformer by its middle layers, not its last.

## Station 11 — The verdict

```
[CLS] vector (768) → dense, 2 neurons → logits → softmax → verdict
```

Each neuron asks one learned question of the summary ("how negative does it smell?" / "how positive?"). Real numbers from the report, sentence "I do not love this movie":

```
logits:   NEG +1.19   POS −1.19
softmax:  e^1.19=3.29  e^−1.19=0.30   sum=3.59
          NEG 91.5%    POS 8.5%
verdict:  NEGATIVE ☹️
```

Note: softmax only cares about the **gap** (2.38), not the absolute values — (+5.00, +2.62) gives the same 91.5/8.5.

## Station 12 — The two families

| | BERT family (this repo) | GPT family (ChatGPT, Claude) |
|---|---|---|
| mission | judge finished text | continue unfinished text |
| attention | bidirectional | masked ("no peeking at the future") |
| summary | [CLS] secretary | none — last word's vector holds the past |
| head | dense 2 → verdict | dense ~30k → next-word distribution, looped |

The judge reads the whole file; the writer only knows what's written so far. For sentiment (a verdict on finished text, where a late "not" can flip everything), the bidirectional reader wins.

## Station 13 — Where the weights came from

**Act 1 — Pretraining (not us):** random weights + millions of sentences + one game: fill the blank (`the ___ roared in the jungle`). To win the game you are forced to learn the language. No human labels — raw text teaches itself. Result: meaningful embeddings, smart Q/K/V makers.

**Act 2 — Fine-tuning (us, ~3 minutes):** replace the fill-the-blank head with 2 fresh random neurons (`newly initialized` in the training log), show 3000 labeled IMDB reviews, 3 epochs. Old weights shift slightly; new neurons learn their job. 90% accuracy from 3000 examples — because the language was already there.

> Pretraining is literacy; fine-tuning is a job. We hired a literate model and told it what to focus on.
