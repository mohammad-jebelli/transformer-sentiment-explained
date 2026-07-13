"""
explain_sentence.py
===================
Take ONE sentence, push it through the fine-tuned model, and capture what
happens at *every* station of the journey. Then build an HTML report that
shows the whole trip -- from raw text to the final decision.

This is the sentiment-analysis parallel of the Cat/Dog `explain_prediction.py`:
where that one showed the feature maps of each Conv layer, this one shows the
attention maps of each Transformer layer -- i.e. which word looked at which.

Run:
    python explain_sentence.py "I do not love this movie"
    python explain_sentence.py            # uses a default sentence

Outputs:
    explain_output/report.html   <- the full report (open this in a browser)
    explain_output/*.png         <- per-step images
"""

import sys
import os
import html

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed -- save straight to file
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_PATH = "sentiment-model"      # the folder train_model.py produced
OUT_DIR = "explain_output"
DEFAULT_SENTENCE = "I do not love this movie"

os.makedirs(OUT_DIR, exist_ok=True)

sentence = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_SENTENCE

# ---------------------------------------------------------------------------
# 1) Load the fine-tuned model + tokenizer
#    output_attentions=True is the key: it makes the model hand back the
#    attention weights of every layer, not just the final answer.
# ---------------------------------------------------------------------------
if not os.path.isdir(MODEL_PATH):
    print(f"[ERROR] '{MODEL_PATH}/' not found. Run train_model.py first.")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, output_attentions=True
)
model.eval()

id2label = model.config.id2label

# ---------------------------------------------------------------------------
# STATION 1-2: text -> tokens
#    Note the tokenizer adds [CLS] at the front and [SEP] at the end.
# ---------------------------------------------------------------------------
tokens_plain = tokenizer.tokenize(sentence)              # just the words
inputs = tokenizer(sentence, return_tensors="pt")        # the real thing
input_ids = inputs["input_ids"][0]
tokens_full = tokenizer.convert_ids_to_tokens(input_ids)  # with [CLS]/[SEP]
ids_full = input_ids.tolist()

# ---------------------------------------------------------------------------
# STATION 3: tokens -> embedding vectors (before any attention)
# ---------------------------------------------------------------------------
with torch.no_grad():
    raw_embeddings = model.distilbert.embeddings.word_embeddings(inputs["input_ids"])
emb = raw_embeddings[0].numpy()   # shape: (n_tokens, 768)

# ---------------------------------------------------------------------------
# STATION 4-7: full forward pass -- attention, [CLS] vector, logits, softmax
# ---------------------------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

logits = outputs.logits[0]                       # the two raw scores
probs = torch.softmax(logits, dim=-1)            # -> percentages
pred_id = int(torch.argmax(probs))
pred_label = id2label[pred_id]

# attentions: tuple of length n_layers, each (batch, n_heads, n_tok, n_tok)
attentions = outputs.attentions
n_layers = len(attentions)
n_heads = attentions[0].shape[1]

# the final [CLS] vector -- the "summary of the whole sentence"
cls_vector = outputs.hidden_states[-1][0][0].numpy()   # (768,)

# ---------------------------------------------------------------------------
# Helper: draw one attention heatmap (rows = who is looking, cols = looked at)
# ---------------------------------------------------------------------------
def draw_attention(matrix, tokens, title, path):
    fig, ax = plt.subplots(figsize=(1.0 * len(tokens) + 2, 1.0 * len(tokens) + 1.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=matrix.max())
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("looked AT (Key)", fontsize=9)
    ax.set_ylabel("who is looking (Query)", fontsize=9)
    ax.set_title(title, fontsize=11)
    # write the percentage inside each cell
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < matrix.max() * 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Draw: embeddings, attention (first layer, last layer, head-average), CLS
# ---------------------------------------------------------------------------
# (a) raw embedding vectors -- one row per token, first 60 dimensions
fig, ax = plt.subplots(figsize=(11, 0.5 * len(tokens_full) + 1.5))
ax.imshow(emb[:, :60], aspect="auto", cmap="RdBu")
ax.set_yticks(range(len(tokens_full)))
ax.set_yticklabels(tokens_full)
ax.set_xlabel("first 60 of the 768 dimensions")
ax.set_title("Embeddings: every token is now a vector (before attention)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/embeddings.png", dpi=110); plt.close(fig)

# (b) attention of the FIRST layer, averaged over its heads
first_att = attentions[0][0].mean(dim=0).numpy()
draw_attention(first_att, tokens_full,
               "Layer 1 attention (average of all heads)",
               f"{OUT_DIR}/attention_first.png")

# (c) attention of the LAST layer, averaged over its heads
last_att = attentions[-1][0].mean(dim=0).numpy()
draw_attention(last_att, tokens_full,
               f"Layer {n_layers} attention (average of all heads)",
               f"{OUT_DIR}/attention_last.png")

# (d) what [CLS] paid attention to in the LAST layer -- the summary decision
cls_att = last_att[0]  # row 0 = [CLS] as the Query
fig, ax = plt.subplots(figsize=(1.1 * len(tokens_full) + 2, 3))
bars = ax.bar(range(len(tokens_full)), cls_att, color="#4c72b0")
ax.set_xticks(range(len(tokens_full)))
ax.set_xticklabels(tokens_full, rotation=45, ha="right")
ax.set_ylabel("attention weight")
ax.set_title("What [CLS] listened to (last layer) -- this becomes the summary")
for b, v in zip(bars, cls_att):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
            ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/cls_attention.png", dpi=110); plt.close(fig)

# (e) the final [CLS] summary vector
fig, ax = plt.subplots(figsize=(11, 1.6))
ax.imshow(cls_vector[:80].reshape(1, -1), aspect="auto", cmap="RdBu")
ax.set_yticks([]); ax.set_xlabel("first 80 of the 768 dimensions")
ax.set_title("The final [CLS] vector -- the whole sentence squeezed into one vector")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/cls_vector.png", dpi=110); plt.close(fig)

# (f) logits and probabilities
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
labels = [id2label[i] for i in range(len(logits))]
c = ["#c44e52", "#55a868"]
ax1.bar(labels, logits.numpy(), color=c)
ax1.set_title("Logits (raw scores, before softmax)")
ax1.axhline(0, color="black", linewidth=0.8)
for i, v in enumerate(logits.numpy()):
    ax1.text(i, v, f"{v:.2f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=10)
ax2.bar(labels, probs.numpy() * 100, color=c)
ax2.set_title("After softmax (percentages)")
ax2.set_ylim(0, 105)
for i, v in enumerate(probs.numpy() * 100):
    ax2.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/decision.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\nSentence : {sentence}")
print(f"Tokens   : {tokens_full}")
print(f"IDs      : {ids_full}")
print(f"Logits   : {[round(float(x), 3) for x in logits]}")
print(f"Softmax  : {[f'{float(p)*100:.1f}%' for p in probs]}")
print(f"Verdict  : {pred_label}\n")

# ---------------------------------------------------------------------------
# Build the HTML report
# ---------------------------------------------------------------------------
tok_rows = "\n".join(
    f"<tr><td>{i}</td><td><code>{html.escape(t)}</code></td><td>{d}</td>"
    f"<td>{'added by the model' if t in ('[CLS]', '[SEP]') else ''}</td></tr>"
    for i, (t, d) in enumerate(zip(tokens_full, ids_full))
)

prob_rows = "\n".join(
    f"<tr><td>{labels[i]}</td><td>{float(logits[i]):.4f}</td>"
    f"<td><b>{float(probs[i])*100:.2f}%</b></td></tr>"
    for i in range(len(labels))
)

report = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>A sentence's journey through a Transformer</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 20px;
         line-height: 1.65; color: #24292f; }}
  h1 {{ border-bottom: 2px solid #eaecef; padding-bottom: .3em; }}
  h2 {{ margin-top: 2.2em; border-bottom: 1px solid #eaecef; padding-bottom: .2em; }}
  .verdict {{ font-size: 1.5em; padding: 16px 20px; border-radius: 10px;
              background: #f6f8fa; border-left: 5px solid #4c72b0; }}
  .sentence {{ font-size: 1.25em; background: #fff8dc; padding: 10px 16px;
               border-radius: 8px; display: inline-block; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #d0d7de; padding: 7px 10px; text-align: left; }}
  th {{ background: #f6f8fa; }}
  img {{ max-width: 100%; border: 1px solid #eaecef; border-radius: 8px; }}
  figcaption {{ color: #57606a; font-size: .9em; margin-top: .4em; }}
  .note {{ background: #f6f8fa; border-left: 4px solid #d0d7de;
           padding: 12px 16px; border-radius: 6px; margin: 1em 0; }}
  code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 4px; }}
</style></head><body>

<h1>A sentence's journey through a Transformer</h1>
<p>Input sentence: <span class="sentence">{html.escape(sentence)}</span></p>
<div class="verdict"><b>Final verdict:</b> {pred_label}
&nbsp;({float(probs[pred_id])*100:.1f}% confident)</div>

<h2>Station 1 &mdash; Text becomes tokens</h2>
<p>The model cannot read letters, only numbers. First the text is chopped into
<b>tokens</b>, and each token is looked up in the model's ~30k-entry vocabulary
to get its row number. Notice two tokens the model adds on its own:</p>
<table>
<tr><th>#</th><th>token</th><th>id</th><th>note</th></tr>
{tok_rows}
</table>
<div class="note"><b>[CLS]</b> is an artificial token placed at the front. Its job is to
listen to every word during attention and collect a summary of the whole sentence
into its own vector. At the end, only this vector is handed to the decision layer.
<b>[SEP]</b> simply marks where the sentence ends.</div>

<h2>Station 2 &mdash; Tokens become vectors (embeddings)</h2>
<p>A row number carries no meaning &mdash; it is just an address, like a seat number.
The <b>embedding table</b> turns each address into a vector of 768 numbers that
<em>does</em> carry meaning: words with similar meanings get nearby vectors.</p>
<figure><img src="embeddings.png" alt="embeddings">
<figcaption>Each row is one token, now a vector. Red = positive values, blue = negative.
These are the <em>raw</em> vectors: at this point the model still does not know which
sense of a word is meant &mdash; that is attention's job.</figcaption></figure>

<h2>Station 3 &mdash; Attention: the words look at each other</h2>
<p>This is the heart of the Transformer. Each word asks a question (<b>Query</b>),
every other word offers a label (<b>Key</b>), the two are compared to see how
relevant they are, and the word then borrows meaning (<b>Value</b>) in proportion
to that relevance. The result: a new, sentence-aware vector for every word.</p>

<p>Read the heatmap like this: each <b>row</b> is a word doing the looking, each
<b>column</b> is a word being looked at. A bright cell means "this word paid a lot
of attention to that one." Every row adds up to 1.00 (100% of its attention).</p>

<figure><img src="attention_first.png" alt="first layer attention">
<figcaption>Layer 1 &mdash; the earliest pass. Attention here tends to be broad and
mostly positional.</figcaption></figure>

<figure><img src="attention_last.png" alt="last layer attention">
<figcaption>Layer {n_layers} &mdash; the final pass, after the sentence has been
re-read {n_layers} times. The pattern is now much more semantic.</figcaption></figure>

<div class="note">DistilBERT has <b>{n_layers} attention layers</b>, each with
<b>{n_heads} heads</b>. Each head learns to look for a different kind of relationship;
the maps above average all heads together.</div>

<h2>Station 4 &mdash; [CLS] builds the summary</h2>
<p>Since <code>[CLS]</code> also participates in attention, it listens to the words
and folds them into a single vector. This bar chart is row 0 of the last attention
map &mdash; literally <em>what the summary token listened to</em>.</p>
<figure><img src="cls_attention.png" alt="cls attention">
<figcaption>The taller the bar, the more that word shaped the final decision.</figcaption></figure>
<figure><img src="cls_vector.png" alt="cls vector">
<figcaption>The resulting <code>[CLS]</code> vector: the entire sentence squeezed
into 768 numbers. This is the <em>only</em> thing the decision layer sees.</figcaption></figure>

<h2>Station 5 &mdash; The decision: dense &rarr; logits &rarr; softmax</h2>
<p>A small <b>dense</b> layer takes the summary vector, multiplies each number by a
learned weight, adds them up, adds a bias, and produces two raw scores called
<b>logits</b>. Logits are not percentages &mdash; they are just "how much the model
leans each way." <b>Softmax</b> then converts them into percentages that add to 100%,
exaggerating the gap between them along the way.</p>
<figure><img src="decision.png" alt="logits and softmax">
<figcaption>Left: the raw logits. Right: after softmax. The bigger the gap in the
logits, the more confident the model is.</figcaption></figure>
<table>
<tr><th>class</th><th>logit (raw)</th><th>after softmax</th></tr>
{prob_rows}
</table>

<h2>The whole journey</h2>
<pre>
  "{html.escape(sentence)}"
        |  tokenize
        v   {html.escape(str(tokens_full))}
        |  vocabulary lookup
        v   {ids_full}
        |  embedding table
        v   {len(tokens_full)} vectors x 768 numbers
        |  {n_layers} layers of self-attention
        v   {len(tokens_full)} sentence-aware vectors
        |  take the [CLS] vector
        v   1 summary vector
        |  dense
        v   logits {[round(float(x), 2) for x in logits]}
        |  softmax
        v   {[f'{float(p)*100:.1f}%' for p in probs]}
        |  pick the biggest
        v   {pred_label}
</pre>

</body></html>
"""

with open(f"{OUT_DIR}/report.html", "w", encoding="utf-8") as f:
    f.write(report)

print(f"[OK] Report written to: {OUT_DIR}/report.html")
print("     Open it in a browser to see the full journey.")
