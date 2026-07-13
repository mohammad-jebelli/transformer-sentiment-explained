"""
train_model.py
==============
Fine-tune DistilBERT for binary sentiment analysis (positive / negative)
on the IMDB movie-review dataset.

This is the "improved" training script — the sentiment-analysis parallel of
the Cat/Dog `train_improved.py`. Each professional touch is commented and
maps to a concept:

    * dynamic padding (DataCollatorWithPadding) -> pad each batch to its own
      longest sequence instead of a fixed huge length  (the "padding" idea)
    * GPU-aware + fp16                              -> train fast on a GPU
    * load_best_model_at_end + metric_for_best_model-> keep the BEST epoch,
      not just the last one                          (like "save best epoch")
    * warmup + weight_decay                          -> stable training
    * id2label / label2id                            -> model outputs the words
      "POSITIVE" / "NEGATIVE" instead of LABEL_0 / LABEL_1  (like class_indices)
    * set_seed                                       -> reproducible runs
    * training-history plot                          -> like training_history.png

Run:
    python train_model.py

Outputs:
    sentiment-model/           <- the trained model + tokenizer (load this later)
    training_history.png       <- loss / f1 curves
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed -- save straight to file
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
import evaluate

# ---------------------------------------------------------------------------
# 0) Reproducibility + device info
# ---------------------------------------------------------------------------
set_seed(42)  # same random start every run -> comparable results

use_gpu = torch.cuda.is_available()
if use_gpu:
    print(f"[OK] GPU found: {torch.cuda.get_device_name(0)} -- training on GPU")
else:
    print("[WARN] No GPU found -- training on CPU (this will be slow)")

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "sentiment-model"

# ---------------------------------------------------------------------------
# 1) Data -- IMDB: 50k movie reviews labelled 0 (negative) / 1 (positive)
#    We take a subset so training finishes quickly. Raise these numbers
#    (or use the full split) once everything works.
# ---------------------------------------------------------------------------
imdb = load_dataset("stanfordnlp/imdb")

N_TRAIN = 3000
N_TEST = 3000
train_ds = imdb["train"].shuffle(seed=42).select(range(N_TRAIN))
test_ds = imdb["test"].shuffle(seed=42).select(range(N_TEST))

# ---------------------------------------------------------------------------
# 2) Tokenizer -- turns text into token IDs (the first station of the journey)
#    truncation=True  : cut reviews longer than the model's max length
#    NO padding here   : we pad *per batch* later, which is far more efficient
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess(batch):
    return tokenizer(batch["text"], truncation=True)


tokenized_train = train_ds.map(preprocess, batched=True)
tokenized_test = test_ds.map(preprocess, batched=True)

# Dynamic padding: pad every batch to the longest sequence *in that batch*.
# This is the "padding" concept -- a batch must be a neat rectangle of numbers,
# so shorter sentences get filler tokens to match the longest one.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------------------------------------------------------
# 3) Model -- DistilBERT with a fresh 2-class classification head on top.
#    id2label / label2id make the outputs human-readable.
# ---------------------------------------------------------------------------
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# ---------------------------------------------------------------------------
# 4) Metrics -- accuracy + F1 (F1 is a fairer score when classes are uneven)
# ---------------------------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # pick the class with the bigger logit
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": acc, "f1": f1}


# ---------------------------------------------------------------------------
# 5) Training configuration
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,            # gently discourage huge weights -> less overfit
    warmup_ratio=0.1,             # ease the learning rate up at the start -> stable
    eval_strategy="epoch",        # evaluate after every epoch
    save_strategy="epoch",        # save a checkpoint after every epoch
    load_best_model_at_end=True,  # keep the BEST epoch, not the last one
    metric_for_best_model="f1",   # "best" is judged by F1
    greater_is_better=True,
    logging_strategy="epoch",
    fp16=use_gpu,                 # half-precision on GPU -> faster, less memory
    report_to="none",            # no external logging service
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------------------------
# 6) Train
# ---------------------------------------------------------------------------
trainer.train()

# ---------------------------------------------------------------------------
# 7) Save the best model + tokenizer together, so predict.py can load one folder
# ---------------------------------------------------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n[OK] Best model saved to: {OUTPUT_DIR}/")

# ---------------------------------------------------------------------------
# 8) Plot the training history (loss + F1 per epoch) -> training_history.png
# ---------------------------------------------------------------------------
history = trainer.state.log_history
train_loss = [(h["epoch"], h["loss"]) for h in history if "loss" in h]
eval_f1 = [(h["epoch"], h["eval_f1"]) for h in history if "eval_f1" in h]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
if train_loss:
    ep, loss = zip(*train_loss)
    ax1.plot(ep, loss, marker="o")
    ax1.set_title("Training loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
if eval_f1:
    ep, f1s = zip(*eval_f1)
    ax2.plot(ep, f1s, marker="o", color="green")
    ax2.set_title("Validation F1")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("F1")
plt.tight_layout()
plt.savefig("training_history.png", dpi=120)
print("[OK] Saved training_history.png")

# ---------------------------------------------------------------------------
# 9) Quick sanity check on two obvious sentences
# ---------------------------------------------------------------------------
print("\n[Sanity check]")
model.eval()
device = model.device
for text in ["I love this movie, it was fantastic!", "What a boring, terrible film."]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = id2label[int(torch.argmax(logits, dim=-1))]
    print(f"  {pred:<9} <- {text}")
