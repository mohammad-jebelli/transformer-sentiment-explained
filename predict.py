"""
predict.py
==========
Quick sentiment check using OUR fine-tuned model (the one train_model.py made).

Run:
    python predict.py "this film was wonderful"
    python predict.py                 # runs a few built-in examples
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "sentiment-model"

if not os.path.isdir(MODEL_PATH):
    print(f"[ERROR] '{MODEL_PATH}/' not found. Run train_model.py first.")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
id2label = model.config.id2label

if len(sys.argv) > 1:
    sentences = [" ".join(sys.argv[1:])]
else:
    sentences = [
        "I love this movie, it was fantastic!",
        "I do not love this movie",
        "What a boring, terrible film.",
        "The acting was weak but the story saved it.",
    ]

for text in sentences:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred = id2label[int(torch.argmax(probs))]
    conf = float(probs.max()) * 100
    print(f"{pred:<9} {conf:5.1f}%   <- {text}")
