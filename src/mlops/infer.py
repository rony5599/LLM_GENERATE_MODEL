from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path("artifacts/model")


@lru_cache(maxsize=1)
def load_artifacts():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text: str) -> dict:
    tokenizer, model = load_artifacts()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs))
        confidence = float(probs[idx])
    label = model.config.id2label[idx]
    return {"label": label, "confidence": confidence}
