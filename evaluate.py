"""Evaluate either saved model on the untouched test split and save artifacts."""
from __future__ import annotations

import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

from src.config import LABEL_NAMES, RESULTS_DIR
from src.metrics import classification_metrics


def predict_bert(model_path: str, texts: list[str], batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model_path, tokenizer=model_path,
                          device=device, truncation=True, max_length=256)
    output = classifier(texts, batch_size=batch_size)
    labels = np.array([int(item["label"].split("_")[-1]) if item["label"].startswith("LABEL_")
                       else LABEL_NAMES.index(item["label"].title()) for item in output])
    confidence = np.array([item["score"] for item in output])
    return labels, confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "bert"], required=True)
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.test)
    if args.model == "baseline":
        with open(args.model_path or "models/tfidf_logreg.pkl", "rb") as file:
            model = pickle.load(file)
        predictions = model.predict(frame.review)
        confidence = model.predict_proba(frame.review).max(axis=1)
    else:
        predictions, confidence = predict_bert(args.model_path or "models/bert_sentiment", frame.review.tolist())
    metrics = classification_metrics(frame.sentiment_id, predictions)
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{args.model}_test"
    with open(RESULTS_DIR / f"{stem}_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)
    report = classification_report(frame.sentiment_id, predictions, target_names=LABEL_NAMES,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(RESULTS_DIR / f"{stem}_classification_report.csv")
    cm = confusion_matrix(frame.sentiment_id, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{args.model.title()} confusion matrix")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"{stem}_confusion_matrix.png", dpi=160); plt.close()
    errors = frame.loc[predictions != frame.sentiment_id, ["review", "sentiment_id"]].copy()
    errors["predicted_id"] = predictions[predictions != frame.sentiment_id]
    errors["confidence"] = confidence[predictions != frame.sentiment_id]
    errors.sort_values("confidence", ascending=False).head(100).to_csv(RESULTS_DIR / f"{stem}_errors.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
