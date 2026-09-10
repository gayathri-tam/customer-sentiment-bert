"""Metrics shared by baseline, BERT training and final evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return accuracy and macro scores so every class has equal influence."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def trainer_metrics(prediction) -> dict[str, float]:
    """Adapter used by Hugging Face Trainer."""
    logits, labels = prediction
    return classification_metrics(labels, np.argmax(logits, axis=-1))
