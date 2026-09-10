"""Download, clean, label and split Yelp Review Full data.

Yelp labels are 0--4 (one to five stars). This project maps 1--2 stars to
Negative, 3 stars to Neutral and 4--5 stars to Positive.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_SEED


def clean_text(text: str) -> str:
    """Apply deliberately light cleaning; preserve wording and punctuation for BERT."""
    text = str(text).replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def star_to_sentiment(star_label: int) -> int:
    """Map Yelp's zero-indexed 1--5 star label to a 3-class sentiment ID."""
    if star_label in (0, 1):
        return 0
    if star_label == 2:
        return 1
    if star_label in (3, 4):
        return 2
    raise ValueError(f"Unexpected Yelp star label: {star_label}")


def load_yelp(limit: int | None = None) -> pd.DataFrame:
    """Download Yelp Review Full from Hugging Face and return a labelled frame.

    `limit` takes a stratified-sized sample after the official splits are merged.
    Use it for a laptop/Colab smoke run; omit it for the full dataset.
    """
    dataset = load_dataset("Yelp/yelp_review_full")
    raw: Dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    # Selecting before converting to pandas avoids materialising all 700k reviews
    # during a quick experiment. Pick an equal number from each original star label.
    if limit is not None and limit < len(raw):
        per_star, remainder = divmod(limit, 5)
        labels = np.asarray(raw["label"])
        chosen: list[int] = []
        rng = np.random.default_rng(RANDOM_SEED)
        for star in range(5):
            candidates = np.flatnonzero(labels == star)
            take = per_star + (1 if star < remainder else 0)
            chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
        raw = raw.select(chosen)
    frame = raw.to_pandas().rename(columns={"text": "review", "label": "star_label"})
    frame["sentiment_id"] = frame["star_label"].map(star_to_sentiment)
    frame["review"] = frame["review"].map(clean_text)
    frame = frame.dropna(subset=["review"]).drop_duplicates(subset=["review"])
    frame = frame.loc[frame["review"].str.len().gt(0)].reset_index(drop=True)
    return frame


def make_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Make stratified 80/10/10 train, validation and held-out test splits."""
    train, temporary = train_test_split(
        frame, test_size=0.20, stratify=frame["sentiment_id"],
        random_state=RANDOM_SEED,
    )
    validation, test = train_test_split(
        temporary, test_size=0.50, stratify=temporary["sentiment_id"],
        random_state=RANDOM_SEED,
    )
    return {"train": train, "validation": validation, "test": test}


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path = PROCESSED_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}.csv", index=False)
