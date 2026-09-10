"""Run transparent EDA and save figures/tables for the README and report."""
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import LABEL_NAMES, RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame["review_length_chars"] = frame.review.str.len()
    frame["review_length_words"] = frame.review.str.split().str.len()
    summary = {
        "rows": len(frame),
        "missing_values": frame.isna().sum().to_dict(),
        "duplicate_reviews": int(frame.review.duplicated().sum()),
        "class_distribution": frame.sentiment_id.value_counts().sort_index().to_dict(),
        "text_statistics": frame[["review_length_chars", "review_length_words"]].describe().to_dict(),
    }
    with open(RESULTS_DIR / "eda_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    sns.set_theme(style="whitegrid")
    labels = frame.sentiment_id.map(dict(enumerate(LABEL_NAMES)))
    ax = sns.countplot(x=labels, order=LABEL_NAMES, hue=labels, legend=False,
                       palette=["#d9534f", "#f0ad4e", "#5cb85c"])
    ax.set(title="Training-set sentiment distribution", xlabel="Sentiment", ylabel="Reviews")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=160); plt.close()
    ax = sns.histplot(frame.review_length_words, bins=60, color="#337ab7")
    ax.set(title="Review length distribution", xlabel="Words per review")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "review_length.png", dpi=160); plt.close()
    print(f"Wrote EDA outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
