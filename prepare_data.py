"""Create reproducible training files from the public Yelp dataset."""
import argparse

from src.data import load_yelp, make_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional stratified sample size for a quick run.")
    args = parser.parse_args()
    frame = load_yelp(args.limit)
    splits = make_splits(frame)
    save_splits(splits)
    print(f"Saved {len(frame):,} cleaned reviews.")
    for name, split in splits.items():
        print(f"{name}: {len(split):,} rows; {split.sentiment_id.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
