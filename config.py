"""Project-wide constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

RANDOM_SEED = 42
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
ID2LABEL = dict(enumerate(LABEL_NAMES))
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
