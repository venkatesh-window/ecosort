import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS / "splits"
RUNS_DIR = ARTIFACTS / "runs"
METRICS_DIR = ARTIFACTS / "metrics"
UPLOADS_DIR = ARTIFACTS / "uploads"
DB_PATH = ARTIFACTS / "ecosort.db"

# Dataset location: ECOSORT_DATASET_ROOT wins (production/portable hosts);
# falls back to the original development path.
DATASET_ROOT = Path(os.environ.get("ECOSORT_DATASET_ROOT") or ROOT / "dataset")
DATASET_VARIANT = "standardized_256"
DATASET_DIR = DATASET_ROOT / DATASET_VARIANT

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Discovered from the dataset directory, sorted for a deterministic label mapping.
CLASSES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

DISPLAY_NAMES = {
    "battery": "Battery",
    "biological": "Biological",
    "cardboard": "Cardboard",
    "clothes": "Clothes",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "shoes": "Shoes",
    "trash": "Trash",
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Both thresholds can be overridden at runtime through the robot config API.
CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15

# Configurable class -> bin mapping for the robot sorting engine.
BIN_MAP = {
    "battery": "BIN_HAZARDOUS",
    "biological": "BIN_ORGANIC",
    "cardboard": "BIN_PAPER",
    "clothes": "BIN_TEXTILE",
    "glass": "BIN_GLASS",
    "metal": "BIN_METAL",
    "paper": "BIN_PAPER",
    "plastic": "BIN_PLASTIC",
    "shoes": "BIN_TEXTILE",
    "trash": "BIN_LANDFILL",
}

BIN_LABELS = {
    "BIN_HAZARDOUS": "Hazardous",
    "BIN_ORGANIC": "Organic",
    "BIN_PAPER": "Paper & Card",
    "BIN_TEXTILE": "Textile",
    "BIN_GLASS": "Glass",
    "BIN_METAL": "Metal",
    "BIN_PLASTIC": "Plastic",
    "BIN_LANDFILL": "Landfill",
}


def display(cls: str) -> str:
    return DISPLAY_NAMES.get(cls, cls.title())


for _d in (ARTIFACTS, SPLITS_DIR, RUNS_DIR, METRICS_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
