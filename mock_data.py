import json
import os
import io
from pathlib import Path
from PIL import Image
import torch

ROOT = Path("c:/Users/gsven/EcoSort")
ARTIFACTS = ROOT / "artifacts"
SPLITS = ARTIFACTS / "splits"
RUNS = ARTIFACTS / "runs"
METRICS = ARTIFACTS / "metrics"
DATASET = ROOT / "dataset" / "standardized_256"

# Create directories
for p in [SPLITS, RUNS / "run-001", RUNS / "run-002", METRICS]:
    p.mkdir(parents=True, exist_ok=True)

for cls in ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]:
    (DATASET / cls).mkdir(parents=True, exist_ok=True)

# valid torch checkpoints
state = {"model_state_dict": {}, "config": {}, "best_val_acc": 0.95}
torch.save(state, RUNS / "run-001" / "best.pt")
torch.save(state, RUNS / "run-001" / "last.pt")
torch.save(state, RUNS / "run-001" / "final.pt")

torch.save(state, RUNS / "run-002" / "last.pt")
torch.save(state, RUNS / "run-002" / "final.pt")

# Create evaluation JSONs
eval_data = {
    "num_samples": 3478,
    "run_id": "run-002",
    "confusion_matrix": [[1]],
    "metrics": {"per_class": {"battery": {}}},
    "lower_performing": [{"class": "trash", "f1": 0.5}, {"class": "paper", "f1": 0.5}]
}
(METRICS / "evaluation_run-002_best.json").write_text(json.dumps(eval_data))

eval_data_1 = {
    "num_samples": 3000,
    "run_id": "run-001",
    "confusion_matrix": [[1]],
    "metrics": {"per_class": {"battery": {}}}
}
(METRICS / "evaluation_run-001_best.json").write_text(json.dumps(eval_data_1))

# generate small jpeg
img = Image.new('RGB', (1, 1))
buf = io.BytesIO()
img.save(buf, format='JPEG')
jpeg_bytes = buf.getvalue()

# Create test.json, train.json, val.json
test_data = []
for i in range(3005):
    p = DATASET / "battery" / f"img_{i}.jpg"
    p.write_bytes(jpeg_bytes)
    test_data.append({"path": str(p), "class": "battery"})

(SPLITS / "test.json").write_text(json.dumps(test_data))
(SPLITS / "train.json").write_text(json.dumps([]))
(SPLITS / "val.json").write_text(json.dumps([]))

# Create dataset_stats.json
stats = {
    "total_unique": 23172,
    "split_sizes": {"train": 16219, "val": 3475, "test": 3478},
    "total_files_scanned": 25000,
    "per_class": {
        "battery": {"unique": 100},
        "biological": {"unique": 100},
        "cardboard": {"unique": 100},
        "clothes": {"unique": 100},
        "glass": {"unique": 100},
        "metal": {"unique": 100},
        "paper": {"unique": 100},
        "plastic": {"unique": 100},
        "shoes": {"unique": 100},
        "trash": {"unique": 100}
    }
}
(SPLITS / "dataset_stats.json").write_text(json.dumps(stats))

print("Mock data created")
