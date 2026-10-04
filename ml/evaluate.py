"""Evaluation on the held-out test split.

Produces accuracy, precision, recall and F1 (macro and per-class), the full
confusion matrix, and real misclassified example paths per confusion cell so the
UI can show actual failure cases rather than illustrative ones.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from . import config
from .infer import DEVICE
from .model import build_model
from .train import ManifestDataset

EVAL_TF = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ]
)

MAX_EXAMPLES_PER_CELL = 6


def confusion_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    true_sum = cm.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    recall = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    total = cm.sum()
    accuracy = float(tp.sum() / total) if total else 0.0
    weights = true_sum / total if total else true_sum

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": float((precision * weights).sum()),
        "recall_weighted": float((recall * weights).sum()),
        "f1_weighted": float((f1 * weights).sum()),
        "per_class": {
            config.CLASSES[i]: {
                "precision": round(float(precision[i]), 5),
                "recall": round(float(recall[i]), 5),
                "f1": round(float(f1[i]), 5),
                "support": int(true_sum[i]),
                "predicted_as": int(pred_sum[i]),
            }
            for i in range(cm.shape[0])
        },
    }


@torch.no_grad()
def run_evaluation(checkpoint: Path) -> dict:
    model = build_model(num_classes=config.NUM_CLASSES, pretrained=False)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE).eval()

    ds = ManifestDataset("test", EVAL_TF)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    y_true: list[int] = []
    y_pred: list[int] = []
    probs_all: list[list[float]] = []
    rows: list[dict] = []
    offset = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1)

        y_true.extend(int(v) for v in y)
        y_pred.extend(int(v) for v in pred)
        probs_all.extend([float(v) for v in row] for row in probs.cpu().numpy())

        for i, label in enumerate(y):
            rows.append(ds.rows[offset + i] | {"pred": int(pred[i])})
        offset += x.size(0)

    yt, yp = np.array(y_true), np.array(y_pred)
    cm = confusion_from_predictions(yt, yp, config.NUM_CLASSES)
    metrics = prf(cm)

    # Real misclassified examples, grouped by (true, predicted) cell.
    errors: dict[str, list[dict]] = defaultdict(list)
    for row, prob in zip(rows, probs_all):
        if row["label"] != row["pred"]:
            key = f"{row['class']}__{config.CLASSES[row['pred']]}"
            if len(errors[key]) < MAX_EXAMPLES_PER_CELL:
                errors[key].append(
                    {
                        "path": row["path"],
                        "true_class": row["class"],
                        "predicted_class": config.CLASSES[row["pred"]],
                        "confidence": round(prob[row["pred"]], 5),
                        "true_confidence": round(prob[row["label"]], 5),
                    }
                )

    # Hardest confusion pairs by volume.
    confusion_pairs = []
    for key in errors:
        true_cls, pred_cls = key.split("__")
        ti, pi = config.CLASS_TO_IDX[true_cls], config.CLASS_TO_IDX[pred_cls]
        confusion_pairs.append(
            {"true_class": true_cls, "predicted_class": pred_cls, "count": int(cm[ti, pi])}
        )
    confusion_pairs.sort(key=lambda d: d["count"], reverse=True)

    confidences = [max(p) for p in probs_all]
    correct_flags = yt == yp

    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "run_id": checkpoint.parent.name,
        "weights": checkpoint.name,
        "architecture": "efficientnet_b0",
        "split": "test",
        "num_samples": len(yt),
        "num_classes": config.NUM_CLASSES,
        "classes": config.CLASSES,
        "training_best_val_acc": state.get("best_val_acc"),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_pairs_top": confusion_pairs[:15],
        "misclassified_examples": dict(errors),
        "confidence_distribution": {
            "mean": round(float(np.mean(confidences)), 5) if confidences else None,
            "median": round(float(np.median(confidences)), 5) if confidences else None,
            "below_threshold": int(sum(c < config.CONFIDENCE_THRESHOLD for c in confidences)),
            "threshold": config.CONFIDENCE_THRESHOLD,
        },
        "selective_accuracy": _selective_accuracy(confidences, correct_flags),
    }
    return payload


def _selective_accuracy(confidences: list[float], correct: np.ndarray) -> list[dict]:
    """Accuracy when the model abstains below a threshold - the honest version of
    'how good is it really', and the basis of the robot confidence gate."""
    out = []
    conf = np.array(confidences)
    for t in (0.0, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90):
        keep = conf >= t
        n = int(keep.sum())
        acc = float(correct[keep].mean()) if n else None
        out.append(
            {
                "threshold": t,
                "coverage": round(n / len(conf), 4) if len(conf) else 0.0,
                "accuracy": round(acc, 5) if acc is not None else None,
                "samples": n,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    payload = run_evaluation(ckpt)
    out = Path(args.out) if args.out else config.METRICS_DIR / f"evaluation_{ckpt.parent.name}_{ckpt.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    m = payload["metrics"]
    print(f"checkpoint : {ckpt}")
    print(f"samples    : {payload['num_samples']}")
    print(f"accuracy   : {m['accuracy']:.4f}")
    print(f"f1 (macro) : {m['f1_macro']:.4f}")
    print(f"precision  : {m['precision_macro']:.4f}")
    print(f"recall     : {m['recall_macro']:.4f}")
    print(f"saved      : {out}")
    print("\nper-class F1:")
    for cls, d in m["per_class"].items():
        print(f"  {cls:<12} p={d['precision']:.3f} r={d['recall']:.3f} f1={d['f1']:.3f} n={d['support']}")


if __name__ == "__main__":
    main()
