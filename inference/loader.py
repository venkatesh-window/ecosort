"""Production model loading. Exactly one loaded model per process; reloads are
serialized so requests can never observe a half-loaded model."""

from __future__ import annotations

import threading

from ml.infer import Predictor, resolve_checkpoint

_lock = threading.Lock()
_predictor = Predictor()


def get_model(force: bool = False) -> Predictor:
    with _lock:
        if force or not _predictor.loaded:
            ckpt = resolve_checkpoint()
            if ckpt is None:
                raise RuntimeError("No trained checkpoint available.")
            _predictor.load(ckpt)
        return _predictor


def reload(run_id: str | None = None, weights: str = "best.pt") -> Predictor:
    with _lock:
        ckpt = resolve_checkpoint(run_id, weights)
        if ckpt is None:
            raise FileNotFoundError(f"checkpoint not found: {run_id}/{weights}")
        _predictor.load(ckpt)
        return _predictor


def model_info() -> dict:
    p = get_model()
    return {
        "model": p.meta.get("run_id"),
        "dataset": (p.meta.get("training_config") or {}).get("dataset_version"),
        "version": p.version,
        "device": str(p.meta.get("device")),
        "checkpoint": str(p.checkpoint_path) if p.checkpoint_path else None,
    }
