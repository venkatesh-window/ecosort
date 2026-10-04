"""EcoSort API - serves the real trained model, explainability, history,
feedback loop, evaluation artifacts and the robot/simulation layer."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import db, hardware, runtime_config
from api.robot import ROBOT, SIMULATED_MOTION_PHASES, VALID_STATES, sample_heldout_image
from ml import config
from ml.infer import DEVICE, Predictor, resolve_checkpoint

ROOT = config.ROOT
WEB_DIST = ROOT / "web" / "dist"

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    runtime_config.load()
    ROBOT.sync_config()
    ckpt = resolve_checkpoint()
    if ckpt:
        try:
            predictor.load(ckpt)
            ROBOT._model_loaded = True
            print(f"model loaded: {ckpt}")
        except Exception as exc:
            print(f"model not loaded at startup: {exc}")
    else:
        print("no checkpoint found - API will run in untrained mode")
    yield

app = FastAPI(title="EcoSort API", version="0.1.0", lifespan=lifespan)


def _allowed_origins() -> list[str]:
    """CORS allow-list. Same-origin + local dev by default; production
    frontends opt in via FRONTEND_ORIGINS (preferred) or ALLOWED_ORIGINS
    (comma-separated). The wildcard is only used when explicitly requested,
    for documented local-network testing — never production."""
    raw = os.environ.get("FRONTEND_ORIGINS", "") or os.environ.get("ALLOWED_ORIGINS", "")
    raw = raw.strip()
    if raw == "*":
        print("WARNING: CORS allow-origin '*' enabled via env (local testing only)")
        return ["*"]
    defaults = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
    extra = [o.strip() for o in raw.split(",") if o.strip()]
    return defaults + [o for o in extra if o not in defaults]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor()
_training_proc: subprocess.Popen | None = None
_model_lock = threading.Lock()

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------- model state


def ensure_model(force: bool = False) -> Predictor:
    global predictor
    # Locked so two simultaneous reload/first-predict requests cannot
    # interleave a half-loaded model into serving.
    with _model_lock:
        if force or not predictor.loaded:
            ckpt = resolve_checkpoint()
            if ckpt is None:
                raise HTTPException(
                    status_code=503,
                    detail="No trained checkpoint available yet. Start a training run first.",
                )
            predictor.load(ckpt)
            ROBOT._model_loaded = True
        return predictor





@app.get("/api/health")
def health() -> dict:
    from ml import dataset_registry as reg

    ckpt = resolve_checkpoint()
    training_active = _training_proc is not None and _training_proc.poll() is None
    stats_path = config.SPLITS_DIR / "dataset_stats.json"
    versions = reg.load_versions()
    finalized = [v for v in versions if v.get("kind", "finalized") == "finalized"]
    # Production truth: the dataset behind the LOADED model. Falls back to
    # the latest finalized version only when no model is loaded.
    loaded_ds = (predictor.meta.get("training_config") or {}).get("dataset_version") if predictor.loaded else None
    dataset_version = loaded_ds or (finalized[-1].get("version") if finalized else None)
    try:
        db_status = "ok" if db.activity_summary() is not None else "error"
    except Exception:
        db_status = "error"
    return {
        "api": "operational",
        "model_loaded": predictor.loaded,
        "model_version": predictor.version if predictor.loaded else None,
        "checkpoint": str(ckpt) if ckpt else None,
        "device": str(DEVICE),
        "cuda_available": DEVICE.type == "cuda",
        "dataset_prepared": stats_path.exists(),
        "dataset_version": dataset_version,
        "dataset_stats": json.loads(stats_path.read_text()) if stats_path.exists() else None,
        "training_active": training_active,
        "training_pid": _training_proc.pid if training_active else None,
        "database": str(config.DB_PATH),
        "database_status": db_status,
        "robot_mode": ROBOT.mode,
        "server_time": time.time(),
    }


@app.get("/api/model/status")
def model_status() -> dict:
    ckpt = resolve_checkpoint()
    run_metrics = None
    if ckpt:
        mp = ckpt.parent / "metrics.json"
        if mp.exists():
            run_metrics = json.loads(mp.read_text())
    return {
        "loaded": predictor.loaded,
        "version": predictor.version if predictor.loaded else None,
        "meta": predictor.meta if predictor.loaded else None,
        "checkpoint": str(ckpt) if ckpt else None,
        "architecture": "efficientnet_b0",
        "input_size": config.IMAGE_SIZE,
        "num_classes": config.NUM_CLASSES,
        "classes": config.CLASSES,
        "device": str(DEVICE),
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "run_metrics": run_metrics,
    }


@app.post("/api/model/reload")
def model_reload(run_id: str | None = None, weights: str = "best.pt") -> dict:
    ckpt = resolve_checkpoint(run_id, weights)
    if ckpt is None:
        raise HTTPException(404, "checkpoint not found")
    with _model_lock:
        try:
            predictor.load(ckpt)
        except Exception as exc:
            # Old model stays in place: load() only swaps internals on success.
            raise HTTPException(500, f"checkpoint failed to load, previous model kept: {exc}") from exc
        ROBOT._model_loaded = True
    return {"loaded": True, "checkpoint": str(ckpt), "version": predictor.version}


# ------------------------------------------------------------------ prediction


class PredictResponse(BaseModel):
    scan_id: str


def _run_prediction(pil: Image.Image, source: str, image_path: str | None, save: bool):
    p = ensure_model()
    result = p.predict(pil)
    scan_id = None
    if save:
        scan_id = db.insert_scan(
            {
                "source": source,
                "image_path": image_path,
                "predicted_class": result["prediction"]["class"],
                "confidence": result["prediction"]["confidence"],
                "state": result["prediction"]["state"],
                "action": result["sorting"]["action"],
                "target_bin": result["sorting"]["target_bin"],
                "top5": result["prediction"]["top5"],
                "latency": result["timings_ms"],
                "quality": result["input_quality"],
                "centroid": result["explainability"]["activation_centroid"],
                "model_version": result["model"]["version"],
                "dataset_version": result["model"].get("dataset_version"),
            }
        )
    result["scan_id"] = scan_id
    return result


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), source: str = "upload", save: bool = True) -> dict:
    from inference.preprocessing import decode_upload

    raw = await file.read()
    try:
        pil, ext = decode_upload(raw, file.filename)
    except ValueError as exc:
        raise HTTPException(413 if "exceeds" in str(exc) else 400, str(exc)) from exc

    saved_path = None
    if save:
        sid = db.new_id()
        dest = config.UPLOADS_DIR / f"{sid}{ext}"
        dest.write_bytes(raw)
        saved_path = str(dest)

    return _run_prediction(pil, source, saved_path, save)


@app.get("/api/scans")
def scans(
    limit: int = Query(100, ge=1, le=500),
    cls: str | None = None,
    state: str | None = None,
) -> dict:
    rows = db.list_scans(limit=limit, cls=cls, state=state)
    return {"count": len(rows), "scans": rows, "classes": config.CLASSES}


@app.get("/api/scans/{scan_id}")
def scan_detail(scan_id: str) -> dict:
    row = db.get_scan(scan_id)
    if not row:
        raise HTTPException(404, "scan not found")
    return row


@app.get("/api/scans/{scan_id}/explain")
def scan_explain(scan_id: str) -> dict:
    """Re-run the model on a stored scan to regenerate Grad-CAM and timings.

    Heatmaps are not persisted, so this recomputes them from the saved image.
    The response is explicitly flagged as a re-analysis, and the original
    recorded latency is returned alongside for comparison.
    """
    row = db.get_scan(scan_id)
    if not row:
        raise HTTPException(404, "scan not found")

    image_path = row.get("image_path")
    if not image_path or not Path(image_path).exists():
        raise HTTPException(404, "source image for this scan is no longer available")

    try:
        with Image.open(image_path) as im:
            im.load()
            pil = im.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(500, f"could not read stored image: {exc}") from exc

    result = _run_prediction(pil, source="reanalysis", image_path=image_path, save=False)
    result["scan_id"] = scan_id
    result["reanalysis"] = {
        "is_reanalysis": True,
        "original_created_at": row["created_at"],
        "original_confidence": row["confidence"],
        "original_predicted_class": row["predicted_class"],
        "original_latency": row.get("latency"),
        "note": "Grad-CAM and timings were recomputed just now; the class and "
        "confidence from the original scan are shown for comparison.",
    }
    result["stored_feedback"] = {
        "was_correct": row.get("was_correct"),
        "corrected_class": row.get("corrected_class"),
    }
    return result


@app.get("/api/scans/{scan_id}/image")
def scan_image(scan_id: str) -> FileResponse:
    """Serve the image stored for a scan. The path is read from the database
    rather than accepted from the client, so it can only ever be a file this
    server wrote during a prediction."""
    row = db.get_scan(scan_id)
    if not row:
        raise HTTPException(404, "scan not found")

    image_path = row.get("image_path")
    if not image_path or not Path(image_path).is_file():
        raise HTTPException(404, "source image for this scan is no longer available")

    return FileResponse(image_path)


class Feedback(BaseModel):
    correct: bool
    correct_class: str | None = None
    source: str = "user"


@app.post("/api/scans/{scan_id}/feedback")
def feedback(scan_id: str, body: Feedback) -> dict:
    row = db.get_scan(scan_id)
    if not row:
        raise HTTPException(404, "scan not found")

    predicted = row["predicted_class"]
    if body.correct:
        final = predicted
    else:
        if not body.correct_class:
            raise HTTPException(422, "correct_class is required when correct=false")
        if body.correct_class not in config.CLASSES:
            raise HTTPException(422, f"unknown class '{body.correct_class}'")
        final = body.correct_class

    cid = db.insert_correction(scan_id, predicted, final, body.source)
    return {
        "correction_id": cid,
        "scan_id": scan_id,
        "predicted_class": predicted,
        "verified_class": final,
        "was_correct": body.correct,
        "added_to_review_queue": not body.correct,
    }


@app.get("/api/corrections")
def corrections(limit: int = Query(100, ge=1, le=500)) -> dict:
    rows = db.list_corrections(limit)
    return {"count": len(rows), "corrections": rows}


@app.get("/api/review-queue")
def review_queue(limit: int = Query(50, ge=1, le=500)) -> dict:
    rows = db.review_queue(limit)
    return {
        "count": len(rows),
        "items": rows,
        "note": "Verified labels from this queue become retraining candidates.",
    }


@app.get("/api/activity")
def activity() -> dict:
    return db.activity_summary()


# --------------------------------------------------------------------- dataset


@app.get("/api/dataset")
def dataset() -> dict:
    from ml import dataset_registry as reg

    stats_path = config.SPLITS_DIR / "dataset_stats.json"
    if not stats_path.exists():
        raise HTTPException(503, "dataset split not built - run `python -m ml.data`")
    stats = json.loads(stats_path.read_text())

    rng = random.Random(config.SEED)
    gallery = {}
    for cls in config.CLASSES:
        cls_dir = config.DATASET_DIR / cls
        files = sorted(p.name for p in cls_dir.iterdir() if p.suffix.lower() in ALLOWED_EXT) if cls_dir.is_dir() else []
        samples = rng.sample(files, min(8, len(files)))
        gallery[cls] = [
            f"/dataset-images/{config.DATASET_VARIANT}/{cls}/{name}" for name in samples
        ]

    counts = stats.get("per_class", {})
    n = {c: counts[c]["unique"] for c in counts}
    largest = max(n.values()) if n else 0
    smallest = min(n.values()) if n else 0
    balance_ratio = round(largest / smallest, 2) if smallest else None

    prog = reg.progress(stats)
    per_class_target_view = {}
    for c in reg.CANONICAL_13:
        per_class_target_view[c] = {
            "have": prog["per_class"][c]["have"],
            "target": prog["per_class"][c]["target"],
            "gap": prog["per_class"][c]["gap"],
            "pct_of_dataset": prog["per_class"][c]["pct_of_dataset"],
            "pct_of_target": prog["per_class"][c]["pct_of_target"],
            "display": reg.CANONICAL_DISPLAY[c],
            "train": 0,
            "val": 0,
            "test": 0,
        }
    # Fill train/val/test from the model-compatible per_class keys
    # (e.g. on-disk "biological" reports under canonical "organic").
    for raw, d in counts.items():
        canon = reg.canonical(raw)
        if canon and canon in per_class_target_view and isinstance(d, dict):
            per_class_target_view[canon]["train"] = d.get("train", 0)
            per_class_target_view[canon]["val"] = d.get("val", 0)
            per_class_target_view[canon]["test"] = d.get("test", 0)

    return {
        **stats,
        "gallery": gallery,
        "health": {
            "duplicates_removed": {"value": stats.get("duplicates_removed"), "ok": True},
            "near_duplicates_removed": {"value": stats.get("near_duplicates_removed", 0), "ok": True},
            "corrupt_files": {"value": stats.get("corrupt_unreadable"), "ok": True},
            "low_quality_removed": {"value": stats.get("low_quality_removed", 0), "ok": True},
            "unmapped_skipped": {"value": stats.get("unmapped_skipped", 0), "ok": True},
            "missing_images": {"value": 0, "ok": True},
            "class_balance": {
                "value": balance_ratio,
                "detail": f"largest/smallest class ratio {balance_ratio}:1",
                "ok": balance_ratio is not None and balance_ratio < 3.0,
            },
            "image_dimensions": {
                "value": f"{stats.get('variant')} standardized",
                "ok": True,
            },
        },
        # 100K goal: honest actual-vs-target accounting. `verified` is the
        # only number that counts; augmented copies are never included.
        "goal_100k": {
            "target": prog["target"],
            "verified": prog["verified"],
            "remaining": prog["remaining"],
            "reached": prog["reached"],
            "progress_frac": prog["progress_frac"],
            "train": prog["train"],
            "val": prog["val"],
            "test": prog["test"],
            "per_class": per_class_target_view,
            "strongest_classes": prog["strongest_classes"],
            "underrepresented_classes": prog["underrepresented_classes"],
            "missing_classes": prog["missing_classes"],
        },
        "robot_ready": reg.robot_ready_summary(),
    }


@app.get("/api/dataset/sources")
def dataset_sources() -> dict:
    from ml import dataset_registry as reg

    rows = reg.load_sources()
    return {
        "count": len(rows),
        "sources": rows,
        "note": "Every imported dataset must record name, source, licence and class mapping. No scraping; no unclear rights.",
    }


@app.get("/api/dataset/versions")
def dataset_versions() -> dict:
    from ml import dataset_registry as reg

    rows = reg.load_versions()
    current = reg.current_stats()
    return {
        "count": len(rows),
        "versions": rows,
        "current_verified": int(current.get("total_unique", 0)) if current else 0,
        "target": reg.TARGET_TOTAL,
    }


@app.get("/api/dataset/health")
def dataset_health() -> dict:
    from ml import dataset_registry as reg

    stats = reg.current_stats()
    if not stats:
        raise HTTPException(503, "dataset split not built - run `python -m ml.data`")
    return {"report": reg.health_report(stats), "target": reg.TARGET_TOTAL}


@app.get("/api/dataset/targets")
def dataset_targets() -> dict:
    from ml import dataset_registry as reg

    stats = reg.current_stats()
    prog = reg.progress(stats)
    return {
        "target": prog["target"],
        "verified": prog["verified"],
        "remaining": prog["remaining"],
        "reached": prog["reached"],
        "per_class": {
            c: {
                "display": reg.CANONICAL_DISPLAY[c],
                "have": v["have"],
                "target": v["target"],
                "gap": v["gap"],
            }
            for c, v in prog["per_class"].items()
        },
        "missing_classes": prog["missing_classes"],
        "underrepresented_classes": prog["underrepresented_classes"],
        "strongest_classes": prog["strongest_classes"],
    }


@app.get("/api/classes")
def classes() -> dict:
    from ml.recycling import GUIDE

    return {
        "classes": [
            {
                "id": c,
                "display": config.display(c),
                "index": config.CLASS_TO_IDX[c],
                "bin": config.BIN_MAP[c],
                "bin_label": config.BIN_LABELS[config.BIN_MAP[c]],
                "guide": GUIDE[c],
            }
            for c in config.CLASSES
        ]
    }


# -------------------------------------------------------------------- training


@app.get("/api/training/status")
def training_status() -> dict:
    runs = sorted((p for p in config.RUNS_DIR.glob("run-*") if p.is_dir()), key=lambda p: p.name)
    active = _training_proc is not None and _training_proc.poll() is None
    out = {"active": active, "pid": _training_proc.pid if active else None, "runs": []}
    for run in runs:
        mp = run / "metrics.json"
        entry = {"run_id": run.name, "metrics": None}
        if mp.exists():
            m = json.loads(mp.read_text())
            entry["metrics"] = m
        entry["checkpoints"] = sorted(p.name for p in run.glob("*.pt"))
        out["runs"].append(entry)
    out["latest"] = out["runs"][-1] if out["runs"] else None
    return out


@app.get("/api/training/runs/{run_id}")
def training_run(run_id: str) -> dict:
    run = config.RUNS_DIR / run_id
    mp = run / "metrics.json"
    if not mp.exists():
        raise HTTPException(404, "run not found")
    return json.loads(mp.read_text())


class TrainRequest(BaseModel):
    epochs_head: int = 5
    epochs_finetune: int = 15
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    resume: str | None = None


@app.post("/api/training/start")
def training_start(body: TrainRequest) -> dict:
    global _training_proc
    if _training_proc is not None and _training_proc.poll() is None:
        raise HTTPException(409, f"training already running (pid {_training_proc.pid})")
    if DEVICE.type != "cuda":
        raise HTTPException(
            503, "CUDA device not available - refusing to start a CPU-only training run from the API."
        )

    run_id = body.resume or f"run-{len(list(config.RUNS_DIR.glob('run-*'))) + 1:03d}"
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log = config.ARTIFACTS / f"train_{run_id}.log"

    cmd = [
        sys.executable, "-u", "-m", "ml.train",
        "--epochs-head", str(body.epochs_head),
        "--epochs-finetune", str(body.epochs_finetune),
        "--batch-size", str(body.batch_size),
        "--lr-head", str(body.lr_head),
        "--lr-finetune", str(body.lr_finetune),
        "--run-id", run_id,
    ]
    if body.resume:
        cmd += ["--resume", body.resume]

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    handle = log.open("wb")
    _training_proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, env=env)
    return {"started": True, "run_id": run_id, "pid": _training_proc.pid, "log": str(log), "command": cmd}


# ------------------------------------------------------------------ evaluation


@app.get("/api/metrics")
def metrics() -> dict:
    files = sorted(config.METRICS_DIR.glob("evaluation_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {
            "available": False,
            "note": "No evaluation has been run yet. Evaluate a trained checkpoint with `python -m ml.evaluate`.",
        }
    payload = json.loads(files[-1].read_text())
    return {"available": True, "file": files[-1].name, "evaluation": payload}


@app.get("/api/metrics/misclassified")
def misclassified(true_class: str, predicted_class: str, limit: int = Query(6, ge=1, le=24)) -> dict:
    files = sorted(config.METRICS_DIR.glob("evaluation_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise HTTPException(404, "no evaluation available")
    payload = json.loads(files[-1].read_text())
    key = f"{true_class}__{predicted_class}"
    items = payload.get("misclassified_examples", {}).get(key, [])[:limit]
    return {
        "true_class": true_class,
        "predicted_class": predicted_class,
        "count": len(items),
        "examples": [
            {
                **it,
                "url": f"/dataset-images/{config.DATASET_VARIANT}/{it['true_class']}/{Path(it['path']).name}",
            }
            for it in items
        ],
    }


# ---------------------------------------------------------------------- models


def _served_checkpoint() -> str | None:
    if predictor.loaded and predictor.checkpoint_path:
        return str(predictor.checkpoint_path)
    ckpt = resolve_checkpoint()
    return str(ckpt) if ckpt else None


@app.get("/api/models")
def models() -> dict:
    from ml import model_registry as mreg

    rows = mreg.list_models(_served_checkpoint())
    deployed = next((m for m in rows if m["status"] == "DEPLOYED"), None)
    return {"count": len(rows), "models": rows,
            "deployed": deployed["run_id"] if deployed else None}


@app.get("/api/models/compare")
def models_compare(ids: str = Query("run-001,run-002")) -> dict:
    from ml import model_registry as mreg

    return mreg.compare_models([i.strip() for i in ids.split(",") if i.strip()],
                               _served_checkpoint())


class DeployRequest(BaseModel):
    run_id: str
    weights: str = "best.pt"
    note: str = ""


@app.post("/api/models/deploy")
def models_deploy(body: DeployRequest) -> dict:
    """Explicit deployment decision: pin production to a checkpoint.

    Pinning alone does not swap the live model; call POST /api/model/reload
    (no run_id — it follows the pin) to activate. Rollback = deploy the
    previous run again. Nothing here trains or auto-deploys.
    """
    from ml import model_registry as mreg

    try:
        pin = mreg.set_pin(body.run_id, body.weights, body.note)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    db.log_system("info", "deployment", f"pinned production to {pin['checkpoint']} ({body.note})")
    return {"pinned": pin, "active_after_reload": True,
            "note": "Call POST /api/model/reload to activate in the live process."}


@app.get("/api/models/deployment")
def models_deployment() -> dict:
    from ml import model_registry as mreg

    return {"pin": mreg.get_pin(), "served_checkpoint": _served_checkpoint()}


@app.get("/api/models/error-analysis")
def models_error_analysis(run_id: str = "run-002") -> dict:
    from ml import model_registry as mreg

    return mreg.error_analysis(run_id)


# ------------------------------------------------- predictions log / analytics


@app.get("/api/predictions")
def predictions(
    limit: int = Query(200, ge=1, le=500),
    model: str | None = None,
    cls: str | None = None,
    min_conf: float | None = None,
    max_conf: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    feedback: str | None = None,
) -> dict:
    rows = db.query_predictions(limit, model, cls, min_conf, max_conf,
                                date_from, date_to, feedback)
    return {"count": len(rows), "predictions": rows}


@app.get("/api/analytics")
def analytics() -> dict:
    return db.prediction_analytics()


# ------------------------------------------------------- active learning


@app.get("/api/active-learning/queue")
def active_learning_queue(limit: int = Query(100, ge=1, le=500)) -> dict:
    items = db.active_learning_queue(limit)
    return {"count": len(items), "items": items,
            "note": "Prioritised by measured signals only: confidence, user corrections, "
                    "class frequency, confusion recurrence, robot relevance."}


@app.get("/api/active-learning/stats")
def active_learning_stats() -> dict:
    return db.review_stats()


class ReviewAction(BaseModel):
    scan_id: str
    action: str  # accept | correct | reject | uncertain
    correct_class: str | None = None


@app.post("/api/active-learning/review")
def active_learning_review(body: ReviewAction) -> dict:
    if body.action not in ("accept", "correct", "reject", "uncertain"):
        raise HTTPException(422, "action must be accept|correct|reject|uncertain")
    row = db.get_scan(body.scan_id)
    if not row:
        raise HTTPException(404, "scan not found")
    predicted = row["predicted_class"]
    if body.action == "accept":
        cid = db.insert_correction(body.scan_id, predicted, predicted, "review", "verified")
        return {"correction_id": cid, "scan_id": body.scan_id, "verified_class": predicted,
                "candidate": True}
    if body.action == "correct":
        if not body.correct_class or body.correct_class not in config.CLASSES:
            raise HTTPException(422, "a valid correct_class is required")
        cid = db.insert_correction(body.scan_id, predicted, body.correct_class, "review", "verified")
        return {"correction_id": cid, "scan_id": body.scan_id, "verified_class": body.correct_class,
                "candidate": True}
    status = "rejected" if body.action == "reject" else "uncertain"
    cid = db.insert_correction(body.scan_id, predicted, body.correct_class or predicted,
                               "review", status)
    return {"correction_id": cid, "scan_id": body.scan_id, "status": status, "candidate": False}


@app.get("/api/dataset/candidates")
def dataset_candidates(limit: int = Query(200, ge=1, le=500)) -> dict:
    rows = db.dataset_candidates(limit)
    return {"count": len(rows), "candidates": rows,
            "note": "Verified reviewer labels for a future dataset version. "
                    "The frozen test set is never modified by this path."}


@app.get("/api/acquisition/targets")
def acquisition_targets() -> dict:
    from ml import dataset_registry as reg

    targets = reg.load_acq_targets()
    cands = db.dataset_candidates(10000)
    verified: dict[str, int] = {}
    for c in cands:
        canon = reg.canonical(c["correct_class"]) or c["correct_class"]
        verified[canon] = verified.get(canon, 0) + 1
    rows = {}
    for cls in reg.CANONICAL_13:
        have = verified.get(cls, 0)
        tgt = targets.get(cls, 0)
        rows[cls] = {"display": reg.CANONICAL_DISPLAY[cls], "verified": have,
                     "target": tgt, "remaining": max(0, tgt - have)}
    return {"per_class": rows,
            "note": "Targets are configurable planning goals (PUT here). Verified counts are live reviewer-verified labels."}


class AcqTargetsUpdate(BaseModel):
    targets: dict


@app.put("/api/acquisition/targets")
def acquisition_targets_put(body: AcqTargetsUpdate) -> dict:
    from ml import dataset_registry as reg

    try:
        return {"targets": reg.save_acq_targets(body.targets)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


# ----------------------------------------------------------------------- robot


@app.get("/api/robot/status")
def robot_status() -> dict:
    ROBOT._model_loaded = predictor.loaded
    return ROBOT.status()


class EStop(BaseModel):
    engage: bool = True


@app.post("/api/robot/emergency-stop")
def robot_estop(body: EStop) -> dict:
    result = ROBOT.emergency_stop(body.engage)
    db.log_robot_event("emergency_stop", state=result["state"], detail={"engage": body.engage})
    return result


@app.post("/api/robot/reset")
def robot_reset() -> dict:
    result = ROBOT.reset()
    db.log_robot_event("reset", state=result["state"])
    return result


class BinMapUpdate(BaseModel):
    mapping: dict


@app.put("/api/robot/bin-map")
def robot_bin_map(body: BinMapUpdate) -> dict:
    try:
        return {"bin_map": ROBOT.set_bin_map(body.mapping)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class PauseBody(BaseModel):
    paused: bool = True


@app.post("/api/robot/pause")
def robot_pause(body: PauseBody) -> dict:
    result = ROBOT.pause(body.paused)
    db.log_robot_event("pause" if body.paused else "resume", state=result["state"])
    return result


@app.get("/api/robot/config")
def robot_config_get() -> dict:
    return {
        **runtime_config.current(),
        "states": list(VALID_STATES),
        "simulated_motion_phases": list(SIMULATED_MOTION_PHASES),
        "capabilities": ROBOT.capabilities(),
    }


class ThresholdUpdate(BaseModel):
    confidence_threshold: float | None = None
    margin_threshold: float | None = None


@app.put("/api/robot/config")
def robot_config_put(body: ThresholdUpdate) -> dict:
    try:
        updated = runtime_config.save_thresholds(body.confidence_threshold, body.margin_threshold)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    db.log_robot_event(
        "config_change",
        state=ROBOT.state,
        detail={
            "confidence_threshold": updated["confidence_threshold"],
            "margin_threshold": updated["margin_threshold"],
        },
    )
    return updated


@app.get("/api/robot/capabilities")
def robot_capabilities() -> dict:
    return ROBOT.capabilities()


@app.post("/api/robot/sort")
async def robot_sort(file: UploadFile = File(...)) -> dict:  # noqa: B008
    """Real end-to-end sorting command: upload -> inference -> gated decision.

    The returned `command` block is the payload a physical controller would
    receive. Nothing is actuated here because no hardware is attached.
    """
    if ROBOT.emergency:
        raise HTTPException(409, "emergency stop engaged - release it before sorting")
    if ROBOT.paused:
        raise HTTPException(409, "robot is paused")

    from inference.preprocessing import decode_upload

    raw = await file.read()
    try:
        pil, ext = decode_upload(raw, file.filename)
    except ValueError as exc:
        raise HTTPException(413 if "exceeds" in str(exc) else 400, str(exc)) from exc

    saved = config.UPLOADS_DIR / f"robot_{db.new_id()}{ext}"
    saved.write_bytes(raw)

    p = ensure_model()
    ROBOT.set_state("SCANNING")
    ROBOT.set_state("CLASSIFYING")
    result = p.predict(pil)
    decision = ROBOT.decide(result, source="robot_api")
    db.log_robot_event(
        "sort_command",
        state=ROBOT.state,
        cls=decision["class"],
        confidence=decision["confidence"],
        target_bin=decision["target_bin"],
        detail={"decision": decision["decision"], "reason": decision["reason"], "image": str(saved)},
    )
    return {
        "command": {
            "action": decision["decision"],
            "target_bin": decision["target_bin"],
            "bin_label": decision["bin_label"],
            "class": decision["class"],
            "confidence": decision["confidence"],
            "centroid": decision["centroid"],
            "reason": decision["reason"],
            "issued_at": decision["timestamp"],
            "actuated": False,
            "note": "no hardware attached - command logged only",
        },
        "robot_state": ROBOT.state,
        "prediction": result["prediction"],
        "image_url": f"/uploads/{saved.name}",
        "telemetry": ROBOT.status()["telemetry"],
    }


@app.get("/api/system/camera")
def system_camera(index: int = Query(0, ge=0, le=4)) -> dict:
    """Actually try to open a camera device and grab a frame."""
    return hardware.probe_camera(index)


@app.get("/api/robot/events")
def robot_events(limit: int = Query(100, ge=1, le=500)) -> dict:
    rows = db.list_robot_events(limit)
    return {"count": len(rows), "events": rows}


@app.post("/api/robot/simulate/step")
def robot_simulate_step(save: bool = True) -> dict:
    """Pull a real held-out image, run the real model, and make a real decision."""
    if ROBOT.emergency:
        raise HTTPException(409, "emergency stop engaged - release it before running the simulator")
    if ROBOT.paused:
        raise HTTPException(409, "robot is paused")
    path = sample_heldout_image()
    if path is None:
        raise HTTPException(503, "test split manifest unavailable")

    p = ensure_model()
    ROBOT.set_state("SCANNING")
    try:
        with Image.open(path) as im:
            im.load()
            pil = im.convert("RGB")
    except OSError as exc:
        ROBOT.set_state("ERROR")
        raise HTTPException(500, f"could not read sample: {exc}") from exc

    ROBOT.set_state("CLASSIFYING")
    result = p.predict(pil)
    decision = ROBOT.decide(result, source="simulation")

    scan_id = None
    if save:
        scan_id = db.insert_scan(
            {
                "source": "simulation",
                "image_path": str(path),
                "predicted_class": result["prediction"]["class"],
                "confidence": result["prediction"]["confidence"],
                "state": result["prediction"]["state"],
                "action": decision["decision"],
                "target_bin": decision["target_bin"],
                "top5": result["prediction"]["top5"],
                "latency": result["timings_ms"],
                "quality": result["input_quality"],
                "centroid": result["explainability"]["activation_centroid"],
                "model_version": result["model"]["version"],
                "dataset_version": result["model"].get("dataset_version"),
            }
        )

    db.log_robot_event(
        "sort_decision",
        state=ROBOT.state,
        cls=decision["class"],
        confidence=decision["confidence"],
        target_bin=decision["target_bin"],
        detail={"decision": decision["decision"], "reason": decision["reason"], "source_file": str(path)},
    )

    return {
        "decision": decision,
        "robot_state": ROBOT.state,
        "motion_phases": list(SIMULATED_MOTION_PHASES) if decision["decision"] == "SORT" else [],
        "scan_id": scan_id,
        "source_image": f"/dataset-images/{config.DATASET_VARIANT}/{path.parent.name}/{path.name}",
        "prediction": result["prediction"],
        "overlay_png_b64": result["explainability"]["overlay_png_b64"],
        "centroid": result["explainability"]["activation_centroid"],
        "timings_ms": result["timings_ms"],
        "telemetry": ROBOT.status()["telemetry"],
    }


@app.get("/api/robot/compatibility")
def robot_compatibility() -> dict:
    ckpt = resolve_checkpoint()
    stats_path = config.SPLITS_DIR / "dataset_stats.json"
    checks = {
        "model_checkpoint_present": ckpt is not None,
        "model_loaded": predictor.loaded,
        "input_size_supported": predictor.loaded and predictor.meta.get("training_config", {}).get("image_size") == config.IMAGE_SIZE,
        "classes_available": predictor.loaded and len(predictor.meta.get("classes", [])) == config.NUM_CLASSES,
        "robot_api_compatible": True,
        "simulation_mode_active": ROBOT.mode == "simulation",
        "sorting_map_configured": all(c in ROBOT.bin_map for c in config.CLASSES),
        "dataset_split_available": stats_path.exists(),
        "cuda_available": DEVICE.type == "cuda",
    }
    ready = all(checks.values())
    return {
        "model_version": predictor.version if predictor.loaded else None,
        "checks": checks,
        "ready_for_deployment": ready,
        "blocking": [k for k, v in checks.items() if not v],
        "hardware_connected": False,
        "interface": "SIMULATION",
        "note": "Software readiness only - no physical robot or arm is attached.",
    }


@app.post("/api/robot/predict")
async def robot_predict(file: UploadFile = File(...)) -> dict:  # noqa: B008
    """Classify one image for the robot pipeline without issuing a motion command.

    Always marked simulation:true - perception only, no actuation path here.
    """
    from inference.preprocessing import decode_upload

    raw = await file.read()
    try:
        pil, _ext = decode_upload(raw, file.filename)
    except ValueError as exc:
        raise HTTPException(413 if "exceeds" in str(exc) else 400, str(exc)) from exc
    p = ensure_model()
    ROBOT.set_state("CLASSIFYING")
    result = p.predict(pil)
    return {
        "simulation": True,
        "prediction": result["prediction"],
        "centroid": result["explainability"]["activation_centroid"],
        "timings_ms": result["timings_ms"],
        "model_version": result["model"]["version"],
        "note": "perception only - use /api/robot/sort for a gated sorting decision",
    }


@app.post("/api/robot/stop")
def robot_stop() -> dict:
    """Stop all motion (engages the emergency stop). Simulation: flags only."""
    from api.robot import ADAPTER

    out = ADAPTER.stop()
    db.log_robot_event("emergency_stop", state=ROBOT.state, detail={"via": "stop"})
    return {"simulation": True, **out}


class RobotFeedback(BaseModel):
    scan_id: str | None = None
    correct: bool = True
    correct_class: str | None = None


@app.post("/api/robot/feedback")
def robot_feedback(body: RobotFeedback) -> dict:
    """Feedback on a robot-pipeline prediction. Stored like any other feedback."""
    if not body.scan_id:
        last = ROBOT.current
        if not last:
            raise HTTPException(404, "no robot decision to give feedback on")
        return {"simulation": True, "stored": False,
                "note": "robot decisions are in-memory; upload-linked scans carry feedback via /api/scans/{id}/feedback"}
    row = db.get_scan(body.scan_id)
    if not row:
        raise HTTPException(404, "scan not found")
    predicted = row["predicted_class"]
    final = predicted if body.correct else body.correct_class
    if not body.correct and (not final or final not in config.CLASSES):
        raise HTTPException(422, "correct_class is required when correct=false")
    cid = db.insert_correction(body.scan_id, predicted, final, "robot")
    return {"simulation": True, "stored": True, "correction_id": cid,
            "verified_class": final}


@app.get("/api/robot/adapter")
def robot_adapter() -> dict:
    from api.robot import ADAPTER

    return {"adapter": ADAPTER.name, "simulation": True,
            "status": ADAPTER.get_status(),
            "note": "SimulationRobotAdapter only. ROS2/Arduino/RaspberryPi adapters do not exist yet."}


# --------------------------------------------------------------- static mounts

if config.DATASET_ROOT.is_dir():
    app.mount("/dataset-images", StaticFiles(directory=str(config.DATASET_ROOT)), name="dataset")
config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(config.UPLOADS_DIR)), name="uploads")


@app.get("/api")
def api_index() -> dict:
    return {
        "service": "EcoSort API",
        "endpoints": sorted({r.path for r in app.routes if getattr(r, "path", "").startswith("/api")}),
    }


if WEB_DIST.is_dir():

    class SPAStaticFiles(StaticFiles):
        """Serve the SPA bundle with index.html as the fallback for client-side routes."""

        async def get_response(self, path: str, scope):
            try:
                return await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code == 404:
                    return await super().get_response("index.html", scope)
                raise

    app.mount("/", SPAStaticFiles(directory=str(WEB_DIST), html=True), name="web")
else:

    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "service": "EcoSort API",
                "frontend": "not built - run the Vite dev server or build web/",
                "docs": "/docs",
            }
        )
