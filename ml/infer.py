"""Inference engine: loads a trained checkpoint and produces predictions with
Grad-CAM explanations, activation centroid, measured latency and input-quality
signals. All values are computed at request time - nothing is hardcoded.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from . import config
from .gradcam import GradCAM, activation_centroid, colorize_cam, resize_cam
from .model import build_model
from .recycling import UNCERTAIN_TIPS, guidance


def _pick_device() -> torch.device:
    # device_count() matters: a driver can be present while no GPU is visible
    # (e.g. CUDA_VISIBLE_DEVICES=""), in which case cuda is unusable.
    if torch.cuda.is_available():
        try:
            if torch.cuda.device_count() > 0:
                return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


DEVICE = _pick_device()

EVAL_TF = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ]
)


def encode_png(arr_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("png encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def input_quality(rgb: np.ndarray) -> dict:
    """Real, measurable image-quality signals used to advise rescans."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean()) / 255.0
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape[:2]
    resolution_ok = min(h, w) >= 128

    issues = []
    if brightness < 0.18:
        issues.append("Image is very dark")
    elif brightness > 0.90:
        issues.append("Image is overexposed")
    if lap_var < 60:
        issues.append("Image looks blurry or low-detail")
    if not resolution_ok:
        issues.append("Source resolution is low")

    score = 1.0
    score -= min(abs(brightness - 0.5) * 0.9, 0.45)
    score -= max(0.0, min((120 - lap_var) / 120, 1.0)) * 0.4
    if not resolution_ok:
        score -= 0.15
    score = round(float(np.clip(score, 0.0, 1.0)), 3)

    return {
        "score": score,
        "label": "Good" if score >= 0.66 else ("Fair" if score >= 0.4 else "Poor"),
        "brightness": round(brightness, 4),
        "sharpness": round(lap_var, 2),
        "resolution": {"width": w, "height": h},
        "issues": issues,
    }


class Predictor:
    def __init__(self, checkpoint: Path | None = None):
        self.checkpoint_path = checkpoint
        self.model = None
        self.meta: dict = {}
        self.loaded = False
        if checkpoint:
            self.load(checkpoint)

    def load(self, checkpoint: Path) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = build_model(num_classes=config.NUM_CLASSES, pretrained=False)
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(DEVICE).eval()
        self.model = model
        self.checkpoint_path = checkpoint
        self.meta = {
            "run_id": checkpoint.parent.name,
            "weights": checkpoint.name,
            "classes": state.get("classes", config.CLASSES),
            "best_val_acc": state.get("best_val_acc"),
            "epochs_completed": (state.get("epoch") or 0) + 1,
            "training_config": state.get("config", {}),
            "architecture": "efficientnet_b0",
            "input_size": config.IMAGE_SIZE,
            "device": str(DEVICE),
        }
        self.loaded = True
        self._warmup()

    def _warmup(self) -> None:
        """Absorb cuDNN autotune + lazy kernel compilation so the first real
        request reports meaningful latency instead of cold-start overhead."""
        try:
            dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=DEVICE)
            with torch.no_grad():
                for _ in range(3):
                    self.model(dummy)
            with GradCAM(self.model, DEVICE) as cam:
                cam.generate(dummy, class_idx=0)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
        except Exception as exc:
            print(f"warmup skipped: {exc}")

    @property
    def version(self) -> str:
        run = self.meta.get("run_id", "untrained")
        val = self.meta.get("best_val_acc")
        suffix = f"-val{val:.3f}".replace("0.", "") if isinstance(val, float) else ""
        return f"ecosort-efficientnetb0-{run}{suffix}"

    def _to_tensor(self, pil: Image.Image) -> torch.Tensor:
        return EVAL_TF(pil).unsqueeze(0).to(DEVICE)

    @torch.no_grad()
    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, pil: Image.Image, top_k: int = 5, cam_alpha: float = 0.45) -> dict:
        if not self.loaded:
            raise RuntimeError("no model checkpoint loaded")

        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        original = np.array(pil.convert("RGB"))
        x = self._to_tensor(pil.convert("RGB"))
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        probs = F.softmax(self._logits(x), dim=1).squeeze(0)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        timings["model_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        order = torch.argsort(probs, descending=True)
        top = order[:top_k]
        top5 = [
            {
                "class": config.CLASSES[int(i)],
                "display": config.display(config.CLASSES[int(i)]),
                "probability": round(float(probs[int(i)]), 6),
                "index": int(i),
            }
            for i in top
        ]
        best = top5[0]
        confidence = best["probability"]
        entropy = float(-(probs * torch.clamp(probs, min=1e-12).log()).sum().item())
        max_entropy = float(np.log(config.NUM_CLASSES))

        confident = confidence >= config.CONFIDENCE_THRESHOLD
        margin = confidence - top5[1]["probability"] if len(top5) > 1 else confidence

        if confident and margin >= config.MARGIN_THRESHOLD:
            state, state_reason = "high", "Clear margin over the next-best class."
        elif confident:
            state, state_reason = "moderate", "Above threshold but the runner-up is close."
        else:
            state, state_reason = "low", "Below the configured confidence threshold."

        timings["postprocess_ms"] = (time.perf_counter() - t2) * 1000

        # Grad-CAM requires gradients, so it runs outside the no_grad block above.
        t3 = time.perf_counter()
        with GradCAM(self.model, DEVICE) as cam:
            cam_map, _cam_class = cam.generate(x, class_idx=best["index"])
        cam_resized = resize_cam(cam_map, config.IMAGE_SIZE)
        resized_original = np.array(
            pil.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
        )
        overlay = colorize_cam(cam_resized, alpha=cam_alpha, image_rgb=resized_original)
        heatmap = colorize_cam(cam_resized)
        centroid = activation_centroid(cam_resized)
        timings["gradcam_ms"] = (time.perf_counter() - t3) * 1000
        timings["total_ms"] = sum(timings.values())

        cls = best["class"]
        quality = input_quality(original)

        return {
            "prediction": {
                "class": cls,
                "display": best["display"],
                "confidence": confidence,
                "confidence_pct": round(confidence * 100, 2),
                "state": state,
                "state_reason": state_reason,
                "margin": round(margin, 6),
                "entropy": round(entropy, 4),
                "normalized_entropy": round(entropy / max_entropy, 4),
                "threshold": config.CONFIDENCE_THRESHOLD,
                "top5": top5,
            },
            "explainability": {
                "method": "gradcam",
                "target_layer": "features[-1] (final conv block)",
                "heatmap_png_b64": encode_png(heatmap),
                "overlay_png_b64": encode_png(overlay),
                "cam_size": [int(cam_resized.shape[1]), int(cam_resized.shape[0])],
                "activation_centroid": centroid,
                "opacity": cam_alpha,
            },
            "guidance": guidance(cls),
            "sorting": {
                "target_bin": config.BIN_MAP[cls],
                "bin_label": config.BIN_LABELS[config.BIN_MAP[cls]],
                "action": "SORT" if confident else "HOLD_FOR_REVIEW",
                "reason": (
                    "Confidence above threshold - safe to issue a sorting command."
                    if confident
                    else "Confidence below threshold - routing to human review instead of actuating."
                ),
                "centroid": centroid,
            },
            "input_quality": quality,
            "uncertainty_tips": UNCERTAIN_TIPS if state == "low" else [],
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
            "model": {
                "version": self.version,
                "run_id": self.meta.get("run_id"),
                "architecture": "efficientnet_b0",
                "input_size": config.IMAGE_SIZE,
                "device": str(DEVICE),
                "num_classes": config.NUM_CLASSES,
                "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
                "dataset_version": (self.meta.get("training_config") or {}).get("dataset_version"),
                "dataset_verified_images": (self.meta.get("training_config") or {}).get(
                    "dataset_verified_images"
                ),
            },
        }

    def predict_file(self, path: str | Path, **kw) -> dict:
        with Image.open(path) as im:
            im.load()
            return self.predict(im, **kw)


def resolve_checkpoint(run_id: str | None = None, weights: str = "best.pt") -> Path | None:
    """Explicit pin wins; otherwise the newest completed run's checkpoint."""
    if run_id:
        p = config.RUNS_DIR / run_id / weights
        return p if p.exists() else None
    try:
        from .model_registry import get_pin

        pin = get_pin()
        if pin:
            p = Path(pin["checkpoint"])
            if p.exists():
                return p
    except Exception:
        pass
    candidates = []
    for run in sorted(config.RUNS_DIR.glob("run-*")):
        for name in ("best.pt", "final.pt", "last.pt"):
            c = run / name
            if c.exists():
                candidates.append(c)
                break
    return candidates[-1] if candidates else None
