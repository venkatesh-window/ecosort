"""Genuine Grad-CAM for EfficientNetB0.

Registers hooks on the final conv block, backpropagates the target class score,
and produces a class activation map. The activation-weighted centroid is used as
the object-position estimate for the robot layer - this is an honest proxy, not a
bounding box from a detector.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import last_conv_block


class GradCAM:
    def __init__(self, model: torch.nn.Module, device: torch.device, target_layer=None):
        self.model = model
        self.device = device
        self.layer = target_layer or last_conv_block(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd = self.layer.register_forward_hook(self._save_fwd)
        self._bwd = self.layer.register_full_backward_hook(self._save_bwd)

    def _save_fwd(self, module, inp, out):
        self._activations = out.detach()

    def _save_bwd(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """Returns (cam in [0,1] as HxW float32, class index used)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)

        grads = self._gradients
        acts = self._activations
        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        denom = cam.max()
        if float(denom) > 0:
            cam = cam / denom
        return cam.squeeze().detach().cpu().numpy().astype(np.float32), class_idx


def resize_cam(cam: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return out.squeeze().numpy()


def activation_centroid(cam: np.ndarray) -> dict:
    """Intensity-weighted centroid of the activation map, in normalized [0,1]
    coords plus pixel coords for the given map resolution."""
    h, w = cam.shape
    total = float(cam.sum())
    if total <= 0:
        return {
            "x_norm": 0.5,
            "y_norm": 0.5,
            "x_px": w // 2,
            "y_px": h // 2,
            "concentration": 0.0,
            "source": "fallback_center",
        }

    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((cam * xs).sum() / total)
    cy = float((cam * ys).sum() / total)

    # Fraction of total activation inside one std-dev radius = focus measure.
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    spread = float(np.sqrt((cam * dist**2).sum() / total))
    concentration = float(np.clip(1.0 - spread / (max(h, w) / 2.0), 0.0, 1.0))

    return {
        "x_norm": round(cx / max(w - 1, 1), 4),
        "y_norm": round(cy / max(h - 1, 1), 4),
        "x_px": round(cx),
        "y_px": round(cy),
        "concentration": round(concentration, 4),
        "source": "gradcam_activation_centroid",
    }


def colorize_cam(cam: np.ndarray, alpha: float = 0.5, image_rgb: np.ndarray | None = None) -> np.ndarray:
    """Jet-colormapped heatmap, optionally alpha-blended over the source image.
    Returns uint8 HxWx3 RGB."""
    import cv2

    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    if image_rgb is None:
        return heat
    return cv2.addWeighted(image_rgb, 1 - alpha, heat, alpha, 0)
