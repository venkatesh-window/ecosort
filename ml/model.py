"""EfficientNetB0 with a 10-class waste head, plus Grad-CAM hook target."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from . import config


def build_model(num_classes: int = config.NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    if not pretrained:
        nn.init.xavier_uniform_(model.classifier[1].weight)
        nn.init.zeros_(model.classifier[1].bias)

    return model


def freeze_backbone(model: nn.Module, frozen: bool = True) -> None:
    for p in model.features.parameters():
        p.requires_grad = not frozen


def last_conv_block(model: nn.Module) -> nn.Module:
    """Final conv block of EfficientNetB0 - the standard Grad-CAM target."""
    return model.features[-1]


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def load_checkpoint(path, device: torch.device, num_classes: int = config.NUM_CLASSES):
    model = build_model(num_classes=num_classes, pretrained=False)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model, state
