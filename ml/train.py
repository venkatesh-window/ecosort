"""Real EfficientNetB0 transfer-learning trainer with checkpointing, resume
and live metrics that the web Training Center reads.

Two phases:
  1. frozen ImageNet backbone, train the 10-class head
  2. unfreeze, fine-tune everything at a lower LR

Class imbalance is handled with a weighted random sampler.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from . import config
from .model import build_model, count_parameters, freeze_backbone

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ManifestDataset(Dataset):
    def __init__(self, split: str, transform):
        manifest = config.SPLITS_DIR / f"{split}.json"
        if not manifest.exists():
            raise SystemExit(f"missing split manifest {manifest} - run `python -m ml.data` first")
        self.rows = json.loads(manifest.read_text())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, row["label"]


def build_transforms() -> tuple:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
            transforms.RandomErasing(p=0.15),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ]
    )
    return train_tf, eval_tf


def next_run_id() -> str:
    existing = sorted(p.name for p in config.RUNS_DIR.glob("run-*") if p.is_dir())
    return f"run-{len(existing) + 1:03d}"


def make_sampler(dataset: ManifestDataset) -> WeightedRandomSampler:
    counts = Counter(r["label"] for r in dataset.rows)
    weight_per_class = {c: 1.0 / n for c, n in counts.items()}
    weights = [weight_per_class[r["label"]] for r in dataset.rows]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


@torch.no_grad()
def evaluate(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def write_metrics(run_dir: Path, payload: dict) -> None:
    tmp = run_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(run_dir / "metrics.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--resume", type=str, default=None, help="run id to resume")
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)

    run_id = args.resume or args.run_id or next_run_id()
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms()
    train_ds = ManifestDataset("train", train_tf)
    val_ds = ManifestDataset("val", eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train_ds),
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    total_epochs = args.epochs_head + args.epochs_finetune
    start_epoch = 0
    history: list[dict] = []
    best_val_acc = 0.0

    ckpt_path = run_dir / "last.pt"
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        start_epoch = state["epoch"] + 1
        best_val_acc = state.get("best_val_acc", 0.0)
        history = state.get("history", [])
        print(f"resumed {run_id} at epoch {start_epoch} (best val acc {best_val_acc:.4f})")

    params_meta = count_parameters(model)
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    epoch_times: list[float] = []

    print(f"device={DEVICE} run={run_id} params(total)={params_meta['total']}")
    print(f"train={len(train_ds)} val={len(val_ds)} epochs={total_epochs} bs={args.batch_size}")

    run_meta = {
        "image_size": config.IMAGE_SIZE,
        "architecture": "efficientnet_b0",
        "pretrained": "IMAGENET1K_V1",
        "dataset_variant": config.DATASET_VARIANT,
        "dataset_dir": str(config.DATASET_DIR),
        "num_classes": config.NUM_CLASSES,
        "classes": config.CLASSES,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "optimizer": "adamw",
        "augmentation": {
            "enabled": True,
            "counted_as_source_images": False,
            "transforms": "random-resized-crop, hflip, rotation, color-jitter, random-erasing",
        },
    }
    try:
        from . import dataset_registry as _reg

        _stats = _reg.current_stats()
        _versions = _reg.load_versions()
        run_meta["dataset_version"] = _versions[-1].get("version") if _versions else None
        run_meta["dataset_verified_images"] = int(_stats.get("total_unique", 0)) if _stats else None
        run_meta["dataset_target"] = _reg.TARGET_TOTAL
    except Exception:
        run_meta["dataset_version"] = None

    for epoch in range(start_epoch, total_epochs):
        phase = "head" if epoch < args.epochs_head else "finetune"
        frozen = phase == "head"
        freeze_backbone(model, frozen)
        params_meta = count_parameters(model)

        trainable = [p for p in model.parameters() if p.requires_grad]
        lr = args.lr_head if phase == "head" else args.lr_finetune
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs - epoch, 1))

        model.train()
        running, correct, total, t0 = 0.0, 0, 0, time.time()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

            if step % 50 == 0:
                done = epoch + step / max(len(train_loader), 1)
                remaining_epochs = total_epochs - done
                avg = (sum(epoch_times) / len(epoch_times)) if epoch_times else (time.time() - t0) / max(step, 1)
                write_metrics(
                    run_dir,
                    {
                        "run_id": run_id,
                        "status": "running",
                        "phase": phase,
                        "epoch": epoch + 1,
                        "total_epochs": total_epochs,
                        "step": step,
                        "steps_per_epoch": len(train_loader),
                        "train_loss": running / max(total, 1),
                        "train_acc": correct / max(total, 1),
                        "best_val_acc": best_val_acc,
                        "lr": optimizer.param_groups[0]["lr"],
                        "progress": done / total_epochs,
                        "eta_seconds": int(avg * remaining_epochs),
                        "started_at": started_at,
                        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                        "device": str(DEVICE),
                        "params": params_meta,
                        "run": run_meta,
                        "history": history,
                    },
                )

        scheduler.step()
        epoch_seconds = time.time() - t0
        epoch_times.append(epoch_seconds)

        train_loss = running / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        best_val_acc = max(best_val_acc, val_acc)

        record = {
            "epoch": epoch + 1,
            "phase": phase,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "lr": round(lr, 8),
            "seconds": round(epoch_seconds, 1),
        }
        history.append(record)
        print(
            f"epoch {epoch + 1:>2}/{total_epochs} [{phase:<9}] "
            f"loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | {epoch_seconds:.0f}s"
        )

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "history": history,
            "classes": config.CLASSES,
            "config": {
                "image_size": config.IMAGE_SIZE,
                "batch_size": args.batch_size,
                "epochs_head": args.epochs_head,
                "epochs_finetune": args.epochs_finetune,
                "lr_head": args.lr_head,
                "lr_finetune": args.lr_finetune,
                "seed": config.SEED,
                "device": str(DEVICE),
                "architecture": "efficientnet_b0",
                "pretrained": "IMAGENET1K_V1",
                "optimizer": "adamw",
                "augmentation": "random-crop/hflip/rotate/jitter/erasing (train-time only, not counted as source images)",
                "dataset_version": run_meta.get("dataset_version"),
                "dataset_verified_images": run_meta.get("dataset_verified_images"),
            },
        }
        torch.save(payload, run_dir / "last.pt")
        if val_acc >= best_val_acc:
            torch.save(payload, run_dir / "best.pt")

        write_metrics(
            run_dir,
            {
                "run_id": run_id,
                "status": "running",
                "phase": phase,
                "epoch": epoch + 1,
                "total_epochs": total_epochs,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "lr": lr,
                "progress": (epoch + 1) / total_epochs,
                "eta_seconds": int((sum(epoch_times) / len(epoch_times)) * (total_epochs - epoch - 1)),
                "started_at": started_at,
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "device": str(DEVICE),
                "params": params_meta,
                "run": run_meta,
                "history": history,
            },
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": total_epochs - 1,
            "best_val_acc": best_val_acc,
            "history": history,
            "classes": config.CLASSES,
        },
        run_dir / "final.pt",
    )

    write_metrics(
        run_dir,
        {
            "run_id": run_id,
            "status": "completed",
            "phase": "done",
            "epoch": total_epochs,
            "total_epochs": total_epochs,
            "best_val_acc": best_val_acc,
            "progress": 1.0,
            "eta_seconds": 0,
            "started_at": started_at,
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "device": str(DEVICE),
            "params": params_meta,
            "run": run_meta,
            "history": history,
        },
    )
    print(f"done. best val acc {best_val_acc:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
