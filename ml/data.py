"""Deterministic dataset audit + stratified split builder.

Hashes every image to find true duplicates, detects near-duplicates with a
dHash, filters unreadable/low-quality files, maps directory names onto the
canonical 13-class taxonomy, and writes reproducible train/val/test manifests
so training never depends on directory ordering.

Near-duplicate groups are kept inside a single split: the same (or
near-identical) image can never appear in both training and test sets.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageStat

from . import config
from . import dataset_registry as reg

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NEAR_HAMMING = 4
MIN_SIDE = 96


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _dhash(path: Path, size: int = 8) -> str | None:
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
            px = list(im.getdata())
            bits = 0
            for r in range(size):
                for c in range(size):
                    bits = (bits << 1) | (1 if px[r * (size + 1) + c] < px[r * (size + 1) + c + 1] else 0)
            return f"{bits:016x}"
    except Exception:
        return None


def _hamming(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except ValueError:
        return 64


def _quality(path: Path) -> tuple[str, str]:
    """Return (status, reason): ok | corrupt | low_quality."""
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            if min(w, h) < MIN_SIDE:
                return "low_quality", f"too small ({w}x{h})"
            g = im.convert("L")
            stat = ImageStat.Stat(g)
            if (stat.mean[0] if stat.mean else 0) < 8.0:
                return "low_quality", "too dark"
            hist = g.histogram()
            total = sum(hist) or 1
            mean = sum(i * c for i, c in enumerate(hist)) / total
            var = sum(c * (i - mean) ** 2 for i, c in enumerate(hist)) / total
            if var < 12.0:
                return "low_quality", "flat/low-detail"
            return "ok", "ok"
    except Exception as exc:
        return "corrupt", f"unreadable ({exc})"


def _assess(path: Path, size: int = 8) -> tuple[str, str, str | None]:
    """Single-open quality gate + dHash. Returns (status, reason, dhash)."""
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            if min(w, h) < MIN_SIDE:
                return "low_quality", f"too small ({w}x{h})", None
            g = im.convert("L")
    except Exception as exc:
        return "corrupt", f"unreadable ({exc})", None
    try:
        stat = ImageStat.Stat(g)
        if (stat.mean[0] if stat.mean else 0) < 8.0:
            return "low_quality", "too dark", None
        hist = g.histogram()
        total = sum(hist) or 1
        mean = sum(i * c for i, c in enumerate(hist)) / total
        var = sum(c * (i - mean) ** 2 for i, c in enumerate(hist)) / total
        if var < 12.0:
            return "low_quality", "flat/low-detail", None
        small = g.resize((size + 1, size), Image.BILINEAR)
        if hasattr(small, "get_flattened_data"):
            px = list(small.get_flattened_data())
        else:
            px = list(small.getdata())
        bits = 0
        for r in range(size):
            for c in range(size):
                bits = (bits << 1) | (1 if px[r * (size + 1) + c] < px[r * (size + 1) + c + 1] else 0)
        return "ok", "ok", f"{bits:016x}"
    except Exception as exc:
        return "corrupt", f"unreadable ({exc})", None


class _UnionFind:
    def __init__(self) -> None:
        self.p: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def audit_and_split(near_hamming: int = NEAR_HAMMING) -> dict:
    root = config.DATASET_DIR
    if not root.is_dir():
        raise SystemExit(f"dataset not found: {root}")

    # NOTE: class keys stay as the on-disk directory names (e.g. "biological")
    # so labels remain compatible with the trained model, BIN_MAP and GUIDE.
    # The canonical 13-class view is derived in dataset_registry.canonical().
    by_class: dict[str, list[Path]] = defaultdict(list)
    unmapped = 0
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        raw = cls_dir.name.lower()
        if reg.canonical(raw) is None:
            print(f"  ! skipping unknown class dir: {cls_dir.name}")
            unmapped += sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in EXTS)
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in EXTS:
                by_class[raw].append(f)

    # Pass 1: quality gating + exact dedup.
    seen: dict[str, Path] = {}
    duplicates: list[dict] = []
    candidates: list[dict] = []  # {path, class, md5, dhash}
    corrupt: list[str] = []
    low_quality: list[str] = []

    for cls, files in by_class.items():
        for f in files:
            status, _reason, dh = _assess(f)
            if status == "corrupt":
                corrupt.append(str(f))
                continue
            if status == "low_quality":
                low_quality.append(str(f))
                continue
            try:
                digest = _md5(f)
            except OSError:
                corrupt.append(str(f))
                continue
            if digest in seen:
                duplicates.append({"path": str(f), "duplicate_of": str(seen[digest])})
            else:
                seen[digest] = f
                candidates.append({"path": f, "class": cls, "md5": digest, "dhash": dh})

    # Pass 2: near-duplicate grouping (dHash hamming, same canonical class
    # only). Keep only the first file per group as verified; record the rest
    # as removed. Grouping within a class keeps this O(sum n_c^2) and, more
    # importantly, keeps near-identical images inside a single split later.
    #
    # Blocking makes it linear: with threshold <= 4 over 64 bits, any
    # near-duplicate pair must match exactly on at least 4 of 8 byte-windows
    # (pigeonhole), so only pairs sharing a (class, window, byte) bucket are
    # ever compared. Recall is exact, not heuristic.
    uf = _UnionFind()
    buckets: dict[tuple, list[int]] = defaultdict(list)
    hint: dict[int, int] = {}
    for i, c in enumerate(candidates):
        dh = c["dhash"]
        if not dh:
            continue
        try:
            h = int(dh, 16)
        except ValueError:
            continue
        hint[i] = h
        cls = c["class"]
        for k in range(8):
            buckets[(cls, k, (h >> (k * 8)) & 0xFF)].append(i)
    seen_pairs: set[int] = set()
    for members in buckets.values():
        m = len(members)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                i, j = members[a], members[b]
                key = (i << 32) | j if i < j else (j << 32) | i
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if (hint[i] ^ hint[j]).bit_count() <= near_hamming:
                    uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(candidates)):
        groups[uf.find(i)].append(i)

    unique: dict[str, list[Path]] = defaultdict(list)
    near_dupes: list[dict] = []
    for members in groups.values():
        members.sort(key=lambda i: str(candidates[i]["path"]))
        keep = members[0]
        unique[candidates[keep]["class"]].append(candidates[keep]["path"])
        for other in members[1:]:
            near_dupes.append(
                {"path": str(candidates[other]["path"]), "near_duplicate_of": str(candidates[keep]["path"])}
            )

    # Leakage-safe stratified split with STABLE assignments: files already
    # present in a previous manifest keep their split, so ingesting new data
    # can never move a frozen test image into training (or vice versa). Only
    # previously-unassigned files are distributed, per class, toward the
    # configured ratios. Near-duplicate groups were already collapsed above,
    # so no near-identical pair can straddle splits either.
    rng = random.Random(config.SEED)
    prior: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        mp = config.SPLITS_DIR / f"{split_name}.json"
        if mp.exists():
            try:
                for row in json.loads(mp.read_text()):
                    prior[row["path"]] = split_name
            except ValueError:
                pass
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    per_class: dict[str, Counter] = {c: Counter() for c in unique}

    for cls, files in unique.items():
        label = config.CLASS_TO_IDX.get(cls, -1)
        old = [p for p in files if str(p) in prior]
        new = [p for p in files if str(p) not in prior]
        for p in old:
            splits[prior[str(p)]].append({"path": str(p), "label": label, "class": cls})
            per_class[cls][prior[str(p)]] += 1
        if new:
            counts = {s: per_class[cls][s] for s in ("train", "val", "test")}
            total = len(files)
            want = {
                "train": round(total * config.SPLIT_RATIOS[0]),
                "val": round(total * config.SPLIT_RATIOS[1]),
            }
            need = {
                "train": max(0, want["train"] - counts["train"]),
                "val": max(0, want["val"] - counts["val"]),
            }
            shuffled = new[:]
            rng.shuffle(shuffled)
            buckets = (
                [("train", p) for p in shuffled[: need["train"]]]
                + [("val", p) for p in shuffled[need["train"] : need["train"] + need["val"]]]
                + [("test", p) for p in shuffled[need["train"] + need["val"] :]]
            )
            for split_name, path in buckets:
                splits[split_name].append({"path": str(path), "label": label, "class": cls})
                per_class[cls][split_name] += 1

    for split_name in splits:
        rng.shuffle(splits[split_name])

    total_unique = sum(len(v) for v in unique.values())
    stats = {
        "dataset_dir": str(root),
        "variant": config.DATASET_VARIANT,
        "image_size": config.IMAGE_SIZE,
        "seed": config.SEED,
        "split_ratios": list(config.SPLIT_RATIOS),
        "classes": sorted(unique),
        "num_classes": len(unique),
        "canonical_classes": list(reg.CANONICAL_13),
        "total_files_scanned": sum(len(v) for v in by_class.values()),
        "total_unique": total_unique,
        "duplicates_removed": len(duplicates),
        "near_duplicates_removed": len(near_dupes),
        "corrupt_unreadable": len(corrupt),
        "low_quality_removed": len(low_quality),
        "unmapped_skipped": unmapped,
        "augmentation": {"enabled": True, "counted_as_source_images": False},
        "stable_splits": True,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "per_class": {
            cls: {
                "unique": len(unique[cls]),
                "train": per_class[cls]["train"],
                "val": per_class[cls]["val"],
                "test": per_class[cls]["test"],
            }
            for cls in sorted(unique)
        },
    }

    for split_name, rows in splits.items():
        out = config.SPLITS_DIR / f"{split_name}.json"
        out.write_text(json.dumps(rows, indent=0))

    (config.SPLITS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    if duplicates:
        (config.SPLITS_DIR / "duplicates.json").write_text(json.dumps(duplicates, indent=0))
    if near_dupes:
        (config.SPLITS_DIR / "near_duplicates.json").write_text(json.dumps(near_dupes, indent=0))

    try:
        reg.ensure_seed_version(stats)
        reg.health_report(stats)
    except Exception as exc:
        print(f"  ! registry update skipped: {exc}")

    return stats


if __name__ == "__main__":
    s = audit_and_split()
    print(json.dumps({k: v for k, v in s.items() if k != "per_class"}, indent=2))
    print("\nper-class:")
    for cls, d in s["per_class"].items():
        print(f"  {cls:<12} unique={d['unique']:<5} train={d['train']:<5} val={d['val']:<4} test={d['test']}")
