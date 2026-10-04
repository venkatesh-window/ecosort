"""Legitimate dataset ingestion for the 100K goal.

Copies only real source images (never duplicates-as-new-rows), validates
each file, maps source labels onto the canonical 13 classes, removes exact
(MD5) and near-duplicates (dHash), filters low-quality images, records
licence/source provenance, and rebuilds the split via ml.data.

Example:
    python -m ml.ingest --source-dir C:/data/taco --dataset-name TACO \\
        --license "CC BY 4.0" --url https://example.org/taco \\
        --class-map plastic:plastic glass:glass --copy

Nothing here downloads or scrapes images. The operator provides a local
directory of lawfully obtained images; this tool vets and registers them.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from ml import config, data
from ml import dataset_registry as reg

EXTS = set(data.EXTS)
MIN_SIDE = data.MIN_SIDE
_md5 = data._md5
_hamming = data._hamming
_assess = data._assess


def parse_class_map(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"bad --class-map entry {item!r}, expected src:canonical")
        src, dst = item.split(":", 1)
        c = reg.canonical(dst.strip())
        if not c:
            raise ValueError(f"unknown canonical class {dst!r}; choose from {reg.CANONICAL_13}")
        mapping[src.strip().lower()] = c
    return mapping


def ingest(
    source_dir: Path,
    dataset_name: str,
    license: str,
    url: str = "",
    class_map: dict[str, str] | None = None,
    copy: bool = True,
    near_hamming: int = 4,
    robot_ready: bool = False,
    dry_run: bool = False,
) -> dict:
    class_map = class_map or {}
    if not source_dir.is_dir():
        raise SystemExit(f"source dir not found: {source_dir}")

    # Index existing corpus hashes so imports never re-add duplicates.
    # Near-duplicate search is scoped per canonical class to stay tractable.
    from collections import defaultdict as _dd

    existing_md5: set[str] = set()
    existing_dh: dict[str, list[str]] = _dd(list)
    for cls_dir in sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir()):
        canon = reg.canonical(cls_dir.name) or cls_dir.name.lower()
        for f in cls_dir.iterdir():
            if f.suffix.lower() in EXTS:
                try:
                    existing_md5.add(_md5(f))
                except OSError:
                    continue
                status, _reason, dh = _assess(f)
                if status == 'ok' and dh:
                    existing_dh[canon].append(dh)

    seen_md5 = set(existing_md5)
    seen_dh: dict[str, list[str]] = _dd(list, {k: list(v) for k, v in existing_dh.items()})

    report = {
        "dataset": dataset_name,
        "scanned": 0,
        "imported": 0,
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "corrupt": 0,
        "low_quality": 0,
        "unmapped": 0,
        "per_class": {},
        "files": [],
    }

    for src_file in sorted(source_dir.rglob("*")):
        if not src_file.is_file() or src_file.suffix.lower() not in EXTS:
            continue
        report["scanned"] += 1
        raw_label = src_file.parent.name.lower()
        mapped = class_map.get(raw_label, reg.canonical(raw_label))
        if not mapped:
            report["unmapped"] += 1
            continue
        ok, _reason, dh = _assess(src_file)
        if not ok:
            # Distinguish unreadable from merely low-quality.
            try:
                with Image.open(src_file) as im:
                    im.load()
                report["low_quality"] += 1
            except Exception:
                report["corrupt"] += 1
            continue
        try:
            digest = _md5(src_file)
        except OSError:
            report["corrupt"] += 1
            continue
        if digest in seen_md5:
            report["exact_duplicates"] += 1
            continue
        if dh and any(_hamming(dh, e) <= near_hamming for e in seen_dh[mapped]):
            report["near_duplicates"] += 1
            continue
        seen_md5.add(digest)
        if dh:
            seen_dh[mapped].append(dh)
        dest = config.DATASET_DIR / reg.disk_dir(mapped) / f"{dataset_name}_{digest[:12]}{src_file.suffix.lower()}"
        if not dry_run and copy:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(src_file, dest)
        report["imported"] += 1
        report["per_class"][mapped] = report["per_class"].get(mapped, 0) + 1
        report["files"].append({"src": str(src_file), "class": mapped, "md5": digest, "dhash": dh})

    if robot_ready and report["files"]:
        tags = reg._read_json(reg.TAGS_FILE, {})
        if not isinstance(tags, dict):
            tags = {}
        for item in report["files"]:
            tags.setdefault(item["class"], []).append({"md5": item["md5"], "dataset": dataset_name, "robot_ready": True})
        if not dry_run:
            reg.TAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            reg.TAGS_FILE.write_text(json.dumps(tags, indent=2))

    if not dry_run:
        reg.record_source(
            {
                "name": dataset_name,
                "source": str(source_dir),
                "url": url,
                "license": license,
                "original_image_count": report["scanned"],
                "imported_image_count": report["imported"],
                "mapped_classes": sorted(report["per_class"]),
                "per_class": dict(report["per_class"]),
                "exact_duplicates_skipped": report["exact_duplicates"],
                "near_duplicates_skipped": report["near_duplicates"],
                "corrupt_skipped": report["corrupt"],
                "low_quality_skipped": report["low_quality"],
                "unmapped_skipped": report["unmapped"],
                "robot_ready_tagged": bool(robot_ready),
            }
        )
        stats = data.audit_and_split()
        reg.ensure_seed_version(stats)
        reg.record_version(
            {
                "version": f"working-set+{dataset_name}",
                "kind": "staging",
                "total_images": stats["total_unique"],
                "class_distribution": reg.canonical_distribution(stats),
                "sources": [r["name"] for r in reg.load_sources()],
                "duplicates_removed": stats.get("duplicates_removed", 0),
                "near_duplicates_removed": stats.get("near_duplicates_removed", 0),
                "corrupt_removed": stats.get("corrupt_unreadable", 0),
                "low_quality_removed": stats.get("low_quality_removed", 0),
                "split": stats.get("split_sizes", {}),
                "note": f"ingest:{dataset_name}",
            }
        )
        report["split_rebuilt"] = stats.get("split_sizes", {})
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a lawfully obtained image folder into EcoSort.")
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--license", required=True, help="Licence string, e.g. 'CC BY 4.0'")
    ap.add_argument("--url", default="")
    ap.add_argument("--class-map", nargs="*", default=[], help="src:canonical pairs")
    ap.add_argument("--no-copy", action="store_true", help="validate only, do not copy files")
    ap.add_argument("--near-hamming", type=int, default=4)
    ap.add_argument("--robot-ready", action="store_true", help="tag imported images as robot-ready diversity data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report = ingest(
        source_dir=args.source_dir,
        dataset_name=args.dataset_name,
        license=args.license,
        url=args.url,
        class_map=parse_class_map(args.class_map),
        copy=not args.no_copy,
        near_hamming=args.near_hamming,
        robot_ready=args.robot_ready,
        dry_run=args.dry_run,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    print(f"\nimported={report['imported']} scanned={report['scanned']} "
          f"exact_dup={report['exact_duplicates']} near_dup={report['near_duplicates']} "
          f"lowq={report['low_quality']} corrupt={report['corrupt']} unmapped={report['unmapped']}")


if __name__ == "__main__":
    main()
