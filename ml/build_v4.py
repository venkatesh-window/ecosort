"""Dataset v4 finalization: v3 + verified non-test candidates, gated.

Gate (pipeline parameters, documented — not claims):
  MIN_NEW_SAMPLES = 200 verified samples
  MIN_CLASSES = 2 distinct classes, each with >= 10 samples

Protections (absolute):
  - candidates come only from reviewer-verified deployment-time records
  - md5 exclusion against every v3 file AND the test manifest (auto-reject)
  - near-duplicate exclusion against v3 and among candidates
  - new data splits into train/val ONLY; the v3 test split is never touched
  - manifest is immutable: refuses to overwrite an existing v4

Outcome: writes artifacts/analysis/v4-candidate-report.json always; writes
artifacts/dataset_v4.json + a registry version only if the gate passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from api import db
from ml import config, data, quality_gates
from ml import dataset_registry as reg

MIN_NEW_SAMPLES = 200
MIN_CLASSES = 2
MIN_PER_CLASS = 10

V4_MANIFEST = config.ARTIFACTS / 'dataset_v4.json'
V4_REPORT = config.ARTIFACTS / 'analysis' / 'v4-candidate-report.json'
V4_DIRNAME = 'v4_additions'


def _error_weight() -> dict[str, float]:
    """Priority weight from measured run-002 per-class F1 (lower F1 → higher
    need). Falls back to uniform weights if the eval artifact is absent."""
    try:
        ev = json.loads((config.METRICS_DIR / 'evaluation_run-002_best.json').read_text())
        per = ev['metrics']['per_class']
        return {c: round(2.0 - v['f1'], 3) for c, v in per.items()}
    except (OSError, ValueError, KeyError):
        return {}


def collect_candidates() -> tuple[list[dict], list[dict]]:
    """Split reviewer records into (considered, rejected_before_checks).

    Status mapping: verified→VERIFIED/CORRECTED, uncertain→UNCERTAIN,
    rejected→REJECTED. Only VERIFIED/CORRECTED proceed.
    """
    rows = db.dataset_candidates(10000)
    with db.db() as conn:  # read-only use of stored review verdicts
        extra = [dict(r) for r in conn.execute(
            "SELECT c.*, s.image_path, s.confidence, s.model_version, s.source AS scan_source "
            "FROM corrections c LEFT JOIN scans s ON s.id = c.scan_id "
            "WHERE c.source = 'review' AND c.status != 'verified'")]
    considered, rejected = [], []
    weights = _error_weight()
    for r in rows:
        actual = r['correct_class']
        entry = {
            'candidate_id': f"v4c-{r['id'][:8]}",
            'image_id': r['scan_id'],
            'source': 'deployment-scan',
            'scan_source': None,
            'actual_label': actual,
            'predicted_label': r['predicted_class'],
            'confidence': r['confidence'],
            'review_status': 'CORRECTED' if actual != r['predicted_class'] else 'VERIFIED',
            'priority_value': weights.get(actual, 1.0),
            'priority_reason': f"reviewer-verified {'correction' if actual != r['predicted_class'] else 'confirmation'}; class error-weight {weights.get(actual, 1.0)}",
            'model_version': r['model_version'],
            'image_path': r['image_path'],
        }
        considered.append(entry)
    for r in extra:
        status = {'uncertain': 'UNCERTAIN', 'rejected': 'REJECTED'}.get(r['status'], r['status'])
        rejected.append({'candidate_id': f"v4c-{r['id'][:8]}", 'image_id': r['scan_id'],
                         'review_status': status, 'reason': f'reviewer marked {status} — excluded from supervised training'})
    return considered, rejected


def finalize() -> dict:
    stats = reg.current_stats() or {}
    v3_entry = next((v for v in reg.load_versions() if v.get("version") == "EcoSort Dataset v3"), {})
    v3_total = int(v3_entry.get("total_images") or stats.get("total_unique", 0))
    working_total = int(stats.get("total_unique", 0))
    considered, rejected = collect_candidates()

    # Index v3 hashes (files + explicit test-manifest set).
    v3_hashes: set[str] = set()
    for cls_dir in sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir()):
        for f in cls_dir.iterdir():
            if f.suffix.lower() in data.EXTS:
                h = quality_gates.md5_bytes(f)
                if h:
                    v3_hashes.add(h)
    test_paths = {r['path'] for r in json.loads((config.SPLITS_DIR / 'test.json').read_text())}

    accepted, auto_rejected = [], []
    seen_new: set[str] = set()
    seen_dh: dict[str, list[str]] = {}
    for c in considered:
        ok, gates, h, dh = quality_gates.validate_candidate(
            c['image_path'], c['actual_label'], v3_hashes, test_paths, seen_new, seen_dh,
            provenance={'origin': f"deployment scan {c['image_id']} ({c['source']})",
                        'license': 'user-submitted deployment scan — reuse for training only after review'})
        if not ok:
            failed = next((g for g in gates if not g['passed']), {'detail': 'unknown'})
            auto_rejected.append({**c, 'review_status': 'REJECTED',
                                  'reason': f"{failed['gate']}: {failed['detail']}",
                                  'gates': gates})
            continue
        seen_new.add(h)
        if dh:
            canon = reg.canonical(c['actual_label']) or c['actual_label']
            seen_dh.setdefault(canon, []).append(dh)
        accepted.append({**c, 'md5': h, 'dhash': dh, 'gates': gates})

    by_class: dict[str, int] = {}
    for c in accepted:
        by_class[c['actual_label']] = by_class.get(c['actual_label'], 0) + 1
    gate = {'min_new_samples': MIN_NEW_SAMPLES, 'min_classes': MIN_CLASSES,
            'min_per_class': MIN_PER_CLASS,
            'new_samples': len(accepted),
            'classes_covered': sum(1 for n in by_class.values() if n >= MIN_PER_CLASS)}
    gate['passed'] = gate['new_samples'] >= MIN_NEW_SAMPLES and gate['classes_covered'] >= MIN_CLASSES

    report = {
        'parent_version': 'EcoSort Dataset v3', 'v3_size': v3_total,
        'working_size': working_total,
        'considered': len(considered), 'accepted': len(accepted),
        'auto_rejected': [{k: r[k] for k in ('candidate_id', 'review_status', 'reason')} for r in auto_rejected],
        'pre_rejected': [{k: r[k] for k in ('candidate_id', 'review_status', 'reason')} for r in rejected],
        'by_class': by_class,
        'gate': gate,
        'test_set': 'untouched — new data splits into train/val only',
    }
    V4_REPORT.parent.mkdir(parents=True, exist_ok=True)
    V4_REPORT.write_text(json.dumps(report, indent=2))

    if not gate['passed']:
        report['v4_created'] = False
        report['note'] = ('Gate refused: verified pool too small for a meaningful v4. '
                          'Review more queue items; v3 stays the training dataset.')
        V4_REPORT.write_text(json.dumps(report, indent=2))
        return report

    if V4_MANIFEST.exists():
        raise SystemExit('dataset_v4.json already exists — v4 is immutable, refusing overwrite')
    dest_dir = config.DATASET_DIR / V4_DIRNAME
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for c in accepted:
        src = Path(c['image_path'])
        dst = dest_dir / reg.disk_dir(reg.canonical(c['actual_label']) or c['actual_label']) / f"{c['candidate_id']}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_entries.append({**c, 'v4_path': str(dst)})
    manifest = {'version': 'EcoSort Dataset v4', 'parent': 'EcoSort Dataset v3',
                'new_samples': len(manifest_entries), 'entries': manifest_entries,
                'splits': 'new data → train/val only; v3 test split untouched'}
    V4_MANIFEST.write_text(json.dumps(manifest, indent=2))
    reg.record_version({'version': 'EcoSort Dataset v4', 'total_images': v3_total + len(manifest_entries),
                        'sources': ['v4-verified-candidates'], 'note': 'v3 + verified candidates'})
    report['v4_created'] = True
    V4_REPORT.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    report = finalize()
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != 'auto_rejected'}, indent=2))
    else:
        print(f"v3={report['v3_size']} considered={report['considered']} accepted={report['accepted']} "
              f"gate_passed={report['gate']['passed']} v4_created={report.get('v4_created')}")
        for r in report['auto_rejected'][:10]:
            print(f"  rejected {r['candidate_id']}: {r['reason']}")


if __name__ == '__main__':
    main()
