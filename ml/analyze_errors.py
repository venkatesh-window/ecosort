"""Forensic error analysis for a completed run. Reads ONLY the evaluation
artifact (+ test manifest for source attribution). Writes a JSON report and
prints the tables. No causal claims — counts and rates only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ml import config


def _source_of(path: str) -> str:
    """Measured derivation: ingest prefixes imported files <DatasetName>_<md5>.ext."""
    name = Path(path).name
    if '_' in name:
        prefix = name.split('_')[0]
        if prefix and not prefix.islower():
            return prefix
    return 'base-corpus'


def analyze(run_id: str, weights: str = 'best') -> dict:
    ev_path = config.METRICS_DIR / f'evaluation_{run_id}_{weights}.json'
    if not ev_path.exists():
        raise SystemExit(f'no evaluation artifact: {ev_path}')
    ev = json.loads(ev_path.read_text())
    classes: list[str] = ev['classes']
    cm: list[list[int]] = ev['confusion_matrix']
    per: dict = ev['metrics']['per_class']

    # Class table with FP/FN derived from the artifact's own numbers.
    table = []
    for i, c in enumerate(classes):
        support = sum(cm[i])
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(classes))) - tp
        fn = support - tp
        m = per[c]
        table.append({'class': c, 'support': support, 'tp': tp, 'fp': fp, 'fn': fn,
                      'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'],
                      'predicted_as': m['predicted_as']})

    # Confusion pairs with % of actual class.
    pairs = []
    for i, tc in enumerate(classes):
        for j, pc in enumerate(classes):
            if i != j and cm[i][j] > 0:
                pairs.append({'true_class': tc, 'predicted_class': pc,
                              'count': cm[i][j],
                              'pct_of_true': round(cm[i][j] / max(1, sum(cm[i])) * 100, 2)})
    pairs.sort(key=lambda p: -p['count'])

    # High/low-confidence errors from stored misclassified examples.
    hi, lo = [], []
    for items in ev.get('misclassified_examples', {}).values():
        for it in items:
            rec = {'true_class': it['true_class'], 'predicted_class': it['predicted_class'],
                   'confidence': it['confidence'], 'true_confidence': it.get('true_confidence'),
                   'path': it['path'], 'source': _source_of(it['path'])}
            (hi if it['confidence'] >= 0.60 else lo).append(rec)
    hi.sort(key=lambda r: -r['confidence'])
    lo.sort(key=lambda r: r['confidence'])

    # Source-level: test manifest support by source + errors by source.
    manifest = json.loads((config.SPLITS_DIR / 'test.json').read_text())
    support_by_src = Counter(_source_of(r['path']) for r in manifest)
    errors_by_src = Counter(r['source'] for r in hi + lo)
    sources = {s: {'test_images': support_by_src.get(s, 0),
                   'stored_misclassified': errors_by_src.get(s, 0),
                   'observed_error_rate': round(errors_by_src.get(s, 0) / max(1, support_by_src.get(s, 0)), 4)}
               for s in sorted(set(support_by_src) | set(errors_by_src))}

    return {
        'evaluation': str(ev_path.name),
        'run_id': run_id, 'weights': weights,
        'model': 'efficientnet_b0',
        'dataset': 'EcoSort Dataset v3 (from serving checkpoint metadata)',
        'test_count': ev['num_samples'],
        'overall': {'accuracy': ev['metrics']['accuracy'],
                    'macro_f1': ev['metrics']['f1_macro'],
                    'macro_precision': ev['metrics']['precision_macro'],
                    'macro_recall': ev['metrics']['recall_macro']},
        'class_table': table,
        'confusion_pairs': pairs,
        'high_confidence_errors': hi,
        'low_confidence_errors': lo,
        'sources': sources,
        'metadata_availability': {
            'lighting': 'Not available', 'background': 'Not available',
            'occlusion': 'Not available', 'image_quality': 'Not available',
            'single_vs_multiple_object': 'Not available',
            'camera_angle': 'Not available', 'robot_ready_status': 'Not available',
            'source_dataset': 'derived from ingest filename prefix (measured convention)',
        },
        'limitations': [
            'misclassified_examples stores a capped sample per confusion pair, not every error',
            'source attribution relies on the ingest filename convention, not embedded provenance',
            'no per-sample lighting/background/occlusion metadata exists in the eval artifact',
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', default='run-002')
    ap.add_argument('--weights', default='best')
    ap.add_argument('--out', default='artifacts/analysis/run-002-error-analysis.json')
    args = ap.parse_args()
    report = analyze(args.run_id, args.weights)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"{'class':<12}{'sup':>6}{'prec':>8}{'rec':>8}{'f1':>8}{'fp':>6}{'fn':>6}")
    for r in sorted(report['class_table'], key=lambda r: r['f1']):
        print(f"{r['class']:<12}{r['support']:>6}{r['precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['f1']:>8.3f}{r['fp']:>6}{r['fn']:>6}")
    print('\nTop confusion pairs (true -> pred : count, % of true class):')
    for p in report['confusion_pairs'][:15]:
        print(f"  {p['true_class']} -> {p['predicted_class']}: {p['count']} ({p['pct_of_true']}%)")
    print(f"\nhigh-conf errors: {len(report['high_confidence_errors'])}, "
          f"low-conf errors: {len(report['low_confidence_errors'])}")
    print('sources:', json.dumps(report['sources'], indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
