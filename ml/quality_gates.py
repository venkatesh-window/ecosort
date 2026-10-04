"""Shared candidate quality gates (Phase 7).

The same ten checks apply to every new candidate regardless of entry path
(dataset ingest or v4 candidate assembly):

 1. valid image        5. exact duplicate check   9. provenance
 2. supported format   6. near duplicate check   10. test-set isolation
 3. resolution check   7. class mapping
 4. corruption check   8. license

A candidate reaches human review only if all gates pass.
"""

from __future__ import annotations

from pathlib import Path

from ml import data
from ml import dataset_registry as reg

ALLOWED_EXT = set(data.EXTS)
NEAR_HAMMING = 4


def md5_bytes(path: Path) -> str | None:
    import hashlib

    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def validate_candidate(
    image_path: str | Path | None,
    actual_label: str,
    v3_hashes: set[str],
    test_paths: set[str],
    seen_hashes: set[str],
    seen_dhash: dict[str, list[str]],
    provenance: dict | None = None,
) -> tuple[bool, list[dict], str | None, str | None]:
    """Run all gates. Returns (ok, results, md5, dhash)."""
    results: list[dict] = []

    def gate(name: str, passed: bool, detail: str = '') -> bool:
        results.append({'gate': name, 'passed': passed, 'detail': detail})
        return passed

    p = Path(image_path) if image_path else None
    if not p or not p.exists():
        gate('valid image', False, 'file missing')
        return False, results, None, None
    gate('valid image', True, str(p))
    if p.suffix.lower() not in ALLOWED_EXT:
        gate('supported format', False, p.suffix)
        return False, results, None, None
    gate('supported format', True, p.suffix.lower())

    status, reason, dh = data._assess(p)
    if status == 'corrupt':
        gate('corruption check', False, reason)
        return False, results, None, None
    gate('corruption check', True, 'decodes cleanly')
    if status == 'low_quality':
        gate('resolution check', False, reason)
        return False, results, None, None
    gate('resolution check', True, 'meets minimum size/quality')

    h = md5_bytes(p)
    if not h:
        gate('exact duplicate check', False, 'unreadable for hashing')
        return False, results, None, None
    if h in v3_hashes or h in seen_hashes:
        gate('exact duplicate check', False, 'md5 already in v3 or candidate pool')
        return False, results, h, dh
    gate('exact duplicate check', True, 'novel md5')

    canon = reg.canonical(actual_label)
    if not canon:
        gate('class mapping', False, f'unknown label {actual_label!r}')
        return False, results, h, dh
    gate('class mapping', True, f'{actual_label} -> {canon}')
    bucket = seen_dhash.setdefault(canon, [])
    if dh and any(data._hamming(dh, e) <= NEAR_HAMMING for e in bucket):
        gate('near duplicate check', False, 'dhash within 4 of another candidate')
        return False, results, h, dh
    gate('near duplicate check', True, 'no near-duplicate found')

    prov = provenance or {}
    if not prov.get('origin'):
        gate('provenance', False, 'no recorded origin')
        return False, results, h, dh
    gate('provenance', True, str(prov['origin']))
    if not prov.get('license'):
        gate('license', False, 'no recorded licence')
        return False, results, h, dh
    gate('license', True, str(prov['license']))

    if str(p) in test_paths:
        gate('test-set isolation', False, 'path is in the protected test manifest')
        return False, results, h, dh
    gate('test-set isolation', True, 'not in test set')
    return True, results, h, dh
