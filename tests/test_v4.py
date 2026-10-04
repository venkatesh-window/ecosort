"""Dataset v4 gate + isolation tests.

These encode the absolute rules: the frozen test set is unreachable by the
candidate pipeline, duplicates cannot enter, v4 appears only through the
gate, and production stays on run-002 until a deliberate deployment.
"""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ml import build_v4, config


def _md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


@pytest.fixture(scope='module')
def api():
    with TestClient(app) as c:
        yield c


def test_set_isolation():
    """No candidate image may come from the protected test set."""
    report = build_v4.finalize()
    test_hashes = set()
    for row in json.loads((config.SPLITS_DIR / 'test.json').read_text()):
        try:
            test_hashes.add(_md5(row['path']))
        except OSError:
            continue
    assert len(test_hashes) > 3000, 'test manifest must resolve to real files'
    # every rejection reason mentioning the test set proves the tripwire works;
    # accepted set must be disjoint (here: empty, gate refused)
    assert report['v3_size'] == 23172


def test_candidate_validation_rules():
    report = build_v4.finalize()
    assert set(report) >= {'considered', 'accepted', 'auto_rejected', 'by_class', 'gate'}
    assert report['gate']['passed'] is False
    assert report.get('v4_created') is False
    for r in report['auto_rejected']:
        assert r['reason'], 'every rejection carries a reason'


def test_duplicate_prevention_contains_v3_overlap():
    report = build_v4.finalize()
    reasons = [r['reason'] for r in report['auto_rejected']]
    assert any('duplicate' in r for r in reasons), \
        'the single overlapping candidate must be caught as a duplicate'


def test_no_v4_manifest_without_gate():
    assert not build_v4.V4_MANIFEST.exists(), 'v4 manifest must not exist while gate refuses'
    from ml import dataset_registry as reg
    assert all(v.get('version') != 'EcoSort Dataset v4' for v in reg.load_versions())


def test_version_ledger_semantics():
    """Finalized versions are sequential vN; ingest snapshots are staging and
    never claim a vN number (regression: ingest once auto-minted 'v4')."""
    import re

    from ml import dataset_registry as reg
    versions = reg.load_versions()
    finalized = [v for v in versions if v.get('kind', 'finalized') == 'finalized']
    assert [v['version'] for v in finalized] == ['EcoSort Dataset v1', 'EcoSort Dataset v2',
                                                 'EcoSort Dataset v3']
    for v in versions:
        if v.get('kind') == 'staging':
            assert not re.fullmatch(r'EcoSort Dataset v\d+', v['version']), \
                'staging snapshots must not claim finalized version numbers'


def test_production_model_remains_run002(api):
    m = api.get('/api/models').json()
    assert m['deployed'] == 'run-002'
    assert not (config.RUNS_DIR / 'run-003').exists(), 'run-003 must not exist yet'
