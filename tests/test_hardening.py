"""Hardening tests from the forensic audit.

These probe failure modes, boundaries and protection guarantees — malformed
input, unknown classes, missing references, threshold edges, reload safety,
robot safety interlocks. All expectations are structural (status codes,
state transitions), never invented model-quality numbers.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.robot import ROBOT
from ml import config


@pytest.fixture(scope='module')
def api():
    with TestClient(app) as c:
        yield c


def _pred_dict(cls='plastic', conf=0.9, state='high'):
    return {
        'prediction': {'class': cls, 'display': cls.title(), 'confidence': conf,
                       'state': state},
        'explainability': {'activation_centroid': {'x_px': 1, 'y_px': 1,
                                                  'x_norm': 0.1, 'y_norm': 0.1}},
        'timings_ms': {'total_ms': 5.0},
        'input_quality': {'label': 'Good'},
    }


# ------------------------------------------------------- confidence boundary


def test_confidence_gate_boundary_exact():
    thr = config.CONFIDENCE_THRESHOLD
    at = ROBOT.decide(_pred_dict(conf=thr, state='moderate'), source='audit')
    assert at['decision'] == 'SORT', 'at-threshold is defined as sortable'
    below = ROBOT.decide(_pred_dict(conf=thr - 1e-9, state='low'), source='audit')
    assert below['decision'] == 'HOLD_FOR_REVIEW', 'epsilon below threshold must hold'
    ROBOT.reset()


def test_low_confidence_never_sorts():
    for conf in (0.0, 0.01, 0.3, 0.59999):
        d = ROBOT.decide(_pred_dict(conf=conf, state='low'), source='audit')
        assert d['decision'] != 'SORT', f'conf={conf} must never sort'
    ROBOT.reset()


def test_estop_blocks_decisions():
    ROBOT.emergency_stop(True)
    try:
        d = ROBOT.decide(_pred_dict(conf=0.99, state='high'), source='audit')
        assert d['decision'] == 'BLOCKED_ESTOP'
    finally:
        ROBOT.emergency_stop(False)
        ROBOT.reset()


# ------------------------------------------------------- feedback protection


def test_feedback_unknown_scan_404(api):
    r = api.post('/api/scans/does-not-exist/feedback',
                 json={'correct': False, 'correct_class': 'paper'})
    assert r.status_code == 404
    assert 'detail' in r.json()


def test_feedback_unknown_class_422(api):
    scans = api.get('/api/scans?limit=1').json()['scans']
    if not scans:
        pytest.skip('no scans to attach feedback to')
    r = api.post(f"/api/scans/{scans[0]['id']}/feedback",
                 json={'correct': False, 'correct_class': 'unobtainium'})
    assert r.status_code == 422


def test_feedback_missing_class_422(api):
    scans = api.get('/api/scans?limit=1').json()['scans']
    if not scans:
        pytest.skip('no scans to attach feedback to')
    r = api.post(f"/api/scans/{scans[0]['id']}/feedback", json={'correct': False})
    assert r.status_code == 422


def test_al_review_invalid_action_422(api):
    r = api.post('/api/active-learning/review',
                 json={'scan_id': 'x', 'action': 'teleport'})
    assert r.status_code == 422


def test_al_review_unknown_scan_404(api):
    r = api.post('/api/active-learning/review',
                 json={'scan_id': 'nope', 'action': 'accept'})
    assert r.status_code == 404


def test_candidates_never_touch_test_split(api):
    before = json.loads((config.SPLITS_DIR / 'test.json').read_text())
    cands = api.get('/api/dataset/candidates?limit=500').json()['candidates']
    after = json.loads((config.SPLITS_DIR / 'test.json').read_text())
    assert before == after, 'candidate flow must not mutate the frozen test manifest'
    assert all(c['source'] == 'review' and c['status'] == 'verified' for c in cands)


# ------------------------------------------------------- reload safety


def test_reload_missing_checkpoint_404(api):
    r = api.post('/api/model/reload?run_id=run-999&weights=best.pt')
    assert r.status_code == 404


def test_reload_swaps_and_restores_safely(api):
    try:
        r1 = api.post('/api/model/reload?run_id=run-001&weights=best.pt')
        assert r1.status_code == 200
        assert 'run-001' in r1.json()['version']
        h = api.get('/api/health').json()
        assert h['model_loaded'] is True
    finally:
        r2 = api.post('/api/model/reload?run_id=run-002&weights=best.pt')
        assert r2.status_code == 200
        assert 'run-002' in r2.json()['version']


# ------------------------------------------------------- upload validation


def test_predict_rejects_non_image(api):
    r = api.post('/api/predict', files={'file': ('x.txt', b'hello world', 'text/plain')})
    assert r.status_code == 400
    assert 'detail' in r.json()


def test_predict_rejects_corrupt_bytes(api):
    r = api.post('/api/predict', files={'file': ('x.jpg', os.urandom(2048), 'image/jpeg')})
    assert r.status_code == 400


def test_predict_rejects_empty(api):
    r = api.post('/api/predict', files={'file': ('x.jpg', b'', 'image/jpeg')})
    assert r.status_code == 400


def test_predict_rejects_oversize(api):
    big = b'\xff' * (13 * 1024 * 1024)
    r = api.post('/api/predict', files={'file': ('x.jpg', big, 'image/jpeg')})
    assert r.status_code == 413


# ------------------------------------------------------- robot interlocks


def test_robot_paused_blocks_simulation(api):
    api.post('/api/robot/pause', json={'paused': True})
    try:
        r = api.post('/api/robot/simulate/step')
        assert r.status_code == 409
    finally:
        api.post('/api/robot/pause', json={'paused': False})


def test_robot_binmap_rejects_unknown_class(api):
    r = api.put('/api/robot/bin-map', json={'mapping': {'unobtainium': 'BIN_X'}})
    assert r.status_code == 422


def test_robot_feedback_invalid_class_422(api):
    scans = api.get('/api/scans?limit=1').json()['scans']
    if not scans:
        pytest.skip('no scans available')
    r = api.post('/api/robot/feedback',
                 json={'scan_id': scans[0]['id'], 'correct': False, 'correct_class': 'nope'})
    assert r.status_code == 422


def test_error_responses_share_detail_schema(api):
    for method, url, kwargs in [
        ('POST', '/api/scans/nope/feedback', {'json': {'correct': True}}),
        ('POST', '/api/predict', {'files': {'file': ('x', b'zzz', 'text/plain')}}),
        ('GET', '/api/scans/nope', {}),
    ]:
        r = getattr(api, method.lower())(url, **kwargs)
        assert r.status_code >= 400
        body = r.json()
        assert 'detail' in body, f'{url} must return a detail error schema'
