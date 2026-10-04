"""EcoSort integrity tests.

Every assertion is measured from the repository's own artifacts and runtime:
checkpoints, evaluation JSONs, split manifests, the dataset registry and the
live API. Nothing here hardcodes model quality beyond reading the eval files,
and no test fabricates data.
"""

import io
import json
import random

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from ml import config
from ml import dataset_registry as reg
from ml import model_registry as mreg
from ml.infer import Predictor, resolve_checkpoint


def _noise_jpeg(seed: int = 11, size: int = 300) -> bytes:
    rnd = random.Random(seed)
    im = Image.frombytes('RGB', (size, size),
                         bytes(rnd.randrange(256) for _ in range(size * size * 3)))
    buf = io.BytesIO()
    im.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture(scope='module')
def api():
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------ data integrity


def test_run_checkpoints_exist():
    for run in ('run-001', 'run-002'):
        for w in ('best.pt', 'last.pt', 'final.pt'):
            assert (config.RUNS_DIR / run / w).exists(), f'{run}/{w} missing'


def test_evaluation_files_exist_and_parse():
    for f in ('evaluation_run-001_best.json', 'evaluation_run-002_best.json'):
        p = config.METRICS_DIR / f
        assert p.exists(), f'{f} missing'
        payload = json.loads(p.read_text())
        assert payload['confusion_matrix'], f'{f} has no confusion matrix'
        assert payload['metrics']['per_class'], f'{f} has no per-class metrics'


def test_frozen_records_intact_and_working_set_consistent():
    """Frozen records never move; the working set only grows and stays
    internally consistent. (The TrashNet import preceded stable-split
    enforcement, so working manifests were rebuilt once — documented in
    docs/DATASET.md. Stable assignment now prevents recurrence.)"""
    from ml import dataset_registry as reg

    # Frozen: run-002 evaluation artifact still describes its own 3,478.
    ev = json.loads((config.METRICS_DIR / 'evaluation_run-002_best.json').read_text())
    assert ev['num_samples'] == 3478
    assert ev['run_id'] == 'run-002'
    # Frozen: registry v3 entry still records the v3 audit.
    v3 = next(v for v in reg.load_versions() if v.get('version') == 'EcoSort Dataset v3')
    assert v3['total_images'] == 23172
    # Working set: internally consistent and monotonically grown from v3.
    stats = json.loads((config.SPLITS_DIR / 'dataset_stats.json').read_text())
    assert stats['total_unique'] == sum(stats['split_sizes'].values())
    assert stats['total_unique'] >= v3['total_images']
    for split in ('train', 'val', 'test'):
        rows = json.loads((config.SPLITS_DIR / f'{split}.json').read_text())
        assert len(rows) == stats['split_sizes'][split]


def test_registry_reports_deployed_run002():
    models = mreg.list_models(str(resolve_checkpoint()))
    by_id = {m['run_id']: m for m in models}
    assert by_id['run-002']['status'] == 'DEPLOYED'
    assert by_id['run-001']['status'] == 'ARCHIVED'
    assert by_id['run-002']['test_accuracy'] == pytest.approx(0.9727, abs=1e-3)
    assert by_id['run-002']['macro_f1'] == pytest.approx(0.9681, abs=1e-3)
    assert by_id['run-002']['dataset_version'] == 'EcoSort Dataset v3'
    # run-001 predates dataset versioning: N/A, never invented
    assert by_id['run-001']['dataset_version'] is None


def test_dataset_registry_counts_are_live():
    p = reg.progress()
    assert p['verified'] == p['train'] + p['val'] + p['test']
    assert p['target'] == 100000
    assert not p['reached']
    assert set(p['missing_classes']) == {'brown_glass', 'green_glass', 'white_glass'}


# ------------------------------------------------------------ model + policy


def test_model_loads():
    ckpt = resolve_checkpoint()
    assert ckpt is not None and 'run-002' in str(ckpt)
    pred = Predictor()
    pred.load(ckpt)
    assert pred.loaded
    assert pred.meta['run_id'] == 'run-002'


def test_prediction_payload_shape():
    ckpt = resolve_checkpoint()
    pred = Predictor()
    pred.load(ckpt)
    out = pred.predict(Image.open(io.BytesIO(_noise_jpeg())))
    assert out['prediction']['class'] in config.CLASSES
    assert out['prediction']['state'] in ('high', 'moderate', 'low')
    assert out['model']['run_id'] == 'run-002'
    assert out['model']['dataset_version'] == 'EcoSort Dataset v3'
    assert out['explainability']['overlay_png_b64']
    assert out['timings_ms']['total_ms'] > 0


def test_confidence_policy_gates_low_predictions():
    ckpt = resolve_checkpoint()
    pred = Predictor()
    pred.load(ckpt)
    out = pred.predict(Image.open(io.BytesIO(_noise_jpeg())))
    state = out['prediction']['state']
    action = out['sorting']['action']
    if state == 'low':
        assert action == 'HOLD_FOR_REVIEW'
    else:
        assert action == 'SORT'


# ------------------------------------------------------------ API + lifecycle


def test_api_health_reports_real_state(api):
    h = api.get('/api/health').json()
    assert h['api'] == 'operational'
    assert h['model_loaded'] is True
    assert 'run-002' in (h['model_version'] or '')
    assert h['dataset_version'] == 'EcoSort Dataset v3'
    assert h['database_status'] == 'ok'
    assert h['robot_mode'] == 'simulation'


def test_api_models_match_files(api):
    m = api.get('/api/models').json()
    assert m['deployed'] == 'run-002'
    assert {x['run_id'] for x in m['models']} >= {'run-001', 'run-002'}


def test_upload_predict_feedback_loop(api):
    r = api.post('/api/predict', files={'file': ('t.jpg', _noise_jpeg(), 'image/jpeg')}).json()
    assert r['scan_id']
    assert r['model']['run_id'] == 'run-002'
    fb = api.post(f"/api/scans/{r['scan_id']}/feedback",
                  json={'correct': False, 'correct_class': 'paper', 'source': 'pytest'}).json()
    assert fb['added_to_review_queue'] is True
    q = api.get('/api/active-learning/queue').json()
    assert any(i['id'] == r['scan_id'] for i in q['items'])
    # reviewer verifies -> becomes a dataset candidate, never touches the test set
    rv = api.post('/api/active-learning/review',
                  json={'scan_id': r['scan_id'], 'action': 'correct', 'correct_class': 'paper'}).json()
    assert rv['candidate'] is True
    cands = api.get('/api/dataset/candidates').json()
    assert any(c['scan_id'] == r['scan_id'] for c in cands['candidates'])


def test_robot_simulation_is_marked(api):
    st = api.get('/api/robot/status').json()
    assert st['hardware_connected'] is False
    assert st['interface'] == 'SIMULATION'
    rp = api.post('/api/robot/predict',
                  files={'file': ('t.jpg', _noise_jpeg(), 'image/jpeg')}).json()
    assert rp['simulation'] is True
    step = api.post('/api/robot/simulate/step').json()
    assert step['decision']['source'] == 'simulation'
    stopped = api.post('/api/robot/stop').json()
    assert stopped['simulation'] is True
    api.post('/api/robot/emergency-stop', json={'engage': False})


def test_error_analysis_uses_measured_pairs(api):
    ea = api.get('/api/models/error-analysis?run_id=run-002').json()
    assert ea['available'] is True
    assert ea['num_samples'] == 3478
    weak = {r['class'] for r in ea['lower_performing']}
    assert {'trash', 'paper'} <= weak
