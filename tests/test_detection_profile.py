"""Profiling must exercise the real finite pipeline without touching held-out labels."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from sleepkit.artifacts.package import sha256


MISSING = [name for name in ("tensorflow", "keras", "h5py") if importlib.util.find_spec(name) is None]


@pytest.mark.skipif(
    bool(MISSING) and os.environ.get("SLEEPKIT_REQUIRE_DETECTION") != "1", reason="Needs detection extras"
)
@pytest.mark.parametrize("tamper", ["source", "declaration", "environment", "code"])
def test_profile_real_finite_pipeline(tmp_path, tamper):
    assert not MISSING, f"Missing detection dependencies: {MISSING}"
    code = r"""
import json
from pathlib import Path
import sys
import numpy as np
from sleepkit.recipes.detection.target_dataset import AnnotatedDataset
from sleepkit.recipes.detection.profile import profile_membership
from sleepkit.recipes.detection.recipe import Config, dataset
from sleepkit.recipes.detection.preprocessing import Normalizer
from sleepkit.recipes.detection.data import examples, subject_features
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_detection_target_dataset import make_dataset_fixture
from sleepkit.artifacts.package import sha256
paths = make_dataset_fixture(Path(sys.argv[1]))
originals = {path: sha256(path) for path in paths[0].glob("*.h5")}
source = AnnotatedDataset(*paths)
output = Path(sys.argv[1]) / "profile"
read = source.read
train = set(source.split['train'])
reads = []
def training_only(subject, *, labels=True):
    assert subject in train, "Profiler read held-out data"
    reads.append((subject, labels))
    return read(subject, labels=labels)
source.read = training_only
cfg = Config(context=2, batch_size=5)
report = profile_membership(source, output, cfg, resident_steps=2, warmup_steps=1, data_kind='synthetic')
cache = output / 'private-feature-cache'
normalizer = Normalizer.fit(subject_features(source.root, source.split['train'], cache, reader=source.read))
native = list(examples(source.root, source.split['train'], normalizer, cfg.context, cache, reader=source.read))
count = len(native)
# A real shuffled training epoch covers every context once, including its partial batch.
batches = list(dataset(source.root, source.split['train'], normalizer, cfg, cache,
                       training=True, count=count, reader=source.read))
def keys(items):
    return sorted((x.tobytes(), y.tobytes()) for x, y in items)
actual = [(x, y) for batch_x, batch_y in batches for x, y in zip(batch_x.numpy(), batch_y.numpy())]
assert keys(actual) == keys(native)
assert report['counts']['contexts'] == count
assert report['counts']['batches'] == len(batches)
assert report['counts']['last_batch_contexts'] == len(batches[-1][0]) < cfg.batch_size
assert report['counts']['normalization_frames'] == normalizer.count
assert report['counts']['feature_frames_trained'] == count * cfg.context
assert all(stage['wall_seconds'] > 0 for stage in report['stages'].values())
assert report['stages']['warm_loader_epoch']['batch_wait']['count'] == len(batches)
assert report['stages']['resident_model_steps']['contexts'] == 2 * cfg.batch_size
assert report['stages']['end_to_end_fit_epoch']['batches'] == len(batches)
assert set(subject for subject, _ in reads) == train
assert len(list(cache.glob('*.npz'))) == len(train)
assert all(subject not in json.dumps(report) for subject in train)
assert all(subject not in (output/'declaration.json').read_text() for subject in train)
try:
    profile_membership(source, output, cfg)
except FileExistsError:
    pass
else:
    raise AssertionError('Profile overwrote evidence')
# A mid-run mutation must not produce a successful report.
checks = 0
verify = source.verify_unchanged
def fail_at_end():
    global checks
    checks += 1
    if checks == 2:
        tamper = sys.argv[2]
        if tamper == 'source':
            raise ValueError('Source data changed')
        if tamper in ('declaration', 'environment'):
            path = failed / (tamper + '.json')
            path.write_text(path.read_text() + '\n')
        if tamper == 'code':
            from sleepkit.recipes.detection import profile
            original_hash = profile.sha256
            profile.sha256 = lambda path: '0'*64 if path.name == 'profile.py' else original_hash(path)

    return verify()
source.verify_unchanged = fail_at_end
failed = output.parent / 'failed'
try:
    profile_membership(source, failed, cfg, resident_steps=1, warmup_steps=1, data_kind='synthetic')
except ValueError as error:
    assert 'changed' in str(error)
else:
    raise AssertionError('Mutation was ignored')
assert (failed / 'declaration.json').is_file()
assert not (failed / 'report.json').exists()
assert {path: sha256(path) for path in originals} == originals
"""
    env = {**os.environ, "TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), tamper], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "input ran out" not in result.stderr
    report = json.loads((tmp_path / "profile" / "report.json").read_text())
    assert report["declaration_sha256"] == sha256(tmp_path / "profile" / "declaration.json")
    assert report["environment_sha256"] == sha256(tmp_path / "profile" / "environment.json")


@pytest.mark.parametrize("kwargs", [{"resident_steps": 0}, {"warmup_steps": False}, {"data_kind": "unknown"}])
def test_invalid_profile_is_not_declared(tmp_path, kwargs):
    from sleepkit.recipes.detection.profile import profile_membership
    from sleepkit.recipes.detection.recipe import Config

    class Source:
        context = 240

    with pytest.raises(ValueError):
        profile_membership(Source(), tmp_path / "output", Config(), **kwargs)
    assert not (tmp_path / "output").exists()
