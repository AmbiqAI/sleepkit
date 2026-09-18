import json
import subprocess
import sys

import numpy as np
import pytest

from sleepkit.data import Annotation, Record, Signal, SubjectSplit, split_subjects
from sleepkit.data.readers import read_record, save_record
from sleepkit.preprocessing import Channel, Standardizer, WindowFeatures, model_windows, prepare


def record(subject="a", values=None, annotations=True):
    values = np.arange(8) if values is None else values
    return Record(
        "test",
        subject,
        "night",
        {"x": Signal(values, 2, "g", "acc", "wrist")},
        (Annotation(0, 2, 0), Annotation(2, 4, 1)) if annotations else (),
    )


def transform(**kwargs):
    return WindowFeatures((Channel("x", "g", "acc", "wrist"),), window_seconds=2, stride_seconds=2, **kwargs)


def test_imports_are_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sleepkit; import sleepkit.preprocessing; import sleepkit.runtime; "
            "assert not {'tensorflow','keras','helia_edge','pyedflib','matplotlib','wandb'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_alignment_exact_window_and_unlabeled():
    result = transform()(record())
    np.testing.assert_allclose(result.values[:, 0], [1.5, 5.5])
    np.testing.assert_array_equal(result.targets, [0, 1])
    np.testing.assert_array_equal(result.times, [1, 3])
    assert len(model_windows(result, 2)[0]) == 1
    unlabeled = transform()(record(annotations=False))
    assert len(model_windows(unlabeled, 2)[0]) == 0
    assert len(model_windows(unlabeled, 2, require_targets=False)[0]) == 1


def test_multirate_and_offset():
    rec = record()
    rec.signals["y"] = Signal([10, 20, 30], 1, "%", "oximetry", "finger", start=1)
    tx = WindowFeatures((Channel("x", "g", "acc", "wrist"), Channel("y", "%", "oximetry", "finger")), 2, 2)
    result = tx(rec)
    np.testing.assert_allclose(result.values[0, [0, 2]], [3.5, 15])
    assert result.times.tolist() == [2]
    assert result.targets.tolist() == [1]


def test_invalid_and_short_records():
    result = transform()(record(values=[0, np.nan, 2, 3, 4, 5, 6, 7]))
    assert result.valid.tolist() == [False, True]
    assert len(model_windows(result, 2)[0]) == 0
    assert transform()(record(values=[1])).values.shape == (0, 2)
    with pytest.raises(ValueError, match="modality"):
        WindowFeatures((Channel("x", "g", "ppg", "wrist"),))(record())
    with pytest.raises(ValueError, match="Missing channel"):
        WindowFeatures((Channel("z", "g", "acc", "wrist"),))(record())


def test_cache_equivalence_and_invalidation(tmp_path):
    first = prepare(record(), transform(), tmp_path)
    cached = prepare(record(), transform(), tmp_path)
    np.testing.assert_array_equal(first.values, cached.values)
    assert first.spec == cached.spec
    prepare(record(values=np.ones(8)), transform(), tmp_path)
    prepare(record(annotations=False), transform(), tmp_path)
    prepare(record(), WindowFeatures(transform().channels, 1, 1), tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 4
    assert not list(tmp_path.glob("tmp*"))


def test_normalization_uses_only_train_and_checks_schema(tmp_path):
    records = [record("a", np.ones(8)), record("b", np.full(8, 100)), record("c", np.full(8, -100))]
    split = SubjectSplit((("test", "a"),), (("test", "b"),), (("test", "c"),))
    norm = Standardizer.fit(records, transform(), split, tmp_path)
    assert norm.mean.tolist() == [1, 0]
    restored = Standardizer.from_dict(json.loads(json.dumps(norm.to_dict())))
    np.testing.assert_array_equal(restored.transform(transform()(records[0])).values, 0)
    with pytest.raises(ValueError, match="schema"):
        restored.transform(WindowFeatures(transform().channels, 1, 1)(records[0]))


def test_splits_keep_subject_sessions_together_and_are_order_independent():
    records = [record(str(i)) for i in range(10)]
    records.append(Record("test", "0", "night2", records[0].signals))
    a = split_subjects(records, seed=5)
    b = split_subjects(list(reversed(records)), seed=5)
    assert a == b
    assert SubjectSplit.from_dict(a.to_dict()) == a
    with pytest.raises(ValueError, match="disjoint"):
        SubjectSplit(a.train, a.train, a.test)


def test_array_adapter_roundtrip(tmp_path):
    path = tmp_path / "input.npz"
    save_record(record(), path)
    restored = read_record(path)
    np.testing.assert_array_equal(transform()(restored).values, transform()(record()).values)
    assert restored.annotations == record().annotations
