"""Independent sample clocks retain local features without weakening physical cadence."""

from copy import deepcopy

import numpy as np
import pytest

from sleepkit.recipes.detection.preprocessing import Normalizer, V2_SPEC, extract, prepare


def sensors(n=60):
    return np.stack([np.arange(n) * 5, np.arange(n), np.arange(n) * 2]).astype(np.float32)


def clock(n=60):
    return np.arange(n, dtype=np.int64) * 5 + 1_600_000_000


@pytest.mark.parametrize("shift", [3600, -3600])
def test_local_shift_uses_independent_clock_and_preserves_features(shift):
    data = sensors()
    data[0, 6:] = (data[0, 6:] + shift) % 86400
    with pytest.raises(ValueError, match="contiguous"):
        extract(data)
    result = extract(data, sample_time=clock())
    assert result.values[0, 0] == pytest.approx(np.cos(2 * np.pi * data[0, :12].astype(float) / 86400).mean())
    np.testing.assert_array_equal(result.times, np.arange(9) * 30 + 55)
    data[0, 20] = np.nan
    assert not extract(data, sample_time=clock()).valid.all()
    data[0, 20] = 86400
    with pytest.raises(ValueError, match="seconds of day"):
        extract(data, sample_time=clock())


@pytest.mark.parametrize(
    "bad",
    [
        np.arange(60, dtype=float) * 5,
        np.zeros(60, dtype=bool),
        np.arange(60, dtype=np.uint64) * 5,
        np.arange(60, dtype=np.int32) * 5,
        clock().reshape(1, -1),
        clock(59),
        clock()[::-1],
        np.full(60, 100, np.int64),
        np.r_[clock()[:30], clock()[30:] + 5],
    ],
)
def test_invalid_independent_clock_rejected_before_cache(tmp_path, bad):
    prepare(sensors(), tmp_path, sample_time=clock())
    with pytest.raises(ValueError, match="sample_time"):
        prepare(sensors(), tmp_path, sample_time=bad)
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_integer_wraparound_is_not_continuity():
    wrapped = np.array([np.iinfo(np.int64).max - 2, np.iinfo(np.int64).min + 2], dtype=np.int64)
    assert np.diff(wrapped)[0] == 5
    with pytest.raises(ValueError, match="increasing"):
        extract(sensors(2), sample_time=wrapped)


def test_cache_binds_clock_presence_and_epoch(tmp_path):
    results = [prepare(sensors(), tmp_path, sample_time=value) for value in (None, clock(), clock() + 3600)]
    assert len(list(tmp_path.glob("*.npz"))) == 3
    for result in results:
        np.testing.assert_array_equal(results[0].values, result.values)
    prepare(sensors(), tmp_path, sample_time=clock())
    assert len(list(tmp_path.glob("*.npz"))) == 3


def test_v2_state_preserves_clock_contract():
    state = Normalizer.fit([extract(sensors())]).to_dict()
    state["spec"] = deepcopy(V2_SPEC)
    normalizer = Normalizer.from_dict(state)
    assert normalizer.to_dict() == state
    prepare(sensors(), spec=normalizer.spec)
    with pytest.raises(ValueError, match="v2"):
        prepare(sensors(), sample_time=clock(), spec=normalizer.spec)
    shifted = sensors()
    shifted[0, 30:] += 3600
    with pytest.raises(ValueError, match="contiguous"):
        prepare(shifted, spec=normalizer.spec)


def test_reader_and_generators_retain_clock(tmp_path):
    h5py = pytest.importorskip("h5py")
    from sleepkit.recipes.detection.data import examples, read_recording, subject_features

    data = sensors()
    data[0, 6:] += 3600
    with h5py.File(tmp_path / "a.h5", "w") as stream:
        stream["data"] = data
        stream["sleep_stages"] = np.arange(60, dtype=np.int32) % 2
        stream["sample_time"] = clock()
        stream["sample_time"].attrs["units"] = "unix_seconds"
    recording = read_recording(tmp_path, "a", labels=False)
    assert recording.labels is None
    np.testing.assert_array_equal(recording.sample_time, clock())
    normalizer = Normalizer.fit(subject_features(tmp_path, ["a"]))
    assert len(list(examples(tmp_path, ["a"], normalizer, 3))) == 3
    with h5py.File(tmp_path / "a.h5", "a") as stream:
        stream["sample_time"].attrs["units"] = "milliseconds"
    with pytest.raises(ValueError, match="units"):
        read_recording(tmp_path, "a", labels=False)
