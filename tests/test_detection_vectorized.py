"""Compare optimized preparation with the original independent scalar reductions."""

import numpy as np
import pytest

from sleepkit.recipes.detection.preprocessing import Features, Normalizer, contexts, extract, prepare, validate_input


def scalar_extract(data, *, sample_time=None, spec=None):
    data, _ = validate_input(data, sample_time, spec=spec)
    starts = np.arange(0, max(0, data.shape[1] - 11), 6)
    values = np.zeros((len(starts), 5), np.float32)
    valid = np.ones(len(starts), bool)
    for i, start in enumerate(starts):
        window = data[:, start : start + 12]
        if not np.isfinite(window).all():
            valid[i] = False
            continue
        values[i] = [
            np.mean(np.cos(2 * np.pi * window[0].astype(float) / 86400)),
            np.mean(window[1], dtype=float),
            np.std(window[1], dtype=float),
            np.mean(window[2], dtype=float),
            np.std(window[2], dtype=float),
        ]
    return Features(values, valid, starts + 11)


def assert_identical(actual, expected):
    for key in ("values", "valid", "ends"):
        left, right = getattr(actual, key), getattr(expected, key)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes(), key


def sensors(n, layout="C"):
    rng = np.random.default_rng(12)
    data = np.vstack(((86350 + np.arange(n) * 5) % 86400, rng.normal(size=(2, n)))).astype(np.float32)
    if layout == "F":
        return np.asfortranarray(data)
    if layout == "strided":
        storage = np.zeros((3, n * 2), np.float32)
        storage[:, ::2] = data
        return storage[:, ::2]
    if layout == "negative":
        return data[:, ::-1].copy()[:, ::-1]
    return data


@pytest.mark.parametrize("n", [0, 1, 11, 12, 13, 17, 18, 31, 4096 * 6 + 5, 4096 * 6 + 12, 8192 * 6 + 17])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "negative"])
def test_exact_values_clock_tail_and_chunk_boundaries(n, layout):
    data = sensors(n, layout)
    assert_identical(extract(data), scalar_extract(data))


@pytest.mark.parametrize("channel", [0, 1, 2])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_windows_keep_zero_values_and_native_coordinates(channel, value):
    data = sensors(4096 * 6 + 90)
    data[channel, [0, 11, 12, 4096 * 6 - 1, 4096 * 6 + 12, -1]] = value
    assert_identical(extract(data), scalar_extract(data))
    data[channel] = value
    assert_identical(extract(data), scalar_extract(data))


@pytest.mark.parametrize("layout", ["C", "F", "strided", "negative"])
def test_extreme_finite_values_cancellation_and_subnormals(layout):
    data = sensors(360, layout)
    limit = np.finfo(np.float32).max
    data[1] = np.resize([limit, -limit, 0, 1e-40, -1e-40, 1, -1, 1e30, -1e30, 1e-10, -1e-10, 0], 360)
    data[2] = np.resize([1e12, 1e-30, -1e12, -1e-30, 0, -0.0], 360)
    assert_identical(extract(data), scalar_extract(data))


def test_normalization_context_targets_and_cache_compatibility(tmp_path, monkeypatch):
    from sleepkit.recipes.detection import preprocessing

    records = [sensors(4096 * 6 + 120), sensors(700)]
    records[0][1, 30] = np.nan
    clocks = [1600000000 + np.arange(data.shape[1], dtype=np.int64) * 5 for data in records]
    records[0][0, 100:] = (records[0][0, 100:] + 3600) % 86400
    old = [scalar_extract(data, sample_time=clock) for data, clock in zip(records, clocks)]
    new = [extract(data, sample_time=clock) for data, clock in zip(records, clocks)]
    for left, right in zip(new, old):
        assert_identical(left, right)
    assert Normalizer.fit(new).to_dict() == Normalizer.fit(old).to_dict()
    normalizer = Normalizer.fit(old)
    labels = np.arange(records[0].shape[1], dtype=np.int32) % 2
    labels[60:90] = -1
    actual = list(contexts(normalizer.transform(new[0]), 7, labels))
    expected = list(contexts(normalizer.transform(old[0]), 7, labels))
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)
    # Cache files created by the historical extractor are still valid byte-equivalent inputs.
    monkeypatch.setattr(preprocessing, "extract", scalar_extract)
    cached = prepare(records[0], tmp_path, sample_time=clocks[0])
    monkeypatch.setattr(preprocessing, "extract", lambda *a, **k: pytest.fail("Should read the existing cache"))
    assert_identical(prepare(records[0], tmp_path, sample_time=clocks[0]), cached)
    assert_identical(new[0], cached)


def test_invalid_windows_skip_arithmetic_with_strict_numpy_errors():
    data = sensors(4096 * 6 + 120)
    data[0, 20] = np.nan
    data[1, : 4096 * 6 + 1] = np.inf
    clock = 1600000000 + np.arange(data.shape[1], dtype=np.int64) * 5
    with np.errstate(all="raise"):
        assert_identical(extract(data, sample_time=clock), scalar_extract(data, sample_time=clock))
