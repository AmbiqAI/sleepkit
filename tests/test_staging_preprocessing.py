"""Saved-feature staging contracts using synthetic records only."""

from dataclasses import replace

import h5py
import numpy as np
import pytest

from sleepkit.recipes.staging import SubjectRecord, prepare_subject, read_subject, window_subject


def make_record(length=7):
    return SubjectRecord(
        np.tile(np.arange(length, dtype=np.float32)[:, None], (1, 14)),
        np.arange(length, dtype=np.int32) % 7,
        np.ones(length, dtype=np.int32),
    )


def test_independent_masked_statistics_imputation_mapping_and_no_mutation():
    # Valid values 1, 3, 8 have mean 4, median 3, population variance 26/3.
    values = np.tile(np.array([1, 3, 8, np.nan], np.float32)[:, None], (1, 14))
    values[3, 1] = np.inf
    record = SubjectRecord(values, np.array([0, 1, 6, 5], np.int32), np.array([1, 1, 1, 0], np.int32))
    before = [a.copy() for a in (record.features, record.stage_labels, record.mask)]
    prepared = prepare_subject(record)
    np.testing.assert_array_equal(prepared.mean, np.full(14, 4, np.float32))
    np.testing.assert_array_equal(prepared.median, np.full(14, 3, np.float32))
    np.testing.assert_allclose(prepared.variance, 26 / 3, rtol=1e-7)
    expected = np.array([-3, -1, 4, -1]) / np.sqrt(26 / 3 + 1e-6)
    np.testing.assert_allclose(prepared.features, np.tile(expected[:, None], (1, 14)), rtol=2e-7)
    np.testing.assert_array_equal(prepared.targets, [0, 1, -1, 2])
    np.testing.assert_array_equal(prepared.mask, [True, True, True, False])
    assert prepared.features.dtype == np.float32
    for original, snapshot in zip((record.features, record.stage_labels, record.mask), before):
        np.testing.assert_array_equal(original, snapshot)
    for output, original in ((prepared.features, record.features), (prepared.mask, record.mask)):
        assert not np.shares_memory(output, original)


def test_all_source_classes_and_constant_features():
    record = replace(make_record(), features=np.full((7, 14), 7, np.float32))
    prepared = prepare_subject(record)
    np.testing.assert_array_equal(prepared.targets, [0, 1, 1, 1, 1, 2, -1])
    np.testing.assert_array_equal(prepared.features, np.zeros((7, 14), np.float32))


@pytest.mark.parametrize("length, windows, remainder", [(0, 0, 0), (239, 0, 239), (240, 1, 0), (481, 2, 1)])
def test_window_boundaries_and_coverage_without_stitching(length, windows, remainder):
    # Construct a prepared record from a nonempty source, then vary only sequence
    # length. An empty prepared input has a well-defined zero-window report.
    prepared = prepare_subject(make_record(max(length, 1)))
    targets = np.zeros(length, np.int32)
    mask = np.ones(length, bool)
    if length > 4:
        targets[2] = -1
        mask[4] = False
    prepared = replace(prepared, features=prepared.features[:length], targets=targets, mask=mask)
    result = window_subject(prepared)
    assert result.features.shape == (windows, 240, 14)
    assert result.targets.shape == result.scoring_mask.shape == (windows, 240)
    np.testing.assert_array_equal(result.starts, np.arange(windows) * 240)
    np.testing.assert_array_equal(result.features.reshape(-1, 14), prepared.features[: windows * 240])
    assert result.coverage == {
        "input_epochs": length,
        "window_count": windows,
        "windowed_epochs": windows * 240,
        "remainder_epochs": remainder,
        "eligible_epochs": length - (2 if length > 4 else 0),
        "scored_epochs": windows * 240 - (2 if windows else 0),
        "unscored_epochs": length - windows * 240 + (2 if windows else 0),
    }
    if windows:
        assert result.targets[0, 2] == -1
        assert not result.scoring_mask[0, 2] and not result.scoring_mask[0, 4]
        assert result.scoring_mask[0, 3]
    if windows == 2:
        np.testing.assert_array_equal(result.features[1, 0], prepared.features[240])


@pytest.mark.parametrize(
    "field,value",
    [
        ("features", np.zeros((7, 14), np.float64)),
        ("features", np.zeros((7, 13), np.float32)),
        ("features", np.zeros((7, 1, 14), np.float32)),
        ("stage_labels", np.zeros(7, np.float32)),
        ("stage_labels", np.zeros(7, bool)),
        ("stage_labels", np.zeros((7, 1), np.int32)),
        ("stage_labels", np.full(7, -1, np.int32)),
        ("stage_labels", np.full(7, 7, np.int32)),
        ("stage_labels", np.zeros(6, np.int32)),
        ("mask", np.ones(7, np.float32)),
        ("mask", np.full(7, 2, np.int32)),
        ("mask", np.zeros((7, 1), np.int32)),
    ],
)
def test_reader_and_direct_preparation_reject_malformed_arrays(tmp_path, field, value):
    record = replace(make_record(), **{field: value})
    with pytest.raises(ValueError):
        prepare_subject(record)
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as stream:
        for name in ("features", "stage_labels", "mask"):
            stream[name] = getattr(record, name)
    with pytest.raises(ValueError):
        read_subject(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_valid_rows_rejected(bad):
    record = make_record()
    record.features[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        prepare_subject(record)


def test_no_valid_rows_and_float32_overflow_rejected():
    for record in (make_record(0), replace(make_record(), mask=np.zeros(7, bool))):
        with pytest.raises(ValueError, match="no quality-valid"):
            prepare_subject(record)
    record = replace(make_record(2), features=np.tile(np.array([-3e38, 3e38], np.float32)[:, None], (1, 14)))
    with pytest.raises(ValueError, match="Nonfinite normalization"):
        prepare_subject(record)


def test_reader_roundtrip_is_detached_and_ignores_unrelated_data(tmp_path):
    record = make_record()
    path = tmp_path / "subject.h5"
    with h5py.File(path, "w") as stream:
        for name in ("features", "stage_labels", "mask"):
            stream[name] = getattr(record, name)
        stream["apnea_labels"] = np.zeros(7, np.int32)
    loaded = read_subject(path)
    path.unlink()
    np.testing.assert_array_equal(prepare_subject(loaded).targets, [0, 1, 1, 1, 1, 2, -1])


@pytest.mark.parametrize("kind", ["missing", "group", "soft", "external", "virtual", "external_storage"])
def test_reader_rejects_missing_or_indirect_required_datasets(tmp_path, kind):
    path = tmp_path / "subject.h5"
    with h5py.File(path, "w") as stream:
        stream["stage_labels"] = np.zeros(7, np.int32)
        stream["mask"] = np.ones(7, np.int32)
        if kind == "group":
            stream.create_group("features")
        elif kind == "soft":
            stream["other"] = np.zeros((7, 14), np.float32)
            stream["features"] = h5py.SoftLink("/other")
        elif kind == "external":
            stream["features"] = h5py.ExternalLink("absent.h5", "/features")
        elif kind == "virtual":
            layout = h5py.VirtualLayout(shape=(7, 14), dtype=np.float32)
            layout[:] = h5py.VirtualSource("absent.h5", "/features", shape=(7, 14))
            stream.create_virtual_dataset("features", layout)
        elif kind == "external_storage":
            stream.create_dataset(
                "features", (7, 14), dtype=np.float32, external=[("absent.bin", 0, h5py.h5f.UNLIMITED)]
            )
    with pytest.raises(ValueError):
        read_subject(path)


def test_windowing_validates_constructed_prepared_records():
    prepared = prepare_subject(make_record(240))
    with pytest.raises(ValueError, match="Unsupported stage"):
        window_subject(replace(prepared, targets=np.full(240, 3, np.int32)))
    with pytest.raises(ValueError, match="finite"):
        window_subject(replace(prepared, features=np.full((240, 14), np.nan, np.float32)))
