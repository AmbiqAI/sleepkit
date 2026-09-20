"""Scoring coordinates must exactly identify outputs retained by native contexts."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sleepkit.recipes.detection.data import Recording
from sleepkit.recipes.detection.preprocessing import contexts, prepare
from sleepkit.recipes.detection.scoring import write_index


def source(recordings):
    return SimpleNamespace(
        source_hashes={subject: hashlib.sha256(subject.encode()).hexdigest() for subject in recordings},
        provenance={"kind": "test"},
        target={"kind": "annotated-period membership"},
        read=lambda subject, labels=True: recordings[subject],
    )


def recording(samples=103):
    data = np.stack([np.arange(samples) * 5, np.arange(samples), np.arange(samples) * 2]).astype(np.float32)
    return Recording(data, (np.arange(samples) % 2).astype(np.int8))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_index_matches_native_contexts_with_missing_sensors_targets_and_tails(tmp_path):
    first = recording()
    first.data[1, 20] = np.nan
    first.labels[53] = -1
    second = recording(43)
    adapter = source({"first": first, "second": second})
    output = tmp_path / "scoring.jsonl"
    summary = write_index(adapter, ["first", "second"], 3, output)
    actual = rows(output)
    expected = []
    for subject, item in (("first", first), ("second", second)):
        features = prepare(item.data)
        for _, targets, times in contexts(features, 3, item.labels):
            expected.extend((subject, int(time / 5), int(target)) for target, time in zip(targets, times))
    scored = [row for row in actual if row["context_eligible"]]
    assert [(row["subject"], row["feature_end_sample"], row["target"]) for row in scored] == expected
    assert [row["eligible_output_index"] for row in scored] == list(range(len(expected)))
    assert [row["output_index"] for row in actual] == list(range(len(actual)))
    assert all(row["eligible_output_index"] is None for row in actual if not row["context_eligible"])
    assert any(row["sensor_valid"] and not row["context_eligible"] for row in actual)
    assert any(row["target_known"] and "unknown_target" in row["exclusion_reasons"] for row in actual)
    assert [(row["context_start_sample"], row["context_end_sample"]) for row in actual[:3]] == [(0, 23)] * 3
    assert summary["outputs"] == 21  # 15 first-subject and 6 second-subject native frames
    assert summary["eligible_outputs"] == len(expected)
    assert summary["complete_contexts"] == 7
    assert summary["tail_feature_frames"] == 1
    assert summary["samples_after_last_feature"] == 2
    assert summary["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_short_recordings_have_no_outputs_but_account_for_tails(tmp_path):
    adapter = source({"short": recording(10), "partial": recording(18)})
    output = tmp_path / "scoring.jsonl"
    summary = write_index(adapter, ["short", "partial"], 3, output)
    assert output.read_bytes() == b""
    assert summary["outputs"] == 0
    assert summary["tail_feature_frames"] == 2
    assert summary["samples_after_last_feature"] == 10


def test_clock_shift_requires_independent_clock_and_true_gap_still_fails(tmp_path):
    item = recording(60)
    item.data[0, 24:] += 3600
    adapter = source({"subject": item})
    output = tmp_path / "scoring.jsonl"
    with pytest.raises(ValueError, match="contiguous"):
        write_index(adapter, ["subject"], 3, output)
    assert not output.exists()
    clock = 1_500_000_000 + np.arange(60, dtype=np.int64) * 5
    adapter = source({"subject": Recording(item.data, item.labels, clock)})
    summary = write_index(adapter, ["subject"], 3, output, cache=tmp_path / "cache")
    assert summary["eligible_outputs"] == 9
    clock[30:] += 5
    with pytest.raises(ValueError, match="contiguous"):
        write_index(adapter, ["subject"], 3, tmp_path / "gap.jsonl")
    assert not (tmp_path / "gap.jsonl").exists()


def test_output_is_exclusive_and_duplicates_are_rejected(tmp_path):
    adapter = source({"subject": recording()})
    output = tmp_path / "scoring.jsonl"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        write_index(adapter, ["subject"], 3, output)
    assert output.read_text() == "keep"
    with pytest.raises(ValueError, match="unique"):
        write_index(adapter, ["subject", "subject"], 3, tmp_path / "duplicate.jsonl")
    assert not (tmp_path / "duplicate.jsonl").exists()


@pytest.mark.parametrize("context", [0, -1, 1.5, True])
def test_context_validation(context, tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        write_index(source({}), [], context, tmp_path / "index.jsonl")


@pytest.mark.parametrize("labels", [None, np.zeros(10, np.int8), np.full(103, 2), np.zeros(103, float)])
def test_target_shape_and_values_are_checked(labels, tmp_path):
    item = recording()
    adapter = source({"subject": Recording(item.data, labels)})
    with pytest.raises(ValueError, match="Labels"):
        write_index(adapter, ["subject"], 3, tmp_path / "index.jsonl")
    assert not (tmp_path / "index.jsonl").exists()


@pytest.mark.parametrize("change", ["reorder", "short", "long"])
def test_expected_targets_bind_every_eligible_output_and_remove_partial_file(change, tmp_path):
    item = recording(60)
    item.labels[17] = 0
    adapter = source({"subject": item})
    expected = np.concatenate([targets for _, targets, _ in contexts(prepare(item.data), 3, item.labels)])
    if change == "reorder":
        expected = expected[::-1]
    elif change == "short":
        expected = expected[:-1]
    else:
        expected = np.concatenate([expected, [1]])
    path = tmp_path / "index.jsonl"
    with pytest.raises(ValueError, match="evaluated targets"):
        write_index(adapter, ["subject"], 3, path, expected_targets=expected)
    assert not path.exists()


def test_expected_targets_match_and_source_is_verified_before_and_after(tmp_path):
    item = recording(60)
    item.labels[17] = 0
    adapter = source({"subject": item})
    calls = []
    adapter.verify_unchanged = lambda: calls.append(True)
    expected = np.concatenate([targets for _, targets, _ in contexts(prepare(item.data), 3, item.labels)])
    path = tmp_path / "index.jsonl"
    summary = write_index(adapter, ["subject"], 3, path, expected_targets=expected)
    assert calls == [True, True]
    assert summary["eligible_outputs"] == len(expected)
    assert [row["target"] for row in rows(path)] == expected.tolist()


def test_source_change_during_index_removes_closed_output(tmp_path):
    adapter = source({"subject": recording(60)})
    calls = []

    def verify():
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("Source changed")

    adapter.verify_unchanged = verify
    path = tmp_path / "index.jsonl"
    with pytest.raises(ValueError, match="Source changed"):
        write_index(adapter, ["subject"], 3, path)
    assert not path.exists()


@pytest.mark.parametrize("expected", [np.zeros((1, 9), int), np.zeros(9, float), np.full(9, -1)])
def test_expected_targets_validate_before_creating_output(expected, tmp_path):
    path = tmp_path / "index.jsonl"
    with pytest.raises(ValueError, match="Expected targets"):
        write_index(source({"subject": recording(60)}), ["subject"], 3, path, expected_targets=expected)
    assert not path.exists()


def test_frozen_source_context_is_enforced(tmp_path):
    adapter = source({"subject": recording()})
    adapter.context = 3
    output = tmp_path / "index.jsonl"
    with pytest.raises(ValueError, match="frozen"):
        write_index(adapter, ["subject"], 2, output)
    assert not output.exists()
