"""Frozen group assignment cannot depend on scores, coverage, or input order."""

from copy import deepcopy
import json

import numpy as np
import pytest

from sleepkit.artifacts.package import sha256
from sleepkit.recipes.detection.candidate_labels import POLICY
from sleepkit.recipes.detection.split import assign, freeze


def fixture(tmp_path):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "source"
    root.mkdir()
    subjects = {}
    for i in range(10):
        subject = f"s{i}"
        path = root / f"{subject}.h5"
        with h5py.File(path, "w") as stream:
            stream["data"] = np.zeros((3, 60), np.float32)
        subjects[subject] = {
            "source_h5_sha256": sha256(path),
            "counts": {
                "samples": 60,
                "sleep_candidate_samples": 12,
                "wake_candidate_samples": 12,
                "unknown_samples": 36,
                "feature_windows": 9,
                "complete_contexts": 4,
                "incomplete_context_feature_windows": 1,
                "nonfinite_sensor_contexts": 0,
                "unknown_target_contexts": 2,
                "nonfinite_and_unknown_contexts": 0,
                "retained_contexts": 2,
                "retained_sleep_targets": 2,
                "retained_wake_targets": 2,
            },
        }
    report = {
        "schema": "sleepkit.candidate_label_audit/v1",
        "policy": deepcopy(POLICY),
        "source_subjects": 10,
        "subjects": subjects,
        "context_policy": {
            "features_per_context": 2,
            "window_samples": 12,
            "stride_samples": 6,
            "target": "last source sample in each window",
            "grouping": "nonoverlapping native contexts; drop incomplete tail; never stitch",
            "exclusion": "any nonfinite sensor sample or unknown target; reasons counted independently and jointly",
        },
    }
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(report))
    return root, path, report


def test_assign_is_order_independent_and_keeps_groups_together():
    groups = {f"s{i}": f"g{i // 3}" for i in range(30)}
    result = assign(groups)
    assert result == assign(dict(reversed(list(groups.items()))))
    assert sorted(sum(result.values(), [])) == sorted(groups)
    group_sets = [{groups[s] for s in subjects} for subjects in result.values()]
    assert all(not a & b for i, a in enumerate(group_sets) for b in group_sets[i + 1 :])
    assert [len(x) for x in group_sets] == [7, 1, 2]


@pytest.mark.parametrize("seed", [-1, True, 1.5, "0"])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        assign({str(i): str(i) for i in range(10)}, seed=seed)


@pytest.mark.parametrize("group", [None, "", " ", "a\0b", 1])
def test_invalid_groups(group):
    with pytest.raises(ValueError):
        assign({"a": group})


def test_too_few_groups_is_explicit():
    with pytest.raises(ValueError, match="seven"):
        assign({str(i): str(i) for i in range(6)})


def test_freeze_hashes_compatible_split_and_preserves_zero_coverage_subjects(tmp_path):
    from sleepkit.recipes.detection.data import load_split

    root, path, report = fixture(tmp_path)
    out = tmp_path / "frozen"
    manifest = freeze(root, path, out)
    split = load_split(out / "split.json", root)
    assert split == assign({s: s for s in report["subjects"]})
    assert manifest["seed"] == 0
    assert all(sha256(out / name) == value for name, value in manifest["files"].items())
    for detail in report["subjects"].values():
        counts = detail["counts"]
        counts.update(
            sleep_candidate_samples=0,
            wake_candidate_samples=0,
            unknown_samples=60,
            unknown_target_contexts=4,
            retained_contexts=0,
            retained_sleep_targets=0,
            retained_wake_targets=0,
        )
    path.write_text(json.dumps(report))
    second = tmp_path / "empty-coverage"
    other = freeze(root, path, second)
    assert (second / "split.json").read_bytes() == (out / "split.json").read_bytes()
    assert not any(p["both_candidate_classes_retained"] for p in other["partitions"].values())


@pytest.mark.parametrize("mutation", ["hash", "policy", "context", "count", "missing_subject"])
def test_changed_evidence_rejected(tmp_path, mutation):
    root, path, report = fixture(tmp_path)
    if mutation == "hash":
        report["subjects"]["s0"]["source_h5_sha256"] = "0" * 64
    elif mutation == "policy":
        report["policy"]["kind"] = "other"
    elif mutation == "context":
        report["context_policy"]["stride_samples"] = 1
    elif mutation == "count":
        report["subjects"]["s0"]["counts"]["retained_sleep_targets"] = 3
    else:
        del report["subjects"]["s0"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        freeze(root, path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate"])
def test_group_mapping_must_be_complete_and_unique(tmp_path, kind):
    root, path, report = fixture(tmp_path)
    mapping = {s: s for s in report["subjects"]}
    if kind == "missing":
        del mapping["s0"]
    elif kind == "extra":
        mapping["extra"] = "extra"
    text = json.dumps(mapping)
    if kind == "duplicate":
        text = text[:-1] + ', "s0": "different"}'
    group_path = tmp_path / "groups.json"
    group_path.write_text(text)
    with pytest.raises(ValueError):
        freeze(root, path, tmp_path / "out", groups_path=group_path)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_destination_preserved(tmp_path, kind):
    root, path, _ = fixture(tmp_path)
    out = tmp_path / "out"
    if kind == "directory":
        out.mkdir()
    elif kind == "file":
        out.write_text("preserve")
    else:
        out.symlink_to(tmp_path / "nonexistent")
    with pytest.raises(FileExistsError):
        freeze(root, path, out)
    if kind == "file":
        assert out.read_text() == "preserve"
    elif kind == "symlink":
        assert out.is_symlink()
    else:
        assert out.is_dir()


def test_mutating_source_during_freeze_leaves_no_published_output(tmp_path, monkeypatch):
    import sleepkit.recipes.detection.split as module

    root, path, _ = fixture(tmp_path)
    original = module.write_json

    def changing_write(target, value):
        original(target, value)
        if target.name == "split.json":
            with (root / "s0.h5").open("ab") as stream:
                stream.write(b"changed")

    monkeypatch.setattr(module, "write_json", changing_write)
    with pytest.raises(ValueError, match="changed"):
        freeze(root, path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".split-*"))


def test_impossible_context_coverage_rejected(tmp_path):
    root, path, report = fixture(tmp_path)
    report["subjects"]["s0"]["counts"].update(
        retained_contexts=1000000, retained_sleep_targets=1000000, retained_wake_targets=1000000
    )
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Impossible"):
        freeze(root, path, tmp_path / "out")


def test_supplied_group_provenance_and_isolation(tmp_path):
    root, path, report = fixture(tmp_path)
    groups = {s: s for s in report["subjects"]}
    groups["s1"] = groups["s0"]
    mapping = tmp_path / "groups.json"
    mapping.write_text(json.dumps(groups))
    out = tmp_path / "out"
    result = freeze(root, path, out, groups_path=mapping)
    assert result["supplied_groups_sha256"] == sha256(mapping)
    assert result["grouping"] == "supplied_group_mapping_unverified"
    split = json.loads((out / "split.json").read_text())
    assert any("s0" in part and "s1" in part for part in split.values())


def test_subject_id_compatible_with_existing_reader():
    with pytest.raises(ValueError, match="Subject IDs"):
        assign({"contains.dot": "g"})
