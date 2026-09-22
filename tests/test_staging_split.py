"""Subject membership must be fixed and disjoint before features are read."""

import copy
import json

import pytest

from sleepkit.recipes.staging.split import create_split, load_split, validate_split


def test_split_is_deterministic_order_independent_and_complete(tmp_path):
    subjects = [f"subject-{n}" for n in range(12)]
    split = create_split(subjects, dataset="synthetic", validation_count=3, test_count=2, seed=4)
    assert split == create_split(subjects[::-1], dataset="synthetic", validation_count=3, test_count=2, seed=4)
    assert [len(split["partitions"][p]) for p in ("train", "validation", "test")] == [7, 3, 2]
    assert set(sum(split["partitions"].values(), [])) == set(subjects)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    assert load_split(path) == split
    original = copy.deepcopy(split)
    validate_split(split)["partitions"]["train"].clear()
    assert split == original


@pytest.mark.parametrize("kind", ["overlap", "duplicate", "empty", "traversal", "float", "extra", "dataset", "schema"])
def test_bad_manifests_rejected(kind):
    split = create_split(["a", "b", "c", "d"], dataset="synthetic", validation_count=1, test_count=1)
    groups = split["partitions"]
    if kind == "overlap":
        groups["test"] = groups["train"][:1]
    elif kind == "duplicate":
        groups["train"] *= 2
    elif kind == "empty":
        groups["validation"] = []
    elif kind in {"traversal", "float"}:
        groups["train"] = ["../outside"] if kind == "traversal" else [1.2]
    elif kind == "extra":
        groups["other"] = ["z"]
    else:
        split[kind] = ""
    with pytest.raises(ValueError):
        validate_split(split)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "split.json"
    path.write_text('{"partitions": {}, "partitions": {}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_split(path)


@pytest.mark.parametrize(
    "kwargs",
    [{"seed": True}, {"seed": -1}, {"seed": 2**32}, {"validation_count": 0}, {"test_count": True}, {"test_count": 9}],
)
def test_invalid_split_generation_rejected(kwargs):
    options = {"dataset": "synthetic", "validation_count": 1, "test_count": 1, **kwargs}
    with pytest.raises(ValueError):
        create_split(["a", "b", "c", "d"], **options)
