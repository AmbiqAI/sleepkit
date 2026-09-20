"""Versioned semantics must agree in inference metadata, not just tensor shapes."""

from copy import deepcopy

import pytest

from sleepkit.recipes.detection.output_contract import output_contract, validate_output
from sleepkit.recipes.detection.split import TARGET


def test_legacy_and_membership_contracts():
    for target in (None, TARGET):
        version, classes = output_contract(target)
        assert (
            validate_output({"recipe": version, "class_names": classes, "output": "logits", "target": target})
            == classes
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("recipe", "sleepkit.detection/v1"),
        ("class_names", ["WAKE", "SLEEP"]),
        ("output", "probabilities"),
        ("target", None),
    ],
)
def test_mixed_target_metadata_rejected(field, value):
    contract = {
        "recipe": "sleepkit.detection/v2",
        "class_names": TARGET["classes"],
        "output": "logits",
        "target": deepcopy(TARGET),
    }
    contract[field] = value
    with pytest.raises(ValueError, match="Unsupported"):
        validate_output(contract)


def test_unknown_target_rejected():
    target = deepcopy(TARGET)
    target["kind"] = "unrecognized/v1"
    with pytest.raises(ValueError, match="Unsupported"):
        output_contract(target)


def test_saved_predictions_preserve_evaluated_precision():
    import numpy as np
    from sleepkit.recipes.detection.recipe import evaluate

    logits = np.array([[[1.0, 1.0 + 1e-8]]], dtype=np.float64)
    saved = []
    metrics = evaluate(lambda x, training: x, [(logits, np.array([[1]]))], prediction_batches=saved)
    assert metrics["accuracy"] == 1
    np.testing.assert_array_equal(saved[0][0], logits.reshape(-1, 2))
    assert np.argmax(saved[0][0], axis=1)[0] == 1
