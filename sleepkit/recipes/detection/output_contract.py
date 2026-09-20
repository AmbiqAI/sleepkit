"""Explicit output semantics shared by export and inference."""

from .split import TARGET

LEGACY_CLASSES = ["WAKE", "SLEEP"]


def output_contract(target=None):
    if target is None:
        return "sleepkit.detection/v1", list(LEGACY_CLASSES)
    if target != TARGET:
        raise ValueError("Unsupported detection target contract")
    return "sleepkit.detection/v2", list(TARGET["classes"])


def validate_output(recipe):
    version, classes = output_contract(recipe.get("target"))
    if recipe.get("recipe") != version or recipe.get("class_names") != classes or recipe.get("output") != "logits":
        raise ValueError("Unsupported detection output contract")
    return classes
