"""SleepKit: composable sleep experiments and deployment tools.

Legacy exports are resolved only when requested. Importing the package does not
initialize TensorFlow, dataset clients, logging, plotting, or hardware backends.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sleepkit")
except PackageNotFoundError:
    __version__ = "1.0.0a1"

_LEGACY = {
    **dict.fromkeys(["DatasetFactory", "Dataset"], "datasets"),
    **dict.fromkeys(
        ["QuantizationParams", "FeatureParams", "TaskParams", "TaskMode", "NamedParams", "SleepApnea", "SleepStage"],
        "defines",
    ),
    **dict.fromkeys(["FeatureFactory", "FeatureSet", "H5Dataloader"], "features"),
    "ModelFactory": "models",
    **dict.fromkeys(["TaskFactory", "Task", "ApneaTask", "StageTask"], "tasks"),
}


def __getattr__(name):
    if name not in _LEGACY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_LEGACY[name]}", __name__), name)
    globals()[name] = value
    return value
