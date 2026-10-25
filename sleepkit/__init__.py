"""SleepKit public API, resolved lazily to keep artifact tools independent."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sleepkit")
except PackageNotFoundError:
    __version__ = "0.12.0"

_MODULES = {"cli", "datasets", "models", "backends", "tasks", "features"}
_EXPORTS = {
    **dict.fromkeys(["DatasetFactory", "Dataset"], "datasets"),
    **dict.fromkeys(
        ["QuantizationParams", "FeatureParams", "TaskParams", "TaskMode", "NamedParams", "SleepApnea", "SleepStage"],
        "defines",
    ),
    **dict.fromkeys(["FeatureFactory", "FeatureSet", "H5Dataloader"], "features"),
    "ModelFactory": "models",
    **dict.fromkeys(["TaskFactory", "Task", "ApneaTask", "StageTask"], "tasks"),
}
__all__ = ["__version__", *sorted(_MODULES), *_EXPORTS]


def __getattr__(name):
    if name in _MODULES:
        value = import_module(f".{name}", __name__)
    elif name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
