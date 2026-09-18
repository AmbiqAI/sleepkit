"""Small artifact contracts independent of training, runtimes, and domains."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re

SCHEMA = "sleepkit.artifacts/v1"
DTYPES = {"float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"}


def filename(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Expected a simple artifact filename: {value!r}")
    return value


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | None, ...]
    dtype: str
    meaning: str
    scale: float | None = None
    zero_point: int | None = None

    def __post_init__(self):
        if not self.name or not self.meaning or self.dtype not in DTYPES:
            raise ValueError("Tensor name, meaning, and supported dtype are required")
        if any(d is not None and (type(d) is not int or d < 1) for d in self.shape):
            raise ValueError("Tensor dimensions must be positive integers or null")
        if (self.scale is None) != (self.zero_point is None):
            raise ValueError("Quantization requires scale and zero point together")
        if self.scale is not None:
            if not math.isfinite(self.scale) or self.scale <= 0 or type(self.zero_point) is not int:
                raise ValueError("Quantization scale must be positive and zero point must be an integer")
            limits = {"int8": (-128, 127), "uint8": (0, 255), "int16": (-32768, 32767)}
            if self.dtype not in limits or not limits[self.dtype][0] <= self.zero_point <= limits[self.dtype][1]:
                raise ValueError("Unsupported quantization dtype or zero point")


@dataclass(frozen=True)
class Artifact:
    source: Path
    name: str
    role: str
    format: str
    origin: str
    inputs: tuple[TensorSpec, ...] = ()
    outputs: tuple[TensorSpec, ...] = ()
    expected_sha256: str | None = None

    def __post_init__(self):
        filename(self.name)
        if not all((self.role, self.format, self.origin)):
            raise ValueError("Artifact role, format, and source provenance are required")
        for tensors in (self.inputs, self.outputs):
            if len({t.name for t in tensors}) != len(tensors):
                raise ValueError("Tensor names must be unique within an input/output signature")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    reason: str
    artifacts: dict[str, str] = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in {"passed", "failed", "not_run", "not_applicable"} or not self.name or not self.reason:
            raise ValueError("Checks need a name, explicit status, and reason")
        if self.status == "passed" and not self.artifacts:
            raise ValueError("Passing evidence must be bound to artifact hashes")
        for name, digest in self.artifacts.items():
            filename(name)
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid evidence artifact hash")

    def to_dict(self):
        return asdict(self)
