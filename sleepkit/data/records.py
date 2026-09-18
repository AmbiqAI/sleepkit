"""Small, framework-independent signal and annotation records."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Signal:
    values: np.ndarray
    sample_rate: float
    unit: str
    modality: str
    location: str
    start: float = 0.0
    valid: np.ndarray | None = None

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.float32).copy()
        if values.ndim != 1 or not values.size:
            raise ValueError("Signals must be nonempty one-dimensional arrays")
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0 or not np.isfinite(self.start):
            raise ValueError("Signal rate must be positive and start must be finite")
        if not all((self.unit, self.modality, self.location)):
            raise ValueError("Signal units, modality, and location are required")
        valid = np.isfinite(values)
        if self.valid is not None:
            supplied = np.asarray(self.valid, dtype=bool)
            if supplied.shape != values.shape:
                raise ValueError("Signal validity must match signal shape")
            valid &= supplied
        values.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)

    @property
    def stop(self):
        return self.start + len(self.values) / self.sample_rate


@dataclass(frozen=True)
class Annotation:
    start: float
    stop: float
    label: int

    def __post_init__(self):
        if not np.isfinite([self.start, self.stop]).all() or self.stop <= self.start:
            raise ValueError("Annotations require a finite, positive interval")
        if not isinstance(self.label, (int, np.integer)) or self.label < 0:
            raise ValueError("Annotation labels must be nonnegative integers")


@dataclass(frozen=True)
class Record:
    dataset: str
    subject: str
    recording: str
    signals: dict[str, Signal]
    annotations: tuple[Annotation, ...] = ()
    source_revision: str = "unknown"

    def __post_init__(self):
        if not all((self.dataset, self.subject, self.recording)) or not self.signals:
            raise ValueError("Dataset, subject, recording, and signals are required")
        annotations = tuple(sorted(self.annotations, key=lambda a: a.start))
        if any(a.stop > b.start for a, b in zip(annotations, annotations[1:])):
            raise ValueError("Target annotation intervals must not overlap")
        object.__setattr__(self, "annotations", annotations)

    @property
    def subject_key(self):
        return (self.dataset, self.subject)


@dataclass(frozen=True)
class SubjectSplit:
    train: tuple[tuple[str, str], ...]
    validation: tuple[tuple[str, str], ...]
    test: tuple[tuple[str, str], ...]

    def __post_init__(self):
        groups = [self.train, self.validation, self.test]
        flat = [subject for group in groups for subject in group]
        if any(not group for group in groups) or len(set(flat)) != len(flat):
            raise ValueError("Subject splits must be nonempty and disjoint")

    def select(self, records, partition):
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {partition}")
        keys = set(getattr(self, partition))
        selected = [record for record in records if record.subject_key in keys]
        if {record.subject_key for record in selected} != keys:
            raise ValueError(f"Missing subjects in {partition}")
        return selected

    def to_dict(self):
        return {name: [list(key) for key in getattr(self, name)] for name in ("train", "validation", "test")}

    @classmethod
    def from_dict(cls, value):
        return cls(**{name: tuple(tuple(key) for key in value[name]) for name in ("train", "validation", "test")})


def split_subjects(records, seed=0, validation_fraction=0.2, test_fraction=0.2):
    """Split sorted dataset-qualified subjects; keep every session together."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one")
    keys = sorted({record.subject_key for record in records})
    if len(keys) < 3:
        raise ValueError("At least three subjects are required")
    order = np.random.default_rng(seed).permutation(len(keys))
    shuffled = [keys[index] for index in order]
    nv = max(1, int(len(keys) * validation_fraction))
    nt = max(1, int(len(keys) * test_fraction))
    return SubjectSplit(tuple(shuffled[nv + nt :]), tuple(shuffled[:nv]), tuple(shuffled[nv : nv + nt]))
