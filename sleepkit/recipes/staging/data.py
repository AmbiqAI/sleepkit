"""In-memory preparation for a small array-backed Keras staging recipe."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import prepare_subject, read_subject, window_subject


@dataclass(frozen=True)
class PreparedPartition:
    """Complete contexts with supervision; coordinates remain private run evidence."""

    features: np.ndarray
    targets: np.ndarray
    scoring_mask: np.ndarray
    coordinates: tuple[tuple[str, int], ...]
    coverage: dict[str, int]

    def fit_arrays(self):
        if (
            self.features.dtype != np.float32
            or self.features.ndim != 3
            or self.features.shape[1:] != (240, 14)
            or not len(self.features)
            or not np.isfinite(self.features).all()
            or self.targets.shape != self.features.shape[:2]
            or self.targets.dtype.kind not in "iu"
            or not np.isin(self.targets, (-1, 0, 1, 2)).all()
            or self.scoring_mask.dtype != bool
            or self.scoring_mask.shape != self.targets.shape
            or not self.scoring_mask.any(axis=1).all()
            or (self.targets[self.scoring_mask] < 0).any()
        ):
            raise ValueError(
                "Expected finite supervised float32 [windows,240,14] with mapped targets and boolean masks"
            )
        # Sparse CE validates targets even where their sample weight is zero.
        targets = np.where(self.scoring_mask, self.targets, 0).astype(np.int32)
        return self.features, targets, self.scoring_mask.astype(np.float32)


def prepare_partition(root, subjects):
    """Preserve each record's context; omit only windows with no scored epochs.

    Materializes the partition in memory. No filtering by class, quality fraction,
    or split-dependent statistics; normalization remains whole-subject/offline.
    """
    features, targets, masks, coordinates, coverage = [], [], [], [], {}
    for subject in subjects:
        windows = window_subject(prepare_subject(read_subject(Path(root) / f"{subject}.h5")))
        for key, value in windows.coverage.items():
            coverage[key] = coverage.get(key, 0) + value
        retained = windows.scoring_mask.any(axis=1)
        features.append(windows.features[retained])
        targets.append(windows.targets[retained])
        masks.append(windows.scoring_mask[retained])
        coordinates.extend((subject, int(start)) for start in windows.starts[retained])
    if not coordinates:
        raise ValueError("Every partition needs at least one supervised complete window")
    coverage.update(
        subjects=len(subjects),
        supervised_windows=len(coordinates),
        zero_supervision_windows=coverage["window_count"] - len(coordinates),
    )
    result = PreparedPartition(
        np.concatenate(features), np.concatenate(targets), np.concatenate(masks), tuple(coordinates), coverage
    )
    result.coverage["prepared_array_bytes"] = sum(
        array.nbytes for array in (result.features, result.targets, result.scoring_mask)
    )
    return result
