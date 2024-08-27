"""
# H5Dataloader API

The `H5Dataloader` class is used to load features from a folder containing HDF5 files.
Each file is treated as a subject and contains features, labels, and an optional mask.
The features should be stored in a dataset with key `feat_key`, labels with key `label_key`, and mask with key `mask_key`.
The mask is used to filter out invalid frames.

Classes:
    H5Dataloader: Dataloader to load features from folder containing HDF5 files.

"""

import functools
from pathlib import Path
from typing import Generator, Callable

import h5py
import numpy as np
import numpy.typing as npt


XYType = tuple[npt.NDArray, npt.NDArray]
XYMaskType = tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]

SampleGenerator = Generator[XYType, None, None]
MaskSampleGenerator = Generator[XYMaskType, None, None]
PreprocessorType = Callable[[XYMaskType], XYMaskType]


class H5Dataloader:
    def __init__(
        self,
        path: Path,
        frame_size: int = 128,
        feat_key: str = "features",
        label_key: str = "labels",
        mask_key: str | None = None,
        feat_cols: list[int] | None = None,
        class_map: dict[int, int] | None = None,
        mask_threshold: float = 0.90,
        cacheable: bool = True,
    ):
        """Dataloader to load features from folder containing HDF5 files.

        Each file is treated as a subject and contains features, labels, and optional mask.
        The features should be stored in a dataset with key `feat_key`, labels with key `label_key`,
        and mask with key `mask_key`. The mask is used to filter out invalid frames.

        The features format should have shape (time, features) and labels should have shape (time,).

        Args:
            path (Path): Path to folder containing HDF5 files.
            frame_size (int, optional): Frame size. Defaults to 128.
            feat_key (str, optional): Feature key. Defaults to "features".
            label_key (str, optional): Label key. Defaults to "labels".
            mask_key (str | None, optional): Mask key. Defaults to None.
            feat_cols (list[int] | None, optional): Feature columns. Defaults to None.
            class_map (dict[int, int] | None, optional): Class map. Defaults to None.
            mask_threshold (float, optional): Mask threshold. Defaults to 0.90.
            cacheable (bool, optional): Cacheable. Defaults to True.

        """
        self.path = path
        self.frame_size = frame_size
        self.feat_key = feat_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.feat_cols = feat_cols
        self.class_map = class_map
        self.mask_threshold = mask_threshold

        self._subject_paths = dict()
        self._cached_data: dict[str, np.ndarray] = {}
        self.cacheabe = cacheable
        self.build()

    def build(self) -> None:
        """Build dataloader"""
        self._subject_paths = {p.stem: p for p in self.path.rglob("*.h5")}

    @property
    def subject_ids(self) -> list[str]:
        """Get all subject IDs.

        Returns:
            list[str]: Subject IDs
        """
        return list(self._subject_paths.keys())

    @property
    def train_subject_ids(self) -> list[str]:
        """Get train subject IDs.

        Returns:
            list[str]: Train subject IDs
        """
        return self.subject_ids[: int(0.8 * len(self.subject_ids))]

    @property
    def test_subject_ids(self) -> list[str]:
        """Get test subject ids.

        Returns:
            list[str]: Test subject IDs

        """
        return self.subject_ids[int(0.8 * len(self.subject_ids)) :]

    @functools.cached_property
    def feature_shape(self) -> tuple[int, int]:
        """Get feature shape.

        Returns:
            tuple[int, ...]: Feature shape
        """
        # Grab the first subject to get the feature shape
        with h5py.File(self._subject_paths[self.subject_ids[0]], mode="r") as h5:
            feat_shape = (self.frame_size, h5[self.feat_key].shape[-1])
        if self.feat_cols:
            feat_shape = (feat_shape[0], len(self.feat_cols))
        return feat_shape

    def load_subject_data(
        self,
        subject_id: str,
        preprocessor: PreprocessorType | None = None,
    ) -> XYMaskType:
        """Load subject data

        Args:
            subject_id (str): Subject ID
            preprocessor (PreprocessorType|None): Preprocessor function

        Returns:
            XYMaskType: Tuple of features, labels, and mask
        """

        # Skip if data is cached
        if subject_id in self._cached_data:
            x, y, mask = self._cached_data[subject_id]
        else:
            # Load data from HDF5 file
            mask = None
            with h5py.File(self._subject_paths[subject_id], mode="r") as h5:
                x = h5[self.feat_key][:]
                y = h5[self.label_key][:]
                if self.mask_key:
                    mask = h5[self.mask_key][:]
                # END IF
            # END WITH

            # Grab target feature columns
            if self.feat_cols:
                x = x[:, self.feat_cols]
            # END IF

            # Apply class map
            if self.class_map:
                y = np.vectorize(self.class_map.get)(y)
            # END IF

            # Cache data if cacheable
            if self.cacheabe:
                self._cached_data[subject_id] = x, y, mask
            # END IF
        # END IF

        # Dont cache preprocessed data as it may be different for each epoch
        if preprocessor:
            x, y, mask = preprocessor(x, y, mask)

        return x, y, mask

    def signal_generator(
        self,
        subject_generator: Generator[str, None, None],
        samples_per_subject: int = 1,
        preprocessor: Callable[[XYMaskType], XYMaskType] | None = None,
    ) -> SampleGenerator:
        """Generate random frames of signals from subject data.

        Args:
            subject_generator (Generator[str, None, None]): Generator of subject ids.
            samples_per_subject (int, optional): Number of samples per subject. Defaults to 1.
            preprocessor (PreprocessorType|None, optional): Preprocessor function. Defaults to None.

        Yields:
            Iterator[SampleGenerator]: Iterator of frames and labels.

        Returns:
            SampleGenerator: Generator of frames and labels.
        """
        for subject_id in subject_generator:
            x, y, mask = self.load_subject_data(subject_id, preprocessor=preprocessor)
            num_samples = 0
            num_attempts = 0
            while num_samples < samples_per_subject:
                frame_start = np.random.randint(x.shape[0] - self.frame_size)
                frame_end = frame_start + self.frame_size
                x_frame = x[frame_start:frame_end]
                y_frame = y[frame_start:frame_end]

                is_invalid = np.isnan(x_frame).any() or (np.mean(mask[frame_start:frame_end]) < self.mask_threshold)
                if is_invalid:
                    num_attempts += 1
                    if num_attempts > 10:
                        num_samples += 1
                        num_attempts = 0
                else:
                    num_samples += 1
                    num_attempts = 0
                    yield x_frame, y_frame
                # END IF
            # END WHILE
        # END FOR

    def close(self):
        """Close dataloader"""
        self._cached_data.clear()
        self._subject_paths.clear()
