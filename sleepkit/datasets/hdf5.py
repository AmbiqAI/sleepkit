import functools
import glob
import os
import random
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from .dataset import SKDataset
from .defines import SampleGenerator, SubjectGenerator


class Hdf5Dataset(SKDataset):
    """Subject feature sets saved in HDF5 format."""

    def __init__(
        self,
        ds_path: Path,
        frame_size: int = 128,
        feat_key: str = "features",
        label_key: str = "labels",
        mask_key: str | None = None,
        feat_cols: list[int] | None = None,
        mask_threshold: float = 0.90,
        **kwargs,
    ) -> None:
        super().__init__(ds_path, frame_size)
        self.ds_path = ds_path
        self.frame_size = frame_size
        self.feat_key = feat_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.feat_cols = feat_cols
        self.mask_threshold = mask_threshold

    @property
    def subject_ids(self) -> list[str]:
        """Get dataset subject IDs

        Returns:
            list[str]: Subject IDs
        """
        subj_paths = glob.glob(str(self.ds_path / "*.h5"), recursive=True)
        subjs = [os.path.splitext(os.path.basename(p))[0] for p in subj_paths]
        subjs.sort()
        return subjs

    @property
    def train_subject_ids(self) -> list[str]:
        """Get train subject ids.

        Returns:
            list[str]: Train subject ids
        """
        return self.subject_ids[: int(0.8 * len(self.subject_ids))]

    @property
    def test_subject_ids(self) -> list[str]:
        """Get test subject ids.

        Returns:
            list[str]: Test subject ids

        """
        return self.subject_ids[int(0.8 * len(self.subject_ids)) :]

    @functools.cached_property
    def feature_shape(self) -> tuple[int, ...]:
        """Get feature shape.

        Returns:
            tuple[int, ...]: Feature shape
        """
        with h5py.File(os.path.join(self.ds_path, f"{self.subject_ids[0]}.h5"), mode="r") as h5:
            feat_shape = (self.frame_size, h5[self.feat_key].shape[-1])
        if self.feat_cols:
            feat_shape = (feat_shape[0], len(self.feat_cols))
        return feat_shape

    @functools.lru_cache(maxsize=2000)
    def subject_stats(self, subject_id: str) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Get subject feature stats.

        Args:
            subject_id (str): Subject ID

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]: Tuple of feature mean and std
        """
        with h5py.File(os.path.join(self.ds_path, f"{subject_id}.h5"), mode="r") as h5:
            features = h5[self.feat_key][:]
            if self.mask_key:
                mask = h5[self.mask_key][:]
                features = features[mask == 1, :]
        feats_mu = np.nanmean(features, axis=0)
        feats_var = np.nanvar(features, axis=0)
        feats_med = np.nanmedian(features, axis=0)
        feats_iqr = np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0)
        return feats_mu, feats_var, feats_med, feats_iqr

    def uniform_subject_generator(
        self,
        subject_ids: list[str] | None = None,
        repeat: bool = True,
        shuffle: bool = True,
    ) -> SubjectGenerator:
        """Yield data for each subject in the array.

        Args:
            subject_ids (pt.ArrayLike): Array of subject ids
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle subject ids.. Defaults to True.

        Returns:
            SubjectGenerator: Subject generator

        Yields:
            Iterator[SubjectGenerator]
        """
        if subject_ids is None:
            subject_ids = self.subject_ids
        subject_idxs = list(range(len(subject_ids)))
        while True:
            if shuffle:
                random.shuffle(subject_idxs)
            for subject_idx in subject_idxs:
                subject_id = subject_ids[subject_idx]
                subject_id = subject_id.decode("ascii") if isinstance(subject_id, bytes) else subject_id
                with h5py.File(os.path.join(self.ds_path, f"{subject_id}.h5"), mode="r") as h5:
                    yield subject_id, h5
            # END FOR
            if not repeat:
                break
        # END WHILE

    def _preprocess_data(
        self,
        subject_id: str,
        x: npt.NDArray,
        y: npt.NDArray,
        mask: npt.NDArray | None = None,
        normalize: bool = False,
        epsilon: float = 1e-3,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]:
        if self.feat_cols:
            x = x[:, self.feat_cols]

        mask_x = x[mask == 1] if mask is not None else x

        # Impute missing values with median
        if mask is not None:
            x_med = np.nanmedian(mask_x, axis=0)
            x[mask == 0, :] = x_med

        if normalize:
            # x = self.normalize_signals(x)
            x_mu = np.nanmean(mask_x, axis=0)
            x_var = np.nanvar(mask_x, axis=0)
            # x_med = np.nanmedian(mask_x, axis=0)
            # x_iqr = np.nanpercentile(mask_x, 75, axis=0) - np.nanpercentile(mask_x, 25, axis=0)
            x = (x - x_mu) / np.sqrt(x_var + epsilon)
            # x = (x - x_med) / (x_iqr + epsilon)
        # END IF

        return x, y, mask

    def load_subject_data(
        self, subject_id: str, normalize: bool = True, epsilon: float = 1e-3
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]:
        """Load subject data
        Args:
            subject_id (str): Subject ID
        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]: Tuple of features, labels, and mask
        """
        mask = None
        with h5py.File(os.path.join(self.ds_path, f"{subject_id}.h5"), mode="r") as h5:
            x = h5[self.feat_key][:]
            y = h5[self.label_key][:]
            if self.mask_key:
                mask = h5[self.mask_key][:]

        return self._preprocess_data(subject_id, x, y, mask, normalize, epsilon)

    def signal_generator(
        self, subject_generator, samples_per_subject: int = 1, normalize: bool = True, epsilon: float = 1e-3
    ) -> SampleGenerator:
        """Generate frames using subject generator from the segments in subject data by
        placing a frame in a random location within one of the segments.

        Args:
            subject_generator (SubjectGenerator): Generator that yields a tuple of subject id and subject data.
                    subject data may contain only signals, since labels are not used.
            samples_per_subject (int): Samples per subject.
        Yields:
            Iterator[SampleGenerator]: Iterator of frames and labels.

        Returns:
            SampleGenerator: Generator of frames and labels.
        """
        in_mem = True
        for subject_id, subject_data in subject_generator:
            xx = subject_data[self.feat_key][:] if in_mem else subject_data[self.feat_key]
            yy = subject_data[self.label_key][:] if in_mem else subject_data[self.label_key]
            mm: npt.NDArray = subject_data[self.mask_key][:] if self.mask_key else None
            xx, yy, mm = self._preprocess_data(subject_id, xx, yy, mm, normalize, epsilon)

            num_samples = 0
            num_attempts = 0
            while num_samples < samples_per_subject:
                frame_start = np.random.randint(xx.shape[0] - self.frame_size)
                frame_end = frame_start + self.frame_size
                x = xx[frame_start:frame_end]
                y = yy[frame_start:frame_end]

                is_invalid = np.isnan(x).any() or (np.mean(mm[frame_start:frame_end]) < self.mask_threshold)
                if is_invalid:
                    num_attempts += 1
                    if num_attempts > 10:
                        num_samples += 1
                        num_attempts = 0
                else:
                    num_samples += 1
                    num_attempts = 0
                    yield x, y
                # END IF
            # END WHILE
        # END FOR
