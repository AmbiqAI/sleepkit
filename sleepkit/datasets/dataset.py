import abc
from pathlib import Path

import numpy.typing as npt

from .defines import SubjectGenerator


class SKDataset(abc.ABC):
    """Base SK dataset."""

    def __init__(self, ds_path: Path, frame_size: int = 128, **kwargs) -> None:
        self.ds_path = ds_path
        self.frame_size = frame_size

    @property
    def subject_ids(self) -> list[str]:
        """Get dataset subject IDs

        Returns:
            list[str]: Subject IDs
        """
        raise NotImplementedError()

    @property
    def train_subject_ids(self) -> list[str]:
        """Get train subject ids"""
        raise NotImplementedError()

    @property
    def test_subject_ids(self) -> list[str]:
        """Get test subject ids"""
        raise NotImplementedError()

    @property
    def feature_shape(self) -> tuple[int, int]:
        """Get feature shape"""
        raise NotImplementedError()

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
        raise NotImplementedError()

    def load_subject_data(
        self, subject_id: str, normalize: bool = True, epsilon: float = 1e-6
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]:
        """Load subject data
        Args:
            subject_id (str): Subject ID
        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]: Tuple of features and labels
        """
        raise NotImplementedError()

    def signal_generator(
        self, subject_generator, samples_per_subject: int = 1, normalize: bool = True, epsilon: float = 1e-6
    ):
        """
        Generate frames using subject generator.
        from the segments in subject data by placing a frame in a random location within one of the segments.
        Args:
            subject_generator (SubjectGenerator): Generator that yields a tuple of subject id and subject data.
                    subject data may contain only signals, since labels are not used.
            samples_per_subject (int): Samples per subject.
        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """
        raise NotImplementedError()
