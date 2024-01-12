import abc
import numpy as np
from ..defines import SKFeatureParams

class PoorSignalError(Exception):
    """Poor signal error."""


class NoSignalError(Exception):
    """No signal error."""

class FeatSet(abc.ABC):

    @staticmethod
    def name() -> str:
        """Feature set name."""
        raise NotImplementedError()

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        raise NotImplementedError()

    @staticmethod
    def compute_features(
        ds_name: str, subject_id: str, args: SKFeatureParams
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute features for subject.

        Args:
            ds_name (str): Dataset name
            subject_id (str): Subject ID
            args (SKFeatureParams): Feature generation parameters

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Features, labels, masks
        """
        raise NotImplementedError()
