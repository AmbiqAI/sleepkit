"""
# Feature Set Abstract Class API

The `FeatureSet` abstract class defines the interface for feature sets in sleep analysis tasks.

Classes:
    FeatureSet: Abstract class for feature

"""

import abc

from ..defines import TaskParams


class FeatureSet(abc.ABC):
    """Feature set abstract class."""

    @staticmethod
    def name() -> str:
        """Feature set name."""
        raise NotImplementedError()

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        raise NotImplementedError()

    @staticmethod
    def generate_subject_features(subject_id: str, ds_name: str, params: TaskParams):
        """Generate features for given dataset and subject.

        Args:
            subject_id (str): Subject ID
            ds_name (str): Dataset name
            params (TaskParams): Feature generation parameters
        """
        raise NotImplementedError()
