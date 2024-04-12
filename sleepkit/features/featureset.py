import abc

from ..defines import SKFeatureParams


class PoorSignalError(Exception):
    """Poor signal error."""


class NoSignalError(Exception):
    """No signal error."""


class SKFeatureSet(abc.ABC):
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
    def generate_features(ds_subject: tuple[str, str], args: SKFeatureParams):
        """Generate features for dataset subject.

        Args:
            ds_subject (tuple[str, str]): Dataset name and subject ID
            args (SKFeatureParams): Feature generation parameters
        """
        raise NotImplementedError()
