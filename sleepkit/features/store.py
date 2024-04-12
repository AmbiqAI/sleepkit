from ..utils import setup_logger
from .featureset import SKFeatureSet

_fsets: dict[str, type[SKFeatureSet]] = {}

logger = setup_logger(__name__)


class FeatureStore:
    """Feature store"""

    @staticmethod
    def register(name: str, fset: type[SKFeatureSet]) -> None:
        """Register a feature set

        Args:
            name (str): feature set name
            fset (type[SKFeatures]): feature set
        """
        _fsets[name] = fset

    @staticmethod
    def list() -> list[str]:
        """List registered feature sets

        Returns:
            list[str]: feature set names
        """
        return list(_fsets.keys())

    @staticmethod
    def get(name: str) -> type[SKFeatureSet]:
        """Get a feature set

        Args:
            name (str): feature set name

        Returns:
            type[SKFeatures]: feature set
        """
        return _fsets[name]

    @staticmethod
    def has(name: str) -> bool:
        """Check if a feature set is registered

        Args:
            name (str): feature set name

        Returns:
            bool: True if feature set is registered
        """
        return name in _fsets
