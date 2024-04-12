from .dataset import SKDataset

_datasets: dict[str, type[SKDataset]] = {}


class DatasetFactory:
    """Dataset factory enables registering, creating, and listing datasets. It is a singleton class."""

    @staticmethod
    def register(name: str, dataset: type[SKDataset]) -> None:
        """Register a dataset

        Args:
            name (str): dataset name
            dataset (type[SKDataset]): dataset
        """
        _datasets[name] = dataset

    @staticmethod
    def create(name: str, **kwargs) -> SKDataset:
        """Create a dataset

        Args:
            name (str): dataset name

        Returns:
            SKDataset: dataset
        """
        return _datasets[name](**kwargs)

    @staticmethod
    def list() -> list[str]:
        """List registered datasets

        Returns:
            list[str]: dataset names
        """
        return list(_datasets.keys())

    @staticmethod
    def get(name: str) -> type[SKDataset]:
        """Get a dataset

        Args:
            name (str): dataset name

        Returns:
            type[SKDataset]: dataset
        """
        return _datasets[name]

    @staticmethod
    def has(name: str) -> bool:
        """Check if a dataset is registered

        Args:
            name (str): dataset name

        Returns:
            bool: True if dataset is registered
        """
        return name in _datasets
