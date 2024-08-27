import abc
import os
import functools

from tqdm.contrib.concurrent import process_map
import neuralspot_edge as nse

from ..features import FeatureFactory
from ..defines import TaskParams
from ..datasets import DatasetFactory, Dataset


class Task(abc.ABC):
    """Task base class. All tasks should inherit from this class."""

    @staticmethod
    def download(params: TaskParams) -> None:
        """ "Download datasets for task

        Args:
            params (TaskParams): Task parameters

        """
        os.makedirs(params.job_dir, exist_ok=True)
        logger = nse.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "download.log")
        logger.debug(f"Creating working directory in {params.job_dir}")

        for ds in params.datasets:
            if DatasetFactory.has(ds.name):
                logger.debug(f"Downloading dataset: {ds.name}")
                ds: Dataset = DatasetFactory.get(ds.name)(**ds.params)
                ds.download(
                    num_workers=params.num_workers,
                    force=params.force_download,
                )
            # END IF
        # END FOR

    @staticmethod
    def feature(params: TaskParams) -> None:
        """Generate features for task

        Args:
            params (TaskParams): Task parameters
        """
        os.makedirs(params.job_dir, exist_ok=True)
        logger = nse.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "download.log")
        logger.debug(f"Creating working directory in {params.job_dir}")

        if not FeatureFactory.has(params.feature.name):
            raise NotImplementedError(f"Feature set {params.feature.name} not implemented")
        fset = FeatureFactory.get(params.feature.name)

        for dataset in params.datasets:
            if not DatasetFactory.has(dataset.name):
                logger.error(f"Dataset {dataset.name} not found")
                continue
            # END IF
            os.makedirs(params.feature.save_path / dataset.name, exist_ok=True)
            ds: Dataset = DatasetFactory.get(dataset.name)(**dataset.params)
            fn = functools.partial(fset.generate_subject_features, ds_name=dataset.name, params=params)
            process_map(fn, ds.subject_ids, max_workers=params.num_workers, desc=f"Gen features for {dataset.name}")
        # END FOR

    @staticmethod
    def train(params: TaskParams) -> None:
        """Train a model

        Args:
            params (TaskParams): Task parameters

        """
        raise NotImplementedError()

    @staticmethod
    def evaluate(params: TaskParams) -> None:
        """Evaluate a model

        Args:
            params (TaskParams): Task parameters

        """
        raise NotImplementedError()

    @staticmethod
    def export(params: TaskParams) -> None:
        """Export a model

        Args:
            params (TaskParams): Task parameters

        """
        raise NotImplementedError()

    @staticmethod
    def demo(params: TaskParams) -> None:
        """Run a demo

        Args:
            params (TaskParams): Task parameters

        """
        raise NotImplementedError()
