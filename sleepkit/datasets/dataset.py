import abc
import os
from pathlib import Path

from .defines import SubjectGenerator


class Dataset(abc.ABC):
    path: Path

    def __init__(self, path: os.PathLike | None = None, **kwargs) -> None:
        """Dataset serves as a base class to download and provide unified access to datasets.

        Args:
            path (os.PathLike|None, optional): Path to dataset base path. Defaults to None.

        Example:

        ```python
        import numpy as np
        import sleepkit as sk

        class MyDataset(sk.Dataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            @property
            def name(self) -> str:
                return 'my-dataset'

            @property
            def sampling_rate(self) -> int:
                return 100

            def get_train_patient_ids(self) -> npt.NDArray:
                return np.arange(80)

            def get_test_patient_ids(self) -> npt.NDArray:
                return np.arange(80, 100)

            @contextlib.contextmanager
            def patient_data(self, patient_id: int) -> Generator[PatientData, None, None]:
                data = np.random.randn(1000)
                segs = np.random.randint(0, 1000, (10, 2))
                yield {"data": data, "segmentations": segs}

            def signal_generator(
                self,
                patient_generator: PatientGenerator,
                frame_size: int,
                samples_per_patient: int = 1,
                target_rate: int | None = None,
            ) -> Generator[npt.NDArray, None, None]:
                for patient in patient_generator:
                    for _ in range(samples_per_patient):
                        with self.patient_data(patient) as pt:
                            yield pt["data"]

            def download(self, num_workers: int | None = None, force: bool = False):
                pass

        # Register dataset
        sk.DatasetFactory.register("my-dataset", MyDataset)
        ```

        """

        if path is None:
            path = os.environ.get("SK_DATASET_PATH", None)
        if path is None:
            raise ValueError("Root dataset path is not set")
        self.path = Path(path)

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

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        This will download preprocessed HDF5 files from S3.

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        raise NotImplementedError()
