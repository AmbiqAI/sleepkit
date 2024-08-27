import functools
import glob
import logging
import math
import os
from enum import IntEnum
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
import scipy.io
import neuralspot_edge as nse
from tqdm.contrib.concurrent import process_map

from ..defines import SleepStage
from .dataset import Dataset
from .defines import SubjectGenerator

logger = logging.getLogger(__name__)


class YsywSleepStage(IntEnum):
    """YSYW sleep stages"""

    nonrem1 = 0  # N1
    nonrem2 = 1  # N2
    nonrem3 = 2  # N3/4
    rem = 3
    undefined = 4
    wake = 5


YsywStageMap = {
    YsywSleepStage.wake: SleepStage.wake,
    YsywSleepStage.nonrem1: SleepStage.stage1,
    YsywSleepStage.nonrem2: SleepStage.stage2,
    YsywSleepStage.nonrem3: SleepStage.stage3,
    YsywSleepStage.rem: SleepStage.rem,
    YsywSleepStage.undefined: SleepStage.wake,
}


class YsywDataset(Dataset):
    def __init__(
        self,
        target_rate: int = 128,
        **kwargs,
    ) -> None:
        """YSYW dataset

        Args:
            target_rate (int, optional): Target rate. Defaults to 128.

        """

        super().__init__(**kwargs)
        # If last folder is not "ysyw", then add it
        if self.path.parts[-1] != "ysyw":
            self.path = self.path / "ysyw"
        self.target_rate = target_rate

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 200

    @functools.cached_property
    def subject_ids(self) -> list[str]:
        """Get dataset subject IDs

        Returns:
            list[str]: Subject IDs
        """
        pts = glob.glob(os.path.join(self.path, "*.h5"))
        pts = [os.path.splitext(os.path.basename(p))[0] for p in pts]
        pts.sort()
        return pts

    @property
    def train_subject_ids(self) -> list[str]:
        """Get train subject ids"""
        return self.subject_ids[: int(0.8 * len(self.subject_ids))]

    @property
    def test_subject_ids(self) -> list[str]:
        """Get test subject ids"""
        return self.subject_ids[int(0.8 * len(self.subject_ids)) :]

    @property
    def signal_names(self) -> list[str]:
        """Signal names"""
        return [
            # EEG
            "F3-M2",
            "F4-M1",
            "C3-M2",
            "C4-M1",
            "O1-M2",
            "O2-M1",
            # EOG
            "E1-M2",
            # EMG
            "Chin1-Chin2",
            # RSP
            "ABD",
            "CHEST",
            "AIRFLOW",
            # SPO2
            "SaO2",
            # ECG
            "ECG",
        ]

    def uniform_subject_generator(
        self,
        subject_ids: list[str] | None = None,
        repeat: bool = True,
        shuffle: bool = True,
    ) -> SubjectGenerator:
        """Yield Subject IDs uniformly.

        Args:
            subject_ids (list[str], optional): Array of subject ids. Defaults to None.
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle subject ids. Defaults to True.

        Returns:
            SubjectGenerator: Subject generator
        """
        if subject_ids is None:
            subject_ids = self.subject_ids
        for idx in nse.utils.uniform_id_generator(list(range(len(subject_ids))), repeat=repeat, shuffle=shuffle):
            subject_id = subject_ids[idx]
            yield (subject_id.decode("ascii") if isinstance(subject_id, bytes) else subject_id)

    def load_signal_for_subject(
        self,
        subject_id: str,
        signal_label: str,
        start: int = 0,
        data_size: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """Load signal into memory for subject at target rate (resampling if needed)

        Args:
            subject_id (str): Subject ID
            signal_label (str): Signal label
            start (int): Start location @ target rate
            data_size (int): Data length @ target rate

        Returns:
            npt.NDArray[np.float32]: Signal
        """
        with h5py.File(self._get_subject_h5_path(subject_id), mode="r") as fp:
            signal_idx = self.signal_names.index(signal_label)
            sample_rate = self.sampling_rate
            sig_start = round(start * (sample_rate / self.target_rate))
            sig_len = fp["/data"].shape[1]  # pylint: disable=no-member
            sig_duration = sig_len if data_size is None else math.ceil(data_size * (sample_rate / self.target_rate))
            # pylint: disable=no-member
            signal = fp["/data"][signal_idx, sig_start : sig_start + sig_duration].astype(np.float32)
        # END WITH
        if sample_rate != self.target_rate:
            signal = pk.signal.resample_signal(signal, sample_rate, self.target_rate)
        if data_size is None:
            return signal
        return signal[:data_size]

    def load_sleep_stages_for_subject(
        self, subject_id: str, start: int = 0, data_size: int | None = None
    ) -> npt.NDArray[np.int32]:
        """Load sleep stages for subject

        Args:
            subject_id (str): Subject ID

        Returns:
            npt.NDArray[np.int32]: Sleep stages
        """
        sample_rate = self.sampling_rate
        with h5py.File(self._get_subject_h5_path(subject_id), mode="r") as fp:
            sig_start = round(start * (sample_rate / self.target_rate))
            sig_len = fp["/sleep_stages"].shape[1]  # pylint: disable=no-member
            sig_duration = sig_len if data_size is None else math.ceil(data_size * (sample_rate / self.target_rate))
            # pylint: disable=no-member
            sleep_stages = fp["/sleep_stages"][:, sig_start : sig_start + sig_duration].astype(np.int32)
        # END WITH
        sleep_stages = np.argmax(sleep_stages, axis=0)

        sleep_stages = np.vectorize(YsywStageMap.get)(sleep_stages)
        if sample_rate != self.target_rate:
            sleep_stages = pk.signal.filter.resample_categorical(sleep_stages, sample_rate, self.target_rate)

        if data_size is None:
            return sleep_stages

        return sleep_stages[:data_size]

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset from S3

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """

        nse.utils.download_s3_objects(
            bucket="ambiqai-ysyw-2018-dataset",
            prefix="training",
            dst=self.path,
            checksum="size",
            num_workers=num_workers,
        )

    def download_raw_dataset(self, src_path: str, num_workers: int | None = None, force: bool = False):
        """Download raw dataset"""
        os.makedirs(self.path, exist_ok=True)

        # 1. Download source data
        # NOTE: Skip for now

        # 2. Extract and convert subject data to H5 files
        logger.debug("Generating YSYW subject data")

        pt_paths = list(filter(os.path.isdir, glob.glob(os.path.join(src_path, "training", "*"))))
        # pt_paths += list(filter(os.path.isdir, glob.glob(os.path.join(src_path, "test", "*"))))

        fn = functools.partial(self._convert_pt_to_hdf5, force=force)
        process_map(fn, pt_paths, max_workers=num_workers, desc="Convert YSYW subject data")
        logger.debug("Finished YSYW subject data")

    def get_subject_duration(self, subject_id: str) -> float:
        """Get subject duration in seconds"""
        with h5py.File(self._get_subject_h5_path(subject_id), mode="r") as fp:
            return fp["/data"].shape[1] / self.sampling_rate  # pylint: disable=no-member
        # END WITH

    def _convert_pt_to_hdf5(self, pt_path: str, force: bool = False):
        """Extract subject data from Physionet.

        Args:
            pt_path (str): Source path
            force (bool, optional): Whether to override destination if it exists. Defaults to False.
        """
        sleep_stage_names = [
            "nonrem1",
            "nonrem2",
            "nonrem3",
            "rem",
            "undefined",
            "wake",
        ]
        pt_id = os.path.basename(pt_path)
        pt_src_data_path = os.path.join(pt_path, f"{pt_id}.mat")
        pt_src_ann_path = os.path.join(pt_path, f"{pt_id}-arousal.mat")
        pt_dst_h5_path = os.path.join(self.path, f"{pt_id}.h5")

        if os.path.exists(pt_dst_h5_path) and not force:
            return

        data = scipy.io.loadmat(pt_src_data_path)
        atr = h5py.File(pt_src_ann_path, mode="r")
        h5 = h5py.File(pt_dst_h5_path, mode="w")

        sleep_stages = np.vstack([atr["data"]["sleep_stages"][stage][:] for stage in sleep_stage_names])
        arousals: npt.NDArray = atr["data"]["arousals"][:]
        arousals = arousals.squeeze().astype(np.int8)  # pylint: disable=no-member
        h5.create_dataset(name="/data", data=data["val"], compression="gzip", compression_opts=5)
        h5.create_dataset(name="/arousals", data=arousals, compression="gzip", compression_opts=5)
        h5.create_dataset(
            name="/sleep_stages",
            data=sleep_stages,
            compression="gzip",
            compression_opts=5,
        )
        h5.close()

    def _get_subject_h5_path(self, subject_id: str) -> Path:
        """Get subject HDF5 data path"""
        return self.path / f"{subject_id}.h5"
