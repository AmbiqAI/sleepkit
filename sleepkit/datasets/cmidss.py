import functools
import glob
import logging
import math
import os
import random
from datetime import datetime
from enum import IntEnum
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import physiokit as pk
from tqdm import tqdm

from .dataset import SKDataset
from .defines import SampleGenerator, SubjectGenerator

logger = logging.getLogger(__name__)


class CmidssSleepStage(IntEnum):
    """CMIDSS sleep stages"""

    wake = 0
    sleep = 1


class CmidssDataset(SKDataset):
    """CMIDSS dataset"""

    def __init__(
        self,
        ds_path: Path,
        frame_size: int = 30,
        target_rate: int = 1,
    ) -> None:
        super().__init__(ds_path, frame_size)
        self.frame_size = frame_size
        self.target_rate = target_rate
        self.ds_path = ds_path / "cmidss"
        self.sleep_mapping = lambda v: v

    @property
    def sampling_rate(self) -> float:
        """Sampling rate in Hz"""
        return 1.0 / 5.0

    @functools.cached_property
    def subject_ids(self) -> list[str]:
        """Get dataset subject IDs

        Returns:
            list[str]: Subject IDs
        """
        subjs = glob.glob(os.path.join(self.ds_path, "*.h5"))
        subjs = [os.path.splitext(os.path.basename(p))[0] for p in subjs]
        subjs.sort()
        return subjs

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
        return ["TS", "ENMO", "ZANGLE"]

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
        subject_idxs = list(range(len(subject_ids)))
        while True:
            if shuffle:
                random.shuffle(subject_idxs)
            for subject_idx in subject_idxs:
                subject_id = subject_ids[subject_idx]
                yield subject_id.decode("ascii") if isinstance(subject_id, bytes) else subject_id
            # END FOR
            if not repeat:
                break
        # END WHILE

    def signal_generator2(
        self, subject_generator: SubjectGenerator, signals: list[str], samples_per_subject: int = 1
    ) -> SampleGenerator:
        """Randomly generate frames of data for given subjects.

        Args:
            subject_generator (SubjectGenerator): Generator that yields subject ids.
            samples_per_subject (int): Samples per subject.

        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, num_signals)
        """
        for subject_id in subject_generator:
            max_size = int(self.target_rate * self.get_subject_duration(subject_id))

            sleep_stages = self.load_sleep_stages_for_subject(subject_id=subject_id)

            x = np.zeros((self.frame_size, len(signals)), dtype=np.float32)
            y = np.zeros((self.frame_size,), dtype=np.int32)
            for _ in range(samples_per_subject):
                frame_start = random.randint(0, max_size - 2 * self.frame_size)
                frame_end = frame_start + self.frame_size
                for i, signal_label in enumerate(signals):
                    signal_label = signal_label.decode("ascii") if isinstance(signal_label, bytes) else signal_label
                    signal = self.load_signal_for_subject(
                        subject_id, signal_label=signal_label, start=frame_start, data_size=self.frame_size
                    )
                    signal_len = min(signal.size, x.shape[0])
                    x[:signal_len, i] = signal[:signal_len]
                # END FOR
                y = sleep_stages[frame_start:frame_end]
                yield x, y
            # END FOR
        # END FOR

    def load_signal_for_subject(
        self, subject_id: str, signal_label: str, start: int = 0, data_size: int | None = None
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
        with h5py.File(self.get_subject_h5_path(subject_id), mode="r") as fp:
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
        with h5py.File(self.get_subject_h5_path(subject_id), mode="r") as fp:
            sig_start = round(start * (sample_rate / self.target_rate))
            sig_len = fp["/sleep_stages"].shape[0]  # pylint: disable=no-member
            sig_duration = sig_len if data_size is None else math.ceil(data_size * (sample_rate / self.target_rate))
            # pylint: disable=no-member
            sleep_stages = fp["/sleep_stages"][sig_start : sig_start + sig_duration].astype(np.int32)
        # END WITH

        if sample_rate != self.target_rate:
            sleep_stages = pk.signal.filter.resample_categorical(sleep_stages, sample_rate, self.target_rate)

        if data_size is None:
            return sleep_stages

        return sleep_stages[:data_size]

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        This will download preprocessed HDF5 files from S3.

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        self.download_raw_dataset(num_workers=num_workers, force=force)

    def download_raw_dataset(self, num_workers: int | None = None, force: bool = False):
        """Download raw dataset"""

        # kaggle import will raise OSError if config is not set...
        import zipfile  # pylint: disable=import-outside-toplevel

        import kaggle  # pylint: disable=import-outside-toplevel

        os.makedirs(self.ds_path, exist_ok=True)

        # 1. Download source data
        logger.info("Downloading CMIDSS dataset")
        competiton_name = "child-mind-institute-detect-sleep-states"
        kaggle.api.competition_download_files(competiton_name, path=self.ds_path, force=force, quiet=False)

        logger.info("Extracting CMIDSS dataset")
        zp_path = self.ds_path / f"{competiton_name}.zip"
        with zipfile.ZipFile(zp_path, "r") as zp:
            zp.extractall(self.ds_path)
        os.remove(zp_path)
        logger.info("CMIDSS dataset downloaded and extracted")

        # 2. Extract and convert subject data to H5 files
        logger.info("Generating CMIDSS subject data")

        df, df_lbls = self._load_raw_train_parquet()
        subject_ids: npt.NDArray[np.str_] = df.series_id.unique()

        for subject_id in tqdm(subject_ids):
            self._convert_subject_to_hdf5(
                subject_id,
                df=df,
                df_lbls=df_lbls,
            )

        logger.info("Finished YSYW subject data")

    def _load_raw_train_parquet(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw training dataset."""
        data_path = self.ds_path / "train_series.parquet"
        label_path = self.ds_path / "train_events.csv"
        df = pd.read_parquet(data_path)
        df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z")
        df_lbls = pd.read_csv(label_path)
        df_lbls["datetime"] = pd.to_datetime(df_lbls["timestamp"], format="%Y-%m-%dT%H:%M:%S%z")
        return df, df_lbls

    def _convert_subject_to_hdf5(self, subject_id: str, df: pd.DataFrame, df_lbls: pd.DataFrame, force: bool = False):
        """Extract subject data from parquet.

        Args:
            subject_id (str): Subject id
            df (pd.DataFrame): Raw data
            df_lbls (pd.DataFrame): Raw labels
            force (bool, optional): Whether to override destination if it exists. Defaults to False.
        """

        sub_h5_path = os.path.join(self.ds_path, f"{subject_id}.h5")
        if os.path.exists(sub_h5_path) and not force:
            return

        # Extract subject data and labels
        sub_df = df[df.series_id == subject_id]
        sub_lbls = df_lbls[df_lbls.series_id == subject_id]

        # Data
        date: list[datetime] = sub_df.datetime.to_list()
        step = sub_df.step.to_numpy()
        duration = step.size
        data = np.zeros((3, duration), dtype=np.float32)
        data[0] = [(td.hour * 3600 + td.minute * 60 + td.second) for td in date]
        data[1] = sub_df.enmo.to_numpy()
        data[2] = sub_df.anglez.to_numpy()

        # Labels
        labels = np.zeros((duration), dtype=np.int32)
        onsets = sub_lbls[sub_lbls.event == "onset"].step.values
        wakeups = sub_lbls[sub_lbls.event == "wakeup"].step.values
        for onset, wakeup in zip(onsets, wakeups):
            if np.isnan(onset) or np.isnan(wakeup):
                continue
            labels[int(onset) : int(wakeup)] = CmidssSleepStage.sleep.value
        # END FOR

        with h5py.File(sub_h5_path, mode="w") as h5:
            h5.create_dataset(name="/data", data=data, compression="gzip", compression_opts=5)
            h5.create_dataset(name="/sleep_stages", data=labels, compression="gzip", compression_opts=5)
        # END WITH

    def get_subject_h5_path(self, subject_id: str) -> Path:
        """Get subject HDF5 data path"""
        return self.ds_path / f"{subject_id}.h5"

    def get_subject_duration(self, subject_id: str) -> float:
        """Get subject duration in seconds"""
        with h5py.File(self.get_subject_h5_path(subject_id), mode="r") as fp:
            return fp["/data"].shape[1] / self.sampling_rate  # pylint: disable=no-member
        # END WITH
