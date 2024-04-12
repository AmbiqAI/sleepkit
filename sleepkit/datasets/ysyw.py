import functools
import glob
import logging
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import IntEnum
from multiprocessing import Pool
from pathlib import Path

import boto3
import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
import scipy.io
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

from ..tasks import SleepStage
from .dataset import SKDataset
from .defines import SampleGenerator, SubjectGenerator

logger = logging.getLogger(__name__)


class YsywSleepStage(IntEnum):
    """YSYW sleep stages"""

    nonrem1 = 0  # N1
    nonrem2 = 1  # N2
    nonrem3 = 2  # N3/4
    rem = 3
    undefined = 4
    wake = 5


class YsywDataset(SKDataset):
    """YSYW dataset"""

    def __init__(
        self,
        ds_path: Path,
        frame_size: int = 30 * 128,
        target_rate: int = 128,
    ) -> None:
        super().__init__(ds_path=ds_path, frame_size=frame_size)
        self.frame_size = frame_size
        self.target_rate = target_rate
        self.ds_path = ds_path / "ysyw"

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
        pts = glob.glob(os.path.join(self.ds_path, "*.h5"))
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
        """Randomly generate frames of sleep data for given subjects.

        Args:
            subject_generator (SubjectGenerator): Generator that yields subject ids.
            samples_per_subject (int): Samples per subject.

        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, num_signals)
        """
        for subject_id in subject_generator:
            max_size = int(self.target_rate * self.get_subject_duration(subject_id))

            sleep_mask = self.load_sleep_stages_for_subject(subject_id=subject_id)

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
                y = sleep_mask[frame_start:frame_end]
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

        stage_label_map = lambda v: {
            YsywSleepStage.wake: SleepStage.wake,
            YsywSleepStage.nonrem1: SleepStage.stage1,
            YsywSleepStage.nonrem2: SleepStage.stage2,
            YsywSleepStage.nonrem3: SleepStage.stage3,
            YsywSleepStage.rem: SleepStage.rem,
            YsywSleepStage.undefined: SleepStage.wake,
        }.get(v, 0)

        sleep_stages = np.vectorize(stage_label_map)(sleep_stages)
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

        def download_s3_file(
            s3_file: str,
            save_path: str,
            bucket: str,
            client: boto3.client,
            force: bool = False,
        ):
            if not force and os.path.exists(save_path):
                return
            client.download_file(
                Bucket=bucket,
                Key=s3_file,
                Filename=save_path,
            )

        s3_bucket = "ambiqai-ysyw-2018-dataset"
        s3_prefix = "training"

        os.makedirs(self.ds_path, exist_ok=True)

        # Creating only one session and one client
        session = boto3.Session()
        client = session.client("s3", config=Config(signature_version=UNSIGNED))

        rst = client.list_objects(Bucket=s3_bucket, Prefix=s3_prefix, MaxKeys=1000)
        pt_s3_paths = list(filter(lambda obj: obj.endswith("h5"), (obj["Key"] for obj in rst["Contents"])))

        func = functools.partial(download_s3_file, bucket=s3_bucket, client=client, force=force)

        with tqdm(desc="Downloading YSYW dataset from S3", total=len(pt_s3_paths)) as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = (
                    executor.submit(
                        func,
                        pt_s3_path,
                        os.path.join(self.ds_path, os.path.basename(pt_s3_path)),
                    )
                    for pt_s3_path in pt_s3_paths
                )
                for future in as_completed(futures):
                    err = future.exception()
                    if err:
                        logger.error(f"Failed on file {err}")
                    pbar.update(1)
                # END FOR
            # END WITH
        # END WITH

    def download_raw_dataset(self, src_path: str, num_workers: int | None = None, force: bool = False):
        """Download raw dataset"""
        os.makedirs(self.ds_path, exist_ok=True)

        # 1. Download source data
        # NOTE: Skip for now

        # 2. Extract and convert subject data to H5 files
        logger.info("Generating YSYW subject data")

        pt_paths = list(filter(os.path.isdir, glob.glob(os.path.join(src_path, "training", "*"))))
        # pt_paths += list(filter(os.path.isdir, glob.glob(os.path.join(src_path, "test", "*"))))

        f = functools.partial(self._convert_pt_to_hdf5, force=force)
        with Pool(processes=num_workers) as pool:
            _ = list(tqdm(pool.imap(f, pt_paths), total=len(pt_paths)))

        logger.info("Finished YSYW subject data")

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
        sleep_stage_names = ["nonrem1", "nonrem2", "nonrem3", "rem", "undefined", "wake"]
        pt_id = os.path.basename(pt_path)
        pt_src_data_path = os.path.join(pt_path, f"{pt_id}.mat")
        pt_src_ann_path = os.path.join(pt_path, f"{pt_id}-arousal.mat")
        pt_dst_h5_path = os.path.join(self.ds_path, f"{pt_id}.h5")

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
        h5.create_dataset(name="/sleep_stages", data=sleep_stages, compression="gzip", compression_opts=5)
        h5.close()

    def _get_subject_h5_path(self, subject_id: str) -> Path:
        """Get subject HDF5 data path"""
        return self.ds_path / f"{subject_id}.h5"
