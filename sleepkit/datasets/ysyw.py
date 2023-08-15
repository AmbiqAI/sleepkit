import os
import glob
import logging
import functools
from enum import IntEnum
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

import boto3
import h5py
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import scipy.io
from botocore import UNSIGNED
from botocore.client import Config


logger = logging.getLogger(__name__)

class YsywSleepStage(IntEnum):
    nonrem1 = 0
    nonrem2 = 1
    nonrem3 = 2
    rem = 3
    undefined = 4
    wake = 5

signal_names = [
    # ECG
    'F3-M2',
    'F4-M1',
    'C3-M2',
    'C4-M1',
    'O1-M2',
    'O2-M1',
    # EOG
    'E1-M2',
    # EMG
    'Chin1-Chin2',
    # RSP
    'ABD',
    'CHEST',
    #
    'AIRFLOW',
    # SPO2
    'SaO2',
    # ECG
    'ECG'
]

# Compute Resiratory rate from ABD/CHEST or from ECG

class YsywDataset:

    def __init__(self, ds_path: str) -> None:
        self.ds_path = os.path.join(ds_path, "ysyw")

    @property
    def patient_ids(self) -> list[str]:
        """Get dataset patient IDs

        Returns:
            list[str]: Patient IDs
        """
        pts = glob.glob(os.path.join(self.ds_path, "*.h5"))
        pts = [os.path.splitext(os.path.basename(p))[0] for p in pts]
        pts.sort()
        return pts


    @property
    def signal_names(self) -> list[str]:
        return [
            # ECG
            'F3-M2',
            'F4-M1',
            'C3-M2',
            'C4-M1',
            'O1-M2',
            'O2-M1',
            # EOG
            'E1-M2',
            # EMG
            'Chin1-Chin2',
            # RSP
            'ABD',
            'CHEST',
            #
            'AIRFLOW',
            # SPO2
            'SaO2',
            # ECG
            'ECG'
        ]

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

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
        pt_s3_paths = list(filter(lambda obj: obj.endswith('h5'), (obj['Key'] for obj in rst['Contents'])))

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
                        print("Failed on file", err)
                    pbar.update(1)
                # END FOR
            # END WITH
        # END WITH

    def download_raw_dataset(self, src_path: str, num_workers: int | None = None, force: bool = False):

        os.makedirs(self.ds_path, exist_ok=True)

        # 1. Download source data
        # NOTE: Skip for now

        # 2. Extract and convert patient data to H5 files
        logger.info("Generating YSYW patient data")

        pt_paths = list(filter(os.path.isdir, glob.glob(os.path.join(src_path, "training", "*"))))
        # pt_paths += list(filter(os.path.isdir, glob.glob(os.path.join(src_path, "test", "*"))))

        f = functools.partial(self._convert_pt_to_hdf5, force=force)
        with Pool(processes=num_workers) as pool:
            _ = list(tqdm(pool.imap(f, pt_paths), total=len(pt_paths)))

        logger.info("Finished YSYW patient data")

    def _convert_pt_to_hdf5(
        self, pt_path: str, force: bool = False
    ):
        """Extract patient data from Physionet.

        Args:
            pt_path (str): Source path
            force (bool, optional): Whether to override destination if it exists. Defaults to False.
        """
        sleep_stage_names = ['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']
        pt_id = os.path.basename(pt_path)
        pt_src_data_path = os.path.join(pt_path, f"{pt_id}.mat")
        pt_src_ann_path = os.path.join(pt_path, f"{pt_id}-arousal.mat")
        pt_dst_h5_path = os.path.join(self.ds_path, f"{pt_id}.h5")

        if os.path.exists(pt_dst_h5_path) and not force:
            return

        data = scipy.io.loadmat(pt_src_data_path)
        atr = h5py.File(pt_src_ann_path, mode='r')
        h5 = h5py.File(pt_dst_h5_path, mode='w')

        sleep_stages = np.vstack([atr['data']['sleep_stages'][stage][:] for stage in sleep_stage_names])
        arousals = atr['data']['arousals'][:].squeeze().astype(np.int8)
        h5.create_dataset(name='/data', data=data['val'], compression="gzip", compression_opts=5)
        h5.create_dataset(name='/arousals', data=arousals, compression="gzip", compression_opts=5)
        h5.create_dataset(name='/sleep_stages', data=sleep_stages, compression="gzip", compression_opts=5)
        h5.close()
