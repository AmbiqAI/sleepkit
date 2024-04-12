import logging
import os

from ..defines import SKDownloadParams
from ..utils import setup_logger
from .factory import DatasetFactory

logger = setup_logger(__name__)


def download_datasets(params: SKDownloadParams):
    """Download specified datasets.

    Args:
        params (HeartDownloadParams): Download parameters

    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "download.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    os.makedirs(params.ds_path, exist_ok=True)

    for ds_name in params.datasets:
        if DatasetFactory.has(ds_name):
            Dataset = DatasetFactory.get(ds_name)
            ds = Dataset(ds_path=params.ds_path, frame_size=1)
            ds.download(
                num_workers=params.data_parallelism,
                force=params.force,
            )
        else:
            logger.warning(f"Dataset {ds_name} not found in DatasetFactory")
        # END IF
    # END FOR
