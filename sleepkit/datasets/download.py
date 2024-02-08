import os
import shutil
import subprocess
from pathlib import Path

from ..defines import SKDownloadParams
from ..utils import setup_logger

logger = setup_logger(__name__)


def download_datasets(params: SKDownloadParams):
    """Download datasets"""
    if "cmidss" in params.datasets:
        download_cmidss(params)

    if  "mesa" in params.datasets:
        download_mesa(params)

    if "stages" in params.datasets:
        download_stages(params)

def download_nssr_dataset(dataset: str, save_path: Path):
    """Download dataset from NSSR using nssr CLI tool"""
    token = os.environ.get("NSSR_TOKEN")
    if token is None:
        raise ValueError("NSSR_TOKEN is not set")

    if shutil.which("nssr") is None:
        raise ValueError("nssr is not installed or not in $PATH")

    logger.info(f"Downloading {dataset} dataset to {save_path}")

    os.makedirs(save_path, exist_ok=True)

    subprocess.run(
        [
            "nssr",
            "d",
            dataset,
            f"--token={os.environ.get('NSSR_TOKEN')}",
        ],
        cwd=save_path.parent,
        check=False,
    )

def download_mesa(args: SKDownloadParams):
    """Download MESA dataset"""
    is_commercial = True
    dataset = "mesa-commercial-use" if is_commercial else "mesa"
    download_nssr_dataset(dataset, args.ds_path)

def download_stages(args: SKDownloadParams):
    """Download STAGES dataset from NSSR"""
    download_nssr_dataset("stages", args.ds_path)

def download_cmidss(args: SKDownloadParams):
    logger.info((
        "Please refer to the CMIDSS dataset website for download instructions.\n"
        "Once downloaded, please place the dataset in the datasets folder:\n"
        f"{args.ds_path.absolute()}{os.path.sep}cmidss"
    ))
