import os
import shutil
import subprocess

from ..defines import SKDownloadParams
from ..utils import setup_logger

logger = setup_logger(__name__)


def download_datasets(params: SKDownloadParams):
    """Download datasets"""


def download_mesa(self, args: SKDownloadParams):
    """Download MESA dataset"""
    is_commercial = True
    token = os.environ.get("NSSR_TOKEN")
    if token is None:
        raise ValueError("NSSR_TOKEN is not set")

    if shutil.which("nssr") is None:
        raise ValueError("nssr is not installed or not in PATH")

    logger.info(f"Downloading MESA dataset to {args.ds_path}")

    os.makedirs(args.ds_path, exist_ok=True)

    subprocess.run(
        [
            "nssr",
            "d",
            "mesa--commercial-use" if is_commercial else "mesa",
            f"--token={os.environ.get('NSSR_TOKEN')}",
        ],
        cwd=args.ds_path,
    )
