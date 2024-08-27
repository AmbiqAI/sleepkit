"""
# NSRR python API for downloading datasets from [sleepdata.org](https://sleepdata.org).

Adapted from [SleepECG](https://github.com/cbrnr/sleepecg/blob/main/sleepecg/io/utils.py)
"""

import os
from fnmatch import fnmatch
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Literal
import requests

from tqdm.contrib.concurrent import thread_map

import neuralspot_edge as nse

_nsrr_token = None


def authenticate_nsrr(token: str | None = None) -> str:
    """Authenticate the [NSRR](https://sleepdata.org) download token.

    Implemented according to [NSRR API specs](https://github.com/nsrr/sleepdata.org/wiki/api-v1-account).

    Args:
        token (str|None): NSRR [download token](https://sleepdata.org/token). Defaults to None.

    Returns:
        str: The verified download token.
    """
    global _nsrr_token  # pylint: disable=global-statement

    # If already authenticated, return the token
    if token is None and _nsrr_token:
        return _nsrr_token

    # If token is not provided, check the environment variable
    if token is None:
        token = os.environ.get("NSRR_TOKEN", "")

    # Attempt to authenticate
    response = requests.get(
        "https://sleepdata.org/api/v1/account/profile.json",
        params={"auth_token": token},
        timeout=30,
    )
    authenticated = response.json()["authenticated"]
    if authenticated:
        username = response.json()["username"]
        email = response.json()["email"]
        print(f"Authenticated at sleepdata.org as {username} ({email})")

        _nsrr_token = token
        return _nsrr_token

    raise RuntimeError("Authentication at sleepdata.org failed, verify token!")


def _get_nsrr_url(db_slug: str) -> str:
    """Get the download URL for a given NSRR database.

    The download token is a part of the URL, so it needs to be already set.

    Args:
        db_slug (str): Short identifier of a database, e.g. `'mesa'`.

    Returns:
        str: The download URL.
    """
    token = authenticate_nsrr(None)
    return f"https://sleepdata.org/datasets/{db_slug}/files/a/{token}/m/sleepkit/"


def list_nsrr_items(
    db_slug: str,
    subfolder: str = "",
    pattern: str = "*",
    shallow: bool = False,
) -> list[dict]:
    """Recursively list items for a dataset to be downloaded.

    Specify a subfolder and/or a filename-pattern to filter results.

    Implemented according to the NSRR API documentation:
    https://github.com/nsrr/sleepdata.org/wiki/api-v1-datasets#list-files-in-folder

    Args:
        db_slug (str): Short identifier of a database, e.g. `'mesa'`.
        subfolder (str, optional): The folder at which to start the search. Defaults to `''`.
        pattern (str, optional): Glob-like pattern applied to basename. Defaults to `'*'`.
        shallow (bool, optional): If `True` no recursion is performed. Defaults to `False`.

    Returns:
        list[dict]: List of dictionary items to be downloaded.
    """
    api_url = f"https://sleepdata.org/api/v1/datasets/{db_slug}/files.json"

    response = requests.get(api_url, params={"path": subfolder}, timeout=30)
    try:
        response_json = response.json()
    except JSONDecodeError:
        raise RuntimeError(f"No API response for dataset {db_slug}.") from None

    files = []
    for item in response_json:
        if not item["is_file"] and not shallow:
            files.extend(list_nsrr_items(db_slug, item["full_path"], pattern))
        elif fnmatch(item["file_name"], pattern):
            files.append(item)
    return files


def download_nsrr_file(url: str, dst: Path, checksum: str, checksum_type: Literal["md5", "size"] = "size") -> None:
    """Download a file from `url` to `dst` and verify `checksum`.

    This is a wrapper to provide a helpful error message in case the currently set token
    does not grant access to the requested file.

    Args:
        url (str): URL to download from.
        dst (Path): Location where the downloaded file will be stored.
        checksum (str): Checksum to verify the file against.
        checksum_type (Literal["md5", "size"], optional): Checksum type. Defaults to `'size'`.
    """

    try:
        nse.utils.download_file(
            src=url,
            dst=dst,
            checksum=checksum,
            checksum_type=checksum_type,
            progress=False,
        )
    except RuntimeError as error:
        # If the token is invalid for the requested dataset, the request is redirected to a
        # files overview page. The response is an HTML-page which doesn't have a
        # "content-disposition" header.
        response = requests.get(url, stream=True, timeout=30)
        if "content-disposition" not in response.headers:
            db_slug = url.split("/")[4]
            raise RuntimeError(f"Make sure you have access to {db_slug}!") from error
        raise error


def download_nsrr(
    db_slug: str,
    subfolder: str = "",
    pattern: str = "*",
    shallow: bool = False,
    data_dir: str | Path = ".",
    checksum_type: Literal["md5", "size"] = "size",
    num_workers: int | None = None,
) -> None:
    """Recursively download files from [NSRR](https://sleepdata.org).

    Specify a subfolder and/or a filename pattern to filter results. Implemented according
    to the [NSRR API specs](https://github.com/nsrr/sleepdata.org/wiki/api-v1-datasets).

    Args:
        db_slug (str): Short identifier of a database, e.g. `'mesa'`.
        subfolder (str, optional): The folder at which to start the search. Defaults to `''`.
        pattern (str, optional): Glob-like pattern applied to basename. Defaults to `'*'`.
        shallow (bool, optional): If `True` no recursion is performed. Defaults to `False`.
        data_dir (str | Path, optional): Root directory to save the dataset folder into. Defaults to `'.'`.
        checksum_type (Literal["md5", "size"], optional): Checksum type. Defaults to `'size'`.
        num_workers (int | None, optional): Number of parallel download workers. Defaults to `None`.

    """
    db_dir = Path(data_dir) / db_slug

    download_url = _get_nsrr_url(db_slug)
    files_to_download = list_nsrr_items(db_slug, subfolder, pattern, shallow)

    checksum_key = "file_checksum_md5" if checksum_type == "md5" else "file_size"

    # Download files in parallel
    thread_map(
        lambda item: download_nsrr_file(
            url=download_url + item["full_path"],
            dst=db_dir / item["full_path"],
            checksum=str(item[checksum_key]),
            checksum_type=checksum_type,
        ),
        files_to_download,
        max_workers=num_workers,
    )
