from pathlib import Path
import logging

import boto3
from argdantic import ArgParser, ArgField

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s: %(message)s")
logger = logging.getLogger("sk.scripts")

parser = ArgParser()


@parser.command()
def upload_to_model_zoo(
    src: Path = ArgField("-s", description="Model path"),
    name: str | None = ArgField("-n", description="Model name", default=None),
    task: str = ArgField("-t", description="Task", default="rhythm"),
    version: str = ArgField("-v", description="Version", default="latest"),
    adk: str = ArgField(description="ADK", default="sleepkit"),
    bucket: str = ArgField("-b", description="Bucket", default="ambiqai-model-zoo"),
    assets: tuple[str, ...] = ArgField("-a", description="Assets", default=()),
    dryrun: bool = ArgField("-d", description="Dry run", default=False),
) -> int:
    """Upload model assets to model zoo on S3

    Args:
        src (Path): Model path
        name (str): Model name
        task (str, optional): Task. Defaults to 'rhythm'.
        version (str, optional): Version. Defaults to 'latest'.
        adk (str, optional): ADK. Defaults to 'sleepkit'.
        bucket (str, optional): Bucket. Defaults to 'ambiqai-model-zoo'.
        assets (tuple[str,...], optional): Assets. Defaults to ().
        dryrun (bool, optional): Dry run. Defaults to False.

    Examples:
        ```bash
        python scripts/upload_to_model_zoo.py \
            --dryrun \
            -s ./results/model \
            -n model_name \
            -t rhythm \
            -v latest \
            -a configuration.json \
            -a model.keras \
            -a model.tflite \
            -a metrics.json \
            -a history.csv \
        ```
    """
    if not assets:
        assets = ("configuration.json", "model.keras", "model.tflite", "metrics.json", "history.csv")

    if not src.exists():
        logger.error(f"Model path {src} not found")
        return -1

    if not name:
        name = src.name

    # Create an S3 client
    s3 = boto3.client("s3")

    # Verify all assets exist
    for asset in assets:
        file_path = src / asset
        if not file_path.exists():
            logger.error(f"Asset {file_path} not found")
            return -1
        # END IF

    dst_prefix = f"{adk}/{task}/{name}/{version}"
    # Upload all assets
    for asset in assets:
        file_path = src / asset
        dst_key = f"{dst_prefix}/{asset}"
        logger.info(f"Uploading s3://{bucket}/{dst_key}")
        if not dryrun:
            s3.upload_file(str(file_path), bucket, dst_key)
    # END FOR
    return 0


if __name__ == "__main__":
    parser()

"""
# Apnea models
python ./scripts/upload_to_model_zoo.py -t apnea -s ./results/sa-2-tcn-sm -v v1.0

# Detect models
python ./scripts/upload_to_model_zoo.py -t detect -s ./results/sd-2-tcn-sm -v v1.0

# Stage models
python ./scripts/upload_to_model_zoo.py -t stage -s ./results/ss-2-tcn-sm -v v1.0
python ./scripts/upload_to_model_zoo.py -t stage -s ./results/ss-3-tcn-sm -v v1.0
python ./scripts/upload_to_model_zoo.py -t stage -s ./results/ss-4-tcn-sm -v v1.0
python ./scripts/upload_to_model_zoo.py -t stage -s ./results/ss-5-tcn-sm -v v1.0

"""
