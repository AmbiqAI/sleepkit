import os
from typing import Type, TypeVar

import pydantic_argparse
from pydantic import BaseModel, Field

from . import apnea, stage
from .datasets import download_datasets
from .defines import (
    SKDemoParams,
    SKDownloadParams,
    SKExportParams,
    SKFeatureParams,
    SKMode,
    SKTask,
    SKTestParams,
    SKTrainParams,
)
from .features.factory import generate_feature_set
from .utils import setup_logger

logger = setup_logger(__name__)


class CliArgs(BaseModel):
    """CLI arguments"""

    task: SKTask = Field(default=SKTask.detect)
    mode: SKMode = Field(default=SKMode.train)
    config: str = Field(description="JSON config file path or string")


B = TypeVar("B", bound=BaseModel)


def parse_content(cls: Type[B], content: str) -> B:
    """Parse file or raw content into Pydantic model.

    Args:
        cls (B): Pydantic model subclasss
        content (str): File path or raw content

    Returns:
        B: Pydantic model subclass instance
    """
    return cls.parse_file(content) if os.path.isfile(content) else cls.parse_raw(content)


def run(inputs: list[str] | None = None):
    """Main CLI app runner

    Args:
        inputs (list[str] | None, optional): App arguments. Defaults to CLI arguments.
    """
    parser = pydantic_argparse.ArgumentParser(
        model=CliArgs,
        prog="SleepKit CLI",
        description="SleepKit leverages AI for sleep monitoring tasks.",
    )
    args = parser.parse_typed_args(inputs)

    # Download datasets
    if args.mode == SKMode.download:
        logger.info("#STARTED download")
        download_datasets(parse_content(SKDownloadParams, args.config))
        logger.info("#FINISHED download")
        return

    # Generate feature set
    if args.mode == SKMode.feature:
        logger.info("#STARTED feature")
        generate_feature_set(parse_content(SKFeatureParams, args.config))
        logger.info("#FINISHED feature")
        return

    # Grab task handler
    match args.task:
        case SKTask.stage:
            task_handler = stage
        case SKTask.apnea:
            task_handler = apnea
        case _:
            raise NotImplementedError()
    # END MATCH
    logger.info(f"#STARTED {args.mode} for task {args.task}")

    match args.mode:
        case SKMode.train:
            task_handler.train(parse_content(SKTrainParams, args.config))

        case SKMode.evaluate:
            task_handler.evaluate(parse_content(SKTestParams, args.config))

        case SKMode.export:
            task_handler.export(parse_content(SKExportParams, args.config))

        case SKMode.demo:
            task_handler.demo(parse_content(SKDemoParams, args.config))

        case _:
            logger.error("Error: Unsupported CLI command")
    # END MATCH

    logger.info(f"#FINISHED {args.mode} for task {args.task}")


if __name__ == "__main__":
    run()
