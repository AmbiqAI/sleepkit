import os
from typing import Type, TypeVar

import pydantic_argparse
from pydantic import BaseModel, Field

from . import apnea, sleepstage
from .datasets import download_datasets
from .defines import (
    SKDownloadParams,
    SKExportParams,
    SKFeatureParams,
    SKMode,
    SKTask,
    SKTestParams,
    SKTrainParams,
)
from .features import generate_feature_set
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

    logger.info(f"#STARTED {args.mode} model")

    # Download datasets
    if args.mode == SKMode.download:
        download_datasets(parse_content(SKDownloadParams, args.config))
        return

    # Generate feature set
    if args.mode == SKMode.feature:
        generate_feature_set(parse_content(SKFeatureParams, args.config))
        return

    # Grab task handler
    match args.task:
        case SKTask.stage:
            task_handler = sleepstage
        case SKTask.apnea:
            task_handler = apnea
        case _:
            raise NotImplementedError()
    # END MATCH

    match args.mode:
        case SKMode.train:
            task_handler.train(parse_content(SKTrainParams, args.config))

        case SKMode.evaluate:
            task_handler.evaluate(parse_content(SKTestParams, args.config))

        case SKMode.export:
            task_handler.export(parse_content(SKExportParams, args.config))

        case SKMode.demo:
            raise NotImplementedError()
            # demo(params=parse_content(SKDemoParams, args.config))

        case _:
            logger.error("Error: Unsupported CLI command")
    # END MATCH

    logger.info(f"#FINISHED {args.mode} model")


if __name__ == "__main__":
    run()
