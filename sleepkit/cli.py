import os
from typing import Type, TypeVar

import pydantic_argparse
from pydantic import BaseModel, Field

from . import sleepstage, apnea
from .defines import (
    SKMode, SKTask, SKTrainParams, SKFeatureParams
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

    # Download dataset(s)
    if args.mode == SKMode.download:
        # download_datasets(parse_content(SKDownloadParams, args.config))
        return

    # Generate feature set
    if args.mode == SKMode.feature:
        generate_feature_set(parse_content(SKFeatureParams, args.config))
        return

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
            raise NotImplementedError()
            # task_handler.evaluate_model(parse_content(SKTestParams, args.config))

        case SKMode.export:
            raise NotImplementedError()
            # task_handler.export_model(parse_content(SKExportParams, args.config))

        case SKMode.demo:
            raise NotImplementedError()
            # demo(params=parse_content(SKDemoParams, args.config))

        case SKMode.predict:
            raise NotImplementedError()

        case _:
            logger.error("Error: Unknown command")

    # END MATCH

    logger.info(f"#FINISHED {args.mode} model")


if __name__ == "__main__":
    run()
