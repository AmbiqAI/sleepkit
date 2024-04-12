import os
from typing import Type, TypeVar

from argdantic import ArgField, ArgParser
from pydantic import BaseModel

from .datasets import download_datasets
from .defines import (
    SKDemoParams,
    SKDownloadParams,
    SKExportParams,
    SKFeatureParams,
    SKMode,
    SKTestParams,
    SKTrainParams,
)
from .features import generate_feature_set
from .tasks import TaskFactory
from .utils import setup_logger

logger = setup_logger(__name__)

cli = ArgParser()

B = TypeVar("B", bound=BaseModel)


def parse_content(cls: Type[B], content: str) -> B:
    """Parse file or raw content into Pydantic model.

    Args:
        cls (B): Pydantic model subclasss
        content (str): File path or raw content

    Returns:
        B: Pydantic model subclass instance
    """
    if os.path.isfile(content):
        with open(content, "r", encoding="utf-8") as f:
            content = f.read()
    return cls.model_validate_json(json_data=content)


@cli.command(name="run")
def _run(
    mode: SKMode = ArgField("-m", description="Mode", default="train"),
    task: str = ArgField("-t", description="Task", default=""),
    config: str = ArgField("-c", description="File path or JSON content", default="{}"),
):
    """ "SleepKit CLI"""

    # Download datasets
    if mode == SKMode.download:
        logger.info("#STARTED MODE=download")
        download_datasets(parse_content(SKDownloadParams, config))
        logger.info("#FINISHED MODE=download")
        return

    # Generate feature set
    if mode == SKMode.feature:
        logger.info("#STARTED MODE=feature")
        generate_feature_set(parse_content(SKFeatureParams, config))
        logger.info("#FINISHED MODE=feature")
        return

    if not TaskFactory.has(task):
        raise ValueError(f"Unknown task {task}")

    logger.info(f"#STARTED MODE={mode} TASK={task}")

    task_handler = TaskFactory.get(task)

    match mode:
        case SKMode.train:
            task_handler.train(parse_content(SKTrainParams, config))

        case SKMode.evaluate:
            task_handler.evaluate(parse_content(SKTestParams, config))

        case SKMode.export:
            task_handler.export(parse_content(SKExportParams, config))

        case SKMode.demo:
            task_handler.demo(parse_content(SKDemoParams, config))

        case _:
            logger.error("Error: Unsupported CLI command")
    # END MATCH

    logger.info(f"#FINISHED MODE={mode} TASK={task}")


def run():
    """Run CLI."""
    cli()


if __name__ == "__main__":
    run()
