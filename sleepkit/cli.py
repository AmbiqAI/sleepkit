"""
# :octicons-terminal-24: SleepKit CLI API

The SleepKit CLI provides a command-line interface to interact with the SleepKit library.

```bash
$ sleepkit --help

SleepKit CLI Options:
    --task [detect, stage, apnea]
    --mode [download, feature, train, evaluate, export, demo]
    --config ["./path/to/config.json", or '{"raw: "json"}']
```

"""

import os
from typing import Type, TypeVar

from argdantic import ArgField, ArgParser
from pydantic import BaseModel
import neuralspot_edge as nse

from .defines import TaskParams, TaskMode

from .tasks import TaskFactory

logger = nse.utils.setup_logger(__name__)

B = TypeVar("B", bound=BaseModel)

parser = ArgParser()


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


@parser.command()
def run(
    mode: TaskMode = ArgField("-m", description="Mode", default="train"),
    task: str = ArgField("-t", description="Task", default="detect"),
    config: str = ArgField("-c", description="File path or JSON content", default="{}"),
):
    """SleepKit CLI"""

    logger.info(f"#STARTED MODE={mode} TASK={task}")

    if not TaskFactory.has(task):
        raise ValueError(f"Unknown task {task}")

    task_handler = TaskFactory.get(task)

    params = parse_content(TaskParams, config)

    match mode:
        case TaskMode.download:
            task_handler.download(params)

        case TaskMode.feature:
            task_handler.feature(params)

        case TaskMode.train:
            task_handler.train(params)

        case TaskMode.evaluate:
            task_handler.evaluate(params)

        case TaskMode.export:
            task_handler.export(params)

        case TaskMode.demo:
            task_handler.demo(params)

        case _:
            logger.error("Error: Unsupported CLI command")
    # END MATCH

    logger.info(f"#FINISHED MODE={mode} TASK={task}")


def main():
    parser()


if __name__ == "__main__":
    main()
