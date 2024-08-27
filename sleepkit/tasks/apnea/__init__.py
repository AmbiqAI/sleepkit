"""
# :nose: Sleep Apnea Task API

The `ApneaTask` class provides the API for the sleep apnea detection task.

Classes:
    ApneaTask: Sleep apnea detection task

"""

from ...defines import TaskParams
from ..task import Task
from . import utils
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class ApneaTask(Task):
    """Sleep Apnea Task"""

    @staticmethod
    def train(params: TaskParams) -> None:
        """Train sleep apnea model.

        Args:
            params (TaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: TaskParams) -> None:
        """Evaluate sleep apnea model.

        Args:
            params (TaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: TaskParams) -> None:
        """Export sleep apnea model.

        Args:
            params (TaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: TaskParams) -> None:
        """Run sleep apnea demo.

        Args:
            params (TaskParams): Task parameters
        """
        demo(params)
