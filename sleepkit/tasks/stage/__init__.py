"""
# :sleeping: Sleep Stage Task API

The `StageTask` class provides the API for the sleep stage classification task.

Classes:
    StageTask: Sleep stage classification task

"""

from ..task import Task, TaskParams
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class StageTask(Task):
    """Sleep Stage Task"""

    @staticmethod
    def train(params: TaskParams) -> None:
        """Train sleep stage model.

        Args:
            params (TaskParams): Task parameters

        """
        train(params)

    @staticmethod
    def evaluate(params: TaskParams) -> None:
        """Evaluate sleep stage model.

        Args:
            params (TaskParams): Task parameters

        """
        evaluate(params)

    @staticmethod
    def export(params: TaskParams) -> None:
        """Export sleep stage model.

        Args:
            params (TaskParams): Task parameters

        """
        export(params)

    @staticmethod
    def demo(params: TaskParams) -> None:
        """Run sleep stage demo.

        Args:
            params (TaskParams): Task parameters
        """
        demo(params)
