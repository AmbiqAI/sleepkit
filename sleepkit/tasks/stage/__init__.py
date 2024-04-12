from ..task import SKTask
from . import defines, utils
from .defines import SleepStage
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class StageTask(SKTask):
    """Sleep Stage Task"""

    @staticmethod
    def train(params) -> None:
        train(params)

    @staticmethod
    def evaluate(params) -> None:
        evaluate(params)

    @staticmethod
    def export(params) -> None:
        export(params)

    @staticmethod
    def demo(params) -> None:
        demo(params)
