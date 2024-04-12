from ..task import SKTask
from . import utils
from .defines import SleepApnea
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class ApneaTask(SKTask):
    """Sleep Apnea Task"""

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
