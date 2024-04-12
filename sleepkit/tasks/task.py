import abc

from ..defines import SKDemoParams, SKExportParams, SKTestParams, SKTrainParams


class SKTask(abc.ABC):
    """HSleepKit Task base class. All tasks should inherit from this class."""

    @staticmethod
    def train(params: SKTrainParams) -> None:
        """Train a model

        Args:
            params (SKTrainParams): train parameters

        """
        raise NotImplementedError

    @staticmethod
    def evaluate(params: SKTestParams) -> None:
        """Evaluate a model

        Args:
            params (SKTestParams): test parameters

        """
        raise NotImplementedError

    @staticmethod
    def export(params: SKExportParams) -> None:
        """Export a model

        Args:
            params (SKExportParams): export parameters

        """
        raise NotImplementedError

    @staticmethod
    def demo(params: SKDemoParams) -> None:
        """Run a demo

        Args:
            params (SKDemoParams): demo parameters

        """
        raise NotImplementedError
