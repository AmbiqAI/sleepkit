import abc

import neuralspot_edge as nse
import numpy.typing as npt

from ..defines import TaskParams

logger = nse.utils.setup_logger(__name__)


class InferenceBackend(abc.ABC):
    def __init__(self, params: TaskParams) -> None:
        """Backend inference engine base class

        Args:
            params (TaskParams): Task parameters
        """
        self.params = params

    def open(self):
        """Open backend"""
        raise NotImplementedError

    def close(self):
        """Close backend"""
        raise NotImplementedError

    def set_inputs(self, inputs: npt.NDArray):
        """Set inputs"""
        raise NotImplementedError

    def perform_inference(self):
        """Perform inference"""
        raise NotImplementedError

    def get_outputs(self) -> npt.NDArray:
        """Get outputs"""
        raise NotImplementedError
