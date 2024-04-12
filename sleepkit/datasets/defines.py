from typing import Any, Callable, Generator

import numpy.typing as npt
from pydantic import BaseModel

SubjectGenerator = Generator[str, None, None]
Preprocessor = Callable[[npt.NDArray], npt.NDArray]
SampleGenerator = Generator[tuple[npt.NDArray, npt.NDArray], None, None]


class PreprocessParams(BaseModel, extra="allow"):
    """Preprocessing parameters"""

    name: str
    params: dict[str, Any]


class AugmentationParams(BaseModel, extra="allow"):
    """Augmentation parameters"""

    name: str
    args: dict[str, tuple[float | int, float | int]]
