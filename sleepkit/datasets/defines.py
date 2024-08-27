from typing import Callable, Generator

import numpy.typing as npt

SubjectGenerator = Generator[str, None, None]
Preprocessor = Callable[[npt.NDArray], npt.NDArray]
