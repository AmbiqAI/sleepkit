import math
import functools

import scipy.signal as sps
import scipy.interpolate as spi
import numpy as np
import numpy.typing as npt

def resample_categorical(data: npt.NDArray, sample_rate: float, target_rate: float, axis: int = 0) -> npt.NDArray:
    """Resample categorical data using nearest neighbor.

    Args:
        data (npt.NDArray): Signal
        sample_rate (float): Signal sampling rate
        target_rate (float): Target sampling rate
        axis (int, optional): Axis to resample along. Defaults to 0.

    Returns:
        npt.NDArray: Resampled signal
    """
    if sample_rate == target_rate:
        return data
    ratio = target_rate / sample_rate
    actual_length = data.shape[axis]
    target_length = int(np.round(data.shape[axis] * ratio))
    interp_fn = spi.interp1d(np.arange(0, actual_length), data, kind='nearest', axis=axis)
    return interp_fn(np.arange(0, target_length)).astype(data.dtype)
