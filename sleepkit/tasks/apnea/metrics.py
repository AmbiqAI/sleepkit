import numpy as np
import numpy.typing as npt
import scipy.signal

from ...defines import SleepApnea


def compute_sleep_apnea_durations(apnea_mask: npt.NDArray) -> dict[int, int]:
    """Compute sleep apnea durations

    Args:
        apnea_mask (npt.NDArray): Sleep mask (1D array of sleep apnea)

    Returns:
        dict[int, int]: Sleep apnea durations (class -> duration)
    """
    bounds = np.diff(apnea_mask).nonzero()[0] + 1
    left_bounds = np.concatenate(([0], bounds))
    right_bounds = np.concatenate((bounds, [apnea_mask.size]))
    dur_bounds = right_bounds - left_bounds
    class_bounds = apnea_mask[left_bounds]
    class_durations = {k: 0 for k in set(class_bounds)}
    for i, c in enumerate(class_bounds):
        class_durations[c] += dur_bounds[i]
    # END FOR
    return class_durations


def compute_apnea_efficiency(apnea_durations: dict[int, int], class_map: dict[int, int]) -> float:
    """Compute apnea efficiency.

    Args:
        apnea_durations (dict[int, int]): Sleep apnea durations (class -> duration)
        class_map (dict[int, int]): Class map (class -> class)

    Returns:
        float: apnea efficiency
    """
    norm_classes = [SleepApnea.none]
    apnea_classes = [
        SleepApnea.hypopnea,
        SleepApnea.central,
        SleepApnea.obstructive,
        SleepApnea.mixed,
    ]
    norm_keys = list(set(class_map.get(s) for s in norm_classes if s in class_map))
    apnea_keys = list(set(class_map.get(s) for s in apnea_classes if s in class_map))
    norm_duration = sum(apnea_durations.get(k, 0) for k in norm_keys)
    apnea_duration = sum(apnea_durations.get(k, 0) for k in apnea_keys)
    efficiency = norm_duration / (apnea_duration + norm_duration)
    return efficiency


def compute_apnea_hypopnea_index(apnea_mask: npt.NDArray, min_duration: int, sample_rate: float) -> float:
    """Compute apnea hypopnea index (AHI).

    Args:
        apnea_mask (npt.NDArray): Sleep apnea mask (1D array of sleep apnea)
        min_duration (int): Minimum duration (in samples) to be considered an event
        sample_rate (float): Sample rate

    Returns:
        float: Sleep efficiency
    """
    med_len = (1 + min_duration // 2) % 2 + min_duration // 2
    med_mask = scipy.signal.medfilt(apnea_mask, kernel_size=med_len)

    bounds = np.diff(med_mask).nonzero()[0] + 1
    left_bounds = np.concatenate(([0], bounds))
    right_bounds = np.concatenate((bounds, [med_mask.size]))
    dur_bounds = right_bounds - left_bounds
    num_events = 0

    for left_bound, dur in zip(left_bounds, dur_bounds):
        if dur > min_duration and med_mask[left_bound] != 0:
            num_events += 1
    num_hours = apnea_mask.size / sample_rate / 3600
    ahi = num_events / num_hours
    return ahi
