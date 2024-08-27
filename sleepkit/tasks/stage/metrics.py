import numpy as np
import numpy.typing as npt

from ...defines import SleepStage


def compute_sleep_stage_durations(sleep_mask: npt.NDArray) -> dict[int, int]:
    """Compute sleep stage durations

    Args:
        sleep_mask (npt.NDArray): Sleep mask (1D array of sleep stages)

    Returns:
        dict[int, int]: Sleep stage durations (class -> duration)
    """
    bounds = np.diff(sleep_mask).nonzero()[0] + 1
    left_bounds = np.concatenate(([0], bounds))
    right_bounds = np.concatenate((bounds, [sleep_mask.size]))
    dur_bounds = right_bounds - left_bounds
    class_bounds = sleep_mask[left_bounds]
    class_durations = {k: 0 for k in set(class_bounds)}
    for i, c in enumerate(class_bounds):
        class_durations[c] += dur_bounds[i]
    # END FOR
    return class_durations


def compute_total_sleep_time(sleep_durations: dict[int, int], class_map: dict[int, int]) -> int:
    """Compute total sleep time (# samples).

    Args:
        sleep_durations (dict[int, int]): Sleep stage durations (class -> duration)
        class_map (dict[int, int]): Class map (class -> class)

    Returns:
        int: Total sleep time (# samples)
    """
    # wake_classes = [SleepStage.wake]
    sleep_classes = [
        SleepStage.stage1,
        SleepStage.stage2,
        SleepStage.stage3,
        SleepStage.stage4,
        SleepStage.rem,
    ]
    # wake_keys = list(set(class_map.get(s) for s in wake_classes if s in class_map))
    sleep_keys = list(set(class_map.get(s) for s in sleep_classes if s in class_map))
    # wake_duration = sum(sleep_durations.get(k, 0) for k in wake_keys)
    sleep_duration = sum(sleep_durations.get(k, 0) for k in sleep_keys)
    tst = sleep_duration
    return tst


def compute_sleep_efficiency(sleep_durations: dict[int, int], class_map: dict[int, int]) -> float:
    """Compute sleep efficiency.

    Args:
        sleep_durations (dict[int, int]): Sleep stage durations (class -> duration)
        class_map (dict[int, int]): Class map (class -> class)

    Returns:
        float: Sleep efficiency
    """
    wake_classes = [SleepStage.wake]
    sleep_classes = [
        SleepStage.stage1,
        SleepStage.stage2,
        SleepStage.stage3,
        SleepStage.stage4,
        SleepStage.rem,
    ]
    wake_keys = list(set(class_map.get(s) for s in wake_classes if s in class_map))
    sleep_keys = list(set(class_map.get(s) for s in sleep_classes if s in class_map))
    wake_duration = sum(sleep_durations.get(k, 0) for k in wake_keys)
    sleep_duration = sum(sleep_durations.get(k, 0) for k in sleep_keys)
    efficiency = sleep_duration / (sleep_duration + wake_duration)
    return efficiency
