import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("helia_edge")

from sleepkit.defines import SleepStage  # noqa: E402
from sleepkit.tasks.stage.metrics import (  # noqa: E402
    compute_sleep_efficiency,
    compute_sleep_stage_durations,
)


def test_compute_sleep_stage_durations_and_efficiency():
    mask = np.array([0, 0, 1, 1, 2, 2, 0], dtype=int)
    durations = compute_sleep_stage_durations(mask)
    assert durations == {0: 3, 1: 2, 2: 2}

    class_map = {
        SleepStage.wake: 0,
        SleepStage.stage1: 1,
        SleepStage.stage2: 2,
    }
    efficiency = compute_sleep_efficiency(durations, class_map)
    assert efficiency == 4 / 7
