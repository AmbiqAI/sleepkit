import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("helia_edge")

from sleepkit.defines import SleepApnea  # noqa: E402
from sleepkit.tasks.apnea.metrics import (  # noqa: E402
    compute_apnea_efficiency,
    compute_apnea_hypopnea_index,
    compute_sleep_apnea_durations,
)


def test_compute_sleep_apnea_durations_counts_segments():
    mask = np.array([0, 0, 1, 1, 1, 0, 2, 2], dtype=int)
    durations = compute_sleep_apnea_durations(mask)
    assert durations == {0: 3, 1: 3, 2: 2}


def test_compute_apnea_efficiency_uses_mapped_classes_only():
    durations = {0: 90, 1: 10, 2: 50}
    class_map = {
        SleepApnea.none: 0,
        SleepApnea.hypopnea: 1,
    }
    efficiency = compute_apnea_efficiency(durations, class_map)
    assert efficiency == 0.9


def test_compute_apnea_hypopnea_index_counts_events_over_min_duration():
    mask = np.zeros(3600, dtype=int)
    mask[100:105] = 1
    mask[200:203] = 1
    ahi = compute_apnea_hypopnea_index(mask, min_duration=4, sample_rate=1)
    assert ahi == 1.0
