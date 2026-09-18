"""CMIDSS metadata compatibility without importing a training runtime."""

import numpy as np
import pytest

from sleepkit.recipes.detection.data import read_subject


@pytest.mark.parametrize(
    "rate,accepted",
    [(0.2, True), (np.float32(0.2), True), (0.25, False), (0.20001, False), (np.nan, False), (np.inf, False)],
)
def test_sample_rate_metadata_allows_float32_rounding(tmp_path, rate, accepted):
    h5py = pytest.importorskip("h5py")
    data = np.stack([np.arange(12) * 5, np.zeros(12), np.zeros(12)]).astype(np.float32)
    with h5py.File(tmp_path / "subject.h5", "w") as stream:
        stream["data"] = data
        stream.attrs["sample_rate_hz"] = rate
    if accepted:
        actual, labels = read_subject(tmp_path, "subject", labels=False)
        np.testing.assert_array_equal(actual, data)
        assert labels is None
    else:
        with pytest.raises(ValueError, match="0.2 Hz"):
            read_subject(tmp_path, "subject", labels=False)
