"""Exercise the legacy Parquet engine explicitly under the current NumPy stack."""

import importlib.util
import os

import numpy as np
import pytest


def test_fastparquet_numpy2_read(tmp_path):
    pd = pytest.importorskip("pandas")
    if importlib.util.find_spec("fastparquet") is None:
        if os.environ.get("SLEEPKIT_REQUIRE_SOURCE_AUDIT") == "1":
            pytest.fail("Required fastparquet reader is missing")
        pytest.skip("fastparquet is not installed")
    expected = pd.DataFrame({"step": np.arange(8, dtype=np.int32), "value": np.arange(8, dtype=np.float32) / 2})
    path = tmp_path / "fastparquet.parquet"
    expected.to_parquet(path, engine="fastparquet", index=False)
    actual = pd.read_parquet(path, engine="fastparquet")
    pd.testing.assert_frame_equal(actual, expected)
