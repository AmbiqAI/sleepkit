"""Verify source order independently of local time-of-day clock jumps."""

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

from sleepkit.recipes.detection.alignment import main, verify


def source(tmp_path, *, shift=3600, gap=False, wrong_channel=False):
    if os.environ.get("SLEEPKIT_REQUIRE_SOURCE_AUDIT") == "1":
        assert importlib.util.find_spec("pyarrow"), "Required source-audit dependency pyarrow is missing"
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    h5py = pytest.importorskip("h5py")
    timestamps, tod = [], []
    base = datetime(2020, 3, 8, 6, 59, 40, tzinfo=timezone.utc)
    for i in range(12):
        utc = base + timedelta(seconds=(i + (1 if gap and i >= 6 else 0)) * 5)
        offset = -5 * 3600 + (shift if i >= 6 else 0)
        local = utc.astimezone(timezone(timedelta(seconds=offset)))
        timestamps.append(local.strftime("%Y-%m-%dT%H:%M:%S%z"))
        tod.append(local.hour * 3600 + local.minute * 60 + local.second)
    enmo = np.arange(12, dtype=np.float32)
    angle = enmo * 2
    pq.write_table(
        pa.table(
            {"series_id": ["a"] * 12, "step": np.arange(12), "timestamp": timestamps, "enmo": enmo, "anglez": angle}
        ),
        tmp_path / "raw.parquet",
        row_group_size=4,
    )
    with h5py.File(tmp_path / "a.h5", "w") as stream:
        stream["data"] = np.asarray([tod, enmo + (1 if wrong_channel else 0), angle], dtype=np.float32)
    return tmp_path / "raw.parquet"


@pytest.mark.parametrize("shift", [3600, -3600, 0])
def test_offset_changes_preserve_physical_cadence_across_batches(tmp_path, shift):
    path = source(tmp_path, shift=shift)
    report = verify(tmp_path, path, batch_size=3)
    assert report["status"] == "passed"
    assert report["offset_changes"] == int(shift != 0)
    assert report["subjects"]["a"]["raw_samples"] == 12


def test_true_gap_fails_even_with_clock_change(tmp_path):
    path = source(tmp_path, gap=True)
    report = verify(tmp_path, path, batch_size=3)
    assert report["status"] == "failed"
    assert report["issues"]["utc_cadence_mismatch"] == 1


def test_equivalent_offset_spellings_are_not_clock_changes(tmp_path):
    path = source(tmp_path, shift=0)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    timestamps = table["timestamp"].to_pylist()
    timestamps = [value[:-2] + ":" + value[-2:] if i % 2 else value for i, value in enumerate(timestamps)]
    index = table.schema.get_field_index("timestamp")
    pq.write_table(table.set_column(index, "timestamp", pa.array(timestamps)), path)
    report = verify(tmp_path, path, batch_size=3)
    assert report["status"] == "passed"
    assert report["offset_changes"] == 0


def test_channel_mismatch_fails(tmp_path):
    path = source(tmp_path, wrong_channel=True)
    report = verify(tmp_path, path, batch_size=3)
    assert report["status"] == "failed"
    assert report["issues"]["ENMO_mismatch"] == 12


def test_steps_are_verified_independently_of_matching_channels(tmp_path):
    path = source(tmp_path)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    index = table.schema.get_field_index("step")
    pq.write_table(table.set_column(index, "step", pa.array(np.arange(12) + 1)), path)
    report = verify(tmp_path, path, batch_size=3)
    assert report["status"] == "failed"
    assert report["issues"]["step_index_mismatch"] == 12


def test_missing_subject_rows_fail(tmp_path):
    path = source(tmp_path)
    import pyarrow.parquet as pq

    pq.write_table(pq.read_table(path).slice(0, 9), path)
    report = verify(tmp_path, path, batch_size=3)
    assert report["status"] == "failed"
    assert report["issues"]["row_count_mismatch"] == 1


@pytest.mark.parametrize("gap", [False, True])
def test_cli_writes_complete_report_and_returns_failure_for_gap(tmp_path, monkeypatch, capsys, gap):
    path = source(tmp_path, gap=gap)
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["alignment", "--data", str(tmp_path), "--parquet", str(path), "--output", str(output)])
    if gap:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1
    else:
        main()
    report = json.loads(output.read_text())
    assert report["status"] == ("failed" if gap else "passed")
    assert report["subjects"]["a"]["local_clock_changes"] == {str(3610 if gap else 3605): 1}
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == report["status"]
    assert "subjects" not in summary
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_bytes() == before
