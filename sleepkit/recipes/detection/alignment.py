"""Verify raw CMIDSS sample order and clocks against legacy HDF5 files in batches."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from sleepkit.artifacts.package import sha256


def verify(data_root, parquet_path, *, batch_size=65536):
    """Read all source samples; the resulting evidence does not validate annotations."""
    import h5py
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    parquet_path = Path(parquet_path)
    source_hash = sha256(parquet_path)
    files = {p.stem: p for p in Path(data_root).glob("*.h5")}
    if not files:
        raise ValueError("No HDF5 subjects found")
    details, previous = {}, {}
    for subject, path in sorted(files.items()):
        with h5py.File(path, "r") as stream:
            shape = stream["data"].shape
            if len(shape) != 2 or shape[0] != 3:
                raise ValueError("Expected HDF5 data in TS/ENMO/ZANGLE order")
        details[subject] = {
            "h5_sha256": sha256(path),
            "h5_samples": shape[1],
            "raw_samples": 0,
            "issues": Counter(),
            "offset_changes": 0,
            "local_clock_changes": Counter(),
        }
    parquet = pq.ParquetFile(parquet_path)
    columns = ["series_id", "step", "timestamp", "enmo", "anglez"]
    if not set(columns) <= set(parquet.schema_arrow.names):
        raise ValueError("Missing required raw-series columns")
    unknown = Counter()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        if any(batch.column(name).null_count for name in columns):
            raise ValueError("Null raw samples require an explicit missing-data policy")
        ids = np.asarray(batch.column("series_id").to_pylist())
        raw_steps = batch.column("step").to_numpy()
        if not np.issubdtype(raw_steps.dtype, np.integer):
            raise ValueError("Raw source steps must be integers")
        timestamps = batch.column("timestamp")
        # Require a timezone suffix; never silently interpret a naive clock as UTC.
        if not pc.all(
            pc.match_substring_regex(timestamps, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:?\d{2}$")
        ).as_py():
            raise ValueError("Raw timestamps require ISO seconds and a numeric timezone offset")
        utc = pc.cast(pc.strptime(timestamps, format="%Y-%m-%dT%H:%M:%S%z", unit="s"), pa.int64()).to_numpy()

        def component(start, stop):
            return pc.cast(pc.utf8_slice_codeunits(timestamps, start, stop), pa.int32()).to_numpy()

        local = component(11, 13) * 3600 + component(14, 16) * 60 + component(17, 19)
        offset_text = pc.replace_substring(pc.utf8_slice_codeunits(timestamps, 19), ":", "")
        offset_hours = pc.cast(pc.utf8_slice_codeunits(offset_text, 1, 3), pa.int32()).to_numpy()
        offset_minutes = pc.cast(pc.utf8_slice_codeunits(offset_text, 3, 5), pa.int32()).to_numpy()
        offset_sign = np.where(pc.starts_with(offset_text, "-").to_numpy(zero_copy_only=False), -1, 1)
        offsets = offset_sign * (offset_hours * 3600 + offset_minutes * 60)
        enmo = batch.column("enmo").to_numpy().astype(np.float32)
        angle = batch.column("anglez").to_numpy().astype(np.float32)
        for subject in np.unique(ids):
            selected = ids == subject
            count = int(selected.sum())
            if subject not in details:
                unknown[str(subject)] += count
                continue
            detail = details[subject]
            start, stop = detail["raw_samples"], detail["raw_samples"] + count
            steps, times, tod, zone = raw_steps[selected], utc[selected], local[selected], offsets[selected]
            detail["issues"]["step_index_mismatch"] += int((steps != np.arange(start, stop)).sum())
            if subject in previous:
                prev_utc, prev_tod, prev_zone = previous[subject]
                times_for_diff = np.r_[prev_utc, times]
                tod_for_diff = np.r_[prev_tod, tod]
                zones_for_diff = np.concatenate(([prev_zone], zone))
            else:
                times_for_diff, tod_for_diff, zones_for_diff = times, tod, zone
            utc_delta = np.diff(times_for_diff)
            tod_delta = np.mod(np.diff(tod_for_diff), 86400)
            zone_changes = zones_for_diff[1:] != zones_for_diff[:-1]
            detail["issues"]["utc_cadence_mismatch"] += int((utc_delta != 5).sum())
            detail["offset_changes"] += int(zone_changes.sum())
            values, counts = np.unique(tod_delta[tod_delta != 5], return_counts=True)
            detail["local_clock_changes"].update({str(int(v)): int(n) for v, n in zip(values, counts)})
            detail["issues"]["unexplained_local_clock_change"] += int(((tod_delta != 5) & ~zone_changes).sum())
            previous[subject] = (times[-1], tod[-1], zone[-1])
            available = max(0, min(count, detail["h5_samples"] - start))
            if available:
                with h5py.File(files[subject], "r") as stream:
                    h5_values = np.asarray(stream["data"][:, start : start + available], dtype=np.float32)
                for i, (name, raw) in enumerate((("TS", tod), ("ENMO", enmo[selected]), ("ZANGLE", angle[selected]))):
                    raw = raw[:available].astype(np.float32)
                    equal = (h5_values[i] == raw) | (np.isnan(h5_values[i]) & np.isnan(raw))
                    detail["issues"][f"{name}_mismatch"] += int((~equal).sum())
            detail["raw_samples"] = stop
    total_issues = Counter()
    for subject, detail in details.items():
        if sha256(files[subject]) != detail["h5_sha256"]:
            raise ValueError("HDF5 source changed during verification")
        if detail["raw_samples"] != detail["h5_samples"]:
            detail["issues"]["row_count_mismatch"] = 1
        detail["issues"] = {name: count for name, count in detail["issues"].items() if count}
        detail["status"] = "passed" if not detail["issues"] else "failed"
        total_issues.update(detail["issues"])
    if sha256(parquet_path) != source_hash:
        raise ValueError("Parquet source changed during verification")
    return {
        "schema": "sleepkit.source_alignment/v1",
        "parquet_sha256": source_hash,
        "status": "passed" if not total_issues and not unknown else "failed",
        "raw_samples": parquet.metadata.num_rows,
        "h5_subjects": len(files),
        "subjects_passed": sum(d["status"] == "passed" for d in details.values()),
        "raw_subjects_without_h5": dict(unknown),
        "issues": dict(total_issues),
        "subjects_with_offset_changes": sum(d["offset_changes"] > 0 for d in details.values()),
        "offset_changes": sum(d["offset_changes"] for d in details.values()),
        "subjects": details,
        "scope": "Step zero origin, contiguous steps/UTC time, row counts, and TS/ENMO/ZANGLE equality; no annotation or person-identity claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new local report path")
    report = verify(args.data, args.parquet)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"subjects", "raw_subjects_without_h5"}}, indent=2))
    if report["status"] != "passed":
        parser.exit(1, "Source alignment failed; inspect the local report.\n")


if __name__ == "__main__":
    main()
