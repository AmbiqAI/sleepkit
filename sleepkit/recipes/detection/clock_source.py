"""Copy explicitly selected, verified CMIDSS recordings with an independent UTC clock.

The alignment report is trusted local verification evidence, not a signed attestation.
Historical labels are copied unchanged; this operation makes no label-semantic claim.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from .hdf5 import require_self_contained

from sleepkit.artifacts.package import sha256


def materialize(data_root, alignment_path, destination, *, subjects):
    """Return provenance after atomically creating a new clock-bearing source folder.

    Destination publication assumes a single local writer.
    No raw Parquet dependency is needed: a successful report binds every selected
    original HDF5 hash to a verified UTC origin and five-second cadence.
    """
    import h5py

    if isinstance(subjects, (str, bytes)):
        raise ValueError("Subjects must be an explicit iterable of subject IDs")
    subjects = list(subjects)
    if not subjects or any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", s) for s in subjects):
        raise ValueError("Select nonempty filename-stem subject IDs")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Subject IDs must be unique")
    root, alignment_path, destination = Path(data_root), Path(alignment_path), Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    report_hash = sha256(alignment_path)
    report = json.loads(alignment_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "sleepkit.source_alignment/v1"
        or report.get("status") != "passed"
        or report.get("issues") != {}
        or report.get("raw_subjects_without_h5") != {}
        or not re.fullmatch(r"[0-9a-f]{64}", str(report.get("parquet_sha256", "")))
    ):
        raise ValueError("A fully passed source-alignment report is required")
    details = report.get("subjects")
    if not isinstance(details, dict) or not details or report.get("h5_subjects") != len(details) or report.get("subjects_passed") != len(details):
        raise ValueError("Report subject counts must describe complete verification")
    if any(d.get("status") != "passed" or d.get("issues") != {} for d in details.values()):
        raise ValueError("Every report subject must have passed alignment")
    selected = {}
    limit = np.iinfo(np.int64)
    for subject in subjects:
        detail = details.get(subject, {})
        count, start = detail.get("h5_samples"), detail.get("first_utc_seconds")
        digest = detail.get("h5_sha256")
        if type(count) is not int or count < 1 or detail.get("raw_samples") != count:
            raise ValueError(f"Missing or inconsistent sample count: {subject}")
        if type(start) is not int or not limit.min <= start <= start + 5 * (count - 1) <= limit.max:
            raise ValueError(f"Missing or overflowing verified UTC origin: {subject}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Missing source HDF5 hash: {subject}")
        path = root / f"{subject}.h5"
        with h5py.File(path, "r") as stream:
            require_self_contained(stream)
        if sha256(path) != digest:
            raise ValueError(f"Source HDF5 differs from alignment report: {subject}")
        selected[subject] = (path, count, start, digest)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        outputs = {}
        for subject, (source, count, start, digest) in selected.items():
            target = stage / f"{subject}.h5"
            shutil.copyfile(source, target)
            if sha256(target) != digest or sha256(source) != digest:
                raise ValueError(f"Source changed while copying: {subject}")
            with h5py.File(target, "r+") as stream:
                require_self_contained(stream)
                if stream["data"].shape != (3, count) or "sample_time" in stream:
                    raise ValueError(f"Expected legacy data without an existing sample clock: {subject}")
                clock = stream.create_dataset("sample_time", shape=(count,), dtype="int64", chunks=(min(count, 65536),))
                clock.attrs["units"] = "unix_seconds"
                for offset in range(0, count, 65536):
                    size = min(65536, count - offset)
                    # Python arithmetic keeps endpoint checking correct at int64 extremes.
                    clock[offset : offset + size] = np.arange(size, dtype=np.int64) * 5 + np.int64(start + offset * 5)
            outputs[subject] = {"source_h5_sha256": digest, "output_h5_sha256": sha256(target), "samples": count, "first_utc_seconds": start}
        if sha256(alignment_path) != report_hash or any(sha256(path) != digest for path, _, _, digest in selected.values()):
            raise ValueError("Verification evidence or source HDF5 changed during materialization")
        provenance = {
            "schema": "sleepkit.clock_source/v1",
            "alignment_sha256": report_hash,
            "parquet_sha256": report["parquet_sha256"],
            "sample_time": {"units": "unix_seconds", "dtype": "int64", "period_seconds": 5},
            "subjects": outputs,
            "labels": "Historical sleep_stages copied unchanged; no new annotation policy applied",
        }
        evidence = stage / "clock-source.json"
        evidence.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "clock-source.sha256").write_text(f"{sha256(evidence)}  clock-source.json\n", encoding="ascii")
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        stage.rename(destination)
        return provenance
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    args = parser.parse_args()
    materialize(args.data, args.alignment, args.output, subjects=args.subjects)


if __name__ == "__main__":
    main()
