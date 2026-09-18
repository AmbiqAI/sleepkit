"""Inventory historical artifacts without loading or modifying model weights.

python scripts/inventory_baselines.py --results results --output /tmp/baselines.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def inventory(root):
    entries = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / "model.keras").exists():
            continue
        files = {}
        for name in ("model.keras", "model.tflite", "model_buffer.h", "configuration.json", "metrics.json"):
            path = folder / name
            if path.exists():
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files[name] = {"sha256": digest, "bytes": path.stat().st_size}
        entry = {"id": folder.name, "artifacts": files}
        with zipfile.ZipFile(folder / "model.keras") as archive:
            entry["keras_metadata"] = json.loads(archive.read("metadata.json"))
        metrics = folder / "metrics.json"
        if metrics.exists():
            entry["historical_metrics"] = json.loads(metrics.read_text())
        entry["evaluation_provenance"] = "Historical metrics; split membership and protocol require reconciliation"
        entries.append(entry)
    return {"schema_version": 1, "baselines": entries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inventory(args.results), indent=2) + "\n")
