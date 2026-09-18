"""Explicitly synthetic fixture for exercising the complete recipe without datasets."""

from pathlib import Path

import numpy as np

from sleepkit.artifacts.package import write_json
from sleepkit.artifacts.runtime import replay_bundle
from .inference import predict
from .recipe import Config, run


def make_fixture(root):
    import h5py

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for index in range(4):
        n = 300
        labels = ((np.arange(n) // 36 + index) % 2).astype(np.int32)
        data = np.stack(
            [np.arange(n) * 5, (1 - labels) + rng.normal(0, 0.05, n), labels * 20 + rng.normal(0, 1, n)]
        ).astype(np.float32)
        with h5py.File(root / f"synthetic-{index}.h5", "w") as stream:
            stream["data"] = data
            stream["sleep_stages"] = labels
            stream.attrs["sample_rate_hz"] = 0.2
            stream.attrs["channel_names"] = ["TS", "ENMO", "ZANGLE"]
    split = {"train": ["synthetic-0", "synthetic-1"], "validation": ["synthetic-2"], "test": ["synthetic-3"]}
    write_json(root / "split.json", split)
    return root / "split.json"


def run_smoke(source, output):
    from .data import read_subject

    split = make_fixture(source)
    result = run(
        source, split, output, Config(context=8, epochs=1, batch_size=2), cache=source / "cache", data_kind="synthetic"
    )
    replay_bundle(result / "bundle")
    data, _ = read_subject(source, "synthetic-3", labels=False)
    predictions = predict(result / "bundle", data)
    assert predictions["logits"].shape == (48, 2)
    print(f"Synthetic recipe completed: {result}; {len(predictions['times'])} predicted epochs")
    return result
