# Bring-Your-Own-Dataset (BYOD)

A custom dataset subclasses `sk.Dataset`, exposes subject IDs, and supplies the signal and label readers consumed by its feature extractor. There is no universal signal schema: pair the adapter with a [custom feature set](../features/byofs.md).

## Define the adapter

This example reads an existing directory of subject `.npz` files with a `signal` array. It makes an explicit subject split and does not download or invent data.

```python
from pathlib import Path
import random
import numpy as np
import sleepkit as sk

class CustomDataset(sk.Dataset):
    @property
    def subject_ids(self) -> list[str]:
        return sorted(p.stem for p in self.path.glob("*.npz"))

    @property
    def train_subject_ids(self) -> list[str]:
        return self.subject_ids[:int(0.8 * len(self.subject_ids))]

    @property
    def test_subject_ids(self) -> list[str]:
        return self.subject_ids[int(0.8 * len(self.subject_ids)):]

    def uniform_subject_generator(self, subject_ids=None, repeat=True, shuffle=True):
        ids = list(self.subject_ids if subject_ids is None else subject_ids)
        if not ids:
            return
        while True:
            if shuffle:
                random.shuffle(ids)
            yield from ids
            if not repeat:
                break

    def load_signal_for_subject(self, subject_id: str) -> np.ndarray:
        with np.load(self.path / f"{subject_id}.npz") as record:
            return record["signal"].copy()

    def download(self, num_workers=None, force=False):
        if not self.subject_ids:
            raise FileNotFoundError(f"Place subject .npz files in {self.path}")

sk.DatasetFactory.register("custom", CustomDataset)
dataset = sk.DatasetFactory.get("custom")(path=Path("./datasets/custom"))
```

Keep every recording from one person in the same partition. Persist the split for reproducible experiments. Extend the reader with the labels, timestamps and signal names required by your feature extractor; existing built-in extractors expect their own dataset-specific methods.

## Use it in a task

Add `custom` and its path to the task's dataset configuration, then select the matching custom feature set. Register both extensions in the Python process before invoking a task. Registering the dataset alone does not make it compatible with every built-in feature set.
