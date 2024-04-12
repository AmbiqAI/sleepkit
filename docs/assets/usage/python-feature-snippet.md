```python
from pathlib import Path
import sleepkit as sk

sk.generate_feature_set(sk.SKFeatureParams(
    job_dir=Path("./results/fs004"),
    ds_path=Path("./datasets"),
    datasets=[{
        "name": "cmidss",
        "params": {}
    }],
    feature_set="fs004",
    feature_params={},
    save_path=Path("./datasets/processed/fs004"),
    sample_rate=0.2,
    frame_size=12
))

```
