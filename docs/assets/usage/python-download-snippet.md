```python
from pathlib import Path
import sleepkit as sk

sk.datasets.download_datasets(sk.SKDownloadParams(
    ds_path=Path("./datasets"),
    datasets=["icentia11k", "ludb", "qtdb", "synthetic"],
    progress=True
))
```
