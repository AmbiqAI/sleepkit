```python
from pathlib import Path
import sleepkit as sk

task = sk.TaskFactory.get("stage")

task.train(sk.SKTrainParams(
    ...
))
```
