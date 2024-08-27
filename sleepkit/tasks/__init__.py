"""
# :simple-task: Tasks API

SleepKit provides several built-in __sleep-monitoring__ tasks.
Each task is designed to address a unique aspect such as ECG denoising, segmentation, and rhythm/beat classification.
The tasks are designed to be modular and can be used independently or in combination to address specific use cases.

## Available Tasks

- **[StageTask](./stage)**: Sleep stage classification task
- **[ApneaTask](./apnea)**: Sleep apnea detection task


## Task Factory

The TaskFactory provides a convenient way to access the built-in tasks.
The factory is a thread-safe singleton class that provides a single point of access to the tasks via the tasks' slug names.
The benefit of using the factory is it allows registering custom tasks that can then be used just like built-in tasks.

```python
import sleepkit as sk

for task in sk.TaskFactory.list():
    print(f"Task name: {task} - {sk.TaskFactory.get(task)}")
```

Classes:
    Task: Base class for all tasks
    StageTask: Sleep stage classification task
    ApneaTask: Sleep apnea detection task

"""

from .apnea import ApneaTask
from .stage import StageTask
from .task import Task

import neuralspot_edge as nse


TaskFactory = nse.utils.create_factory(factory="SKTaskFactory", type=Task)

TaskFactory.register("detect", StageTask)
TaskFactory.register("stage", StageTask)
TaskFactory.register("apnea", ApneaTask)
