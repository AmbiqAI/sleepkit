from .apnea import ApneaTask, SleepApnea
from .factory import TaskFactory
from .stage import SleepStage, StageTask
from .task import SKTask

TaskFactory.register("detect", StageTask)
TaskFactory.register("stage", StageTask)
TaskFactory.register("apnea", ApneaTask)
# TaskFactory.register("arousal", ArousalTask)
