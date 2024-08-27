"""
# SleepKit API

SleepKit is an AI Development Kit (ADK) that enables developers to easily build and deploy real-time __sleep-monitoring__ models on Ambiq's family of ultra-low power SoCs.
SleepKit explores a number of sleep related tasks including sleep staging, and sleep apnea detection.
The kit includes a variety of datasets, efficient model architectures, and a number of pre-trained models.
The objective of the models is to outperform conventional, hand-crafted algorithms with efficient AI models that still fit within the stringent resource constraints of embedded devices.
Furthermore, the included models are trainined using a large variety datasets- using a subset of biological signals that can be captured from a single body location such as head, chest, or wrist/hand.
The goal is to enable models that can be deployed in real-world commercial and consumer applications that are viable for long-term use.

"""

import os
from importlib.metadata import version
import neuralspot_edge as nse

from . import cli, datasets, models, backends, tasks, features
from .datasets import DatasetFactory, Dataset
from .defines import QuantizationParams, FeatureParams, TaskParams, TaskMode, NamedParams, SleepApnea, SleepStage
from .features import FeatureFactory, FeatureSet, H5Dataloader
from .models import ModelFactory
from .tasks import TaskFactory, Task, ApneaTask, StageTask


__version__ = version(__name__)

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
nse.utils.setup_logger(__name__)
