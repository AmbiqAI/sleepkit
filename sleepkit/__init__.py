import os
from importlib.metadata import version

from . import cli, datasets, metrics, models, rpc, tasks, tflite
from .datasets import DatasetFactory, SKDataset
from .defines import (
    DatasetParams,
    ModelArchitecture,
    SKDemoParams,
    SKDownloadParams,
    SKExportParams,
    SKFeatureParams,
    SKMode,
    SKTestParams,
    SKTrainParams,
)
from .features import FeatureStore, SKFeatureSet, generate_feature_set
from .models import ModelFactory
from .tasks import TaskFactory
from .utils import setup_logger

__version__ = version(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
setup_logger(__name__)
