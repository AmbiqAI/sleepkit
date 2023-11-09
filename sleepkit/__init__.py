import os
from importlib.metadata import version

from . import cli, datasets, features, metrics, models, tflite, defines
from .utils import setup_logger

__version__ = version(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
setup_logger(__name__)
