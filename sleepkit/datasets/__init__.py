"""
# :material-database: Datasets API

This module provides the built-in dataset classes.

Classes:
    Dataset: Base class for all datasets.
    CmidssDataset: CMIDSS dataset.
    MesaDataset: MESA dataset.
    StagesDataset: Stages dataset.
    YsywDataset: YSYW dataset.

"""

from .augmentation import create_augmentation_layer, create_augmentation_pipeline
from .cmidss import CmidssDataset
from .dataset import Dataset
from .mesa import MesaDataset
from .stages import StagesDataset
from .ysyw import YsywDataset

import neuralspot_edge as nse

DatasetFactory = nse.utils.create_factory(factory="SKDatasetFactory", type=Dataset)

DatasetFactory.register("cmidss", CmidssDataset)
DatasetFactory.register("mesa", MesaDataset)
DatasetFactory.register("stages", StagesDataset)
DatasetFactory.register("ysyw", YsywDataset)
