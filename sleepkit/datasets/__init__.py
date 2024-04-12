from .augmentation import augment_pipeline, preprocess_pipeline
from .cmidss import CmidssDataset
from .dataset import SKDataset
from .defines import AugmentationParams, PreprocessParams
from .download import download_datasets
from .factory import DatasetFactory
from .hdf5 import Hdf5Dataset
from .mesa import MesaDataset
from .stages import StagesDataset
from .ysyw import YsywDataset

DatasetFactory.register("cmidss", CmidssDataset)
DatasetFactory.register("mesa", MesaDataset)
DatasetFactory.register("hdf5", Hdf5Dataset)
DatasetFactory.register("stages", StagesDataset)
DatasetFactory.register("ysyw", YsywDataset)
