import functools
import os
from multiprocessing import Pool

from tqdm import tqdm

from ..datasets import DatasetFactory
from ..defines import SKFeatureParams
from ..utils import setup_logger
from .store import FeatureStore

logger = setup_logger(__name__)


def generate_feature_set(args: SKFeatureParams):
    """Generate feature set for all subjects in dataset

    Args:
        args (SKFeatureParams): Feature generation parameters
    """
    os.makedirs(args.save_path, exist_ok=True)

    if not FeatureStore.has(args.feature_set):
        raise NotImplementedError(f"Feature set {args.feature_set} not implemented")
    fset = FeatureStore.get(args.feature_set)

    ds_subjects: list[tuple[str, str]] = []
    for dataset in args.datasets:
        if DatasetFactory.has(dataset.name):
            os.makedirs(args.save_path / dataset.name, exist_ok=True)
            ds = DatasetFactory.get(dataset.name)(ds_path=args.ds_path, frame_size=args.frame_size, **dataset.params)
            ds_subjects += [(dataset.name, subject_id) for subject_id in ds.subject_ids]
        # END IF
    # END FOR

    f = functools.partial(fset.generate_features, args=args)
    with Pool(processes=args.data_parallelism) as pool:
        _ = list(tqdm(pool.imap(f, ds_subjects, chunksize=1), total=len(ds_subjects)))
    # END WITH
