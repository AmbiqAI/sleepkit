import functools
import os
from multiprocessing import Pool

import h5py
from tqdm import tqdm

from ..datasets import CmidssDataset, MesaDataset, YsywDataset
from ..defines import SKFeatureParams
from ..utils import setup_logger
from .defines import FeatSet
from .featset01 import FeatSet01
from .featset02 import FeatSet02
from .featset03 import FeatSet03
from .featset04 import FeatSet04

logger = setup_logger(__name__)

feat_factory_map: dict[str, FeatSet] = {
    FeatSet01.name(): FeatSet01,
    FeatSet02.name(): FeatSet02,
    FeatSet03.name(): FeatSet03,
    FeatSet04.name(): FeatSet04,
}


def compute_subject_features(ds_subject: tuple[str, str], args: SKFeatureParams):
    """Compute features for subject.

    Args:
        ds_subject (tuple[str, str]): Dataset name and subject ID
        args (SKFeatureParams): Feature generation parameters
    """
    ds_name, subject_id = ds_subject

    featset = feat_factory_map.get(args.feature_set, None)
    if featset is None:
        raise NotImplementedError(f"Feature set {args.feature_set} not implemented")

    try:
        features, labels, mask = featset.compute_features(ds_name, subject_id=subject_id, args=args)

        with h5py.File(str(args.save_path / f"{subject_id}.h5"), "w") as h5:
            h5.create_dataset("/features", data=features, compression="gzip", compression_opts=6)
            h5.create_dataset("/labels", data=labels, compression="gzip", compression_opts=6)
            h5.create_dataset("/mask", data=mask, compression="gzip", compression_opts=6)
        # END WITH
    # pylint: disable=broad-except
    except Exception as err:
        logger.error(f"Error computing features for subject {subject_id}: {err}")


def generate_feature_set(args: SKFeatureParams):
    """Generate feature set for all subjects in dataset

    Args:
        args (SKFeatureParams): Feature generation parameters
    """
    os.makedirs(args.save_path, exist_ok=True)

    ds_subjects: list[tuple[str, str]] = []
    if "mesa" in args.datasets:
        subject_ids = MesaDataset(args.ds_path, is_commercial=True).subject_ids
        ds_subjects += [("mesa", subject_id) for subject_id in subject_ids]

    if "ysyw" in args.datasets:
        subject_ids = YsywDataset(args.ds_path).subject_ids
        ds_subjects += [("ysyw", subject_id) for subject_id in subject_ids]

    if "cmidss" in args.datasets:
        subject_ids = CmidssDataset(args.ds_path).subject_ids
        ds_subjects += [("cmidss", subject_id) for subject_id in subject_ids]

    f = functools.partial(compute_subject_features, args=args)
    with Pool(processes=args.data_parallelism) as pool:
        _ = list(tqdm(pool.imap(f, ds_subjects), total=len(ds_subjects)))
    # END WITH
