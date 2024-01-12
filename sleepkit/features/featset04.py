import numpy as np
import physiokit as pk
import scipy.signal

from ..datasets import CmidssDataset
from ..defines import SKFeatureParams
from ..utils import setup_logger

logger = setup_logger(__name__)

class FeatSet04:

    @staticmethod
    def name() -> str:
        """Feature set name."""
        return "fs004"

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        return ["tod", "mov_mu", "mov_std", "angle_mu", "angle_std"]


    @staticmethod
    def compute_features(
        ds_name: str, subject_id: str, args: SKFeatureParams
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute features for subject.

        Args:
            ds_name (str): Dataset name
            subject_id (str): Subject ID
            args (SKFeatureParams): Feature generation parameters

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Features, labels, masks
        """
        num_features = 5
        sample_rate = args.sample_rate
        win_size = int(args.frame_size)
        ovl_size = int(win_size / 2)

        if ds_name == "cmidss":

            ds = CmidssDataset(ds_path=args.ds_path, target_rate=sample_rate)

            duration = int(ds.get_subject_duration(subject_id=subject_id) * sample_rate)
            duration = max(0, duration - win_size)

            # Load signals
            ts = ds.load_signal_for_subject(subject_id, "TS", start=0, data_size=duration)
            enmo = ds.load_signal_for_subject(subject_id, "ENMO", start=0, data_size=duration)
            anglez = ds.load_signal_for_subject(subject_id, "ZANGLE", start=0, data_size=duration)
            sleep_labels = ds.load_sleep_stages_for_subject(subject_id, start=0, data_size=duration)

            # Clean signals
        else:
            raise NotImplementedError(f"Dataset {ds_name} not implemented")
        # END IF

        features = np.zeros(((duration - win_size) // ovl_size, num_features), dtype=np.float32)
        labels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
        masks = np.ones((duration - win_size) // ovl_size, dtype=np.int32)
        for i, start in enumerate(range(0, duration - win_size, ovl_size)):
            if i >= features.shape[0]:
                break

            stop = start + win_size
            ts_win = ts[start:stop]
            enmo_win = enmo[start:stop]
            anglez_win = anglez[start:stop]

            # Features
            tod_norm = np.nanmean(ts_win/(24*60*60))
            tod_cos = np.cos(2*np.pi*tod_norm)
            # mov_win = np.abs(mov_win)
            mov_mu, mov_std = np.nanmean(enmo_win), np.nanstd(enmo_win)
            angle_mu, angle_std = np.nanmean(anglez_win), np.nanstd(anglez_win)

            features[i] = [
                tod_cos,
                mov_mu, mov_std,
                angle_mu, angle_std
            ]
            labels[i] = sleep_labels[start:stop][-1]
        # END FOR

        return features, labels, masks
