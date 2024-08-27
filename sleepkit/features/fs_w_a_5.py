"""

#Feature set: FS-W-A-5

* Location: Wrist
* Sensors: Accelerometer
* Features: 5
* Datasets: CMIDSS
* Tasks: Sleep detect

Classes:
    FS_W_A_5: Feature set FS-W-A-5

"""

import h5py
import numpy as np
import neuralspot_edge as nse

from ..datasets import CmidssDataset
from ..defines import TaskParams
from .featureset import FeatureSet

logger = nse.utils.setup_logger(__name__)


class FS_W_A_5(FeatureSet):
    """Feature set: FS-W-A-5
    Location: Wrist
    Sensors: Accelerometer
    Features: 5
    Datasets: CMIDSS
    Tasks: Sleep detect
    """

    @staticmethod
    def name() -> str:
        """Feature set name."""
        return "FS-W-A-5"

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        return [
            "tod",  # Time of day (cosine encoding)
            "mov_mu",  # Mean of movement
            "mov_std",  # Standard deviation of movement
            "angle_mu",  # Mean of z-angle
            "angle_std",  # Standard deviation of z-angle
        ]

    @staticmethod
    def generate_subject_features(subject_id: str, ds_name: str, params: TaskParams):
        """Generate features for given dataset and subject.

        Args:
            subject_id (str): Subject ID
            ds_name (str): Dataset name
            params (TaskParams): Feature generation parameters
        """

        try:
            num_features = 5
            sample_rate = params.feature.sampling_rate
            win_size = int(params.feature.frame_size)
            ovl_size = int(win_size / 2)

            if ds_name == "cmidss":
                ds_params = next((ds.params for ds in params.datasets if ds.name == ds_name), {})
                ds = CmidssDataset(target_rate=sample_rate, **ds_params)

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
                tod_norm = np.nanmean(ts_win / (24 * 60 * 60))
                tod_cos = np.cos(2 * np.pi * tod_norm)
                # mov_win = np.abs(mov_win)
                mov_mu, mov_std = np.nanmean(enmo_win), np.nanstd(enmo_win)
                angle_mu, angle_std = np.nanmean(anglez_win), np.nanstd(anglez_win)

                features[i] = [tod_cos, mov_mu, mov_std, angle_mu, angle_std]
                labels[i] = sleep_labels[start:stop][-1]
            # END FOR

            with h5py.File(str(params.feature.save_path / ds_name / f"{subject_id}.h5"), "w") as h5:
                h5.create_dataset("/features", data=features, compression="gzip", compression_opts=6)
                h5.create_dataset(
                    "/detect_labels",
                    data=labels,
                    compression="gzip",
                    compression_opts=6,
                )
                h5.create_dataset("/mask", data=masks, compression="gzip", compression_opts=6)
            # END WITH

        # pylint: disable=broad-except
        except Exception as err:
            logger.exception(f"Error computing features for subject {subject_id}", err)
