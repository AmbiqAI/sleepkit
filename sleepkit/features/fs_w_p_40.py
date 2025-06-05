"""
# Feature set: FS-W-P-40

* Location: Wrist
* Sensors: PPG
* Features: 40
* Datasets: MESA
* Tasks: Sleep detect, staging, apnea

Classes:
    FS_W_P_40: Feature set FS-W-P-40
"""

import h5py
import numpy as np
import physiokit as pk
import scipy.signal
import neuralspot_edge as nse

from ..datasets import MesaDataset
from ..defines import TaskParams
from .featureset import FeatureSet

logger = nse.utils.setup_logger(__name__)


class FS_W_P_40(FeatureSet):
    """Feature set: FS-W-P-40
    Location: Wrist
    Sensors: PPG
    Features: 40
    Datasets: MESA
    Tasks: Sleep detect, staging, apnea
    """

    @staticmethod
    def name() -> str:
        """Feature set name."""
        return "FS-W-P-40"

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        return [f"ppg_freq_bin_{i + 1}" for i in range(40)]

    @staticmethod
    def generate_subject_features(subject_id: str, ds_name: str, params: TaskParams):
        """Generate features for given dataset and subject.

        Args:
            subject_id (str): Subject ID
            ds_name (str): Dataset name
            params (TaskParams): Feature generation parameters
        """

        try:
            sample_rate = params.feature.sampling_rate

            win_size = int(params.feature.frame_size)
            ovl_size = int(win_size / 2)

            # Load dataset specific signals
            if ds_name == "mesa":
                ds_params = next((ds.params for ds in params.datasets if ds.name == ds_name), {})
                ds = MesaDataset(
                    target_rate=sample_rate,
                    **ds_params,
                )

                # Read signals
                duration = ds.get_subject_duration(subject_id=subject_id) * sample_rate
                duration = int(max(0, duration - win_size))

                ppg = ds.load_signal_for_subject(subject_id, "Pleth", start=0, data_size=duration)
                qos = ds.load_signal_for_subject(subject_id, "OxStatus", start=0, data_size=duration)

                sleep_stages = ds.extract_sleep_stages(subject_id=subject_id)
                sleep_labels = ds.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=duration)

                sleep_apnea = ds.extract_sleep_apneas(subject_id=subject_id)
                apnea_labels = ds.apnea_events_to_mask(apnea_events=sleep_apnea, data_size=duration)

                # Clean signals
                ppg = pk.ppg.clean(ppg, lowcut=None, highcut=5.0, sample_rate=sample_rate)
            else:
                raise NotImplementedError(f"Dataset {ds_name} not implemented")
            # END IF

            # Extract features
            features = np.zeros(((duration - win_size) // ovl_size, 40), dtype=np.float32)
            slabels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
            alabels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
            masks = np.ones((duration - win_size) // ovl_size, dtype=np.int32)
            window_coefs = scipy.signal.windows.blackman(win_size)
            freqs = np.fft.fftfreq(win_size, d=1 / sample_rate)[: win_size // 2 + 1]
            for i, start in enumerate(range(0, duration - win_size, ovl_size)):
                stop = start + win_size
                ppg_win = ppg[start:stop]
                qos_win = qos[start:stop]

                if i >= features.shape[0]:
                    break
                # END IF

                if np.any(qos_win >= 2.8):
                    masks[i] = 0
                    continue
                # END IF

                # Apply blackman window
                ppg_win = ppg_win * window_coefs
                # Compute fft and get power spectrum
                ppg_win = np.abs(np.fft.rfft(ppg_win))
                # Compute frequency bins
                # Group power spectrum into frequency bands from 0 up to 4 by steps of 0.1
                freq_bins = [np.logical_and(freqs >= i, freqs < i + 0.1) for i in np.arange(0, 4, 0.1)]
                ppg_win = [ppg_win[bin].sum() for bin in freq_bins]
                # Normalize across bands
                ppg_win = ppg_win / np.sum(ppg_win)

                features[i] = ppg_win
                slabels[i] = np.nanmedian(sleep_labels[start:stop])
                alabels[i] = np.nanmedian(apnea_labels[start:stop])
            # END FOR

            with h5py.File(str(params.feature.save_path / ds_name / f"{subject_id}.h5"), "w") as h5:
                h5.create_dataset("/features", data=features, compression="gzip", compression_opts=6)
                h5.create_dataset(
                    "/stage_labels",
                    data=slabels,
                    compression="gzip",
                    compression_opts=6,
                )
                h5.create_dataset(
                    "/apnea_labels",
                    data=alabels,
                    compression="gzip",
                    compression_opts=6,
                )
                h5.create_dataset("/mask", data=masks, compression="gzip", compression_opts=6)
            # END WITH

        # pylint: disable=broad-except
        except Exception as err:
            logger.exception(f"Error computing features for subject {subject_id}", err)
