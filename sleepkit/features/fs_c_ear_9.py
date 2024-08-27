"""

# Feature set: FS-C-EAR-9

* Location: Chest
* Sensors: ECG, respiratory, accelerometer
* Features: 9
* Datasets: YSYW
* Tasks: Sleep detect, staging

Classes:
    FS_C_EAR_9: Feature set FS-C-EAR-9

"""

import h5py
import numpy as np
import physiokit as pk
import scipy.signal
import neuralspot_edge as nse
from ..datasets import YsywDataset
from ..defines import TaskParams
from .featureset import FeatureSet

logger = nse.utils.setup_logger(__name__)


class FS_C_EAR_9(FeatureSet):
    """Feature set: FS-C-EAR-9
    Location: Chest
    Sensors: ECG, respiratory, accelerometer
    Features: 9
    Datasets: YSYW
    Tasks: Sleep detect, staging
    """

    @staticmethod
    def name() -> str:
        """Feature set name."""
        return "FS-C-EAR-9"

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        return [
            # 4 HRV
            "hrv_td_mean_nn",  # Mean of NN intervals
            "hrv_td_sd_nn",  # Standard deviation of NN intervals
            "hrv_td_median_nn",  # Median of NN intervals
            "hrv_fd_lfhf_ratio",  # LF/HF ratio
            # 3 MOV
            "mov_mu",  # Mean of movement
            "mov_std",  # Standard deviation of movement
            "mov_med",  # Median of movement
            # 1 RSP
            "rsp_bpm",  # Respiratory rate (BPM)
            # 1 QOS
            "hrv_qos",  # Quality of signal
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
            sample_rate = params.feature.sampling_rate

            win_size = int(params.feature.frame_size)
            ovl_size = int(win_size / 2)

            # Load dataset specific signals
            if ds_name == "ysyw":
                ds_params = next((ds.params for ds in params.datasets if ds.name == ds_name), {})
                ds = YsywDataset(target_rate=sample_rate, **ds_params)

                duration = int(ds.get_subject_duration(subject_id=subject_id) * sample_rate)
                duration = max(0, duration - win_size)

                # Load signals
                ecg = ds.load_signal_for_subject(subject_id, "ECG", start=0, data_size=duration)
                rsp = ds.load_signal_for_subject(subject_id, "ABD", start=0, data_size=duration)

                sleep_labels = ds.load_sleep_stages_for_subject(subject_id, start=0, data_size=duration)

                # Clean signals
                ecg = pk.ppg.clean(ecg, lowcut=0.5, highcut=30, sample_rate=sample_rate)
                mov = pk.signal.filter_signal(rsp, lowcut=2, highcut=11, order=3, sample_rate=sample_rate)
            else:
                raise NotImplementedError(f"Dataset {ds_name} not implemented")
            # END IF

            # Extract features
            features = np.zeros(((duration - win_size) // ovl_size, 9), dtype=np.float32)
            slabels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
            masks = np.ones((duration - win_size) // ovl_size, dtype=np.int32)
            for i, start in enumerate(range(0, duration - win_size, ovl_size)):
                stop = start + win_size

                ecg_win = ecg[start:stop]
                mov_win = mov[start:stop]

                if i >= features.shape[0]:
                    break

                # Extract peaks and RR intervals
                rpeaks = pk.ecg.find_peaks(ecg_win, sample_rate=sample_rate)
                rri = pk.ecg.compute_rr_intervals(rpeaks)
                rri_mask = pk.ecg.filter_rr_intervals(rri, sample_rate=sample_rate, min_rr=0.5, max_rr=2, min_delta=0.3)
                rpeaks = rpeaks[rri_mask == 0]
                rri = rri[rri_mask == 0]
                if rpeaks.size < 4 or rri.size < 4:
                    masks[i] = 0
                    continue
                # END IF
                hrv_qos = 1 - rri_mask.sum() / (rri_mask.size or 1)

                # HRV metrics
                hrv_td = pk.hrv.compute_hrv_time(rri, sample_rate=sample_rate)
                freq_bands = [(0.04, 0.15), (0.15, 0.4)]
                hrv_fd = pk.hrv.compute_hrv_frequency(rpeaks, rri=rri, bands=freq_bands, sample_rate=sample_rate)

                # RSP metrics
                rsp_bpm, _ = pk.ecg.derive_respiratory_rate(
                    peaks=rpeaks,
                    rri=rri,
                    method="rifv",
                    lowcut=0.15,
                    highcut=2.0,
                    sample_rate=sample_rate,
                )
                rsp_bpm = np.clip(rsp_bpm, 3, 120)

                # MOV metrics
                mov_win = np.abs(mov_win)
                mov_mu, mov_std = np.nanmean(mov_win), np.nanstd(mov_win)
                mov_med, _ = np.nanmedian(mov_win), scipy.stats.iqr(mov_win)

                features[i] = [
                    # 4 HRV
                    hrv_td.mean_nn,
                    hrv_td.sd_nn,
                    hrv_td.median_nn,
                    hrv_fd.bands[0].total_power / hrv_fd.bands[1].total_power,
                    # 3 MOV
                    mov_mu,
                    mov_std,
                    mov_med,
                    # 1 RSP
                    rsp_bpm,
                    # 1 QOS
                    hrv_qos,
                ]
                slabels[i] = sleep_labels[start:stop][-1]
            # END FOR

            with h5py.File(str(params.feature.save_path / ds_name / f"{subject_id}.h5"), "w") as h5:
                h5.create_dataset("/features", data=features, compression="gzip", compression_opts=6)
                h5.create_dataset(
                    "/stage_labels",
                    data=slabels,
                    compression="gzip",
                    compression_opts=6,
                )
                h5.create_dataset("/mask", data=masks, compression="gzip", compression_opts=6)
            # END WITH

        # pylint: disable=broad-except
        except Exception as err:
            logger.exception(f"Error computing features for subject {subject_id}", err)
