"""
# Feature set: FS-W-PA-14

* Location: Wrist
* Sensors: PPG, accelerometer
* Features: 14
* Datasets: MESA
* Tasks: Sleep detect, staging, apnea

Classes:
    FS_W_PA_14: Feature set FS-W-PA-14

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


class FS_W_PA_14(FeatureSet):
    """Feature set: FS-W-PA-14
    Location: Wrist
    Sensors: PPG, accelerometer
    Features: 14
    Datasets: MESA
    Tasks: Sleep detect, staging, apnea
    """

    @staticmethod
    def name() -> str:
        """Feature set name."""
        return "FS-W-PA-14"

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        return [
            # 5 HRV
            "hr_bpm",
            "hrv_td_mean_nn",
            "hrv_td_sd_nn",
            "hrv_td_median_nn",
            "hrv_fd_lfhf_ratio",
            # 3 SpO2
            "spo2_mu",
            "spo2_std",
            "spo2_med",
            # 3 Mov
            "mov_mu",
            "mov_std",
            "mov_med",
            # "mov_iqr",
            # 1 RSP
            "rsp_bpm",
            # 2 QOS
            "spo2_qos",
            "hrv_qos",
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
                spo2 = ds.load_signal_for_subject(subject_id, "SpO2", start=0, data_size=duration)
                leg = ds.load_signal_for_subject(subject_id, "Leg", start=0, data_size=duration)

                sleep_stages = ds.extract_sleep_stages(subject_id=subject_id)
                sleep_labels = ds.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=duration)

                sleep_apnea = ds.extract_sleep_apneas(subject_id=subject_id)
                apnea_labels = ds.apnea_events_to_mask(apnea_events=sleep_apnea, data_size=duration)

                # Clean signals
                ppg = pk.ppg.clean(ppg, lowcut=0.5, highcut=3.0, sample_rate=sample_rate)
                mov = pk.signal.filter_signal(leg, lowcut=3, highcut=11, order=3, sample_rate=sample_rate)
                spo2 = np.clip(spo2, 50, 100)
            else:
                raise NotImplementedError(f"Dataset {ds_name} not implemented")
            # END IF

            # Extract features
            features = np.zeros(((duration - win_size) // ovl_size, 14), dtype=np.float32)
            slabels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
            alabels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
            masks = np.ones((duration - win_size) // ovl_size, dtype=np.int32)
            for i, start in enumerate(range(0, duration - win_size, ovl_size)):
                stop = start + win_size

                ppg_win = ppg[start:stop]
                mov_win = mov[start:stop]
                spo2_win = spo2[start:stop]
                qos_win = qos[start:stop]

                if i >= features.shape[0]:
                    break

                if np.any(qos_win >= 2.8):
                    masks[i] = 0
                    continue
                # END IF

                hr_bpm, _ = pk.ppg.compute_heart_rate_from_fft(
                    ppg_win, sample_rate=sample_rate, lowcut=0.5, highcut=2.0
                )

                # Extract peaks and RR intervals
                rpeaks = pk.ppg.find_peaks(ppg_win, sample_rate=sample_rate)
                rri = pk.ppg.compute_rr_intervals(rpeaks)
                rri_mask = pk.ppg.filter_rr_intervals(rri, sample_rate=sample_rate, min_rr=0.5, max_rr=2, min_delta=0.3)
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
                rsp_bpm, _ = pk.ppg.derive_respiratory_rate(
                    ppg=ppg_win,
                    peaks=rpeaks,
                    rri=rri,
                    method="rifv",
                    lowcut=0.15,
                    highcut=2.0,
                    sample_rate=sample_rate,
                )
                rsp_bpm = np.clip(rsp_bpm, 3, 120)

                # SpO2 metrics
                spo2_mu, spo2_std = np.nanmean(spo2_win), np.nanstd(spo2_win)
                spo2_med, _ = np.nanmedian(spo2_win), scipy.stats.iqr(spo2_win)

                # Movement metrics
                mov_win = np.abs(mov_win)
                mov_mu, mov_std = np.nanmean(mov_win), np.nanstd(mov_win)
                mov_med, _ = np.nanmedian(mov_win), scipy.stats.iqr(mov_win)

                features[i] = [
                    # 5 HRV
                    hr_bpm,
                    hrv_td.mean_nn,
                    hrv_td.sd_nn,
                    hrv_td.median_nn,
                    hrv_fd.bands[0].total_power / hrv_fd.bands[1].total_power,
                    # 3 SpO2
                    spo2_mu,
                    spo2_std,
                    spo2_med,
                    # 3 Mov
                    mov_mu,
                    mov_std,
                    mov_med,
                    # 1 RSP
                    rsp_bpm,
                    # 2 QOS
                    qos_win.mean(),
                    hrv_qos,
                    # rsp_qos,
                ]
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
