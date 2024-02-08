import numpy as np
import physiokit as pk
import scipy.signal

from ..datasets import YsywDataset
from ..defines import SKFeatureParams
from ..utils import setup_logger

logger = setup_logger(__name__)


class FeatSet03:
    """Feature set 3."""

    @staticmethod
    def name() -> str:
        """Feature set name."""
        return "fs003"

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        return [
            # 4 HRV
            "hrv_td_mean_nn",
            "hrv_td_sd_nn",
            "hrv_td_median_nn",
            "hrv_fd_lfhf_ratio",
            # 3 SpO2
            "spo2_mu",
            "spo2_std",
            "spo2_med",
            # 3 MOV
            "mov_mu",
            "mov_std",
            "mov_med",
            # 1 RSP
            "rsp_bpm",
            # 1 QOS
            "qos_win",
        ]

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
        sample_rate = args.sample_rate

        win_size = int(args.frame_size)
        ovl_size = int(win_size / 2)

        # Load dataset specific signals
        if ds_name == "ysyw":
            ds = YsywDataset(ds_path=args.ds_path, target_rate=sample_rate)

            duration = int(ds.get_subject_duration(subject_id=subject_id) * sample_rate)
            duration = max(0, duration - win_size)

            # Load signals
            ecg = ds.load_signal_for_subject(subject_id, "ECG", start=0, data_size=duration)
            rsp = ds.load_signal_for_subject(subject_id, "ABD", start=0, data_size=duration)
            spo2 = ds.load_signal_for_subject(subject_id, "SaO2", start=0, data_size=duration)
            sleep_labels = ds.load_sleep_stages_for_subject(subject_id, start=0, data_size=duration)

            # Clean signals
            ecg = pk.ppg.clean(ecg, lowcut=0.5, highcut=30, sample_rate=sample_rate)
            mov = pk.signal.filter_signal(rsp, lowcut=2, highcut=11, order=3, sample_rate=sample_rate)
            spo2 = np.clip(100 * spo2 / 32768, 50, 100)
        else:
            raise NotImplementedError(f"Dataset {ds_name} not implemented")
        # END IF

        # Extract features
        features = np.zeros(((duration - win_size) // ovl_size, 12), dtype=np.float32)
        labels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
        masks = np.ones((duration - win_size) // ovl_size, dtype=np.int32)
        for i, start in enumerate(range(0, duration - win_size, ovl_size)):
            stop = start + win_size

            ecg_win = ecg[start:stop]
            mov_win = mov[start:stop]
            spo2_win = spo2[start:stop]

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

            # SpO2 metrics
            spo2_mu, spo2_std = np.nanmean(spo2_win), np.nanstd(spo2_win)
            spo2_med, _ = np.nanmedian(spo2_win), scipy.stats.iqr(spo2_win)

            # Movement metrics
            mov_win = np.abs(mov_win)
            mov_mu, mov_std = np.nanmean(mov_win), np.nanstd(mov_win)
            mov_med, _ = np.nanmedian(mov_win), scipy.stats.iqr(mov_win)

            features[i] = [
                # 4 HRV
                hrv_td.mean_nn,
                hrv_td.sd_nn,
                hrv_td.median_nn,
                hrv_fd.bands[0].total_power / hrv_fd.bands[1].total_power,
                # 3 SpO2
                spo2_mu,
                spo2_std,
                spo2_med,
                # 3 MOV
                mov_mu,
                mov_std,
                mov_med,
                # 1 RSP
                rsp_bpm,
                # 1 QOS
                hrv_qos,
            ]
            labels[i] = sleep_labels[start:stop][-1]
        # END FOR

        return features, labels, masks
