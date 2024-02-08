import numpy as np
import physiokit as pk
import scipy.signal

from ..datasets import MesaDataset, YsywDataset
from ..defines import SKFeatureParams
from ..utils import setup_logger

logger = setup_logger(__name__)


class FeatSet02:
    """Feature set 2."""

    @staticmethod
    def name() -> str:
        """Feature set name."""
        return "fs002"

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
            # 3 MOV
            "mov_mu",
            "mov_std",
            "mov_med",
            # 1 RSP
            "rsp_bpm",
            # 2 QOS
            "qos_win",
            "hrv_qos",
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
        if ds_name == "mesa":
            ds = MesaDataset(ds_path=args.ds_path, is_commercial=True, target_rate=sample_rate)

            # Read signals
            duration = ds.get_subject_duration(subject_id=subject_id) * sample_rate
            duration = max(0, duration - win_size)

            ppg = ds.load_signal_for_subject(subject_id, "Pleth", start=0, data_size=duration)
            qos = ds.load_signal_for_subject(subject_id, "OxStatus", start=0, data_size=duration)
            spo2 = ds.load_signal_for_subject(subject_id, "SpO2", start=0, data_size=duration)
            # leg = ds.load_signal_for_subject(subject_id, "Leg", start=0, data_size=duration)
            # ecg = ds.load_signal_for_subject(subject_id, "EKG", start=0, data_size=duration)
            rsp = ds.load_signal_for_subject(subject_id, "Abdo", start=0, data_size=duration)

            sleep_stages = ds.extract_sleep_stages(subject_id=subject_id)
            sleep_labels = ds.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=duration)

            # Clean signals
            ppg = pk.ppg.clean(ppg, lowcut=0.5, highcut=3.0, sample_rate=sample_rate)
            mov = pk.signal.filter_signal(rsp, lowcut=2, highcut=11, order=3, sample_rate=sample_rate)
            spo2 = np.clip(spo2, 50, 100)

        elif ds_name == "ysyw":
            ds = YsywDataset(ds_path=args.ds_path, target_rate=sample_rate)

            ecg = ds.load_signal_for_subject(subject_id, "ECG", start=0, data_size=duration)
            rsp = ds.load_signal_for_subject(subject_id, "ABD", start=0, data_size=duration)
            spo2 = ds.load_signal_for_subject(subject_id, "SaO2", start=0, data_size=duration)
            sleep_labels = ds.load_sleep_stages_for_subject(subject_id, start=0, data_size=duration)

            # Clean signals
            ecg = pk.ppg.clean(ecg, lowcut=0.5, highcut=30, sample_rate=sample_rate)
            mov = pk.signal.filter_signal(rsp, lowcut=2, highcut=11, order=3, sample_rate=sample_rate)
            spo2 = np.clip(spo2, 50, 100)
        else:
            raise NotImplementedError(f"Dataset {ds_name} not implemented")
        # END IF

        # Extract features
        features = np.zeros(((duration - win_size) // ovl_size, 14), dtype=np.float32)
        labels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
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

            hr_bpm, _ = pk.ppg.compute_heart_rate_from_fft(ppg_win, sample_rate=sample_rate, lowcut=0.5, highcut=2.0)

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

            # Circadian metrics
            # ts = time.strptime(tod[0], "%H:%M:%S")
            # tod_norm = (ts.tm_hour * 60 * 60 + ts.tm_min * 60 + ts.tm_sec) / (24 * 60 * 60)
            # tod_cos = np.cos(2 * np.pi * tod_norm)

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
                # 3 MOV
                mov_mu,
                mov_std,
                mov_med,
                # 1 RSP
                rsp_bpm,
                # 2 QOS
                qos_win.mean(),
                hrv_qos,
            ]
            labels[i] = sleep_labels[start:stop][-1]
        # END FOR

        return features, labels, masks
