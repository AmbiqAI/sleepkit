import os
import random
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import physiokit as pk
import scipy.stats

from ..utils import load_pkl, save_pkl
from .dataset import SKDataset
from .mesa import MesaDataset


class OfsgDataset(SKDataset):
    """Online feature set generator."""

    def __init__(self, ds_path: Path, frame_size: int = 128, mask_threshold: float = 0.90, **kwargs) -> None:
        super().__init__(ds_path, frame_size)
        self.ds_path = ds_path
        self.frame_size = frame_size
        self.mask_threshold = mask_threshold
        self.mesa = MesaDataset(ds_path=ds_path, frame_size=frame_size, target_rate=64, is_commercial=True)

    @property
    def subject_ids(self) -> list[str]:
        """Get dataset subject IDs

        Returns:
            list[str]: Subject IDs
        """
        return self.mesa.subject_ids

    @property
    def train_subject_ids(self) -> list[str]:
        """Get train subject ids"""
        return self.mesa.train_subject_ids

    @property
    def test_subject_ids(self) -> list[str]:
        """Get test subject ids"""
        return self.mesa.test_subject_ids

    @property
    def feature_shape(self) -> tuple[int, int]:
        """Get feature shape"""
        x, _, _ = self.load_subject_data(subject_id=self.subject_ids[0])
        return (self.mesa.frame_size, x.shape[-1])

    def compute_features(self, subject_id: str):
        """Compute features"""

        # f, t, sxx = scipy.signal.spectrogram(
        #     x=ppg,
        #     fs=self.mesa.target_rate,
        #     nperseg=8*self.mesa.target_rate,
        #     noverlap=4*self.mesa.target_rate,
        #     nfft=8*self.mesa.target_rate
        # )
        # mask = np.where(qos >= 2, 0, 1)

        # l_idx = np.where(f >= 0.2)[0][0]
        # r_idx = np.where(f >= 4.0)[0][0]
        # x = sxx[l_idx:r_idx]
        # x = ((x - np.nanmean(x, axis=0))/np.nanstd(x, axis=0)).T
        # # y = scipy.signal.medfilt(labels, kernel_size=8*self.mesa.target_rate+1)
        # y = labels[::int(4*self.mesa.target_rate)][:x.shape[0]]
        # mask = mask[::int(4*self.mesa.target_rate)][:x.shape[0]]

        win_size = int(60 * self.mesa.target_rate)
        ovl_size = int(30 * self.mesa.target_rate)

        # Read signals
        duration = max(
            0,
            self.mesa.get_subject_duration(subject_id=subject_id) * self.mesa.target_rate - 60 * self.mesa.target_rate,
        )
        ecg = self.mesa.load_signal_for_subject(subject_id, "EKG", start=0, data_size=duration)
        # ppg = self.mesa.load_signal_for_subject(subject_id, "Pleth", start=0, data_size=duration)
        qos = self.mesa.load_signal_for_subject(subject_id, "OxStatus", start=0, data_size=duration)
        spo2 = self.mesa.load_signal_for_subject(subject_id, "SpO2", start=0, data_size=duration)
        rsp = self.mesa.load_signal_for_subject(subject_id, "Abdo", start=0, data_size=duration)
        leg = self.mesa.load_signal_for_subject(subject_id, "Leg", start=0, data_size=duration)

        sleep_stages = self.mesa.extract_sleep_stages(subject_id=subject_id)
        sleep_labels = self.mesa.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=duration)

        # Clean signals
        ecg = pk.ecg.clean(ecg, lowcut=0.5, highcut=30, sample_rate=self.mesa.target_rate, order=5)
        rsp = pk.rsp.clean_signal(rsp, sample_rate=self.mesa.target_rate)
        mov = pk.signal.filter_signal(leg, lowcut=3, highcut=11, order=3, sample_rate=self.mesa.target_rate)
        spo2 = np.clip(spo2, 50, 100)

        # Extract features
        features = np.zeros(((duration - win_size) // ovl_size, 18 + 8 + 4 + 4 + 1 + 1), dtype=np.float32)
        labels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
        masks = np.ones((duration - win_size) // ovl_size, dtype=np.int32)
        for i, start in enumerate(range(0, duration - win_size, ovl_size)):
            ecg_win = ecg[start : start + win_size]
            rsp_win = rsp[start : start + win_size]
            mov_win = mov[start : start + win_size]
            spo2_win = spo2[start : start + win_size]

            if i >= features.shape[0]:
                break

            if np.any(qos[start : start + win_size] >= 3):
                masks[i] = 0
                continue
            # END IF

            # Extract peaks and RR intervals
            rpeaks = pk.ppg.find_peaks(ecg_win, sample_rate=self.mesa.target_rate)
            rri = pk.ppg.compute_rr_intervals(rpeaks)
            rri_mask = pk.ppg.filter_rr_intervals(rri, sample_rate=self.mesa.target_rate)
            if rpeaks[rri_mask == 0].size < 4 or rri[rri_mask == 0].size < 4:
                masks[i] = 0
                continue
            # END IF

            # HRV metrics
            hrv_td = pk.hrv.compute_hrv_time(rri[rri_mask == 0], sample_rate=self.mesa.target_rate)
            freq_bands = [(0.04, 0.15), (0.15, 0.4), (0.4, 0.5)]
            hrv_fd = pk.hrv.compute_hrv_frequency(
                rpeaks[rri_mask == 0], rri=rri[rri_mask == 0], bands=freq_bands, sample_rate=self.mesa.target_rate
            )

            rsp_bpm = pk.rsp.compute_respiratory_rate_from_fft(
                rsp_win, sample_rate=self.mesa.target_rate, lowcut=0.05, highcut=2
            )
            rsp_bpm = np.clip(rsp_bpm, 3, 120)

            spo2_mu, spo2_std = np.nanmean(spo2_win), np.nanstd(spo2_win)
            spo2_med, spo2_iqr = np.nanmedian(spo2_win), scipy.stats.iqr(spo2_win)

            mov_mu, mov_std = np.nanmean(mov_win), np.nanstd(mov_win)
            mov_med, mov_iqr = np.nanmedian(mov_win), scipy.stats.iqr(mov_win)

            features[i] = [
                # 18 time-domain
                hrv_td.mean_nn,
                hrv_td.sd_nn,
                hrv_td.rms_sd,
                hrv_td.sd_sd,
                hrv_td.cv_nn,
                hrv_td.cv_sd,
                hrv_td.meadian_nn,
                hrv_td.mad_nn,
                hrv_td.mcv_nn,
                hrv_td.iqr_nn,
                hrv_td.prc20_nn,
                hrv_td.prc80_nn,
                hrv_td.nn50,
                hrv_td.nn20,
                hrv_td.pnn50,
                hrv_td.pnn20,
                hrv_td.min_nn,
                hrv_td.max_nn,
                # 8 freq-domain
                hrv_fd.bands[0].peak_frequency,
                hrv_fd.bands[1].peak_frequency,
                hrv_fd.bands[2].peak_frequency,
                hrv_fd.bands[0].peak_power / hrv_fd.total_power,
                hrv_fd.bands[1].peak_power / hrv_fd.total_power,
                hrv_fd.bands[2].peak_power / hrv_fd.total_power,
                hrv_fd.bands[0].total_power / hrv_fd.bands[1].total_power,
                hrv_fd.total_power,
                # 4 SpO2
                spo2_mu,
                spo2_std,
                spo2_med,
                spo2_iqr,
                # 4 Mov
                mov_mu,
                mov_std,
                mov_med,
                mov_iqr,
                # 1 RSP
                rsp_bpm,
                # 1 QOS
                qos[start : start + win_size].mean(),
            ]
            labels[i] = sleep_labels[start : start + win_size][win_size // 2]
        # END FOR

        return features, labels, masks

    def uniform_subject_generator(
        self,
        subject_ids: list[str] | None = None,
        repeat: bool = True,
        shuffle: bool = True,
    ):
        """Yield data for each subject in the array.

        Args:
            subject_ids (pt.ArrayLike): Array of subject ids
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle subject ids.. Defaults to True.

        Returns:
            SubjectGenerator: Subject generator

        Yields:
            Iterator[SubjectGenerator]
        """
        if subject_ids is None:
            subject_ids = self.subject_ids
        subject_idxs = list(range(len(subject_ids)))
        while True:
            if shuffle:
                random.shuffle(subject_idxs)
            for subject_idx in subject_idxs:
                subject_id = subject_ids[subject_idx]
                subject_id = subject_id.decode("ascii") if isinstance(subject_id, bytes) else subject_id
                yield subject_id
            # END FOR
            if not repeat:
                break
        # END WHILE

    def load_subject_data(
        self, subject_id: str, normalize: bool = True, epsilon: float = 1e-6
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Load subject data
        Args:
            subject_id (str): Subject ID
        Returns:
            tuple[npt.NDArray, npt.NDArray]: Tuple of features and labels
        """
        fpath = Path(tempfile.gettempdir(), f"sk-hdf52-{subject_id}.pkl")
        if os.path.exists(fpath):
            data = load_pkl(fpath, compress=True)
            x = data["x"]
            y = data["y"]
            mask = data["mask"]
        else:
            x, y, mask = self.compute_features(subject_id=subject_id)
            save_pkl(fpath, compress=True, x=x, y=y, mask=mask)
        # END IF

        if normalize:
            x_mu = np.nanmean(x[mask == 1], axis=0)
            x_std = np.nanstd(x[mask == 1], axis=0)
            x = (x - x_mu) / (x_std + epsilon)
            x[mask == 0, :] = 0
        # END IF

        return x, y, mask

    def signal_generator(
        self, subject_generator, samples_per_subject: int = 1, normalize: bool = True, epsilon: float = 1e-6
    ):
        """
        Generate frames using subject generator.
        from the segments in subject data by placing a frame in a random location within one of the segments.
        Args:
            subject_generator (SubjectGenerator): Generator that yields a tuple of subject id and subject data.
                    subject data may contain only signals, since labels are not used.
            samples_per_subject (int): Samples per subject.
        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """
        for subject_id in subject_generator:
            x, y, mask = self.load_subject_data(subject_id=subject_id, normalize=normalize, epsilon=epsilon)
            num_samples = 0
            num_attempts = 0
            while num_samples < samples_per_subject:
                frame_start = np.random.randint(x.shape[0] - self.mesa.frame_size)
                frame_end = frame_start + self.frame_size
                xx = x[frame_start:frame_end]
                yy = y[frame_start:frame_end]

                if np.isnan(xx).any():
                    num_attempts += 1
                    if num_attempts > 10:
                        num_attempts = 0
                        num_samples += 1
                    continue
                if mask[frame_start:frame_end].mean() < self.mask_threshold:
                    num_attempts += 1
                    if num_attempts > 10:
                        num_attempts = 0
                        num_samples += 1
                    continue
                num_samples += 1
                num_attempts = 0
                yield xx, yy
            # END FOR
        # END FOR
