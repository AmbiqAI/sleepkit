import functools
import os
from multiprocessing import Pool

import h5py
import numpy as np
import physiokit as pk
import scipy.signal
from tqdm import tqdm

from .datasets import MesaDataset, YsywDataset
from .defines import SKFeatureParams
from .utils import setup_logger

logger = setup_logger(__name__)


class PoorSignalError(Exception):
    """Poor signal error."""


class NoSignalError(Exception):
    """No signal error."""


def get_feature_names_001() -> list[str]:
    """Get feature names for feature set 001.

    Returns:
        list[str]: Feature names
    """
    return [
        # 4 time-domain
        "hr_bpm",
        "hrv_td_mean_nn",
        "hrv_td_sd_nn",
        # "hrv_td_cv_sd",
        "hrv_td_median_nn",
        # 1 freq-domain
        # "hrv_fd_lf_pwr",
        # "hrv_fd_hf_pwr",
        "hrv_fd_lfhf_ratio",
        # 3 SpO2
        "spo2_mu",
        "spo2_std",
        "spo2_med",
        # "spo2_iqr",
        # 3 Mov
        "mov_mu",
        "mov_std",
        "mov_med",
        # "mov_iqr",
        # 1 RSP
        "rsp_bpm",
        # 2 QOS
        "qos_win",
        "hrv_qos",
        # "rsp_qos",
    ]


def get_feature_names_002() -> list[str]:
    """Get feature names for feature set 002.

    Returns:
        list[str]: Feature names
    """
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


def get_feature_names_003() -> list[str]:
    """Get feature names for feature set 003.

    Returns:
        list[str]: Feature names
    """
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


def get_feature_names(feature_set: str) -> list[str]:
    """Get feature names for feature set.

    Args:
        feature_set (str): Feature set name

    Returns:
        list[str]: Feature names
    """
    if feature_set == "fs001":
        return get_feature_names_001()
    if feature_set == "fs002":
        return get_feature_names_002()
    if feature_set == "fs003":
        return get_feature_names_003()
    raise NotImplementedError(f"Feature set {feature_set} not implemented")


def compute_features_001(
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
        leg = ds.load_signal_for_subject(subject_id, "Leg", start=0, data_size=duration)
        # ecg = ds.load_signal_for_subject(subject_id, "EKG", start=0, data_size=duration)
        # rsp = ds.load_signal_for_subject(subject_id, "Abdo", start=0, data_size=duration)

        sleep_stages = ds.extract_sleep_stages(subject_id=subject_id)
        sleep_labels = ds.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=duration)

        # Clean signals
        ppg = pk.ppg.clean(ppg, lowcut=0.5, highcut=3.0, sample_rate=sample_rate)
        mov = pk.signal.filter_signal(leg, lowcut=3, highcut=11, order=3, sample_rate=sample_rate)
        # ecg = pk.ecg.clean(ecg, lowcut=0.5, highcut=30, sample_rate=sample_rate)
        # rsp = pk.rsp.clean(rsp, sample_rate=sample_rate, lowcut=0.1, highcut=2.0)
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
        # ecg_win = ecg[start : stop]
        # rsp_win = rsp[start : stop]

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
            # 4 time-domain
            hr_bpm,
            hrv_td.mean_nn,
            hrv_td.sd_nn,
            # hrv_td.cv_sd,
            hrv_td.median_nn,
            # 1 freq-domain
            # hrv_fd.bands[0].total_power / hrv_fd.total_power,
            # hrv_fd.bands[1].total_power / hrv_fd.total_power,
            hrv_fd.bands[0].total_power / hrv_fd.bands[1].total_power,
            # 3 SpO2
            spo2_mu,
            spo2_std,
            spo2_med,
            # spo2_iqr,
            # 3 Mov
            mov_mu,
            mov_std,
            mov_med,
            # mov_iqr,
            # 1 RSP
            rsp_bpm,
            # 2 QOS
            qos_win.mean(),
            hrv_qos,
            # rsp_qos,
        ]
        labels[i] = sleep_labels[start:stop][-1]
    # END FOR

    return features, labels, masks


def compute_features_002(
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


def compute_features_003(
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


def compute_subject_features(ds_subject: tuple[str, str], args: SKFeatureParams):
    """Compute features for subject.

    Args:
        ds_subject (tuple[str, str]): Dataset name and subject ID
        args (SKFeatureParams): Feature generation parameters
    """
    ds_name, subject_id = ds_subject

    if args.feature_set == "fs001":
        compute_features = compute_features_001
    elif args.feature_set == "fs002":
        compute_features = compute_features_002
    elif args.feature_set == "fs003":
        compute_features = compute_features_003
    else:
        raise NotImplementedError(f"Feature set {args.feature_set} not implemented")
    # logger.info(f"Computing features for subject {subject_id}")
    try:
        features, labels, mask = compute_features(ds_name, subject_id=subject_id, args=args)

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

    f = functools.partial(compute_subject_features, args=args)
    with Pool(processes=args.data_parallelism) as pool:
        _ = list(tqdm(pool.imap(f, ds_subjects), total=len(ds_subjects)))
    # END WITH
