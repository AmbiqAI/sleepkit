import os
import functools
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import scipy.stats as sps
import physiokit as pk
from tqdm import tqdm

from .datasets import MesaDataset
from .defines import SKFeatureParams
from .utils import setup_logger

logger = setup_logger(__name__)

class PoorSignalError(Exception):
    pass

def compute_features_001(ds: MesaDataset, subject_id: str, start: int):

    ppg = ds.load_signal_for_subject(subject_id, "Pleth", start=start, data_size=ds.frame_size)
    spo2 = ds.load_signal_for_subject(subject_id, "SpO2", start=start, data_size=ds.frame_size)
    rsp = ds.load_signal_for_subject(subject_id, "Abdo", start=start, data_size=ds.frame_size)
    leg = ds.load_signal_for_subject(subject_id, "Leg", start=start, data_size=ds.frame_size)
    qos = ds.load_signal_for_subject(subject_id, "OxStatus", start=start, data_size=ds.frame_size)

    # Handle if signal is marked as "bad"
    if np.any(qos >= 3) or np.mean(qos) >= 1.5:
        raise PoorSignalError(f"Bad signal qos={np.mean(qos):.2f} {np.any(qos >= 3)} for subject {subject_id}")

    #### Preprocess signals
    ppg = pk.ppg.clean(ppg, sample_rate=ds.target_rate)
    rsp = pk.rsp.clean(rsp, sample_rate=ds.target_rate)
    mov = pk.signal.filter_signal(leg, lowcut=3, highcut=11, order=3, sample_rate=ds.target_rate)

    spo2 = np.clip(spo2, 50, 100)

    hr_bpm = pk.ppg.compute_heart_rate_from_fft(ppg, sample_rate=ds.target_rate, lowcut=0.5, highcut=2.0)
    hr_bpm = np.clip(hr_bpm, 30, 120)
    min_rr, max_rr = max(0.25, 60/hr_bpm - 0.25), min(2.5, 60/hr_bpm + 0.25)

    rsp_bpm = pk.rsp.compute_respiratory_rate_from_fft(rsp, sample_rate=ds.target_rate, lowcut=0.05, highcut=2)
    rsp_bpm = np.clip(rsp_bpm, 3, 120)

    # HRV metrics
    rpeaks = pk.ppg.find_peaks(ppg, sample_rate=ds.target_rate)
    rri = pk.ppg.compute_rr_intervals(rpeaks)
    rri_mask = pk.ppg.filter_rr_intervals(rri, sample_rate=ds.target_rate, min_rr=min_rr, max_rr=max_rr)

    if rpeaks.size < 4 or rri[rri_mask == 0].size < 4:
        raise PoorSignalError("Not enough peaks")

    hrv_td_metrics = pk.hrv.compute_hrv_time(rri[rri_mask == 0], sample_rate=ds.target_rate)
    freq_bands = [(0.04, 0.15), (0.15, 0.4)]
    hrv_fd_metrics = pk.hrv.compute_hrv_frequency(
        rpeaks[rri_mask == 0],
        rri=rri[rri_mask == 0],
        bands=freq_bands,
        sample_rate=ds.target_rate
    )

    #### Compute features
    spo2_mu, spo2_std = np.nanmean(spo2), np.nanstd(spo2)
    spo2_med, spo2_iqr = np.nanmedian(spo2), sps.iqr(spo2)

    mov_mu, mov_std = np.nanmean(mov), np.nanstd(mov)
    mov_med, mov_iqr = np.nanmedian(mov), sps.iqr(mov)

    rri_mu, rri_std  = hrv_td_metrics.mean_nn, hrv_td_metrics.sd_nn
    rri_med, rri_iqr = hrv_td_metrics.meadian_nn, hrv_td_metrics.iqr_nn
    rri_sd_rms, rri_sd_std = hrv_td_metrics.rms_sd, hrv_td_metrics.sd_sd

    hrv_lf = hrv_fd_metrics.bands[0].total_power
    hrv_hf = hrv_fd_metrics.bands[1].total_power
    hrv_lfhf = hrv_lf/hrv_hf

    features = np.array([
        spo2_mu, spo2_std, spo2_med, spo2_iqr,
        mov_mu, mov_std, mov_med, mov_iqr,
        rri_mu, rri_std, rri_med, rri_iqr, rri_sd_rms, rri_sd_std,
        hr_bpm, rsp_bpm,
        hrv_lf, hrv_hf, hrv_lfhf
    ])
    return features

def compute_subject_features_001(subject_id: str, args: SKFeatureParams):

    args.sample_rate = getattr(args, "sample_rate", 64)
    args.frame_size = getattr(args, "frame_size", args.sample_rate*30)

    feat_names = [
        "SPO2-mu", "SPO2-std", "SPO2-med", "SPO2-iqr",
        "MOV-mu", "MOV-std", "MOV-med", "MOV-iqr",
        "RRI-mu", "RRI-std", "RRI-med", "RRI-iqr", "RRI-sd-rms", "RRI-sd-std",
        "HR-bpm", "RSP-bpm",
        "HRV-lf", "HRV-hf", "HRV-lfhf",
    ]

    ds = MesaDataset(
        args.ds_path,
        frame_size=args.frame_size,
        target_rate=args.sample_rate,
        is_commercial=True
    )

    subject_duration = max(0, ds.get_subject_duration(subject_id=subject_id)*ds.target_rate)
    sleep_stages = ds.extract_sleep_stages(subject_id=subject_id)
    apnea_events = ds.extract_sleep_apneas(subject_id=subject_id)
    sleep_mask = ds.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=subject_duration)
    apnea_mask = ds.apnea_events_to_mask(apnea_events=apnea_events, data_size=subject_duration)
    features = np.zeros((subject_duration // args.frame_size, len(feat_names)), dtype=np.float32)
    sleep_stages = np.zeros((subject_duration // args.frame_size), dtype=np.int32)
    apnea_stages = np.zeros((subject_duration // args.frame_size), dtype=np.int32)
    mask = np.zeros((subject_duration // args.frame_size), dtype=np.int32)
    prev_valid = False
    for f_idx, start in enumerate(range(0, subject_duration - args.frame_size, args.frame_size)):
        try:
            apnea_seg = apnea_mask[start:start+args.frame_size]
            features[f_idx, :] = compute_features_001(ds, subject_id, start)
            sleep_stages[f_idx] = sps.mode(sleep_mask[start:start+args.frame_size]).mode
            apnea_stages[f_idx] = sps.mode(apnea_seg[apnea_seg > 0]).mode if np.any(apnea_seg) else 0
            mask[f_idx] = 1
            prev_valid = True
        except PoorSignalError as err:
            # Copy previous once if it's valid
            if prev_valid:
                features[f_idx, :] = features[f_idx - 1, :]
                sleep_stages[f_idx] = sleep_stages[f_idx - 1]
                apnea_stages[f_idx] = apnea_stages[f_idx - 1]
                mask[f_idx] = 1
            prev_valid = False
        except Exception as err:
            prev_valid = False
            logger.warning(f"Error processing subject {subject_id} at {start} ({err}).")
            continue
    # END FOR

    with h5py.File(str(args.save_path / f"{subject_id}.h5"), "w") as h5:
        h5.create_dataset(f"/features", data=features, compression="gzip", compression_opts=6)
        h5.create_dataset(f"/labels", data=sleep_stages, compression="gzip", compression_opts=6)
        h5.create_dataset(f"/apnea", data=apnea_stages, compression="gzip", compression_opts=6)
        h5.create_dataset(f"/mask", data=mask, compression="gzip", compression_opts=6)
    # END WITH


def compute_features_003(ds: MesaDataset, subject_id: str, start: int):

    # Load signals
    ppg = ds.load_signal_for_subject(subject_id, "Pleth", start=start, data_size=ds.frame_size)
    spo2 = ds.load_signal_for_subject(subject_id, "SpO2", start=start, data_size=ds.frame_size)
    rsp = ds.load_signal_for_subject(subject_id, "Abdo", start=start, data_size=ds.frame_size)
    mov = ds.load_signal_for_subject(subject_id, "activity", start=start, data_size=ds.frame_size)
    pos = ds.load_signal_for_subject(subject_id, "Pos", start=start, data_size=ds.frame_size)
    hr = ds.load_signal_for_subject(subject_id, "HR", start=start, data_size=ds.frame_size)
    qos = ds.load_signal_for_subject(subject_id, "OxStatus", start=start, data_size=ds.frame_size)

    # Handle if signal is marked as bad
    if np.any(qos >= 3) or np.mean(qos) >= 1.5:
        raise PoorSignalError(f"Bad signal qos={np.mean(qos):.2f} {np.any(qos >= 3)} for subject {subject_id}")

    #### Preprocess signals
    ppg = pk.ppg.clean(ppg, sample_rate=ds.target_rate)
    rsp = pk.rsp.clean(rsp, sample_rate=ds.target_rate)

    hr_bpm = np.nanmean(np.clip(hr, 30, 120))
    min_rr, max_rr = max(0.25, 60/hr_bpm - 0.25), min(2.5, 60/hr_bpm + 0.25)

    rsp_bpm = pk.rsp.compute_respiratory_rate_from_fft(rsp, sample_rate=ds.target_rate, lowcut=0.05, highcut=2)
    rsp_bpm = np.clip(rsp_bpm, 3, 120)

    # HRV metrics
    rpeaks = pk.ppg.find_peaks(ppg, sample_rate=ds.target_rate)
    rri = pk.ppg.compute_rr_intervals(rpeaks)
    rri_mask = pk.ppg.filter_rr_intervals(rri, sample_rate=ds.target_rate, min_rr=min_rr, max_rr=max_rr)

    if rpeaks.size < 4 or rri[rri_mask == 0].size < 4:
        raise PoorSignalError("Not enough peaks")

    hrv_td_metrics = pk.hrv.compute_hrv_time(rri[rri_mask == 0], sample_rate=ds.target_rate)
    freq_bands = [(0.04, 0.15), (0.15, 0.4)]
    hrv_fd_metrics = pk.hrv.compute_hrv_frequency(
        rpeaks[rri_mask == 0],
        rri=rri[rri_mask == 0],
        bands=freq_bands,
        sample_rate=ds.target_rate
    )

    #### Compute features
    spo2_mu, spo2_std = np.nanmean(spo2), np.nanstd(spo2)
    spo2_med, spo2_iqr = np.nanmedian(spo2), sps.iqr(spo2)

    mov_mu = np.nanmean(mov)
    mov_med = np.nanmedian(mov)

    rri_mu, rri_std  = hrv_td_metrics.mean_nn, hrv_td_metrics.sd_nn
    rri_med, rri_iqr = hrv_td_metrics.meadian_nn, hrv_td_metrics.iqr_nn
    rri_sd_rms, rri_sd_std = hrv_td_metrics.rms_sd, hrv_td_metrics.sd_sd

    hrv_lf = hrv_fd_metrics.bands[0].total_power
    hrv_hf = hrv_fd_metrics.bands[1].total_power
    hrv_lfhf = hrv_lf/hrv_hf

    pos_mode = 1 if sps.mode(pos).mode == 4 else 0
    qos_mu = np.nanmean(qos)

    features = np.array([
        spo2_mu, spo2_std, spo2_med, spo2_iqr,
        mov_mu, mov_med,
        rri_mu, rri_std, rri_med, rri_iqr, rri_sd_rms, rri_sd_std,
        hr_bpm, rsp_bpm,
        hrv_lf, hrv_hf, hrv_lfhf,
        pos_mode, qos_mu
    ])
    return features


def compute_subject_features_003(subject_id: str, args: SKFeatureParams):

    args.sample_rate = getattr(args, "sample_rate", 64)
    args.frame_size = getattr(args, "frame_size", args.sample_rate*30)

    feat_names = [
        "SPO2-mu", "SPO2-std", "SPO2-med", "SPO2-iqr",
        "MOV-mu", "MOV-med",
        "RRI-mu", "RRI-std", "RRI-med", "RRI-iqr", "RRI-sd-rms", "RRI-sd-std",
        "HR-bpm", "RSP-bpm", "HRV-lf", "HRV-hf", "HRV-lfhf",
        "POS-mode", "QOS-mu"
    ]

    ds = MesaDataset(
        args.ds_path,
        frame_size=args.frame_size,
        target_rate=args.sample_rate,
        is_commercial=True
    )

    # Quick check if subject has activity signal
    try:
        ds.load_signal_for_subject(subject_id, "activity", start=0, data_size=ds.frame_size)
    except:
        print(f"Subject {subject_id} does not have activity signal. Skipping...")
        return

    subject_duration = max(0, ds.get_subject_duration(subject_id=subject_id)*ds.target_rate - ds.target_rate*60*60)
    sleep_stages = ds.extract_sleep_stages(subject_id=subject_id)
    sleep_mask = ds.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=subject_duration)
    features = np.zeros((subject_duration // ds.frame_size, len(feat_names)), dtype=np.float32)
    sleep_stages = np.zeros((subject_duration // ds.frame_size), dtype=np.int32)
    mask = np.zeros((subject_duration // ds.frame_size), dtype=np.int32)
    for f_idx, start in enumerate(range(0, subject_duration - ds.frame_size, ds.frame_size)):
        try:
            features[f_idx, :] = compute_features_001(ds, subject_id, start)
            sleep_stages[f_idx] = sps.mode(sleep_mask[start:start+ds.frame_size]).mode
            mask[f_idx] = 1
        except PoorSignalError as err:
            continue
        except Exception as err:
            print(err)
            continue
    # END FOR

    with h5py.File(str(args.save_path / f"{subject_id}.h5"), "w") as h5:
        h5.create_dataset(f"/features", data=features, compression="gzip", compression_opts=6)
        h5.create_dataset(f"/labels", data=sleep_stages, compression="gzip", compression_opts=6)
        h5.create_dataset(f"/mask", data=mask, compression="gzip", compression_opts=6)
    # END WITH

def compute_dataset_001(args: SKFeatureParams):
    ds = MesaDataset(args.ds_path, is_commercial=True)
    subject_ids = ds.subject_ids
    os.makedirs(args.save_path, exist_ok=True)
    f = functools.partial(compute_subject_features_001, args=args)
    with Pool(processes=args.data_parallelism) as pool:
        _ = list(tqdm(pool.imap(f, subject_ids), total=len(subject_ids)))
    # END WITH

def compute_dataset_003(args: SKFeatureParams):
    args.sample_rate = getattr(args, "sample_rate", 64)
    args.frame_size = getattr(args, "frame_size", args.sample_rate*30)

    ds = MesaDataset(
        args.ds_path,
        frame_size=args.frame_size,
        target_rate=args.sample_rate,
        is_commercial=True
    )
    subject_ids = ds.subject_ids
    os.makedirs(args.save_path, exist_ok=True)
    f = functools.partial(compute_subject_features_003,
        ds_path=args.ds_path,
        frame_size=args.frame_size,
        sample_rate=args.sample_rate,
        save_path=args.save_path
    )
    with Pool(processes=args.data_parallelism) as pool:
        _ = list(tqdm(pool.imap(f, subject_ids), total=len(subject_ids)))
    # END WITH

def generate_feature_set(args: SKFeatureParams):
    if args.feature_set == "fs001":
        compute_dataset_001(args)
    elif args.feature_set == "fs003":
        compute_dataset_003(args)
    else:
        raise NotImplementedError(f"Feature set {args.feature_set} not implemented")
