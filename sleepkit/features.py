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

class PoorSignalError(Exception):
    pass

def compute_features_001(ds: MesaDataset, subject_id: str, start: int):
    # Load signals
    ppg = ds.load_signal_for_subject(subject_id, "Pleth", start=start, data_size=ds.frame_size)
    spo2 = ds.load_signal_for_subject(subject_id, "SpO2", start=start, data_size=ds.frame_size)
    rsp = ds.load_signal_for_subject(subject_id, "Abdo", start=start, data_size=ds.frame_size)
    # leg = ds.load_signal_for_subject(subject_id, "Leg", start=start, data_size=ds.frame_size)
    mov = ds.load_signal_for_subject(subject_id, "activity", start=start, data_size=ds.frame_size)
    qos = ds.load_signal_for_subject(subject_id, "OxStatus", start=start, data_size=ds.frame_size)

    # Handle if signal is marked as bad
    if np.sum(qos > 1) > 0.60*qos.size:
        raise PoorSignalError("Bad signal")

    #### Preprocess signals
    ppg = pk.ppg.clean(ppg, sample_rate=ds.target_rate)
    rsp = pk.rsp.clean(rsp, sample_rate=ds.target_rate)
    # mov = pk.signal.filter_signal(leg, lowcut=3, highcut=11, order=3, sample_rate=ds.target_rate)

    hr_bpm = pk.ppg.compute_heart_rate(ppg, sample_rate=ds.target_rate, method="fft")
    rsp_bpm = pk.rsp.compute_respiratory_rate_from_fft(rsp, sample_rate=ds.target_rate, lowcut=0.05, highcut=1.0)

    min_rr, max_rr = max(0.5, 60/hr_bpm - 0.25), min(4, 60/hr_bpm + 0.25)

    # HRV metrics
    rpeaks = pk.ppg.find_peaks(ppg, sample_rate=ds.target_rate)
    rri = pk.ppg.compute_rr_intervals(rpeaks)
    rri_mask = pk.ppg.filter_rr_intervals(rri, sample_rate=ds.target_rate, min_rr=min_rr, max_rr=max_rr)

    if rpeaks.size < 4 or rri[rri_mask == 0].size < 4:
        raise PoorSignalError("Not enough peaks")

    hrv_td_metrics = pk.hrv.compute_hrv_time(rri[rri_mask == 0], sample_rate=ds.target_rate)
    hrv_fd_metrics = pk.hrv.compute_hrv_frequency(rpeaks, rri=rri, bands=[(0.04, 0.15), (0.15, 0.4)], sample_rate=ds.target_rate)

    #### Compute features
    spo2_mu = np.nanmean(spo2)
    spo2_std = np.nanstd(spo2)
    spo2_med = np.nanmedian(spo2)
    spo2_iqr = sps.iqr(spo2)

    mov_mu = np.nanmean(mov)
    mov_std = np.nanstd(mov)
    mov_med = np.nanmedian(mov)
    mov_iqr = sps.iqr(mov)

    rri_mu = hrv_td_metrics.mean_nn
    rri_std = hrv_td_metrics.sd_nn
    rri_med = hrv_td_metrics.meadian_nn
    rri_iqr = hrv_td_metrics.iqr_nn
    rri_sd_rms = hrv_td_metrics.rms_sd
    rri_sd_std = hrv_td_metrics.sd_sd

    hrv_lf = hrv_fd_metrics.bands[0].total_power
    hrv_hf = hrv_fd_metrics.bands[1].total_power
    hrv_lfhf = hrv_lf/hrv_hf

    # Compute resipiratory rate from PPG / ACCEL
    rtroughs = np.zeros_like(rpeaks)
    for i in range(0, len(rpeaks)-1):
        rtroughs[i] = np.argmin(ppg[rpeaks[i]:rpeaks[i+1]]) + rpeaks[i]
    # END FOR
    rtroughs[-1] = rtroughs[-2]
    rpeaks = rpeaks[rri_mask == 0]
    rtroughs = rtroughs[rri_mask == 0]
    rri = rri[rri_mask == 0]
    rsp_bpm, _ = pk.rsp.compute_respiratory_rate_from_ppg(ppg, rpeaks, rtroughs, rri, sample_rate=ds.target_rate)
    features = np.array([
        spo2_mu, spo2_std, spo2_med, spo2_iqr,
        mov_mu, mov_std, mov_med, mov_iqr,
        rri_mu, rri_std, rri_med, rri_iqr, rri_sd_rms, rri_sd_std,
        hr_bpm, rsp_bpm,
        hrv_lf, hrv_hf, hrv_lfhf
    ])
    return features


def compute_subject_features_001(subject_id: str, ds_path: Path, frame_size: int = 64*30, sample_rate: int = 64, save_path: Path = ""):

    feat_names = [
        "SPO2-mu", "SPO2-std", "SPO2-med", "SPO2-iqr",
        "MOV-mu", "MOV-std", "MOV-med", "MOV-iqr",
        "RRI-mu", "RRI-std", "RRI-med", "RRI-iqr", "RRI-sd-rms", "RRI-sd-std",
        "HR-bpm", "RSP-bpm", "HRV-lf", "HRV-hf", "HRV-lfhf"
    ]

    ds = MesaDataset(
        ds_path,
        frame_size=frame_size,
        target_rate=sample_rate,
        is_commercial=True
    )

    # Quick check if subject has activity signal
    try:
        ds.load_signal_for_subject(subject_id, "activity", start=0, data_size=ds.frame_size)
    except:
        return

    subject_duration = ds.get_subject_duration(subject_id=subject_id)*ds.target_rate
    sleep_stages = ds.extract_sleep_stages(subject_id=subject_id)
    sleep_mask = ds.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=subject_duration)
    features = np.zeros((subject_duration // frame_size, len(feat_names)), dtype=np.float32)
    sleep_stages = np.zeros((subject_duration // frame_size), dtype=np.int32)
    mask = np.zeros((subject_duration // frame_size), dtype=np.int32)
    for f_idx, start in enumerate(range(0, subject_duration - frame_size, frame_size)):
        try:
            features[f_idx, :] = compute_features_001(ds, subject_id, start)
            sleep_stages[f_idx] = sps.mode(sleep_mask[start:start+frame_size]).mode
            mask[f_idx] = 1
        except PoorSignalError as err:
            # print(err)
            continue
    # END FOR

    with h5py.File(str(save_path / f"{subject_id}.h5"), "w") as h5:
        h5.create_dataset(f"/features", data=features, compression="gzip", compression_opts=6)
        h5.create_dataset(f"/labels", data=sleep_stages, compression="gzip", compression_opts=6)
        h5.create_dataset(f"/mask", data=mask, compression="gzip", compression_opts=6)
    # END WITH


def compute_dataset_001(ds_path: Path, frame_size: int = 64*30, sample_rate: int = 64, save_path: Path = "", num_workers: int = 8):
    ds = MesaDataset(
        ds_path,
        frame_size=frame_size,
        target_rate=sample_rate,
        is_commercial=True
    )
    subject_ids = ds.subject_ids
    os.makedirs(save_path, exist_ok=True)
    f = functools.partial(compute_subject_features_001,
        ds_path=ds_path,
        frame_size=frame_size,
        sample_rate=sample_rate,
        save_path=save_path
    )
    with Pool(processes=num_workers) as pool:
        _ = list(tqdm(pool.imap(f, subject_ids), total=len(subject_ids)))
    # END WITH

if __name__ == "__main__":
    print("Started")
    compute_dataset_001(
        ds_path=Path("/home/vscode/datasets"),
        frame_size=64*60,
        sample_rate=64,
        save_path=Path("./datasets/processed/mesa-fs002"),
        num_workers=32
    )
    print("Finished")
