import functools
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
import physiokit as pk
import scipy.signal
import scipy.stats as sps
from tqdm import tqdm

from .datasets import MesaDataset
from .defines import SKFeatureParams
from .utils import setup_logger

logger = setup_logger(__name__)


class PoorSignalError(Exception):
    """Poor signal error."""


class NoSignalError(Exception):
    """No signal error."""


def get_feature_names_001():
    """Get feature names for feature set 001."""
    return [
        "SPO2-mu",
        "SPO2-std",
        "SPO2-med",
        "SPO2-iqr",
        "MOV-mu",
        "MOV-std",
        "MOV-med",
        "MOV-iqr",
        "RRI-mu",
        "RRI-std",
        "RRI-med",
        "RRI-iqr",
        "RRI-sd-rms",
        "RRI-sd-std",
        "HR-bpm",
        "RSP-bpm",
        "HRV-lf",
        "HRV-hf",
        "HRV-lfhf",
    ]


def get_feature_names_002():
    """Get feature names for feature set 002."""
    return [
        "EEG1-delta",
        "EEG1-theta",
        "EEG1-alpha",
        "EEG1-beta",
        "EEG2-delta",
        "EEG2-theta",
        "EEG2-alpha",
        "EEG2-beta",
        "EEG3-delta",
        "EEG3-theta",
        "EEG3-alpha",
        "EEG3-beta",
    ]


def get_feature_names_003():
    """Get feature names for feature set 003."""
    return [
        "SPO2-mu",
        "SPO2-std",
        "SPO2-med",
        "SPO2-iqr",
        "MOV-mu",
        "MOV-med",
        "RRI-mu",
        "RRI-std",
        "RRI-med",
        "RRI-iqr",
        "RRI-sd-rms",
        "RRI-sd-std",
        "HR-bpm",
        "RSP-bpm",
        "HRV-lf",
        "HRV-hf",
        "HRV-lfhf",
        "POS-mode",
        "QOS-mu",
        "TOD-cos",
    ]


def get_feature_names_004():
    """Get feature names for feature set 004."""
    return [
        "rsp-mov-mu",
        "rsp-mov-std",
        "leg-mov-mu",
        "leg-mov-std",
        "ppg-mov-mu",
        "ppg-mov-std",
        "rsp-pos-xy",
        "rsp-pos-yz",
        "ppg-hr-mu",
        "ppg-hr-std",
        "ppg-hrv-rms-sd",
        "ppg-hrv-lfhf",
        "spo2-mu",
        "spo2-std",
        "spo2-qos",
        "ppg-qos",
    ]


def compute_features_001(ds: MesaDataset, subject_id: str, start: int):
    """Compute features 001 for subject."""
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
    min_rr, max_rr = max(0.25, 60 / hr_bpm - 0.25), min(2.5, 60 / hr_bpm + 0.25)

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
        rpeaks[rri_mask == 0], rri=rri[rri_mask == 0], bands=freq_bands, sample_rate=ds.target_rate
    )

    #### Compute features
    spo2_mu, spo2_std = np.nanmean(spo2), np.nanstd(spo2)
    spo2_med, spo2_iqr = np.nanmedian(spo2), sps.iqr(spo2)

    mov_mu, mov_std = np.nanmean(mov), np.nanstd(mov)
    mov_med, mov_iqr = np.nanmedian(mov), sps.iqr(mov)

    rri_mu, rri_std = hrv_td_metrics.mean_nn, hrv_td_metrics.sd_nn
    rri_med, rri_iqr = hrv_td_metrics.meadian_nn, hrv_td_metrics.iqr_nn
    rri_sd_rms, rri_sd_std = hrv_td_metrics.rms_sd, hrv_td_metrics.sd_sd

    hrv_lf = hrv_fd_metrics.bands[0].total_power
    hrv_hf = hrv_fd_metrics.bands[1].total_power
    hrv_lfhf = hrv_lf / hrv_hf

    features = np.array(
        [
            spo2_mu,
            spo2_std,
            spo2_med,
            spo2_iqr,
            mov_mu,
            mov_std,
            mov_med,
            mov_iqr,
            rri_mu,
            rri_std,
            rri_med,
            rri_iqr,
            rri_sd_rms,
            rri_sd_std,
            hr_bpm,
            rsp_bpm,
            hrv_lf,
            hrv_hf,
            hrv_lfhf,
        ]
    )
    return features


def compute_features_002(ds: MesaDataset, subject_id: str, start: int):
    """Compute features 002 for subject."""
    eeg1 = ds.load_signal_for_subject(subject_id, "EEG1", start=start, data_size=ds.frame_size)
    eeg2 = ds.load_signal_for_subject(subject_id, "EEG2", start=start, data_size=ds.frame_size)
    eeg3 = ds.load_signal_for_subject(subject_id, "EEG3", start=start, data_size=ds.frame_size)
    eogl = ds.load_signal_for_subject(subject_id, "EOG-L", start=start, data_size=ds.frame_size)
    eogr = ds.load_signal_for_subject(subject_id, "EOG-R", start=start, data_size=ds.frame_size)

    # Filter signals [0.5, 30] Hz
    eeg1 = pk.signal.filter_signal(eeg1, lowcut=0.5, highcut=30, order=3, sample_rate=ds.target_rate)
    eeg2 = pk.signal.filter_signal(eeg2, lowcut=0.5, highcut=30, order=3, sample_rate=ds.target_rate)
    eeg3 = pk.signal.filter_signal(eeg3, lowcut=0.5, highcut=30, order=3, sample_rate=ds.target_rate)
    eogl = pk.signal.filter_signal(eogl, lowcut=0.5, highcut=8, order=3, sample_rate=ds.target_rate)
    eogr = pk.signal.filter_signal(eogr, lowcut=0.5, highcut=8, order=3, sample_rate=ds.target_rate)
    raise NotImplementedError()
    # eeg_bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]
    # eog_bands = []
    # emg_bands = []
    # fft_len = int(2 ** np.ceil(np.log2(ts.size)))
    # fft_win = np.blackman(ts.size)
    # amp_corr = 1.93

    # freqs = np.fft.fftfreq(fft_len, 1 / sample_rate)
    # rri_fft = np.fft.fft(fft_win * rri_int, fft_len) / ts.size
    # rri_ps = 2 * amp_corr * np.abs(rri_fft)

    # features = (3*len(eeg_bands) + 2*len(eog_bands))*[0]
    # for lowcut, highcut in bands:
    #     l_idx = np.where(freqs >= lowcut)[0][0]
    #     r_idx = np.where(freqs >= highcut)[0][0]
    #     f_idx = rri_ps[l_idx:r_idx].argmax() + l_idx
    #     metrics.bands.append(
    #         HrvFrequencyBandMetrics(
    #             peak_frequency=freqs[f_idx],
    #             peak_power=rri_ps[f_idx],
    #             total_power=rri_ps[l_idx:r_idx].sum(),
    #         )
    #     )
    # # END FOR
    # metrics.total_power = reduce(lambda x, y: x + y.total_power, metrics.bands, 0)
    # return metrics
    # Compute power in 1 Hz bins up to 30 Hz


def compute_features_003(ds: MesaDataset, subject_id: str, start: int):
    """Compute features 003 for subject."""

    # Load signals
    ppg = ds.load_signal_for_subject(subject_id, "Pleth", start=start, data_size=ds.frame_size)
    spo2 = ds.load_signal_for_subject(subject_id, "SpO2", start=start, data_size=ds.frame_size)
    rsp = ds.load_signal_for_subject(subject_id, "Abdo", start=start, data_size=ds.frame_size)
    tod = ds.load_signal_for_subject(subject_id, "linetime", start=start, data_size=ds.frame_size)
    pos = ds.load_signal_for_subject(subject_id, "Pos", start=start, data_size=ds.frame_size)
    hr = ds.load_signal_for_subject(subject_id, "HR", start=start, data_size=ds.frame_size)
    qos = ds.load_signal_for_subject(subject_id, "OxStatus", start=start, data_size=ds.frame_size)
    try:
        mov = ds.load_signal_for_subject(subject_id, "activity", start=start, data_size=ds.frame_size)
    except Exception as err:
        raise NoSignalError(f"Missing signal activity for subject {subject_id}") from err

    # Handle if signal is marked as bad
    if np.any(qos >= 3) or np.mean(qos) >= 1.5:
        raise PoorSignalError(f"Bad signal qos={np.mean(qos):.2f} {np.any(qos >= 3)} for subject {subject_id}")

    #### Preprocess signals
    ppg = pk.ppg.clean(ppg, sample_rate=ds.target_rate)
    rsp = pk.rsp.clean(rsp, sample_rate=ds.target_rate)

    hr_bpm = np.nanmean(np.clip(hr, 30, 120))
    min_rr, max_rr = max(0.25, 60 / hr_bpm - 0.25), min(2.5, 60 / hr_bpm + 0.25)

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
        rpeaks[rri_mask == 0], rri=rri[rri_mask == 0], bands=freq_bands, sample_rate=ds.target_rate
    )

    #### Compute features
    spo2_mu, spo2_std = np.nanmean(spo2), np.nanstd(spo2)
    spo2_med, spo2_iqr = np.nanmedian(spo2), sps.iqr(spo2)

    mov_mu = np.nanmean(mov)
    mov_med = np.nanmedian(mov)

    rri_mu, rri_std = hrv_td_metrics.mean_nn, hrv_td_metrics.sd_nn
    rri_med, rri_iqr = hrv_td_metrics.meadian_nn, hrv_td_metrics.iqr_nn
    rri_sd_rms, rri_sd_std = hrv_td_metrics.rms_sd, hrv_td_metrics.sd_sd

    hrv_lf = hrv_fd_metrics.bands[0].total_power
    hrv_hf = hrv_fd_metrics.bands[1].total_power
    hrv_lfhf = hrv_lf / hrv_hf

    pos_mode = 1 if sps.mode(pos).mode == 4 else 0
    qos_mu = np.nanmean(qos)

    ts = time.strptime(tod[0], "%H:%M:%S")
    tod_norm = (ts.tm_hour * 60 * 60 + ts.tm_min * 60 + ts.tm_sec) / (24 * 60 * 60)
    tod_cos = np.cos(2 * np.pi * tod_norm)

    features = np.array(
        [
            spo2_mu,
            spo2_std,
            spo2_med,
            spo2_iqr,
            mov_mu,
            mov_med,
            rri_mu,
            rri_std,
            rri_med,
            rri_iqr,
            rri_sd_rms,
            rri_sd_std,
            hr_bpm,
            rsp_bpm,
            hrv_lf,
            hrv_hf,
            hrv_lfhf,
            pos_mode,
            qos_mu,
            tod_cos,
        ]
    )
    return features


def compute_features_004(ds: MesaDataset, subject_id: str, start: int):
    """Compute features 004 for subject."""

    ppg = ds.load_signal_for_subject(subject_id, "Pleth", start=start, data_size=ds.frame_size)
    pos = ds.load_signal_for_subject(subject_id, "Pos", start=start, data_size=ds.frame_size)
    rsp = ds.load_signal_for_subject(subject_id, "Thor", start=start, data_size=ds.frame_size)
    leg = ds.load_signal_for_subject(subject_id, "Leg", start=start, data_size=ds.frame_size)
    spo2 = ds.load_signal_for_subject(subject_id, "SpO2", start=start, data_size=ds.frame_size)
    qos = ds.load_signal_for_subject(subject_id, "OxStatus", start=start, data_size=ds.frame_size)

    # Handle if signal is marked as "sensor off" (value is 3 but could be lower due to interpolation)
    if np.any(qos >= 2.8):
        raise PoorSignalError(f"No signal qos={np.mean(qos):.2f} {np.any(qos >= 3)} for subject {subject_id}")

    # if np.any(qos >= 3) or np.mean(qos) >= 1.5:
    #     raise PoorSignalError(f"Bad signal qos={np.mean(qos):.2f} {np.any(qos >= 3)} for subject {subject_id}")

    #### Preprocess signals
    ppg = pk.ppg.clean(ppg, lowcut=0.5, highcut=3, sample_rate=ds.target_rate, order=3)
    ppg_mov = pk.ppg.clean(ppg, lowcut=4, highcut=11, sample_rate=ds.target_rate, order=3)

    rsp = pk.rsp.clean(rsp, lowcut=0.067, highcut=3, sample_rate=ds.target_rate)
    rsp_mov = pk.signal.filter_signal(rsp, lowcut=3, highcut=11, order=3, sample_rate=ds.target_rate)

    leg_mov = pk.signal.filter_signal(leg, lowcut=3, highcut=11, order=3, sample_rate=ds.target_rate)

    spo2 = np.clip(spo2, 50, 100) / 100

    pos = sps.mode(pos.astype(np.int32)).mode

    ppg_freq, ppg_sp = pk.ppg.compute_fft(ppg, sample_rate=ds.target_rate)
    l_idx = np.where(ppg_freq >= 0.5)[0][0]
    r_idx = np.where(ppg_freq >= 3)[0][0]
    ppg_sp = 2 * np.abs(ppg_sp)
    ppg_freq = ppg_freq[l_idx:r_idx]
    ppg_sp = ppg_sp[l_idx:r_idx]
    ppg_pk_idx = np.argmax(ppg_sp)
    dom_pk_val = ppg_freq[ppg_pk_idx]
    dom_l_idx = np.where(ppg_freq >= dom_pk_val - 0.1)[0][0]
    dom_r_idx = np.where(ppg_freq >= dom_pk_val + 0.1)[0][0]
    ppg_hr = np.clip(60 * dom_pk_val, 20, 120)
    ppg_qos = np.clip(2 * np.sum(ppg_sp[dom_l_idx:dom_r_idx]) / np.sum(ppg_sp), 0, 1)

    min_rr, max_rr = max(0.25, 60 / ppg_hr - 0.25), min(2.5, 60 / ppg_hr + 0.25)

    rsp_bpm = pk.rsp.compute_respiratory_rate_from_fft(rsp, sample_rate=ds.target_rate, lowcut=0.05, highcut=2)
    rsp_bpm = np.clip(rsp_bpm, 3, 120)

    # HRV metrics
    rpeaks = pk.ppg.find_peaks(ppg, sample_rate=ds.target_rate)
    rri = pk.ppg.compute_rr_intervals(rpeaks)
    rri_mask = pk.ppg.filter_rr_intervals(rri, sample_rate=ds.target_rate, min_rr=min_rr, max_rr=max_rr)

    if rpeaks.size < 4 or rri[rri_mask == 0].size < 4:
        ppg_hrv_std = 0
        ppg_hrv_rms_sd = 0
        ppg_hrv_lfhf = 1
    else:
        hrv_td_metrics = pk.hrv.compute_hrv_time(rri[rri_mask == 0], sample_rate=ds.target_rate)
        freq_bands = [(0.04, 0.15), (0.15, 0.4)]
        hrv_fd_metrics = pk.hrv.compute_hrv_frequency(
            rpeaks[rri_mask == 0], rri=rri[rri_mask == 0], bands=freq_bands, sample_rate=ds.target_rate
        )
        hrv_lf = hrv_fd_metrics.bands[0].total_power
        hrv_hf = hrv_fd_metrics.bands[1].total_power
        ppg_hrv_std = hrv_td_metrics.sd_nn
        ppg_hrv_rms_sd = hrv_td_metrics.rms_sd
        ppg_hrv_lfhf = hrv_lf / hrv_hf

    #### Compute features
    rsp_mov_mu = np.nanmean(np.abs(rsp_mov))
    rsp_mov_std = np.nanstd(np.abs(rsp_mov))

    leg_mov_mu = np.nanmean(np.abs(leg_mov))
    leg_mov_std = np.nanstd(np.abs(leg_mov))

    ppg_hr = (ppg_hr - 20) / 100
    ppg_mov_mu = np.nanmean(np.abs(ppg_mov))
    ppg_mov_std = np.nanstd(np.abs(ppg_mov))

    spo2_med = scipy.signal.medfilt(spo2, kernel_size=ds.target_rate - 1)
    spo2_mu = np.nanmean(spo2_med)
    spo2_std = np.nanstd(spo2_med)

    spo2_qos = np.nanmean(2 - np.clip(qos, 0, 2)) / 2.0

    rsp_pos_xy = -1 if pos == 2 else 1 if pos == 0 else 0
    rsp_pos_yz = 1 if pos == 4 else 0

    features = np.array(
        [
            rsp_mov_mu,
            rsp_mov_std,
            leg_mov_mu,
            leg_mov_std,
            ppg_mov_mu,
            ppg_mov_std,
            rsp_pos_xy,
            rsp_pos_yz,
            ppg_hr,
            ppg_hrv_std,
            ppg_hrv_rms_sd,
            ppg_hrv_lfhf,
            spo2_mu,
            spo2_std,
            ppg_qos,
            spo2_qos,
        ]
    )
    return features


def compute_subject_features(subject_id: str, args: SKFeatureParams, feat_names: list[str], compute_features: callable):
    """Compute features for subject."""

    args.sample_rate = getattr(args, "sample_rate", 64)
    args.frame_size = getattr(args, "frame_size", args.sample_rate * 30)

    ds = MesaDataset(args.ds_path, frame_size=args.frame_size, target_rate=args.sample_rate, is_commercial=True)

    subject_duration = max(0, ds.get_subject_duration(subject_id=subject_id) * ds.target_rate)
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
            apnea_seg = apnea_mask[start : start + args.frame_size]
            features[f_idx, :] = compute_features(ds, subject_id, start)
            sleep_stages[f_idx] = sps.mode(sleep_mask[start : start + args.frame_size]).mode
            apnea_stages[f_idx] = sps.mode(apnea_seg[apnea_seg > 0]).mode if np.any(apnea_seg) else 0
            mask[f_idx] = 1
            prev_valid = True
        except PoorSignalError:
            # Copy previous once if it's valid
            if prev_valid:
                features[f_idx, :] = features[f_idx - 1, :]
                sleep_stages[f_idx] = sleep_stages[f_idx - 1]
                apnea_stages[f_idx] = apnea_stages[f_idx - 1]
                mask[f_idx] = 1
            prev_valid = False
        except NoSignalError:
            # If no signal, skip subject
            return
        # pylint: disable=broad-exception-caught
        except Exception as err:
            prev_valid = False
            logger.warning(f"Error processing subject {subject_id} at {start} ({err}).")
            continue
    # END FOR

    with h5py.File(str(args.save_path / f"{subject_id}.h5"), "w") as h5:
        h5.create_dataset("/features", data=features, compression="gzip", compression_opts=6)
        h5.create_dataset("/labels", data=sleep_stages, compression="gzip", compression_opts=6)
        h5.create_dataset("/apnea", data=apnea_stages, compression="gzip", compression_opts=6)
        h5.create_dataset("/mask", data=mask, compression="gzip", compression_opts=6)
    # END WITH


def generate_feature_set(args: SKFeatureParams):
    """Generate feature set for all subjects in dataset
    Args:
        args (SKFeatureParams): Feature generation parameters
    """
    ds = MesaDataset(args.ds_path, is_commercial=True)
    subject_ids = ds.subject_ids
    if args.feature_set == "fs001":
        feat_names = get_feature_names_001()
        compute_features = compute_features_001
    elif args.feature_set == "fs003":
        feat_names = get_feature_names_003()
        compute_features = compute_features_003
    elif args.feature_set == "fs004":
        feat_names = get_feature_names_004()
        compute_features = compute_features_004
    else:
        raise NotImplementedError(f"Feature set {args.feature_set} not implemented")

    os.makedirs(args.save_path, exist_ok=True)
    f = functools.partial(compute_subject_features, args=args, feat_names=feat_names, compute_features=compute_features)
    with Pool(processes=args.data_parallelism) as pool:
        _ = list(tqdm(pool.imap(f, subject_ids), total=len(subject_ids)))
    # END WITH
