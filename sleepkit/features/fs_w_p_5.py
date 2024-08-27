"""

# Feature set: FS-W-P-5

* Location: Wrist
* Sensors: PPG
* Features: 5
* Datasets: MESA
* Tasks: Sleep apnea

Classes:
    FS_W_P_5: Feature set FS-W-P-5
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


class FS_W_P_5(FeatureSet):
    """Feature set: FS-W-P-5
    Location: Wrist
    Sensors: PPG
    Features: 5
    Datasets: MESA
    Tasks: Sleep apnea
    """

    @staticmethod
    def name() -> str:
        """Feature set name."""
        return "FS-W-P-5"

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        return [
            "spo2",  # SpO2
            "piav",  # Peak-to-trough amplitude delta
            "piiv",  # Peak-to-peak amplitude delta
            "pifv",  # Peak-to-peak interval delta
            "qos",  # Quality of signal
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
            sample_rate = int(params.feature.sampling_rate)

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

                ts = np.arange(0, duration, 1)

                spo2 = ds.load_signal_for_subject(subject_id, "SpO2", start=0, data_size=duration)
                ppg = ds.load_signal_for_subject(subject_id, "Pleth", start=0, data_size=duration)
                spo2_qos = ds.load_signal_for_subject(subject_id, "OxStatus", start=0, data_size=duration)

                ppg = pk.ppg.clean(ppg, lowcut=0.5, highcut=3.0, sample_rate=sample_rate)

                sleep_apnea = ds.extract_sleep_apneas(subject_id=subject_id)
                apnea_labels = ds.apnea_events_to_mask(apnea_events=sleep_apnea, data_size=duration)

                sleep_stages = ds.extract_sleep_stages(subject_id=subject_id)
                sleep_labels = ds.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=duration)

                ppeaks = pk.ppg.find_peaks(ppg, sample_rate=ds.target_rate)
                # Find troughs by picking min between peaks
                ptroughs = np.zeros_like(ppeaks)
                for i in range(1, ppeaks.size):
                    ptroughs[i - 1] = np.argmin(ppg[ppeaks[i - 1] : ppeaks[i]]) + ppeaks[i - 1]
                ptroughs[-1] = np.argmin(ppg[ppeaks[-1] :]) + ppeaks[-1]

                ipi = pk.ppg.compute_rr_intervals(ppeaks)
                ipi_mask = pk.ppg.filter_rr_intervals(
                    ipi, sample_rate=ds.target_rate, min_rr=0.5, max_rr=2, min_delta=0.3
                )

                nn_ppeaks = ppeaks[ipi_mask == 0]
                nn_ptroughs = ptroughs[ipi_mask == 0]
                nn_ipi = ipi[ipi_mask == 0]

                piav = ppg[nn_ppeaks] - ppg[nn_ptroughs]  # Peak to Trough delta
                piiv = ppg[nn_ppeaks[1:]] - ppg[nn_ppeaks[:-1]]  # Peak to Peak delta
                piiv = np.hstack((piiv[0], piiv))
                pifv = nn_ipi.copy()  # Peak to Peak interval
                ppg_qos = pk.signal.signal_smooth_boxzen(1.0 - ipi_mask, size=ds.target_rate * 5)

                spo2 = np.clip(spo2, 50, 100)
                piav = scipy.interpolate.interp1d(
                    nn_ppeaks,
                    piav,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nanmedian(piav),
                )(ts)
                piiv = scipy.interpolate.interp1d(
                    nn_ppeaks,
                    piiv,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nanmedian(piiv),
                )(ts)
                pifv = scipy.interpolate.interp1d(
                    nn_ppeaks,
                    pifv,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nanmedian(pifv),
                )(ts)
                ppg_qos = scipy.interpolate.interp1d(ppeaks, ppg_qos, kind="linear", bounds_error=False, fill_value=0)(
                    ts
                )
            else:
                raise NotImplementedError(f"Dataset {ds_name} not implemented")
            # END IF

            # Extract features
            features = np.zeros(((duration - win_size) // ovl_size, 5), dtype=np.float32)
            alabels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
            slabels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
            masks = np.ones((duration - win_size) // ovl_size, dtype=np.int32)
            for i, start in enumerate(range(0, duration - win_size, ovl_size)):
                stop = start + win_size

                if i >= features.shape[0]:
                    break

                spo2_win = np.nanmean(spo2[start:stop])
                piav_win = np.nanmean(piav[start:stop])
                piiv_win = np.nanmean(piiv[start:stop])
                pifv_win = np.nanmean(pifv[start:stop])
                ppg_qos_win = np.nanmean(ppg_qos[start:stop])
                spo2_qos_win = np.nanmean(spo2_qos[start:stop])

                if np.any(spo2_qos_win >= 2.8):
                    masks[i] = 0
                    continue
                # END IF

                features[i] = [
                    spo2_win,
                    piav_win,
                    piiv_win,
                    pifv_win,
                    ppg_qos_win,
                ]

                if ppg_qos_win <= 0.4:
                    masks[i] = 0

                alabels[i] = apnea_labels[start:stop][-1]
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
