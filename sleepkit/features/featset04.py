import h5py
import numpy as np

from ..datasets import MesaDataset
from ..defines import SKFeatureParams
from ..utils import setup_logger
from .featureset import SKFeatureSet

logger = setup_logger(__name__)


class FeatSet04(SKFeatureSet):
    """Feature set: FS-H-E-10
    Location: Head
    Sensors: EEG, EOG
    Features: 10
    Tasks: Sleep staging
    """

    @staticmethod
    def name() -> str:
        """Feature set name."""
        return "FS-H-E-10"

    @staticmethod
    def feature_names() -> list[str]:
        """Feature names."""
        return [
            "eeg_delta_power",
            "eeg_theta_power",
            "eeg_alpha_power",
            "eeg_beta_power",
            "eeg_gamma_power",
            "eog_delta_power",
            "eog_theta_power",
            "eog_alpha_power",
            "eog_beta_power",
            "eog_gamma_power",
        ]

    @staticmethod
    def generate_features(ds_subject: tuple[str, str], args: SKFeatureParams):
        """Generate features for dataset subject.

        Args:
            ds_subject (tuple[str, str]): Dataset name and subject ID
            args (SKFeatureParams): Feature generation parameters
        """
        try:

            ds_name, subject_id = ds_subject
            num_features = 50
            sample_rate = args.sampling_rate
            win_size = int(args.frame_size)
            ovl_size = int(win_size / 2)

            # Load dataset specific signals
            if ds_name == "mesa":
                ds = MesaDataset(ds_path=args.ds_path, is_commercial=True, target_rate=sample_rate)

                # Read signals
                duration = ds.get_subject_duration(subject_id=subject_id) * sample_rate
                duration = max(0, duration - win_size)

                # eogl = ds.load_signal_for_subject(subject_id, "EOG-L", start=0, data_size=duration)
                # eogr = ds.load_signal_for_subject(subject_id, "EOG-R", start=0, data_size=duration)
                # eeg1 = ds.load_signal_for_subject(subject_id, "EEG1", start=0, data_size=duration)
                # eeg2 = ds.load_signal_for_subject(subject_id, "EEG2", start=0, data_size=duration)
                # eeg3 = ds.load_signal_for_subject(subject_id, "EEG3", start=0, data_size=duration)

                sleep_stages = ds.extract_sleep_stages(subject_id=subject_id)
                sleep_labels = ds.sleep_stages_to_mask(sleep_stages=sleep_stages, data_size=duration)

            features = np.zeros(((duration - win_size) // ovl_size, num_features), dtype=np.float32)
            slabels = np.zeros((duration - win_size) // ovl_size, dtype=np.int32)
            masks = np.ones((duration - win_size) // ovl_size, dtype=np.int32)
            for i, start in enumerate(range(0, duration - win_size, ovl_size)):
                if i >= features.shape[0]:
                    break

                stop = start + win_size
                # eogl_win = eogl[start:stop]
                # Compute spectral features
                # pk.signal.compute_fft(eogl_win, sample_rate=sample_rate, fft_len=)
                features[i] = []

                slabels[i] = sleep_labels[start:stop][-1]
            # END FOR

            with h5py.File(str(args.save_path / ds_name / f"{subject_id}.h5"), "w") as h5:
                h5.create_dataset("/features", data=features, compression="gzip", compression_opts=6)
                h5.create_dataset("/stage_labels", data=slabels, compression="gzip", compression_opts=6)
                h5.create_dataset("/mask", data=masks, compression="gzip", compression_opts=6)
            # END WITH

        # pylint: disable=broad-except
        except Exception as err:
            logger.exception(f"Error computing features for subject {subject_id}", err)
