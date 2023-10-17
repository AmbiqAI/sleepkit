import glob
import logging
import math
import os
import random
from enum import IntEnum
from pathlib import Path
from typing import Callable
from xml.dom.minidom import Element as XmlElement
from xml.dom.minidom import Node as XmlNode
from xml.dom.minidom import parse as xml_parse

import numpy as np
import numpy.typing as npt
import pandas as pd
import physiokit as pk
import pyedflib

from .defines import SampleGenerator, SubjectGenerator

logger = logging.getLogger(__name__)

# OxStatus: 0=Good 1=Marginal 2=Poor 3=Sensor off


class StagesSleepStage(IntEnum):
    """Sleep stage enum"""

    WAKE = 0
    N1 = 1
    N2 = 2
    N3 = 3
    N4 = 4
    REM = 5
    MOVEMENT = 6
    UNSCORED = 9


class StagesDataset:
    """STAGES dataset"""

    def __init__(
        self,
        ds_path: Path,
        frame_size: int = 30 * 128,
        target_rate: int = 128,
        is_commercial: bool = True,
    ) -> None:
        self.frame_size = frame_size
        self.target_rate = target_rate
        self.ds_path = ds_path / "stages"
        self.sleep_mapping = lambda v: {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 5, 6: 0, 9: 0}.get(v, 0)

    @property
    def subject_ids(self) -> list[str]:
        """Get dataset subject IDs

        Returns:
            list[str]: Subject IDs
        """
        subj_paths = glob.glob(str(self.ds_path / "original" / "STAGES PSGs" / "*.edf"), recursive=True)
        # /polysomnography/edfs/mesa-sleep-NNNN.edf -> NNNN
        subjs = [os.path.splitext(os.path.basename(p))[0].split("-")[-1] for p in subj_paths]
        subjs.sort()
        return subjs

    @property
    def train_subject_ids(self) -> list[str]:
        """Get train subject ids"""
        return self.subject_ids[: int(0.8 * len(self.subject_ids))]

    @property
    def test_subject_ids(self) -> list[str]:
        """Get test subject ids"""
        return self.subject_ids[int(0.8 * len(self.subject_ids)) :]

    @property
    def signal_names(self) -> list[str]:
        """Signal names as they appear in the EDF files"""
        return [
            # Actigraphy
            "activity",
            "linetime",
            "whitelight",
            "offwrist",
            "wake",
            # PSG
            "EKG",
            "EOG-L",
            "EOG-R",
            "EMG",
            "EEG1",
            "EEG2",
            "EEG3",
            "Pres",
            "Flow",
            "Snore",
            "Thor",
            "Abdo",
            "Leg",
            "Aux_AC",
            "Therm",
            "Pos",
            "Pleth",
            "OxStatus",
            "SpO2",
            "HR",
            "DHR",
        ]

    def set_sleep_mapping(self, mapping: Callable[[int], int]):
        """Set sleep mapping"""
        self.sleep_mapping = mapping

    def uniform_subject_generator(
        self,
        subject_ids: list[str] | None = None,
        repeat: bool = True,
        shuffle: bool = True,
    ) -> SubjectGenerator:
        """Yield Subject IDs uniformly.

        Args:
            subject_ids (list[str], optional): Array of subject ids. Defaults to None.
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle subject ids. Defaults to True.

        Returns:
            SubjectGenerator: Subject generator
        """
        if subject_ids is None:
            subject_ids = self.subject_ids
        subject_idxs = list(range(len(subject_ids)))
        while True:
            if shuffle:
                random.shuffle(subject_idxs)
            for subject_idx in subject_idxs:
                subject_id = subject_ids[subject_idx]
                yield subject_id.decode("ascii") if isinstance(subject_id, bytes) else subject_id
            # END FOR
            if not repeat:
                break
        # END WHILE

    def signal_generator(
        self, subject_generator: SubjectGenerator, signals: list[str], samples_per_subject: int = 1
    ) -> SampleGenerator:
        """Randomly generate frames of sleep data for given subjects.
        Args:
            subject_generator (SubjectGenerator): Generator that yields subject ids.
            samples_per_subject (int): Samples per subject.
        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, num_signals)
        """
        subjs_sleep_stages = dict()
        # subjs_apnea_events = dict()
        for subject_id in subject_generator:
            max_size = int(self.target_rate * self.get_subject_duration(subject_id))
            if subject_id in subjs_sleep_stages:
                sleep_stages = subjs_sleep_stages[subject_id]
            else:
                sleep_stages = self.extract_sleep_stages(subject_id=subject_id)
                subjs_sleep_stages[subject_id] = sleep_stages
            # END IF
            sleep_mask = self.sleep_stages_to_mask(sleep_stages, max_size)
            # if subject_id in subjs_apnea_events:
            #     apnea_events = subjs_apnea_events[subject_id]
            # else:
            #     apnea_events = self.extract_sleep_apneas(subject_id=subject_id)
            #     subjs_apnea_events[subject_id] = apnea_events
            # # END IF
            # apnea_mask = self.apnea_events_to_mask(apnea_events, max_size)

            x = np.zeros((self.frame_size, len(signals)), dtype=np.float32)
            y = np.zeros((self.frame_size,), dtype=np.int32)
            for _ in range(samples_per_subject):
                frame_start = random.randint(0, max_size - 2 * self.frame_size)
                frame_end = frame_start + self.frame_size
                for i, signal_label in enumerate(signals):
                    signal_label = signal_label.decode("ascii") if isinstance(signal_label, bytes) else signal_label
                    signal = self.load_signal_for_subject(
                        subject_id, signal_label=signal_label, start=frame_start, data_size=self.frame_size
                    )
                    signal_len = min(signal.size, x.shape[0])
                    x[:signal_len, i] = signal[:signal_len]
                # END FOR
                y = sleep_mask[frame_start:frame_end]
                yield x, y
            # END FOR
        # END FOR

    def _load_actigraphy_signal_for_subject(
        self, subject_id: str, signal_label: str, start: int = 0, data_size: int | None = None
    ) -> npt.NDArray[np.float32]:
        if data_size is None:
            raise ValueError("data_size must be specified for actigraphy signals")

        overlap_path = str(self.ds_path / "overlap" / "mesa-actigraphy-psg-overlap.csv")
        df = pd.read_csv(overlap_path)
        line = df[df["mesaid"] == int(subject_id)].line.to_numpy()
        if len(line) != 1:
            raise ValueError(f"Invalid line for subject {subject_id}")

        df = pd.read_csv(self._get_subject_actigraphy_path(subject_id))
        df = df[df["line"] >= line[0]]
        l_idx = math.floor(start / self.target_rate / 30.0)
        r_idx = l_idx + math.ceil(data_size / self.target_rate / 30.0)
        signal = df[signal_label][l_idx:r_idx].to_numpy()
        # Upsample signal to target rate
        signal = np.repeat(signal, 30 * self.target_rate)
        return signal[:data_size]

    def load_signal_for_subject(
        self, subject_id: str, signal_label: str, start: int = 0, data_size: int | None = None
    ) -> npt.NDArray[np.float32]:
        """Load signal into memory for subject at target rate (resampling if needed)
        Args:
            subject_id (str): Subject ID
            signal_label (str): Signal label
            start (int): Start location @ target rate
            data_size (int): Data length @ target rate
        Returns:
            npt.NDArray[np.float32]: Signal
        """
        if signal_label in ["activity", "linetime", "whitelight", "offwrist", "wake"]:
            return self._load_actigraphy_signal_for_subject(subject_id, signal_label, start, data_size)

        with pyedflib.EdfReader(self._get_subject_edf_path(subject_id)) as fp:
            signal_labels = fp.getSignalLabels()
            signal_idx = signal_labels.index(signal_label)
            sample_rate = fp.samplefrequency(signal_idx)
            sig_start = round(start * (sample_rate / self.target_rate))
            sig_len = fp.getNSamples()
            sig_duration = sig_len if data_size is None else math.ceil(data_size * (sample_rate / self.target_rate))
            signal = fp.readSignal(signal_idx, sig_start, sig_duration, digital=False).astype(np.float32)
        # END WITH
        if sample_rate != self.target_rate:
            signal = pk.signal.resample_signal(signal, sample_rate, self.target_rate)
        if data_size is None:
            return signal
        return signal[:data_size]

    def extract_sleep_apneas(self, subject_id: str) -> list[tuple[int, float, float]]:
        """Extract sleep apnea events for subject
        Args:
            subject_id (str): Subject ID
        Returns:
            list[tuple[int, float, float]]: Apnea events (apnea, start_time, duration)
        """

        def get_first_element_by_tag_name(element: XmlElement, tag_name: str) -> XmlNode | None:
            """Get first element matching tag name"""
            elements = element.getElementsByTagName(tag_name)
            return elements[0] if elements else None

        def has_element_by_tag_name(element: XmlElement, tag_name: str) -> bool:
            """Check if element has child element matching tag name"""
            return bool(get_first_element_by_tag_name(element, tag_name))

        def element_has_node_value(element: XmlElement, node_value) -> bool:
            """Check if element has child node with value"""
            return any((node for node in element.childNodes if node.nodeValue == node_value))

        def is_apnea_event(event: XmlElement) -> bool:
            """Determine if event is an apnea event"""
            event_type = get_first_element_by_tag_name(event, "EventType")
            return all(
                (
                    event_type is not None,
                    element_has_node_value(event_type, "Respiratory|Respiratory"),
                    has_element_by_tag_name(event, "EventConcept"),
                    has_element_by_tag_name(event, "Duration"),
                    has_element_by_tag_name(event, "Start"),
                )
            )

        xml_path = self._get_subject_xml_path(subject_id=subject_id)
        doc = xml_parse(xml_path)
        events = doc.getElementsByTagName("ScoredEvent")
        events = [event for event in events if is_apnea_event(event)]
        apneas = []
        for event in events:
            event_label = get_first_element_by_tag_name(event, "EventConcept").childNodes[0].nodeValue
            start_time = float(get_first_element_by_tag_name(event, "Start").childNodes[0].nodeValue)
            duration = float(get_first_element_by_tag_name(event, "Duration").childNodes[0].nodeValue)
            apnea_labels = ["Hypopnea|Hypopnea", "Unsure|Unsure", "Obstructive apnea|Obstructive Apnea"]
            try:
                apnea = apnea_labels.index(event_label) + 1
            except ValueError:
                continue
            apneas.append((apnea, start_time, duration))
        return apneas

    def extract_sleep_stages(self, subject_id: str) -> list[tuple[int, float, float]]:
        """Extract sleep stages for subject
        Args:
            subject_id (str): Subject ID
        Returns:
            list[tuple[int, float, float]]: Sleep stages (stage, start_time, duration)
        """

        def get_first_element_by_tag_name(element: XmlElement, tag_name: str) -> XmlNode | None:
            """Get first element matching tag name"""
            elements = element.getElementsByTagName(tag_name)
            return elements[0] if elements else None

        def has_element_by_tag_name(element: XmlElement, tag_name: str) -> bool:
            """Check if element has child element matching tag name"""
            return bool(get_first_element_by_tag_name(element, tag_name))

        def element_has_node_value(element: XmlElement, node_value):
            """Check if element has child node with value"""
            return any((node for node in element.childNodes if node.nodeValue == node_value))

        def is_sleep_stage_event(event: XmlElement) -> bool:
            """Check if event is a sleep stage event"""
            event_type = get_first_element_by_tag_name(event, "EventType")
            return all(
                (
                    event_type is not None,
                    element_has_node_value(event_type, "Stages|Stages"),
                    has_element_by_tag_name(event, "EventConcept"),
                    has_element_by_tag_name(event, "Duration"),
                    has_element_by_tag_name(event, "Start"),
                )
            )

        xml_path = self._get_subject_xml_path(subject_id=subject_id)
        doc = xml_parse(xml_path)
        events = doc.getElementsByTagName("ScoredEvent")
        events = [event for event in events if is_sleep_stage_event(event)]
        sleep_stages: list[tuple[int, float, float]] = []
        for event in events:
            stage_label = get_first_element_by_tag_name(event, "EventConcept").childNodes[0].nodeValue
            start_time = float(get_first_element_by_tag_name(event, "Start").childNodes[0].nodeValue)
            duration = float(get_first_element_by_tag_name(event, "Duration").childNodes[0].nodeValue)
            sleep_stage = self.sleep_mapping(int(stage_label.split("|")[-1]))
            sleep_stages.append((sleep_stage, start_time, duration))
        return sleep_stages

    def get_subject_duration(self, subject_id: str) -> float:
        """Get subject duration in seconds"""
        with pyedflib.EdfReader(self._get_subject_edf_path(subject_id)) as fp:
            # return int(min(fp.getNSamples()/[fp.samplefrequency(i) for i in range(fp.signals_in_file)]))
            return fp.getFileDuration()

    def _get_subject_actigraphy_path(self, subject_id: str) -> str:
        return str(self.ds_path / "actigraphy" / f"mesa-sleep-{subject_id}.csv")

    def _get_subject_edf_path(self, subject_id: str) -> str:
        """Get subject EDF data path"""
        return str(self.ds_path / "polysomnography" / "edfs" / f"mesa-sleep-{subject_id}.edf")

    def _get_subject_xml_path(self, subject_id: str) -> str:
        """Get subject XML NSRR path"""
        return str(self.ds_path / "polysomnography" / "annotations-events-nsrr" / f"mesa-sleep-{subject_id}-nsrr.xml")

    def sleep_stages_to_mask(
        self, sleep_stages: list[tuple[int, float, float]], data_size: int
    ) -> npt.NDArray[np.int32]:
        """Convert sleep stages to mask array
        Args:
            sleep_stages (list[tuple[int, float, float]]): Sleep stages
            data_size (int): Data size
        Returns:
            npt.NDArray[np.int32]: Sleep mask
        """
        sleep_mask = np.zeros(data_size, dtype=np.int32)
        for sleep_stage, start_time, duration in sleep_stages:
            left_idx = int(self.target_rate * start_time)
            right_idx = left_idx + int(self.target_rate * duration)
            sleep_mask[left_idx : right_idx + 1] = sleep_stage
        # END FOR
        return sleep_mask

    def apnea_events_to_mask(
        self, apnea_events: list[tuple[int, float, float]], data_size: int
    ) -> npt.NDArray[np.int32]:
        """Convert apnea events to mask array
        Args:
            apnea_events (list[tuple[int, float, float]]): Apnea events
            data_size (int): Data size
        Returns:
            npt.NDArray[np.int32]: Apnea mask
        """
        apnea_mask = np.zeros(data_size, dtype=np.int32)
        for apnea_event, start_time, duration in apnea_events:
            left_idx = int(self.target_rate * start_time)
            right_idx = left_idx + int(self.target_rate * duration)
            apnea_mask[left_idx : right_idx + 1] = apnea_event
        # END FOR
        return apnea_mask