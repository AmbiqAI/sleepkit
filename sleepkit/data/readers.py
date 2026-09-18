"""Optional file adapters; preprocessing never needs to know file formats."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .records import Annotation, Record, Signal


def save_record(record, path):
    """Portable array adapter for user data and synthetic examples (no pickle)."""
    metadata = {
        "dataset": record.dataset,
        "subject": record.subject,
        "recording": record.recording,
        "source_revision": record.source_revision,
        "annotations": [asdict(a) for a in record.annotations],
        "signals": {},
    }
    arrays = {}
    for index, (name, signal) in enumerate(record.signals.items()):
        key = f"signal_{index}"
        metadata["signals"][name] = {
            "key": key,
            "sample_rate": signal.sample_rate,
            "unit": signal.unit,
            "modality": signal.modality,
            "location": signal.location,
            "start": signal.start,
        }
        arrays[key] = signal.values
        arrays[key + "_valid"] = signal.valid
    np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)


def read_record(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        signals = {}
        for name, description in metadata.pop("signals").items():
            key = description.pop("key")
            signals[name] = Signal(saved[key], valid=saved[key + "_valid"], **description)
        annotations = tuple(Annotation(**a) for a in metadata.pop("annotations"))
        return Record(signals=signals, annotations=annotations, **metadata)


def read_edf(path, *, dataset, subject, channels, annotations=()):
    """Read native-rate EDF channels with explicit physiological semantics.

    channels maps source EDF labels to Channel objects. Physical units are checked
    rather than silently converted. EDF+D discontinuities are rejected; represent
    gapped data with explicit sample masks in an array Record instead.
    """
    import pyedflib

    path = Path(path)
    with path.open("rb") as stream:
        stream.seek(192)
        if b"EDF+D" in stream.read(44):
            raise ValueError("Discontinuous EDF requires explicit gap handling")
    signals = {}
    with pyedflib.EdfReader(str(path)) as reader:
        names = reader.getSignalLabels()
        for source, channel in channels.items():
            if source not in names:
                raise ValueError(f"Missing EDF signal {source!r}")
            if channel.name in signals:
                raise ValueError(f"Duplicate destination signal {channel.name!r}")
            index = names.index(source)
            unit = reader.getPhysicalDimension(index).strip()
            if unit != channel.unit:
                raise ValueError(f"EDF unit for {source}: {unit!r}, expected {channel.unit!r}")
            signals[channel.name] = Signal(
                reader.readSignal(index), reader.getSampleFrequency(index), unit, channel.modality, channel.location
            )
    with path.open("rb") as stream:
        revision = hashlib.file_digest(stream, "sha256").hexdigest()
    return Record(dataset, subject, path.stem, signals, tuple(annotations), revision)


def read_nsrr_stages(path, class_map):
    """Decode NSRR stage intervals; unmapped stages remain unlabeled gaps.

    The caller owns the target mapping, including whether NREM stages are merged.
    Times are seconds from the associated EDF start.
    """
    annotations = []
    for event in ET.parse(path).getroot().iter("ScoredEvent"):
        if event.findtext("EventType") != "Stages|Stages":
            continue
        source = int(event.findtext("EventConcept").rsplit("|", 1)[-1])
        if source not in class_map:
            continue
        start = float(event.findtext("Start"))
        annotations.append(Annotation(start, start + float(event.findtext("Duration")), class_map[source]))
    return tuple(annotations)
