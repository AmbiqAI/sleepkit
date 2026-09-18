import numpy as np
import pytest

from sleepkit.data.readers import read_edf, read_nsrr_stages
from sleepkit.preprocessing import Channel, WindowFeatures


def test_nsrr_mapping_preserves_unlabeled_gaps(tmp_path):
    path = tmp_path / "stages.xml"
    path.write_text("""<PSGAnnotation><ScoredEvents>
      <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept><Start>0</Start><Duration>30</Duration></ScoredEvent>
      <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Unknown|9</EventConcept><Start>30</Start><Duration>30</Duration></ScoredEvent>
      <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 2|2</EventConcept><Start>60</Start><Duration>30</Duration></ScoredEvent>
    </ScoredEvents></PSGAnnotation>""")
    annotations = read_nsrr_stages(path, {0: 0, 2: 1})
    assert [(a.start, a.stop, a.label) for a in annotations] == [(0, 30, 0), (60, 90, 1)]


def test_edf_adapter_uses_same_transform_as_arrays(tmp_path):
    pyedflib = pytest.importorskip("pyedflib")
    from sleepkit.data import Record, Signal

    path = tmp_path / "signals.edf"
    raw = np.arange(8, dtype=float)
    header = {
        "label": "X",
        "dimension": "g",
        "sample_frequency": 2,
        "physical_min": -10,
        "physical_max": 10,
        "digital_min": -32768,
        "digital_max": 32767,
        "transducer": "",
        "prefilter": "",
    }
    with pyedflib.EdfWriter(str(path), 1) as writer:
        writer.setSignalHeaders([header])
        writer.writeSamples([raw])
    channel = Channel("movement", "g", "accelerometer", "wrist")
    loaded = read_edf(path, dataset="example", subject="one", channels={"X": channel})
    direct = Record("example", "one", "signals", {"movement": Signal(raw, 2, "g", "accelerometer", "wrist")})
    transform = WindowFeatures((channel,), 2, 2)
    np.testing.assert_allclose(transform(loaded).values, transform(direct).values, atol=0.001)
    with pytest.raises(ValueError, match="unit"):
        read_edf(
            path,
            dataset="example",
            subject="one",
            channels={"X": Channel("movement", "m/s2", "accelerometer", "wrist")},
        )
