from dataclasses import dataclass

import pytest

pytest.importorskip("helia_edge")

import sleepkit.backends.utils as utils  # noqa: E402


@dataclass
class DummyPort:
    device: str
    vid: int
    pid: int
    serial_number: str | None = None
    manufacturer: str | None = None
    product: str | None = None


class DummySerialTransport:
    def __init__(self, device, baudrate=None):
        self.device = device
        self.baudrate = baudrate


def test_find_serial_device_matches_fields(monkeypatch):
    ports = [
        DummyPort(
            device="/dev/ttyUSB0",
            vid=1234,
            pid=5678,
            serial_number="ABC123",
            manufacturer="Acme",
            product="Widget",
        ),
        DummyPort(
            device="/dev/ttyUSB1",
            vid=1111,
            pid=2222,
            serial_number="XYZ789",
            manufacturer="Other",
            product="Gadget",
        ),
    ]

    monkeypatch.setattr(utils, "list_ports", lambda: ports)

    port = utils._find_serial_device(vid_pid="1234:5678")
    assert port is ports[0]

    port = utils._find_serial_device(manufacturer="acme", product="widget")
    assert port is ports[0]

    port = utils._find_serial_device(serial_number="XYZ789")
    assert port is ports[1]


def test_get_serial_transport_uses_first_matching_port(monkeypatch):
    dummy_port = DummyPort(device="/dev/ttyUSB9", vid=1, pid=2)
    monkeypatch.setattr(utils, "_find_serial_device", lambda **kwargs: dummy_port)
    monkeypatch.setattr(utils, "SerialTransport", DummySerialTransport)

    transport = utils.get_serial_transport(vid_pid="1:2", baudrate=115200)
    assert transport.device == "/dev/ttyUSB9"
    assert transport.baudrate == 115200


def test_get_serial_transport_times_out(monkeypatch):
    monkeypatch.setattr(utils, "_find_serial_device", lambda **kwargs: None)
    monkeypatch.setattr(utils, "time", utils.time)

    times = iter([0.0, 0.1, 0.2, 10.1])
    monkeypatch.setattr(utils.time, "time", lambda: next(times))
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="Unable to locate serial port"):
        utils.get_serial_transport(timeout=0.3)
