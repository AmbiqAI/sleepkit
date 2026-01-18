from pathlib import Path

import pytest

pytest.importorskip("helia_edge")

from sleepkit.defines import TaskParams  # noqa: E402


def test_taskparams_resolves_relative_paths(tmp_path):
    params = TaskParams(
        job_dir=tmp_path,
        model_file=Path("model.keras"),
        weights_file=Path("weights.ckpt"),
        val_file=Path("val.tfrecord"),
        test_file=Path("test.tfrecord"),
        tflm_file=Path("model_buffer.h"),
    )

    assert params.model_file == tmp_path / "model.keras"
    assert params.weights_file == tmp_path / "weights.ckpt"
    assert params.val_file == tmp_path / "val.tfrecord"
    assert params.test_file == tmp_path / "test.tfrecord"
    assert params.tflm_file == tmp_path / "model_buffer.h"


def test_taskparams_keeps_absolute_paths(tmp_path):
    absolute_path = tmp_path / "abs.keras"
    params = TaskParams(job_dir=tmp_path, model_file=absolute_path)
    assert params.model_file == absolute_path
