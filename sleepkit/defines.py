import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DatasetTypes = Literal["mesa", "stages", "cmidss"]


class SKDownloadParams(BaseModel, extra="allow"):
    """SleepKit download command params"""

    ds_path: Path = Field(default_factory=Path, description="Dataset root directory")
    datasets: list[DatasetTypes] = Field(default_factory=list, description="Datasets")
    progress: bool = Field(True, description="Display progress bar")
    force: bool = Field(False, description="Force download dataset- overriding existing files")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )


class SKFeatureParams(BaseModel, extra="allow"):
    """SleepKit feature command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    ds_path: Path = Field(default_factory=Path, description="Dataset base directory")
    datasets: list[str] = Field(default_factory=list, description="Dataset names")
    feature_set: str = Field(description="Feature set name")
    feature_params: dict[str, Any] | None = Field(default=None, description="Custom feature parameters")
    save_path: Path = Field(default_factory=Path, description="Save directory")
    sampling_rate: float = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )


class SKTrainParams(BaseModel, extra="allow"):
    """SleepKit train command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset base directory")
    ds_handler: str = Field(description="Dataset handler name")
    ds_params: dict[str, Any] | None = Field(default_factory=dict, description="Dataset parameters")
    sampling_rate: float = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(2, description="# of classes")
    samples_per_subject: int | list[int] = Field(1000, description="# train samples per subject")
    val_samples_per_subject: int | list[int] = Field(1000, description="# validation samples per subject")
    train_subjects: float | None = Field(None, description="# or proportion of subjects for training")
    val_subjects: float | None = Field(None, description="# or proportion of subjects for validation")
    val_file: Path | None = Field(None, description="Path to load/store pickled validation file")
    val_size: int | None = Field(None, description="# samples for validation")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    # Model arguments
    model: str | None = Field(default=None, description="Custom model")
    model_file: str | None = Field(None, description="Path to model file")
    model_params: dict[str, Any] | None = Field(default=None, description="Custom model parameters")

    weights_file: Path | None = Field(None, description="Path to a checkpoint weights to load")
    quantization: bool | None = Field(None, description="Enable quantization aware training (QAT)")
    # Training arguments
    batch_size: int = Field(32, description="Batch size")
    buffer_size: int = Field(100, description="Buffer size")
    epochs: int = Field(50, description="Number of epochs")
    steps_per_epoch: int | None = Field(None, description="Number of steps per epoch")
    val_metric: Literal["loss", "acc", "f1"] = Field("loss", description="Performance metric")
    # augmentations: list[AugmentationParams] = Field(default_factory=list, description="Augmentations")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    model_config = ConfigDict(protected_namespaces=())


class SKTestParams(BaseModel, extra="allow"):
    """SleepKit test command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset base directory")
    ds_handler: str = Field(description="Dataset handler name")
    ds_params: dict[str, Any] | None = Field(default_factory=dict, description="Dataset parameters")
    sampling_rate: float = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(2, description="# of classes")
    test_subjects: float | None = Field(None, description="# or proportion of subjects for testing")
    test_size: int = Field(20_000, description="# samples for testing")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    # Model arguments
    model_file: str | None = Field(None, description="Path to model file")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    model_config = ConfigDict(protected_namespaces=())


class SKExportParams(BaseModel, extra="allow"):
    """Export command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset base directory")
    ds_handler: str = Field(description="Dataset handler name")
    ds_params: dict[str, Any] | None = Field(default_factory=dict, description="Dataset parameters")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(2, description="# of classes")
    samples_per_subject: int | list[int] = Field(100, description="# test samples per subject")
    test_subjects: float | None = Field(None, description="# or proportion of subjects for testing")
    test_size: int = Field(20_000, description="# samples for testing")
    model_file: str | None = Field(None, description="Path to model file")
    threshold: float | None = Field(None, description="Model output threshold")
    val_acc_threshold: float | None = Field(0.98, description="Validation accuracy threshold")
    use_logits: bool = Field(True, description="Use logits output or softmax")
    quantization: bool | None = Field(None, description="Enable post training quantization (PQT)")
    tflm_var_name: str = Field("g_model", description="TFLite Micro C variable name")
    tflm_file: Path | None = Field(None, description="Path to copy TFLM header file (e.g. ./model_buffer.h)")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    model_config = ConfigDict(protected_namespaces=())


class SKDemoParams(BaseModel, extra="allow"):
    """Demo command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset base directory")
    ds_handler: str = Field(description="Dataset handler name")
    ds_params: dict[str, Any] | None = Field(default_factory=dict, description="Dataset parameters")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(2, description="# of classes")
    # Model arguments
    model_file: str | None = Field(None, description="Path to model file")
    backend: Literal["pc", "evb"] = Field("pc", description="Backend")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    model_config = ConfigDict(protected_namespaces=())


class SKTask(StrEnum):
    """SleepKit task"""

    detect = "detect"
    stage = "stage"
    apnea = "apnea"


class SKMode(StrEnum):
    """SleepKit Mode"""

    download = "download"
    feature = "feature"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    demo = "demo"
