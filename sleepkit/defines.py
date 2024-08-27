import os
import tempfile
from enum import StrEnum, IntEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from neuralspot_edge.converters.tflite import QuantizationType, ConversionType


class NamedParams(BaseModel, extra="allow"):
    """
    Named parameters is used to store parameters for a specific model, preprocessing, or augmentation.
    Typically name refers to class/method name and params is provided as kwargs.

    Attributes:
        name: Name
        params: Parameters
    """

    name: str = Field(..., description="Name")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters")


class TaskMode(StrEnum):
    """Task run mode"""

    download = "download"
    feature = "feature"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    demo = "demo"


class QuantizationParams(BaseModel, extra="allow"):
    """Quantization parameters

    Attributes:
        enabled: Enable quantization
        qat: Enable quantization aware training (QAT)
        format: Quantization mode
        io_type: I/O type
        conversion: Conversion method
        debug: Debug quantization
        fallback: Fallback to float32
    """

    enabled: bool = Field(False, description="Enable quantization")
    qat: bool = Field(False, description="Enable quantization aware training (QAT)")
    format: QuantizationType = Field(QuantizationType.INT8, description="Quantization mode")
    io_type: str = Field("int8", description="I/O type")
    conversion: ConversionType = Field(ConversionType.KERAS, description="Conversion method")
    debug: bool = Field(False, description="Debug quantization")
    fallback: bool = Field(False, description="Fallback to float32")


class FeatureParams(BaseModel, extra="allow"):
    """Feature configuration params

    Attributes:
        name: Feature set name
        sampling_rate: Target sampling rate (Hz)
        frame_size: Frame size in samples
        loader: Data loader
        feat_key: Feature key
        label_key: Label key
        mask_key: Mask key
        feat_cols: Feature columns
        save_path: Save path
        params: Feature Parameters
    """

    name: str = Field("feature", description="Feature set name")
    sampling_rate: float = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size in samples")
    loader: str = Field("hdf5", description="Data loader")
    feat_key: str = Field("features", description="Feature key")
    label_key: str = Field("labels", description="Label key")
    mask_key: str = Field("mask", description="Mask key")
    feat_cols: list[str] | None = Field(None, description="Feature columns")
    save_path: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()), description="Save path")
    params: dict[str, Any] = Field(default_factory=dict, description="Feature Parameters")


class TaskParams(BaseModel, extra="allow"):
    """Task configuration params"""

    # Common arguments
    name: str = Field("experiment", description="Experiment name")
    project: str = Field("sleepkit", description="Project name")
    job_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()),
        description="Job output directory",
    )

    # Dataset arguments
    datasets: list[NamedParams] = Field(default_factory=list, description="Datasets")
    dataset_weights: list[float] | None = Field(None, description="Dataset weights")
    force_download: bool = Field(False, description="Force download dataset- overriding existing files")

    # Feature arguments
    feature: FeatureParams = Field(default_factory=FeatureParams, description="Feature configuration")

    # Signal arguments
    sampling_rate: float = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size in samples")

    # Dataloader arguments
    samples_per_subject: int | list[int] = Field(1000, description="Number train samples per subject")
    val_samples_per_subject: int | list[int] = Field(1000, description="Number validation samples per subject")
    test_samples_per_subject: int | list[int] = Field(1000, description="Number test samples per subject")

    # Preprocessing/Augmentation arguments
    preprocesses: list[NamedParams] = Field(default_factory=list, description="Preprocesses")
    augmentations: list[NamedParams] = Field(default_factory=list, description="Augmentations")

    # Class arguments
    num_classes: int = Field(1, description="# of classes")
    class_map: dict[int, int] = Field(default_factory=lambda: {1: 1}, description="Class/label mapping")
    class_names: list[str] | None = Field(default=None, description="Class names")

    # Split arguments
    train_subjects: float | None = Field(None, description="# or proportion of subjects for training")
    val_subjects: float | None = Field(None, description="# or proportion of subjects for validation")
    test_subjects: float | None = Field(None, description="# or proportion of subjects for testing")

    # Val/Test dataset arguments
    val_file: Path | None = Field(None, description="Path to load/store TFDS validation data")
    test_file: Path | None = Field(None, description="Path to load/store TFDS test data")
    val_size: int | None = Field(None, description="Number of samples for validation")
    test_size: int = Field(10000, description="# samples for testing")

    # Model arguments
    resume: bool = Field(False, description="Resume training")
    architecture: NamedParams | None = Field(default=None, description="Custom model architecture")
    model_file: Path | None = Field(None, description="Path to load/save model file (.keras)")
    use_logits: bool = Field(True, description="Use logits output or softmax")
    weights_file: Path | None = Field(None, description="Path to a checkpoint weights to load/save")
    quantization: QuantizationParams = Field(default_factory=QuantizationParams, description="Quantization parameters")

    # Training arguments
    lr_rate: float = Field(1e-3, description="Learning rate")
    lr_cycles: int = Field(3, description="Number of learning rate cycles")
    lr_decay: float = Field(0.9, description="Learning rate decay")
    label_smoothing: float = Field(0, description="Label smoothing")
    batch_size: int = Field(32, description="Batch size")
    buffer_size: int = Field(100, description="Buffer cache size")
    epochs: int = Field(50, description="Number of epochs")
    steps_per_epoch: int = Field(10, description="Number of steps per epoch")
    val_steps_per_epoch: int = Field(10, description="Number of validation steps")
    val_metric: Literal["loss", "acc", "f1"] = Field("loss", description="Performance metric")
    class_weights: Literal["balanced", "fixed"] = Field("fixed", description="Class weights")

    # Evaluation arguments
    threshold: float | None = Field(None, description="Model output threshold")
    test_metric: Literal["loss", "acc", "f1"] = Field("acc", description="Test metric")
    test_metric_threshold: float | None = Field(0.98, description="Validation metric threshold")

    # Export arguments
    tflm_var_name: str = Field("g_model", description="TFLite Micro C variable name")
    tflm_file: Path | None = Field(None, description="Path to copy TFLM header file (e.g. ./model_buffer.h)")

    # Demo arguments
    backend: str = Field("pc", description="Backend")
    demo_size: int | None = Field(1000, description="# samples for demo")
    display_report: bool = Field(True, description="Display report")

    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    num_workers: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="Number of workers for parallel processing",
    )
    verbose: int = Field(1, ge=0, le=2, description="Verbosity level")
    model_config = ConfigDict(protected_namespaces=())

    def model_post_init(self, __context: Any) -> None:
        """Post init hook"""

        if self.val_file and len(self.val_file.parts) == 1:
            self.val_file = self.job_dir / self.val_file

        if self.test_file and len(self.test_file.parts) == 1:
            self.test_file = self.job_dir / self.test_file

        if self.model_file and len(self.model_file.parts) == 1:
            self.model_file = self.job_dir / self.model_file

        if self.weights_file and len(self.weights_file.parts) == 1:
            self.weights_file = self.job_dir / self.weights_file

        if self.tflm_file and len(self.tflm_file.parts) == 1:
            self.tflm_file = self.job_dir / self.tflm_file


class SleepApnea(IntEnum):
    """Sleep apnea class

    Attributes:
        none: None
        apnea: Apnea
        hypopnea: Hypopnea
        central: Central
        obstructive: Obstructive
        mixed: Mixed
        noise: Noise
    """

    none = 0
    hypopnea = 1
    central = 2
    obstructive = 3
    mixed = 4
    noise = 5


class SleepStage(IntEnum):
    """Sleep stage class

    Attributes:
        wake: Wake
        stage1: Stage 1
        stage2: Stage 2
        stage3: Stage 3
        stage4: Stage 4
        rem: REM
        noise: Noise
    """

    wake = 0
    stage1 = 1
    stage2 = 2
    stage3 = 3
    stage4 = 4
    rem = 5
    noise = 6
