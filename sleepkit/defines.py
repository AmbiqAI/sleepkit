import os
from enum import IntEnum, StrEnum
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Extra, Field

class SleepStage(IntEnum):
    """Sleep stage class"""
    wake = 0
    stage1 = 1
    stage2 = 2
    stage3 = 3
    stage4 = 4
    rem = 5
    noise = 6

class SleepStageName(StrEnum):
    """Sleep stage name"""
    wake = "wake"
    stage1 = "stage1"
    stage2 = "stage2"
    stage3 = "stage3"
    stage4 = "stage4"
    rem = "rem"
    noise = "noise"

class SleepApnea(IntEnum):
    """Sleep apnea class"""
    none = 0
    hypopnea = 1
    central = 2
    obstructive = 3
    mixed = 4
    noise = 5

class SleepApneaName(StrEnum):
    """Sleep apnea name"""
    none = "none"
    hypopnea = "hypopnea"
    central = "central"
    obstructive = "obstructive"
    mixed = "mixed"
    noise = "noise"

class SKFeatureParams(BaseModel, extra=Extra.allow):
    feature_set: str = Field(description="Feature set name")
    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    save_path: Path = Field(default_factory=Path, description="Save directory")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )

class SKTrainParams(BaseModel, extra=Extra.allow):
    """SleepKit train command params"""
    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    samples_per_subject: int | list[int] = Field(1000, description="# train samples per patient")
    val_samples_per_subject: int | list[int] = Field(1000, description="# validation samples per patient")
    train_subjects: float | None = Field(None, description="# or proportion of patients for training")
    val_subjects: float | None = Field(None, description="# or proportion of patients for validation")
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
    predict = "predict"
    demo = "demo"


def get_sleep_stage_classes(nstages: int) -> list[int]:
    """Get target classes for sleep stage classification
    Args:
        nstages (int): Number of sleep stages
    Returns:
        list[int]: Target classes
    """
    if 2 <= nstages <= 5:
        return list(range(nstages))
    raise ValueError(f"Invalid number of stages: {nstages}")

def get_sleep_stage_class_mapping(nstages: int) -> dict[int, int]:
    """Get class mapping for sleep stage classification
    Args:
        nstages (int): Number of sleep stages
    Returns:
        dict[int, int]: Class mapping
    """
    if nstages == 2:
        return {
            SleepStage.wake: 0,
            SleepStage.stage1: 1,
            SleepStage.stage2: 1,
            SleepStage.stage3: 1,
            SleepStage.stage4: 1,
            SleepStage.rem: 1,
        }
    if nstages == 3:
        return {
            SleepStage.wake: 0,
            SleepStage.stage1: 1,
            SleepStage.stage2: 1,
            SleepStage.stage3: 1,
            SleepStage.stage4: 1,
            SleepStage.rem: 2,
        }
    if nstages == 4:
        return {
            SleepStage.wake: 0,
            SleepStage.stage1: 1,
            SleepStage.stage2: 1,
            SleepStage.stage3: 2,
            SleepStage.stage4: 2,
            SleepStage.rem: 3,
        }
    if nstages == 5:
        return {
            SleepStage.wake: 0,
            SleepStage.stage1: 1,
            SleepStage.stage2: 2,
            SleepStage.stage3: 3,
            SleepStage.stage4: 3,
            SleepStage.rem: 4,
        }
    raise ValueError(f"Invalid number of stages: {nstages}")


def get_sleep_stage_class_names(nstages: int):
    """Get class names for sleep stage classification
    Args:
        nstages (int): Number of sleep stages
    Returns:
        list[str]: Class names
    """
    if nstages == 2:
        return ["WAKE", "SLEEP"]
    if nstages == 3:
        return ["WAKE", "NREM", "REM"]
    if nstages == 4:
        return ["WAKE", "CORE", "DEEP", "REM"]
    if nstages == 5:
        return ["WAKE", "N1", "N2", "N3", "REM"]
    raise ValueError(f"Invalid number of stages: {nstages}")
