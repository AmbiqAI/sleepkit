from enum import IntEnum, StrEnum


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


def get_stage_classes(nstages: int) -> list[int]:
    """Get target classes for sleep stage classification

    Args:
        nstages (int): Number of sleep stages

    Returns:
        list[int]: Target classes
    """
    if 2 <= nstages <= 5:
        return list(range(nstages))
    raise ValueError(f"Invalid number of stages: {nstages}")


def get_stage_class_mapping(nstages: int) -> dict[int, int]:
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


def get_stage_class_names(nstages: int) -> list[str]:
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
