from enum import IntEnum, StrEnum


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


def get_sleep_apnea_classes(nstages: int) -> list[int]:
    """Get target classes for sleep apnea classification
    Args:
        nstages (int): Number of apnea stages
    Returns:
        list[int]: Target classes
    """
    if nstages in (2, 3):
        return list(range(nstages))
    raise ValueError(f"Invalid number of stages: {nstages}")


def get_sleep_apnea_class_mapping(nstages: int) -> dict[int, int]:
    """Get class mapping for sleep apnea classification
    Args:
        nstages (int): Number of sleep apnea stages
    Returns:
        dict[int, int]: Class mapping
    """
    if nstages == 2:
        return {
            SleepApnea.none: 0,
            SleepApnea.hypopnea: 1,
            SleepApnea.central: 1,
            SleepApnea.obstructive: 1,
            SleepApnea.mixed: 1,
        }
    if nstages == 3:
        return {
            SleepApnea.none: 0,
            SleepApnea.central: 1,
            SleepApnea.obstructive: 1,
            SleepApnea.mixed: 1,
            SleepApnea.hypopnea: 2,
        }
    raise ValueError(f"Invalid number of stages: {nstages}")


def get_sleep_apnea_class_names(nstages: int):
    """Get class names for sleep apnea classification
    Args:
        nstages (int): Number of sleep apnea stages
    Returns:
        list[str]: Class names
    """
    if nstages == 2:
        return ["NORM", "APNEA"]
    if nstages == 3:
        return ["NORM", "APNEA", "HYPOPNEA"]
    raise ValueError(f"Invalid number of stages: {nstages}")
