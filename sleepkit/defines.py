from enum import IntEnum, Enum

class SleepStage(IntEnum):
    """Sleep stage class"""
    wake = 0
    stage1 = 1
    stage2 = 2
    stage3 = 3
    rem = 4
    noise = 5

class SleepStageName(str, Enum):
    """Sleep stage name"""
    wake = "wake"
    stage1 = "stage1"
    stage2 = "stage2"
    stage3 = "stage3"
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

class SleepApneaName(str, Enum):
    """Sleep apnea name"""
    none = "none"
    hypopnea = "hypopnea"
    central = "central"
    obstructive = "obstructive"
    mixed = "mixed"
    noise = "noise"
