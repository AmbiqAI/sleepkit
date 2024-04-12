from enum import IntEnum


class SleepStage(IntEnum):
    """Sleep stage class"""

    wake = 0
    stage1 = 1
    stage2 = 2
    stage3 = 3
    stage4 = 4
    rem = 5
    noise = 6
