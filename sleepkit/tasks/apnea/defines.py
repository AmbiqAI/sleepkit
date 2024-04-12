from enum import IntEnum


class SleepApnea(IntEnum):
    """Sleep apnea class"""

    none = 0
    hypopnea = 1
    central = 2
    obstructive = 3
    mixed = 4
    noise = 5
