from enum import Enum


class PMT(Enum):
    Hamamatsu_R3600_PMT = 0
    PTF_Monitor_PMT = 2
    Timing_NotAPMT = 1
    Reference = 9