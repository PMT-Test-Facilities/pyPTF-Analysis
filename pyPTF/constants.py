import numpy as np 
from pyPTF.enums import PMT

"""
    This is just a handful of constants that I stole from the old analysis code 
"""

MAXSAMPLES = 6000
SAMPLESIZE=70

PULSE_LOCATION_CUT= 10

PTF_CAEN_V1730_SAMPLE_RATE = 500 # MS/s
PTF_CAEN_V1730_FULL_SCALE_RANGE = 2.0 # Vpp
PTF_CAEN_V1730_RESOLUTION = 14 # bits
PTF_SCALE = PTF_CAEN_V1730_FULL_SCALE_RANGE / (2**PTF_CAEN_V1730_RESOLUTION)
print("Scaler {}".format(PTF_SCALE))


MPMT_DIGITIZER_SAMPLE_RATE = 125
MPMT_DIGITIZER_FULL_SCALE_RANGE = 2.0
MPMT_DIGITIZER_RESOLUTION = 12

MAX_PULSES = 10

PTF_TS = np.linspace(0, float(SAMPLESIZE)*1000/PTF_CAEN_V1730_SAMPLE_RATE, SAMPLESIZE)
PTF_SAMPLE_WIDTH = PTF_TS[1]-PTF_TS[0]
MPMT_TS = np.linspace(0, float(SAMPLESIZE)*1000/MPMT_DIGITIZER_SAMPLE_RATE, SAMPLESIZE)

def get_PMT_channel(which:PMT):
    if which==PMT.Hamamatsu_R3600_PMT:
        return 0
    elif which==PMT.PTF_Monitor_PMT:
        return 2
    else:
        print(which)
        raise NotImplementedError()
