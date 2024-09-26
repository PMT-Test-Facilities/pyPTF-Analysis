import numpy as np 
from scipy.optimize import minimize, basinhopping
from scipy.fft import fft, fftfreq
from math import pi, log, sqrt
import h5py as h5 
import os 
from time import time
from tqdm import tqdm

from pyPTF.enums import PMT
from pyPTF.constants import get_PMT_channel, PTF_TS,PTF_SCALE, PULSE_LOCATION_CUT 
from pyPTF.utils import PointScan


# from . import DEBUG

fwhm_scaler= 2*sqrt(2*log(2))

DEBUG = False

if DEBUG:
    from matplotlib import pyplot as plt 

BOUNDS = np.array([
    [0.0, 1.0],
    [2.0, 138.0 ],
    [0.5, 20.0],
    [0.9, 1.1 ],
    [ 0.0, 1.1],
    [ 0.15, 0.4],
    [-3*pi, 3*pi]

])

def pulse_location_cut(waveform:np.ndarray)->bool:
    minbin = np.argmin(waveform) # +1 for consistency with the c++ root code
    nbins = len(waveform)
    if (minbin>=1 and minbin<=PULSE_LOCATION_CUT) or ((minbin >= nbins-PULSE_LOCATION_CUT) and minbin<=nbins):
        return False
    else:
        return True

def FFT_cut(waveform:np.ndarray)->bool:
    """
        Examines the waveform in frequency space, makes cut therein
    """
    transformed = np.abs(fft(waveform))
    transformed[0] = 0.0  # cut the pedestal out!

    nbins = len(waveform)
    maxbin = np.argmax(transformed)
    max_val = np.max(transformed)
    

    #       above thresh        peak is small              peak is near max
    good= (max_val>0.01 and ((maxbin>0 and maxbin<=3) or (maxbin>=nbins-3 and maxbin<=nbins)) )

    

    if False: #DEBUG and good:
        print("{} {} - {} {} {} {}".format(maxbin, nbins , max_val>0.01 , maxbin>0 and maxbin<=3, maxbin>=nbins-4,  maxbin<=nbins ))
        xf = fftfreq(len(waveform), PTF_TS[1]-PTF_TS[0])[:len(waveform)//2]
        plt.plot(xf, (len(waveform)*transformed[0:len(waveform)//2]))
        plt.show()
    return good



def get_charge_sum(waveform, bin_low=0, bin_high=0):
    """
        Determines the pedestal first, then the `Qsum` charge sum of the waveform over
            `bin_low` to `bin_high`
        If no `bin_high` is provided, it goes over the whole waveform 
    """
    
    if bin_high==0:
        bin_high = len(waveform)

    # will either go up to the 10th bin, or up to 50 below bin_low, to get the pedestal 
    ped_range = max([10, bin_low-50])

    pedestal = np.mean(waveform[:ped_range])
    
    qsum = 0
    for i in range(bin_low, bin_high):
        qsum += pedestal - waveform[i]
    
    return pedestal, qsum

def process_into_fitseries(meta_data:dict, which_pmt):
    """
        Takes a dictionary of run information and the desired PMT. 
        Then, it processes all of the scan points and fits, one scan point at a time 
    """

    channel = get_PMT_channel(which_pmt)
    dummy = channel==1

    waveform_file = os.path.join(
        meta_data["data_folder"],
        "convert_V1730_wave{}.h5".format(channel))
    
    waveform_data = h5.File(waveform_file, 'r')

    # go over the scan spots
    count = 0

    if dummy:
        keys = ["pulse_times","x","y","z","tilt","rot", "passing", ]
    else:

        keys = [
            "x","y","z","tilt","rot",
            "amplitudes","sigmas","means","peds",
            "n_pass", "pulse_times", "passing",
            "bfield", "charge_sum"
        ]

    outdata = {key:[] for key in keys}
    

    for keyname in tqdm(waveform_data.keys()):
        i = int(keyname.split("_")[1])
        these_waveforms = np.array(waveform_data[keyname], dtype=float)
        if len(these_waveforms)==0:
            print("No waveforms for scanpoint {}".format(i))
            continue
        
        # pointscan is kind of a holdover from a previous version
        # this holds the data for a single point measurement 
        this_ps = PointScan(
            meta_data["gantry0_x"][i],
            meta_data["gantry0_y"][i],
            meta_data["gantry0_z"][i],
            meta_data["gantry0_rot"][i],
            meta_data["gantry0_tilt"][i],
            which_pmt.value
        )

        #print(keyname)
        if dummy:
            this_ps.extract_timing_signal(these_waveforms)
        else:
            # extract waveform values 
            this_ps.extract_values(these_waveforms)

        outdata["x"]+=[meta_data["gantry0_x"][i],]*len(this_ps)
        outdata["y"]+=[meta_data["gantry0_y"][i],]*len(this_ps)
        outdata["z"]+=[meta_data["gantry0_z"][i],]*len(this_ps)
        outdata["rot"]+=[meta_data["gantry0_rot"][i],]*len(this_ps)
        outdata["tilt"]+=[meta_data["phidg0_tilt"][i],]*len(this_ps)
        
        outdata["pulse_times"] += this_ps.pulse_times
        outdata["passing"]+=this_ps.passing

        if not dummy:
            outdata["amplitudes"]+= this_ps.amplitudes
            outdata["bfield"]+=[meta_data["phidg0_Btot"][i],]*len(this_ps)
            outdata["sigmas"]+=this_ps.sigmas
            outdata["charge_sum"] += this_ps.charge_sum
            outdata["means"]+=this_ps.means
            
            if len(this_ps)!=0:
                outdata["n_pass"] +=[this_ps.pass_fract/len(this_ps), ]*len(this_ps)
            else:
                outdata["n_pass"] +=[0, ]*len(this_ps)

            
            

    waveform_data.close()
    return outdata




