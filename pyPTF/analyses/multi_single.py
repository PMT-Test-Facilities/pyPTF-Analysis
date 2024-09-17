
import os 

import h5py as h5 
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

import matplotlib.pyplot as plt 
from pyPTF.analyses.charge import get_analysis_charge, CHARGE_BINNING, fit_binned_data, _ped_20

plt.figure() 

def add_plot(filename, label, norm=1.0):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    name, ext = os.path.splitext(filename)

    print("... loading ")
    data_dict = h5.File(filename, 'r')

    charge = get_analysis_charge(data_dict["pmt0"])


    width = CHARGE_BINNING[1:] - CHARGE_BINNING[:-1]
    b_center = 0.5*(CHARGE_BINNING[:-1] + CHARGE_BINNING[1:])
    

    binned_data = np.histogram(charge, CHARGE_BINNING)[0]/(width)  
    if norm>1:
        binned_data *= norm/np.sum(binned_data)

    # let's fit the pedestal first 
    
    ped, fit = fit_binned_data(binned_data, b_center, False)

    plt.stairs(binned_data , CHARGE_BINNING, fill=False, label=label)


    return np.sum( binned_data )

if __name__=="__main__":
    normal = add_plot("../../data/pulse_series_fit_run5768.hdf5", "250mG -Y")
    add_plot("../../data/pulse_series_fit_run5767.hdf5", "0mG", normal)

    plt.ylim([0, 500])
    #plt.yscale('log')
    plt.legend()
    plt.show()
    