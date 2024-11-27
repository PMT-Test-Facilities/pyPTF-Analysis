
import os 

import h5py as h5 
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

import matplotlib.pyplot as plt 
from pyPTF.analyses.charge import get_analysis_charge, CHARGE_BINNING, fit_binned_data, _ped_20, charge_fit_bellamy, fit_bellamy
from pyPTF.utils import get_color

plt.figure(figsize=(4,3)) 
#bins = np.linspace(0, 200, 200)
bins = CHARGE_BINNING
time_bins = np.arange(140.5, 174.5, 2)
counter = 0

def add_time(filename, label, norm=None):
    global counter 
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    name, ext = os.path.splitext(filename)

    print("... loading ")
    data_dict = h5.File(filename, 'r')

    keep =  np.array(data_dict["pmt0"]["passing"])
    
    charge = get_analysis_charge(data_dict["pmt0"], just_height=False)
    only_under = np.logical_and(charge[keep]<300, charge[keep]>75)
    print("keep {}, only {}".format(len(keep), len(only_under)))

    main_mean =  np.array(data_dict["pmt0"]["pulse_times"]) + 150 #  np.array(data_dict["pmt0"]["means"])[keep]
    monitor_mean = (np.array(data_dict["timing_data"]["pulse_times"])[keep])
    print("pulse times", len(main_mean))

    diff = main_mean - monitor_mean 
    diff = diff[only_under]
    print("remaining: ",len(diff))

    binned_data = np.histogram(diff, time_bins)[0]
    if norm is not None:
        binned_data =binned_data*np.max(norm)/np.max(binned_data)
    plt.stairs(binned_data , time_bins, fill=False, label=label, color=get_color(counter+0.5,3))
    counter +=1 
    return binned_data


def add_time_cut(filename, label, norm=None):
    global counter 
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    name, ext = os.path.splitext(filename)

    print("... loading ")
    data_dict = h5.File(filename, 'r')

    keep =  np.array(data_dict["pmt0"]["passing"])
    
    charge = get_analysis_charge(data_dict["pmt0"], just_height=False)

    cut = np.logical_and(
        np.logical_and((np.array(data_dict["pmt0"]["x"]) <0.41), (np.array(data_dict["pmt0"]["x"]) >0.39)),
        np.logical_and((np.array(data_dict["pmt0"]["y"]) >0.29), (np.array(data_dict["pmt0"]["y"]) <0.31)),
    )
    only_under = np.logical_and(charge[keep]<300, charge[keep]>75)
    print("keep {}, only {}".format(len(keep), len(only_under)))

    main_mean =  np.array(data_dict["pmt0"]["pulse_times"]) + 150 #  np.array(data_dict["pmt0"]["means"])[keep]
    monitor_mean = (np.array(data_dict["timing_data"]["pulse_times"])[keep])
    print("pulse times", len(main_mean))

    diff = main_mean - monitor_mean 
    diff = diff[np.logical_and(only_under, cut[keep])]
    print("remaining: ",len(diff))

    binned_data = np.histogram(diff, time_bins)[0]
    if norm is not None:
        binned_data =binned_data*np.max(norm)/np.max(binned_data)
    plt.stairs(binned_data , time_bins, fill=False, label=label, color=get_color(counter+0.5,6))
    counter +=1 
    return binned_data

def add_plot(filename, label, norm=None):
    global counter 

    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    name, ext = os.path.splitext(filename)

    print("... loading ")
    data_dict = h5.File(filename, 'r')

    charge = get_analysis_charge(data_dict["pmt0"], just_height=False)

    bcenter = 0.5*(bins[:-1]+bins[1:])
    binned_data = np.histogram(charge, bins)[0]
    if norm is not None:
        binned_data =binned_data*np.max(norm)/np.max(binned_data)

    # let's fit the pedestal first 
    
    fitres = charge_fit_bellamy(binned_data, bcenter)
    print(fitres)
    xfine = np.linspace(min(bcenter), max(bcenter), 3000)
    yfine = fit_bellamy(xfine, fitres)
    #plt.plot(xfine, yfine, ls='--', color='red', label="Fit")
    plt.stairs(binned_data , bins, fill=False, label=label, color=get_color(counter+0.5,6))
    counter +=1 


    return binned_data


def add_plot_cut(filename, label, norm=None):
    global counter 

    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    name, ext = os.path.splitext(filename)

    print("... loading ")
    data_dict = h5.File(filename, 'r')

    charge = get_analysis_charge(data_dict["pmt0"], just_height=False)
    print(data_dict["pmt0"].keys())
    cut = np.logical_and(
        np.logical_and((np.array(data_dict["pmt0"]["x"]) <0.41), (np.array(data_dict["pmt0"]["x"]) >0.39)),
        np.logical_and((np.array(data_dict["pmt0"]["y"]) >0.29), (np.array(data_dict["pmt0"]["y"]) <0.31)),
    )

    bcenter = 0.5*(bins[:-1]+bins[1:])
    binned_data = np.histogram(charge[cut], bins)[0]
    if norm is not None:
        binned_data =binned_data*np.max(norm)/np.max(binned_data)

    # let's fit the pedestal first 
    
    
   
    fitres = charge_fit_bellamy(binned_data, bcenter)
    print(fitres)
    xfine = np.linspace(min(bcenter), max(bcenter), 3000)
    yfine = fit_bellamy(xfine, fitres)
    #plt.plot(xfine, yfine, ls='--', color='red', label="Fit")
    plt.stairs(binned_data , bins, fill=False, label=label, color=get_color(counter+0.5,3))
    counter +=1 


    return binned_data


if __name__=="__main__":
    #normal = add_plot("../../data/pulse_series_fit_run5763.hdf5", "laser off")
    
    add_plot("../../data/pulse_series_fit_run5808.hdf5","0mG Y")
    add_plot("../../data/pulse_series_fit_run5803.hdf5","100mG Y")
    add_plot("../../data/pulse_series_fit_run5809.hdf5","200mG Y")
    add_plot("../../data/pulse_series_fit_run5805.hdf5","300mG Y")
    add_plot("../../data/pulse_series_fit_run5806.hdf5","400mG Y")
    add_plot("../../data/pulse_series_fit_run5807.hdf5","500mG Y")
    
    #new_one = add_plot("../../data/pulse_series_fit_run5768.hdf5","laser on")# , norm=normal)
    #plt.stairs(new_one-normal, bins, fill=False, label="Pedestal-Sub")

    #plt.ylim([0, 500])
    plt.yscale('log')
    plt.xlabel("Pulse Charge [ADC]", size=14)
    plt.ylabel("Arb. Units", size=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("multi_superimpose.png", dpi=400)
    plt.show()

    plt.clf() 
    import sys 
    sys.exit()

    counter = 0
    new_one = add_time("../../data/pulse_series_fit_run5797.hdf5","0mG")# , norm=normal)
    new_one = add_time("../../data/pulse_series_fit_run5796.hdf5","250mG Y",norm=new_one)# , norm=normal)
    new_one = add_time_cut("../../data/pulse_series_fit_run5782.hdf5","500mG Y",norm=new_one)
    plt.yscale('log')
    plt.xlabel("Transit Time [ns]", size=14)
    plt.ylabel("Arb. Units", size=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("multitime_superimpose.png", dpi=400)
    plt.show()