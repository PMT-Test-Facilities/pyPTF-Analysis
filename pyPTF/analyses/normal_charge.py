import numpy as np
import matplotlib.pyplot as plt

import json
import os 

from pyPTF.analyses.charge import fit_binned_data, get_analysis_charge, CHARGE_BINNING, BIN_WIDTHS, BIN_CENTERS
from pyPTF.analyses.timing import TIME_EDGES, extract_values
from pyPTF.constants import PTF_TS, PTF_SAMPLE_WIDTH
from pyPTF.utils import get_color


def process_dataset_charge(data_dict):
    xs = np.array(data_dict["x"][:])
    ys = np.array(data_dict["y"][:])

    tilt = np.array(data_dict["tilt"][:])
    rot  = np.array(data_dict["rot"][:])

    unique_x = np.unique(xs)
    unique_y = np.unique(ys)

    charges = get_analysis_charge(data_dict)
    passing = np.array(data_dict["n_pass"])
    print(passing)
    # we bin these in position

    zeniths_deg = [10, 20, 30, 40, 45]

    gains = []
    det_effs = []

    for xpos in unique_x[::-1]:
        these_charges = charges[xpos==xs]

        binned_charges = np.histogram(these_charges, CHARGE_BINNING)[0]/BIN_WIDTHS

        fitped, q1_fit = fit_binned_data(binned_charges)
        

        avg_charge = np.sum(binned_charges*BIN_CENTERS)/np.sum(binned_charges)
        gain = q1_fit[1]
        gain_width = q1_fit[2]
   
        det_eff = sum(passing[xpos==xs])/4 
        mu = det_eff*avg_charge/gain
        hq1pemu = gain*mu

        gains.append(q1_fit[1])
        det_effs.append(sum(passing[xpos==xs])/4 )
    
    return {
        "zeniths":np.array(zeniths_deg),
        "gain":np.array(gains),
        "det_effs":np.array(det_effs)
    }
    
def extract_transit_times(data_dict):
    # timing! 
    main_mean =  np.array(data_dict["pmt0"]["means"])
    monitor_mean = np.array(data_dict["monitor"]["means"])
    diff = main_mean - monitor_mean

    dif_bins = np.arange(-45, 45, PTF_SAMPLE_WIDTH)

    xs = np.array(data_dict["pmt0"]["x"][:])
    ys = np.array(data_dict["pmt0"]["y"][:])

    tilt = np.array(data_dict["pmt0"]["tilt"][:])
    rot  = np.array(data_dict["pmt0"]["rot"][:])

    unique_x = np.unique(xs)
    tt = []
    tts = []

    for xpos in unique_x[::-1]:
        diff_histo = np.histogram(diff[xs==xpos])[0]
        time_centers = 0.5*(dif_bins[1:] + dif_bins[:-1])
        transit_time, spread = extract_values(time_centers, diff_histo)
        tt.append(transit_time)
        tts.append(spread)
    return {
        "transit_times": np.array(tt),
        "transit_time_spread":np.array(tts)
    }

def get_data(filename):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    print("... loading ")
    obj = open(filename, 'r')
    data_dict = json.load(obj)
    obj.close()


    pmt_20 = process_dataset_charge(data_dict["pmt0"])
    monitor = process_dataset_charge(data_dict["monitor"])

    time_data = extract_transit_times(data_dict)

    pmt_20["transit_time"] = time_data["transit_times"]
    pmt_20["transit_time_spread"] = time_data["transit_time_spread"]

    return pmt_20

def make_plots(data):
    b_fields = ["Compensated", "250mG", "500mG"]

    # gain 
    keys = [
        "transit_time", "transit_time_spread", "gain", "det_effs"
    ]

    for key in keys:
        plt.clf()
        for i in range(0,3):            
            plt.plot(data[i]["zeniths"], data[i][key], 
                     label=b_fields[i], color=get_color(i+1, 3), marker="d")
            plt.title(key, size=14)
            #plt.ylim([0.5, 1.5])
            plt.legend()
            plt.xlabel("Zenith [deg]", size=14)
        plt.savefig("./plots/norm_{}.png".format(key),dpi=400)
    

def main():
    root = "/Users/bsmithers/software/pyPTF-Analysis/data/"
    

    files = [
        "pulse_series_fit_run5645.json",
        "pulse_series_fit_run5646.json",
        "pulse_series_fit_run5647.json"
    ]

    charge_data =[]

    for filename in files:
        data = get_data(os.path.join(root, filename))
        charge_data.append(data)

    make_plots(charge_data)


if __name__=="__main__":
    main()
