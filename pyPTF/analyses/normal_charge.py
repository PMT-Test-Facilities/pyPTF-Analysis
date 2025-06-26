import numpy as np
import matplotlib.pyplot as plt

import json
import os 

from pyPTF.analyses.charge import fit_binned_data, get_analysis_charge, CHARGE_BINNING, BIN_WIDTHS, BIN_CENTERS, PTF_SCALE
from pyPTF.analyses.timing import TIME_EDGES, extract_values
from pyPTF.constants import PTF_TS, PTF_SAMPLE_WIDTH
from pyPTF.utils import get_color

from math import sqrt
import h5py as h5 


def process_dataset_charge(data_dict):
    xs = np.array(data_dict["x"][:])
    ys = np.array(data_dict["y"][:])

    tilt = np.array(data_dict["tilt"][:])
    rot  = np.array(data_dict["rot"][:])

    unique_x = np.unique(xs)
    unique_y = np.unique(ys)

    charges = get_analysis_charge(data_dict)
    passing = np.array(data_dict["n_pass"])
    keep =  np.array(data_dict["passing"])
    charge_pass = np.array(data_dict["charge_sum"])[keep]
    
    # we bin these in position

    zeniths_deg = [10, 20, 30, 40, 45, 50, 60, 70, 80, 85]
    zeniths_deg = [85, 80,75, 70, 65, 60, 55, 50,45,40,30,20,10,0][::-1]
    #zeniths_deg = []
    gains = []
    det_effs = []
    stat_errors = []

    for xpos in unique_x[::-1]:
        #zeniths_deg.append(90+np.mean(tilt[xpos==xs]))
        total_wav = sum(keep[xpos==xs].astype(int))
        
        these_charges = charges[xpos==xs]
        these_pass = charges[np.logical_and(keep, xs==xpos)]
        det_eff = sum(data_dict["n_pass"][xpos==xs])/4

        stat_error = sqrt(total_wav*det_eff)/(total_wav*det_eff)
        stat_errors.append( sqrt(stat_error**2 + 0.05**2) )



        binned_charges = np.histogram(these_charges, CHARGE_BINNING)[0]/BIN_WIDTHS

        fitped, q1_fit = fit_binned_data(binned_charges)

        gain = np.sum(-1*these_pass/PTF_SCALE)/len(these_pass)
        #avg_charge = np.sum(-1*charges[xpos==xs]/PTF_SCALE)/len(charges[xpos==xs])
        
   
        gains.append(gain)
        det_effs.append( det_eff )
    
    if True : # len(gains)==10:
        return {
            "zeniths":np.array(zeniths_deg),
            "gain":np.array(gains),
            "det_effs":np.array(det_effs),
            "stat_error":np.array(stat_errors)
        }
    else:
        return {
            "zeniths":np.array(zeniths_deg[:-1]),
            "gain":np.array(gains),
            "det_effs":np.array(det_effs),
            "stat_error":np.array(stat_errors)
        }
        
def extract_transit_times(data_dict):
    # timing! 
    keep =  np.array(data_dict["pmt0"]["passing"])
    main_mean =  np.array(data_dict["pmt0"]["pulse_times"])

    monitor_mean = np.array(data_dict["timing_data"]["pulse_times"])[keep]

    diff = (main_mean+290) - monitor_mean

    dif_bins = np.arange(230, 310,PTF_SAMPLE_WIDTH)

    all_hist = np.histogram(diff, bins=dif_bins)[0]
    #plt.stairs(all_hist, dif_bins)
    plt.xlabel("Transit Time [ns]",size=14)
    #plt.ylabel("Counts",size=14)
    #plt.show()

    xs = np.array(data_dict["pmt0"]["x"][:])
    ys = np.array(data_dict["pmt0"]["y"][:])

    tilt = np.array(data_dict["pmt0"]["tilt"][:])
    rot  = np.array(data_dict["pmt0"]["rot"][:])

    unique_x = np.unique(xs)
    tt = []
    tts = []
    from math import sqrt, log

    fwhm_scaler= 2*sqrt(2*log(2))

    time_centers = 0.5*(dif_bins[1:] + dif_bins[:-1])
    for xpos in unique_x[::-1]:
        diff_histo = np.histogram(diff[xs[keep]==xpos], bins=dif_bins)[0]
        
        transit_time, spread = extract_values(time_centers, diff_histo)
        tt.append(transit_time)
        tts.append(spread*fwhm_scaler)
    return {
        "transit_times": np.array(tt),
        "transit_time_spread":np.array(tts)
    }

def get_data(filename):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    print("... loading ")
    data_dict = h5.File(filename,'r')


    pmt_20 = process_dataset_charge(data_dict["pmt0"])
    monitor = process_dataset_charge(data_dict["monitor"])

    time_data = extract_transit_times(data_dict)

    pmt_20["transit_time"] = time_data["transit_times"]
    pmt_20["transit_time_spread"] = time_data["transit_time_spread"]

    return pmt_20, monitor

def make_plots(data, mon):
    b_fields = ["Compensated"]

    # gain 
    keys = [
        "transit_time", "transit_time_spread", "gain", "det_effs"
    ]

    runlab = [
        "Perpendicular",
        "Diagonal",
    ]

    labels={
        "transit_time":"Relative TT [ns]",
        "transit_time_spread": "TTS (FWHM) [ns]",
        "gain":"Relative Gain [unitless]",
        "det_effs":"Relative Hit Eff. [unitless]"
    }

    takefiles = [
        "take_transit",
        "take_tts",
        "take_gain",
        "take_eff"
    ]

    bounds = [
        [-6, 10],
        [0, 20],
        [0, 1.6],
        [0, 1.5],
        
    ]

    direct = [" Y-Axis", " Diagonal"  ]
    for i in [0, 1]:
        

        for ik, key in enumerate(keys):
            print(key)
            plt.clf()
            if "transit_time" in key and "spread" not in key: 
                print(len(data[i]["zeniths"]), len(data[i][key]-data[i][key][0]))
                plt.errorbar(data[i]["zeniths"], data[i][key]-data[i][key][0],
                      yerr=0.05*(data[i][key]-data[i][key][0]), capsize=5, ecolor='k',
                      color=get_color(1, 3), marker="d", label="PTF")# , label=runlab[i])
            elif "spread" in key:
                plt.errorbar(data[i]["zeniths"], data[i][key], 
                      yerr=0.05*data[i][key], capsize=5, ecolor='k',
                      color=get_color(1, 3), marker="d", label="PTF")# , label=runlab[i])
            else:

                toplot = data[i][key] #/ mon[i][key]

                if "eff" in key:
                    plt.errorbar(data[i]["zeniths"], toplot/toplot[0], xerr=0,
                                yerr=data[i]["stat_error"], capsize=5, ecolor='k',
                            color=get_color(1, 3), marker="d", label="PTF") #, label=runlab[i])
                else:
                    plt.errorbar(data[i]["zeniths"], toplot/toplot[0],
                            yerr=0.05*toplot/toplot[0], capsize=5, ecolor='k',
                            color=get_color(1, 3), marker="d", label="PTF")# , label=runlab[i])

            errors = data[i]["det_effs"]
            plt.title(direct[i], size=14)
            plt.ylim(bounds[ik])


            if i==1:   
                newdat = np.loadtxt("./datafiles/{}_d.dat".format(takefiles[ik]), delimiter=",",comments="#").T
                cut = newdat[0]>-5
                plt.plot(newdat[0][cut], newdat[1][cut],marker="x", color=get_color(2, 3), label="Takenaka-san")
            else:
                # perpendicular first 
                newdat = np.loadtxt("./datafiles/{}_y.dat".format(takefiles[ik]), delimiter=",",comments="#").T
                cut = newdat[0]>-5
                plt.plot(newdat[0][cut], newdat[1][cut],marker="x", color=get_color(2, 3), label="Takenaka-san")

            #plt.ylim([0.5, 1.5])
            
            plt.xlabel("Zenith [deg]", size=14)
            
            #plt.plot([], [], color='k', marker="x", label="Takenaka-san")
            #plt.plot([], [], color='k', marker="d", label="PTF")
            #plt.plot([], [], color=get_color(0+1, 3), marker="o", label="Diagonal")
            #plt.plot([], [], color=get_color(1+1, 3), marker="o", label="Y-axis")
            plt.ylabel(labels[key], size=14)
            plt.legend()
            plt.savefig("./plots/take/norm_{}_{}.png".format(key, direct[i]),dpi=400)
            #plt.show()
    

def main():
    root = "/Users/bsmithers/software/pyPTF-Analysis/data/"


    """files = [
        "pulse_series_fit_run5724.hdf5",
        "pulse_series_fit_run5725.hdf5",
    ]"""

    files = [
        "pulse_series_fit_run5744.hdf5",
        "pulse_series_fit_run5743.hdf5",
    ]
    files = [
        "pulse_series_fit_run5715.hdf5",
        "pulse_series_fit_run5714.hdf5",
    ]


    charge_data =[]
    mon_data = []

    for filename in files:
        print("    running for {}".format(filename))
        data, mon = get_data(os.path.join(root, filename))
        charge_data.append(data)
        mon_data.append(mon)

    make_plots(charge_data, mon_data)


if __name__=="__main__":
    main()
