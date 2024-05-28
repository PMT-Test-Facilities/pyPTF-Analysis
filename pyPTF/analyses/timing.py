import sys 
import os 
import json 
import numpy as np 
import matplotlib.pyplot as plt 
from math import log, sqrt 

from pyPTF.utils import get_color, make_bin_edges
from pyPTF.constants import PTF_TS, PTF_SAMPLE_WIDTH
print("time per sample: ", PTF_SAMPLE_WIDTH)
from scipy.signal import peak_widths

from tqdm import tqdm 

fwhm_scaler= 2*sqrt(2*log(2))

TIME_EDGES = np.linspace(
    min(PTF_TS)-0.5*PTF_SAMPLE_WIDTH, max(PTF_TS)+0.5*PTF_SAMPLE_WIDTH, len(PTF_TS)+1, endpoint=True
)

def extract_values(time_centers, tdiffs):
    index = np.argmax(tdiffs)

    width = peak_widths(tdiffs, [index,])[0]*PTF_SAMPLE_WIDTH/fwhm_scaler

    return time_centers[index], width

def main(filename):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    obj = open(filename, 'r')
    data_dict = json.load(obj)
    obj.close()
    run_no = data_dict["run_no"]

    main_mean =  np.array(data_dict["pmt0"]["means"])
    monitor_mean = np.array(data_dict["monitor"]["means"])

    if len(main_mean)!=len(monitor_mean):
        raise Exception("extract_pulses should be ran keeping all waveforms!")

    diff = main_mean - monitor_mean
    

    histo_data = np.histogram(monitor_mean, TIME_EDGES)[0]
    main_histo_data = np.histogram(main_mean, TIME_EDGES)[0]

    plt.stairs(histo_data, TIME_EDGES, alpha=1, color=get_color(1,3), label="Monitor")
    plt.stairs(main_histo_data,TIME_EDGES, alpha=1, color=get_color(2,3), label="20in")
    plt.legend()
    plt.xlabel("Time [ns]", size=14)
    plt.ylabel("Counts", size=14)
    plt.tight_layout()

    plt.savefig(
        os.path.join(os.path.dirname(__file__), "plots","time_dist{}.png".format(run_no)),
        dpi=400
    )
    #plt.show()

    plt.clf()

    dif_bins = np.arange(-45, 45, PTF_SAMPLE_WIDTH)
    diff_histo = np.histogram(diff, dif_bins)[0]
    plt.stairs(diff_histo, dif_bins)
    plt.xlabel("Time [ns]", size=14)
    plt.ylabel("Counts", size=14)
    plt.title(r"$\mu_{20in} - \mu_{monitor}$")

    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "plots","time_spread{}.png".format(run_no)),
        dpi=400
    )
    #plt.show()
    plt.clf()

    time_centers = 0.5*(dif_bins[1:] + dif_bins[:-1])

    transit_time, spread = extract_values(time_centers, diff_histo)

    print(transit_time)
    print(spread)
    
    xs = np.array(data_dict["pmt0"]["x"])
    ys = np.array(data_dict["pmt0"]["y"])

    x_edges = make_bin_edges(xs)
    y_edges = make_bin_edges(ys)
    n_x = len(x_edges)-1
    n_y = len(y_edges)-1

    sample = np.transpose([xs, ys, diff])
    binned_diffs = np.histogramdd(
        sample, bins=(x_edges,y_edges, dif_bins)
    )[0]
    results={
        "transit_time":np.zeros((n_x,n_y)),
        "transit_time_spread":np.zeros((n_x,n_y))
    }
    for ix in tqdm(range(n_x)):
        for jy in range(n_y):
            tt, tts = extract_values(time_centers, binned_diffs[ix][jy])
            results["transit_time"][ix][jy]=tt - 20*2.014388489208633
            results["transit_time_spread"][ix][jy]=tts

    obj = open("timing_results_{}.json".format(run_no),'wt')

    output = {
        "transit_time":results["transit_time"].tolist(),
        "transit_time_spread":results["transit_time_spread"].tolist(),
        "xs":x_edges.tolist(),
        "ys":y_edges.tolist()

    }
    json.dump(output, obj,indent=4)
    obj.close()

    plt.pcolormesh(x_edges,y_edges, np.transpose(results["transit_time"]), vmin=-10, vmax=10, cmap="coolwarm")
    cbar = plt.colorbar()
    cbar.set_label("[ns]")
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]",size=14)
    plt.gca().set_aspect('equal')
    plt.title("Transit Time", size=14)
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "transit_time_{}.png".format(run_no)), dpi=400)
    plt.clf()

    plt.pcolormesh(x_edges,y_edges, np.transpose(results["transit_time_spread"]), vmin=0, vmax=5)
    cbar = plt.colorbar()
    cbar.set_label("[ns]")
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]",size=14)
    plt.gca().set_aspect('equal')
    plt.title("Transit Time Spread", size=14)
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "transit_time_spread_{}.png".format(run_no)), dpi=400)
    plt.clf()

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("timing.py filename.json")
        sys.exit()
    main(sys.argv[1])