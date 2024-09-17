import sys 
import os 
import json 
import numpy as np 
import matplotlib.pyplot as plt 
from math import log, sqrt 
import h5py as h5 

from pyPTF.utils import get_color, make_bin_edges
from pyPTF.constants import PTF_TS, PTF_SAMPLE_WIDTH, PTF_SCALE
print("time per sample: ", PTF_SAMPLE_WIDTH)
from scipy.signal import peak_widths
from scipy.optimize import minimize

from tqdm import tqdm 

DEBUG = True

pmt_x = 0.417
pmt_y = 0.297 
pmt_radius =0.508/2


fwhm_scaler= 2*sqrt(2*log(2))

TIME_EDGES = np.linspace(
    min(PTF_TS)-0.5*PTF_SAMPLE_WIDTH, max(PTF_TS)+0.5*PTF_SAMPLE_WIDTH, len(PTF_TS)+1, endpoint=True
)
TIME_EDGES = PTF_TS
dif_bins = np.arange(130, 170, PTF_SAMPLE_WIDTH)

def extract_values(time_centers, tdiffs, fancy=False):
    index = np.argmax(tdiffs)

    fwhm = peak_widths(tdiffs, [index,])[0]*PTF_SAMPLE_WIDTH
    width = fwhm/fwhm_scaler
    
    def metric(params):
        values = params[0]*np.exp(-0.5*((params[1] - time_centers)/params[2])**2)
        return np.sum((values - tdiffs)**2)
    
    x0 = [max(tdiffs), time_centers[index], width[0]]
    bounds = [
        (min(tdiffs), np.inf),
        (100, 300),
        (0.5, 15)
    ]
    res = minimize(metric, x0, bounds=bounds).x 

    if fancy:
        import matplotlib.pyplot as plt 
        plt.bar(time_centers, tdiffs,width=time_centers[1]-time_centers[0], label="data", alpha=0.5)
        fit_x = np.linspace(min(time_centers), max(time_centers), 1000)
        fit_y = res[0]*np.exp(-0.5*((res[1] - fit_x)/res[2])**2)
        #plt.plot(fit_x, fit_y, label="fit", color="orange")
        plt.xlabel("Time [ns]", size=14)
        plt.ylabel("Counts", size=14)
        #plt.legend()
        plt.show()

    return res[1], res[2]


def main(filename):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    data_dict = h5.File(filename, 'r')
    run_no = np.int64(data_dict["run_no"])



    main_mean =  np.array(data_dict["pmt0"]["means"])

    keep =  np.array(data_dict["pmt0"]["passing"])
    
    print(data_dict.keys())
    main_mean =  np.array(data_dict["pmt0"]["pulse_times"]) + 150 #  np.array(data_dict["pmt0"]["means"])[keep]
    monitor_mean = np.array(data_dict["timing_data"]["pulse_times"])[keep]

    print(np.nanmean(main_mean))
    print(np.nanmean(monitor_mean))
    diff = main_mean - monitor_mean
    print(np.mean(diff))

    histo_data = np.histogram(monitor_mean, TIME_EDGES)[0]
    main_histo_data = np.histogram(main_mean, TIME_EDGES)[0]

    plt.stairs(histo_data, TIME_EDGES, alpha=1, color=get_color(1,3), label="Trigger")
    plt.stairs(main_histo_data,TIME_EDGES, alpha=1, color=get_color(2,3), label="20in")
    plt.legend()
    plt.xlabel("Time [ns]", size=14)
    plt.ylabel("Counts", size=14)
    plt.yscale('log')
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "plots","time_dist{}.png".format(run_no)),
        dpi=400
    )
    plt.show()

    plt.clf()

    diff_histo = np.histogram(diff, dif_bins)[0]
    plt.stairs(diff_histo, dif_bins)
    plt.xlabel("Time [ns]", size=14)
    plt.ylabel("Counts", size=14)
    plt.title(r"$\tau_{20in}$", size=14)
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "plots","time_spread{}.png".format(run_no)),
        dpi=400
    )
    #plt.show()
    plt.clf()

    time_centers = 0.5*(dif_bins[1:] + dif_bins[:-1])

    transit_time, spread = extract_values(time_centers, diff_histo, True)

    print("Time stuff: ", transit_time, spread*2*sqrt(2*log(2)))

    xs = np.array(data_dict["pmt0"]["x"])
    ys = np.array(data_dict["pmt0"]["y"])


    x_edges = make_bin_edges(xs)
    y_edges = make_bin_edges(ys)


    x_centers = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centers = 0.5*(y_edges[:-1] + y_edges[1:])
    n_x = len(x_edges)-1
    n_y = len(y_edges)-1
    sample = np.transpose([xs[keep], ys[keep], diff])
    binned_diffs = np.histogramdd(
        sample, bins=(x_edges,y_edges, dif_bins)
    )[0]

    facecut = np.logical_and(np.abs(pmt_x-xs[keep]) < 0.045 , np.abs(pmt_y-ys[keep]) < 0.045)
    face_binned = np.histogram(diff[facecut], dif_bins)[0]
    plt.stairs(face_binned, dif_bins)
    plt.xlabel("Time [ns]", size=14)
    plt.ylabel("Counts", size=14)
    plt.title(r"$\tau_{20in}$ Face-Cut", size=14)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "plots","facecut{}.png".format(run_no)),
        dpi=400
    )
    #plt.show()
    plt.clf()

    facecut = (pmt_x-xs[keep]) > 0.045 
    face_binned = np.histogram(diff[facecut], dif_bins)[0]
    plt.stairs(face_binned, dif_bins)
    plt.xlabel("Time [ns]", size=14)
    plt.ylabel("Counts", size=14)
    plt.title(r"$\tau_{20in}$ Face-Cut", size=14)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "plots","facelesscut{}.png".format(run_no)),
        dpi=400
    )
    #plt.show()
    plt.clf()


    results={
        "transit_time":np.zeros((n_x,n_y)),
        "transit_time_spread":np.zeros((n_x,n_y))
    }
    counter = 0
    for ix in tqdm(range(n_x)):
        for jy in range(n_y):
            y_shift = 0

            shifts = [-0.17, 0.0, 0.17]
            diffs = [abs(x_centers[ix] - (pmt_x+entry))<5e-3 for entry in shifts]
            #diffs = [abs(y_centers[jy] - (pmt_y+entry))<5e-3 for entry in shifts]

            if DEBUG and False : # any(diffs) and abs(y_centers[jy]-(pmt_y+0))<5e-3:
                counter+=1
                plt.stairs(binned_diffs[ix][jy], dif_bins, color=get_color(counter, 4), lw=3, label="shift {:.3f}".format(x_centers[ix] - pmt_x))
                plt.xlabel(r"$\tau_{20in}$", size=14)
                plt.ylabel("Counts", size=14)
                plt.yscale('log')
                #plt.ylim([0,250])
                plt.tight_layout()

            fancy = any(diffs) and abs(y_centers[jy]-(pmt_y+0))<5e-3

            tt, tts = extract_values(time_centers, binned_diffs[ix][jy], False)
            results["transit_time"][ix][jy]=tt
            results["transit_time_spread"][ix][jy]=tts
    if DEBUG:
        plt.legend()
        plt.savefig("all_three.png", dpi=400)
        plt.show()
    obj = open("./results/timing_results_{}.json".format(run_no),'wt')

    output = {
        "transit_time":results["transit_time"].tolist(),
        "transit_time_spread":results["transit_time_spread"].tolist(),
        "xs":x_edges.tolist(),
        "ys":y_edges.tolist()

    }
    json.dump(output, obj,indent=4)
    obj.close()
  
    xmesh, ymesh = np.meshgrid(x_centers, y_centers)

    exclude = (xmesh - pmt_x)**2 + (ymesh-pmt_y)**2 > pmt_radius**2
    results["transit_time"][exclude.T]=None
    results["transit_time_spread"][exclude.T] = None
    plt.pcolormesh(x_edges,y_edges, np.transpose(results["transit_time"]),  cmap="coolwarm", vmin=145, vmax=155)
    cbar = plt.colorbar()
    cbar.set_label("[ns]")
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]",size=14)
    plt.gca().set_aspect('equal')
    plt.title("Transit Time", size=14)
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "transit_time_{}.png".format(run_no)), dpi=400)
    plt.clf()

    plt.pcolormesh(x_edges,y_edges, 2*sqrt(2*log(2))*np.transpose(results["transit_time_spread"]), vmin=0, vmax= 10)
    cbar = plt.colorbar()
    cbar.set_label("[ns]")
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]",size=14)
    plt.gca().set_aspect('equal')
    plt.title("Transit Time Spread (FWHM)", size=14)
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "transit_time_spread_{}.png".format(run_no)), dpi=400)
    plt.clf()

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("timing.py filename.json")
        sys.exit()
    main(sys.argv[1])