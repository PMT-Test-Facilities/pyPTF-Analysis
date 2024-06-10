
import os 
import json 
import numpy as np
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt 
from pyPTF.utils import get_centers

DBL_RATIO = False

pmt_x = 0.417
pmt_y = 0.297 

pmt_radius =0.508/2

keys = [
    "det_eff",
    "hq1pemu",
    "avg_charge",
    "mu",
    "Q_1",
    "Sigma_1",
]

time_keys = [
    "transit_time",
    "transit_time_spread"
]

title_keys = {key:key for key in keys}
for time_key in time_keys:
    title_keys[time_key] = time_key
title_keys["det_eff"] = "Detection Efficiency"
title_keys["Q_1"]="Gain"
title_keys["Sigma_1"] = "Gain Width"
title_keys["transit_time"] = "Transit Time"
title_keys["transit_time_spread"] = "Transit Time Spread"

def load_data(filename, timing=False):
    if not os.path.exists(filename):
        print("could not open ", filename)
        raise IOError("File not found")

    numpy_data = {}

    _obj = open(filename, 'rt')
    data = json.load(_obj)
    _obj.close()

    if timing:
        for key in data.keys():
            numpy_data[key] = np.array(data[key])
    else:
        for key in data.keys():
            numpy_data[key] = {}
            for subkey in data[key].keys():
                numpy_data[key][subkey] = np.array(data[key][subkey])

    return numpy_data

def load_other_data(filename, base_data, timing=False):
    """
        Loads the other data, but then interpolates this data along the original compensated data grid 
    """
    interpo_data={"pmt0":{}, "monitor":{}}

    raw_data = load_data(filename, timing)

    if timing:
        for key in time_keys:
            terpo = RectBivariateSpline(
                get_centers(raw_data["xs"]), get_centers(raw_data["ys"]), raw_data[key]
            )
            interpo_data[key] = terpo(get_centers(base_data["pmt0"]["xs"]), get_centers(base_data["pmt0"]["ys"]), grid=True)

    else:
            
        for pmt_key in interpo_data.keys():
            for key in keys:
                terpo = RectBivariateSpline(
                    get_centers(raw_data[pmt_key]["xs"]), get_centers(raw_data[pmt_key]["ys"]), raw_data[pmt_key][key]
                )
                interpo_data[pmt_key][key] = terpo(get_centers(base_data[pmt_key]["xs"]), get_centers(base_data[pmt_key]["ys"]), grid=True)

    return interpo_data

if __name__=="__main__":

    root_folder = os.path.dirname(__file__)

    compensated = "charge_results_5653.json"
    time_comp = "timing_results_5653.json"

    fields = [
        "250 y", 
        "500 y",
        "500 x",
        "250 x" 
    ]

    filenames = [
        "charge_results_5633.json",
        "charge_results_5630.json",
        "charge_results_5650.json",
        "charge_results_5651.json"
    ]
    time_names = [
        "timing_results_5633.json",
        "timing_results_5630.json",
        "timing_results_5650.json",
        "timing_results_5651.json",
    ]

    compensated_data = load_data(os.path.join(root_folder, compensated))
    #compensated_data["pmt0"]["xs"] -= 0.03

    time_comp = load_data(os.path.join(root_folder, time_comp), True)
    for key in time_keys:
        compensated_data["pmt0"][key] = time_comp[key]

    xmesh, ymesh = np.meshgrid(get_centers(compensated_data["pmt0"]["xs"]), get_centers(compensated_data["pmt0"]["ys"]))

    exclude = (xmesh - pmt_x)**2 + (ymesh-pmt_y)**2 > pmt_radius**2

    for i, field in enumerate(fields):
        this_data = load_other_data(
            os.path.join(root_folder, filenames[i]),
            compensated_data
        )

        time_data = load_other_data(
            os.path.join(root_folder, time_names[i]),
            compensated_data, True
        )

        for key in time_keys:
            this_data["pmt0"][key] = time_data[key]

        allkey = keys + time_keys

        for key in allkey:
            if "time" in key and "spread" not in key:
                double_ratio = (this_data["pmt0"][key]) - (compensated_data["pmt0"][key])
            else:
                if DBL_RATIO and not "spread" in key:
                    double_ratio = (this_data["pmt0"][key]/this_data["monitor"][key])/(compensated_data["pmt0"][key]/compensated_data["monitor"][key])
                else:
                    double_ratio = (this_data["pmt0"][key])/(compensated_data["pmt0"][key])
            double_ratio[exclude.T] = None
    
            plt.clf()
            if "time" in key and "spread" not in key:
                scale = 5 if "spread" in key else 8

                plt.pcolormesh(compensated_data["pmt0"]["xs"], compensated_data["pmt0"]["ys"], double_ratio.T, vmin=-scale, vmax=scale, cmap='coolwarm')
            else:
                plt.pcolormesh(compensated_data["pmt0"]["xs"], compensated_data["pmt0"]["ys"], double_ratio.T, vmin=0.5, vmax=1.5, cmap='coolwarm')
            cbar = plt.colorbar()
            cbar.set_label(r"$\frac{(SK_{B+}/Mon_{B+})}{(SK_{B0}/Mon_{B0})}$")
            if "time" in key and "spread" not in key:
                cbar.set_label(r"$SK_{B+} - SK_{B0} $")
            else:
                if DBL_RATIO and not "spread" in key:
                    cbar.set_label(r"$\frac{(SK_{B+}/Mon_{B+})}{(SK_{B0}/Mon_{B0})}$")
                else:
                    cbar.set_label(r"$SK_{B+} / SK_{B0} $")
            plt.xlabel("X [m]", size=14)
            plt.ylabel("Y [m]", size=14)
            plt.title(title_keys[key] +", "+ str(fields[i]) + " mG")
            plt.gca().set_aspect('equal')
            plt.savefig(os.path.join(os.path.dirname(__file__), "meta_plots", "{}_{}_dblratio.png".format(key, field)), dpi=400)


    assert len(fields)==len(filenames), "Filenames and fields should have the same length"

