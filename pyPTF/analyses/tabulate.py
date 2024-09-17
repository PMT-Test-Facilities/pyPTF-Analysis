import json 
import numpy as np 

import os 

root_folder = os.path.join(os.path.dirname(__file__), "results")
template = "charge_results_{}.json"

pmt_x = 0.417
pmt_y = 0.297
pmt_radius =0.508/2
r_sq = pmt_radius**2

baseline_no = 5745

datafiles = [
    [5769, "-80mG in z"],
    [5751, "-100mG in z"],
    [5752, "100mG in z"],
    [5774, "100mG in z"],
    [5773, "250mG in z"],
    [5766, "80mG in x"],
    [5749, "250mG in x"],
    [5750, "40mG in y"],
    [5771, "100mG in y"],
#    [5755, "80mG in y"],
    [5748, "250mG in y"],
]

def get_efficiency(run_no):
    filename = os.path.join(
        root_folder,
        template.format(run_no)
    )

    _obj = open(filename, 'r')
    data = json.load(_obj)
    _obj.close()

    _xs = np.array(data["pmt0"]["xs"])
    _ys = np.array(data["pmt0"]["ys"])

    _xs = 0.5*(_xs[1:] + _xs[:-1])
    _ys = 0.5*(_ys[1:] + _ys[:-1])

    xs, ys = np.meshgrid(_xs, _ys)

    det_eff = np.array(data["pmt0"]["det_eff"]).T/(np.array(data["monitor"]["det_eff"]).T)
    
    det_eff[np.isnan(det_eff)]=None
    det_eff[np.isinf(det_eff)]=None

    mask = (xs-pmt_x)**2 + (ys-pmt_y)**2 < r_sq
    mask = np.logical_and(np.logical_not(np.isnan(det_eff)), mask)
    return np.nanmean(np.array(data["pmt0"]["det_eff"]).T[mask]), np.nanmean(np.array(data["monitor"]["det_eff"]).T[mask]), np.nanmean(det_eff[mask])

base_pmt0, base_monitor, baseline = get_efficiency(baseline_no)
print("baseline: {}".format(baseline))
for entry in datafiles:
    pmt20, monitor, ratio = get_efficiency(entry[0])
    print(entry[0], entry[1], pmt20, monitor, ratio, ratio/baseline)
