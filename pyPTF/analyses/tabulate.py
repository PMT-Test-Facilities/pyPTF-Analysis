import json 
import numpy as np 
from pyPTF.utils import Irregular2DInterpolator
import os 

root_folder = os.path.join(os.path.dirname(__file__), "results")
template = "charge_results_{}.json"

pmt_x = 0.417
pmt_y = 0.297
pmt_radius =0.508/2
r_sq = pmt_radius**2

baseline_no = 5770

charge = False

datafiles = [
    [5745, "0mG"],
    [5769, "-80mG in z"],
    [5751, "-100mG in z"],
    [5752, "100mG in z"],
    [5774, "-100mG in z"],
    [5773, "250mG in z"],
    [5766, "80mG in x"],
    [5749, "250mG in x"],
    [5750, "40mG in y"],
    [5771, "100mG in y"],
    [5776, "-100mG in y"],
#    [5755, "80mG in y"],
    [5748, "250mG in y"],
    [5777, "100mG in (-Y+X)"],
    [5780, "100mG in (+Y+X)"]
]

def get_efficiency(run_no):
    filename = os.path.join(
        root_folder,
        template.format(run_no)
    )

    key = "avg_charge" if charge else "det_eff"

    _obj = open(filename, 'r')
    data = json.load(_obj)
    _obj.close()

    _xs = np.array(data["pmt0"]["xs"])
    _ys = np.array(data["pmt0"]["ys"])

    _xs = 0.5*(_xs[1:] + _xs[:-1])
    _ys = 0.5*(_ys[1:] + _ys[:-1])

    xs, ys = np.meshgrid(_xs, _ys)

    det_eff = np.array(data["pmt0"][key]).T/(np.array(data["monitor"][key]).T)
    
    det_eff[np.isnan(det_eff)]=None
    det_eff[np.isinf(det_eff)]=None

    mask = (xs-pmt_x)**2 + (ys-pmt_y)**2 < r_sq
    mask = np.logical_and(np.logical_not(np.isnan(det_eff)), mask)
    return np.nanmean(np.array(data["pmt0"][key]).T[mask]), np.nanmean(np.array(data["monitor"][key]).T[mask]), np.nanmean(det_eff[mask])


def build_spline():
    # (x,y,z)

    datafiles = [
        #[5769, 80, [0,0,-1]],
        #[5751, 100, [0,0,-1]],
        [5752, 100, [0,0,1]],
        [5774, 100, [0,0,-1]],
        #[5773,250, [0,0,1]],
        #[5766, 80, [1,0,0]],
        [5749, 250, [1,0,0]],
        [5749, 250, [-1,0,0]], # artificially added - x seems unimportant 
        #[5750, 40, [0,1,0]],
        [5771, 100, [0,1,0]],
        [5776, 100,[0,-1,0]],
    #    [5755, "80mG in y"],
       # [5748, 250, [0,1,0]],
        [5777, 100, [1,-1,0]],
        [5780,100,[1,1,0]]
    ]

    azimuths = []
    zeniths = []
    efficiency = []

    base_pmt0, base_monitor, baseline = get_efficiency(baseline_no) 
    for entry in datafiles:
        pmt0, monitor, ratio = get_efficiency(entry[0])
        azimuthal = np.arctan2(entry[2][0],entry[2][1])
        zenith = np.arctan2(entry[2][2] , np.sqrt(entry[2][0]**2 + entry[2][1]**2))

        azimuths.append(azimuthal)
        zeniths.append(zenith)

        addy= ratio / baseline
        if addy<0.78:
            addy = 0.78
        efficiency.append(addy)



    test = Irregular2DInterpolator(zeniths, azimuths, efficiency)

    import matplotlib.pyplot as plt 
    
    if True:
        zen_infill = np.linspace(min(zeniths), max(zeniths), 100)
        azi_infill = np.linspace(min(azimuths), max(azimuths), 101)
        zen_mesh, azi_mesh = np.meshgrid(zen_infill, azi_infill)
        evaluated = test(zen_mesh, azi_mesh)
        print("min: {}".format(np.min(evaluated)))
        plt.pcolormesh(zen_infill, azi_infill, evaluated, shading='nearest', vmin=0.8, vmax=1.0)
        plt.colorbar()
        plt.scatter(zeniths, azimuths)
        plt.xlabel("Zenith [rad]")
        plt.ylabel("Azimuth [rad]")
        plt.show()


    return test 
    


if __name__=="__main__":
    base_pmt0, base_monitor, baseline = get_efficiency(baseline_no)
    print("baseline: {}".format(baseline))
    print("baseline ", base_pmt0, base_monitor, baseline)
    for entry in datafiles:
        pmt20, monitor, ratio = get_efficiency(entry[0])
        print(entry[0], entry[1], pmt20, monitor, ratio, ratio/baseline)
    