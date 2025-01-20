"""
    This script runs on an analysis file
"""

import os 
import json 
import numpy as np
import matplotlib.pyplot as plt 

CHARGE = True

def main():
    field_bin = np.linspace(0,1.4, 199)
    d_eff_bin = np.linspace(0, 0.5, 100)
    charge_bin = np.linspace(-0.02, 0, 101)

    if CHARGE:
        hdata = np.zeros((len(field_bin)-1, len(charge_bin)-1))
    else:
        hdata = np.zeros((len(field_bin)-1, len(d_eff_bin)-1))

    allruns = [
        5750, 5771, 5748,5778, 5782
    ]

    runs_z = [5769, 5751, 5752, 5774]
    runs_y = [5750, 5771, 5748]
    runs_x = [5776, 5749]

    for run_no in allruns:

        path = os.path.join(
            os.path.dirname(__file__),
            "results",
            "charge_results_{}.json".format(run_no)
        )

        if not os.path.exists(path):
            print("Results not available - run")
            print("    charge.py ../../data/pulse_series_fit_run{}.hdf5".format(run_no))
            raise IOError("File not found {}".format(path))

        _obj = open(path, 'rt')
        data =json.load(_obj)
        _obj.close()

        avg_charge = np.array(data["monitor"]["avg_charge"]).flatten()
        det_eff = np.array(data["monitor"]["det_eff"]).flatten()

        bfield = np.array(data["monitor"]["bfield"]).flatten()

        
        _xs = np.array(data["monitor"]["xs"])
        _ys = np.array(data["monitor"]["ys"])

        _xs = 0.5*(_xs[1:] + _xs[:-1])
        _ys = 0.5*(_ys[1:] + _ys[:-1])

        xs, ys = np.meshgrid(_xs, _ys)
        

        mask = np.ones_like(det_eff).astype(bool)
        if CHARGE:
            hdata += np.histogram2d(bfield[mask], avg_charge[mask], bins=(field_bin, charge_bin))[0]
        else:
            hdata += np.histogram2d(bfield[mask], det_eff[mask], bins=(field_bin, d_eff_bin))[0]

    if CHARGE:
        plt.pcolormesh(field_bin,charge_bin, np.log(hdata.T+1))
    else:
        plt.pcolormesh(field_bin,d_eff_bin, hdata.T)
    plt.title("Monitor Rate Stability",size=14)
    cbar = plt.colorbar()
    cbar.set_label("log(Counts +1)")
    plt.xlabel(r"$\left| \vec{B} \right|$ [G]",size=14)
    if CHARGE:
        plt.ylabel("Avg. Charge", size=14)
    else:
        plt.ylabel("Det. Eff", size=14)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":

    main()
