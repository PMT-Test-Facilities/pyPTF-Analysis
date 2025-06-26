
import numpy as np  
import json 
import sys 
import matplotlib.pyplot as plt 
import os 


def main(run_no):
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

    mon_charge = np.array(data["monitor"]["det_eff"])

    print("Monitor {} +/- {}".format(np.mean(mon_charge), np.std(mon_charge)))

    pmt_charge = np.array(data["pmt0"]["det_eff"])
    xs = np.array(data["pmt0"]["xs"])
    ys = np.array(data["pmt0"]["ys"])

    plt.pcolormesh(xs, ys, pmt_charge.T /mon_charge.T, vmin=-0.02, vmax=0.02, cmap="RdBu")
    cbar = plt.colorbar()
    cbar.set_label("Charge Ratio (20in/Mon)", size=14)
    cbar.set_label("Avg Charge [20in]", size=14)
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]",size=14)
    plt.gca().set_aspect('equal')
    plt.show()
    


if __name__=="__main__":
    main(sys.argv[1])