
import os 
import json 
import numpy as np
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt 
from pyPTF.utils import get_centers

from pyPTF.analyses.plot_relative_to_comp import load_data, load_other_data
pmt_x = 0.417
pmt_y = 0.297 

pmt_radius =0.508/2

if __name__=="__main__":
    first  = "charge_results_5630.json"
    second  = "charge_results_5654.json"
    root_folder = os.path.dirname(__file__)

    compensated_data = load_data(os.path.join(root_folder, first))
    xmesh, ymesh = np.meshgrid(get_centers(compensated_data["pmt0"]["xs"]), get_centers(compensated_data["pmt0"]["ys"]))
    exclude = (xmesh - pmt_x)**2 + (ymesh-pmt_y)**2 > pmt_radius**2
    
    this_data = load_other_data(
            os.path.join(root_folder, second),
            compensated_data
    )

    key = "det_eff"
    if False:
        double_ratio = 1-(this_data["pmt0"][key]/this_data["monitor"][key])/(compensated_data["pmt0"][key]/compensated_data["monitor"][key])
    else:
        double_ratio = 1- (this_data["pmt0"][key])/(compensated_data["pmt0"][key])
    double_ratio[exclude.T] = None
    plt.pcolormesh(compensated_data["pmt0"]["xs"], compensated_data["pmt0"]["ys"], 100*double_ratio.T, vmin=-40, vmax=40, cmap='coolwarm')

    plt.xlabel("X [m]", size=14)
    plt.ylabel("Y [m]", size=14)
    #plt.title("1-Ratio",size=14)
    plt.gca().set_aspect('equal')
    color = plt.colorbar()
    color.set_label("% Difference")
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "det_eff_ratio.png"), dpi=400)
    plt.show()
    

