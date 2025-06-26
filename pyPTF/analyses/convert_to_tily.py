

import numpy as np
import matplotlib.pyplot as plt 
from charge import process_analysis
from pyPTF.utils import project_point_to_plane, get_color
from pyPTF.analyses.get_efficiency import is_point_visible
from math import cos, sin, pi
import argparse

import os 

import h5py as h5 


a = 0.254564663
b = 0.254110205
c = 0.186002389

PMT_X = 0.417
PMT_Y = 0.297
PMT_Z_CENTER = 0.332

bounds_dict={
    "Q_1":[0,900],
    "Sigma_1":[0,700],
    "mu":[0,0.5],
    "hq1pemu":[0,500],
    "avg_charge":[0, 600],
    "det_eff":[0., 0.5],
    "bfield":[0, 1.1],
    "error":[-0.01 , 0.01]
}

def project_data(quantity:np.ndarray, xc:np.ndarray, yc:np.ndarray, theta:float, phi:float):
    qty = quantity.T.flatten()
    xmesh, ymesh = np.meshgrid(xc, yc)
    xs = xmesh.flatten() - PMT_X
    ys = ymesh.flatten() - PMT_Y
    # mask out the ones we want to ignore 
    

    mask = is_point_visible(xs ,ys, theta, phi, True)
    mask = mask>0
    print("{} vs {}".format(np.sum(mask.astype(int)),-np.sum(mask.astype(int)-1) ))
    
    qty = qty[mask]
    xs = xs[mask]
    ys = ys[mask]
    zs = PMT_Z_CENTER + c*np.sqrt(1 - (xs/a)**2 - (ys/b)**2) 
    
    points = np.array([xs,ys,zs]).T 

    plane_p0 = np.array([0,0,0])
    plane_norm = np.array([
        cos(phi)*cos(theta),
        sin(phi)*cos(theta),
        sin(theta)
    ])

    proj_pt = []
    for i in range(len(qty)):
        proj_pt.append(project_point_to_plane(
            points[i], plane_p0, plane_norm
        ))
    
    return {
        "points": np.array(proj_pt).T,
        "quantity": qty 
    }


def main(filename, theta:float, phi:float):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename)) 
    
    data = h5.File(filename,'r')
    run_no = np.int64(data["run_no"])

    #monitor = process_analysis(data["monitor"], True)
    pmt_20in_res = process_analysis(data["pmt0"])

    title_keys = {key:key for key in pmt_20in_res.keys()}
    title_keys["Q_1"] = r"$Q_{1}$ Gain [ADC]"
    title_keys["det_eff"] = "Detection Efficiency"
    title_keys["avg_charge"] = "Avg. Charge [ADC]"
    title_keys["error"]=r"d$\delta$/dADC"

    x_centers = 0.5*(pmt_20in_res["xs"][1:] + pmt_20in_res["xs"][:-1])
    y_centers = 0.5*(pmt_20in_res["ys"][1:] + pmt_20in_res["ys"][:-1])


    if not os.path.exists(os.path.join(os.path.dirname(__file__), "plots","{}".format(run_no))):
        os.mkdir(os.path.join(os.path.dirname(__file__),"plots","{}".format(run_no)))

    skip_keys = ["xs", "ys", "error"]
    for ik, key in enumerate(pmt_20in_res.keys()):
        if key in skip_keys:
            continue
        processed_pmt = project_data(pmt_20in_res[key], x_centers, y_centers, theta, phi)
        #processed_mon = project_data(monitor[key], x_centers, y_centers, theta, phi)
        points = processed_pmt["points"]

        cval = (processed_pmt["quantity"]-bounds_dict[key][0])/(bounds_dict[key][1] - bounds_dict[key][0])
        
        plt.clf()
        
        plt.pcolormesh([-10,-9],[-10,-9], [[0]],vmin=bounds_dict[key][0], vmax=bounds_dict[key][1], cmap="inferno")
        plt.colorbar()
        plt.scatter(-1*points[0], -1*points[1], color = get_color(cval, 1,'inferno'))
        plt.xlim([min(-1*points[0]), max(-1*points[0])])
        plt.ylim([min(-1*points[1]), max(-1*points[1])])
        plt.xlabel("Projected X [m]")
        plt.title(title_keys[key])
        plt.gca().set_aspect('equal')
        
        plt.ylabel("Projected Y [m]")
        outname = os.path.join(
            os.path.dirname(__file__),
            "plots",
            "{}".format(run_no),
            "tilted"
        )
        if not os.path.exists(outname):
            os.mkdir(outname)
        outname=os.path.join(outname, "{}_{}_{:.2f}_{:.2f}.png".format(run_no,key, theta, phi))
        plt.savefig(outname, dpi=400)
        #plt.show()

if __name__=="__main__":

    parser = argparse.ArgumentParser(
                    prog='Tilt Converter')
    
    parser.add_argument('filename')
    parser.add_argument('--theta', default=-90, type=float)
    parser.add_argument('--phi', default=0, type=float) 
    args = parser.parse_args()
    main(args.filename, args.theta*pi/180, args.phi*pi/180)
