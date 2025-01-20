import numpy as np 
import h5py as h5 
import os 
from math import sin, cos, pi

import matplotlib.pyplot as plt 

from pyPTF.analyses.charge import fit_binned_data, get_analysis_charge, BIN_WIDTHS, CHARGE_BINNING, gaus, gaus_mu, PTF_SCALE
from pyPTF.analyses.timing import TIME_EDGES, extract_values, PTF_SAMPLE_WIDTH
from pyPTF.utils import project_point_to_plane, get_color
from tqdm import tqdm 

PMT_X = 0.417
PMT_Y = 0.297
PMT_Z_CENTER = 0.332

RATIO = False
DEBUG = False

def process_dataset_charge(data_dict):
    """
        Processes the charge-data for the data provided as a dictionary for a given PMT. 

        This extracts the gain and detection efficiency. 
        Could be modified to calculate the mean number of PEs, and other such quantities 

        NOTE - the point determination won't work anymore if you run a scan on a vertical plane (which we'll never do)
    """
    xs = np.array(data_dict["x"][:])
    ys = np.array(data_dict["y"][:])
    zs = np.array(data_dict["z"][:])

    unique_points = np.unique(np.array([xs, ys, zs]).T , axis=0)
    tilt = np.array(data_dict["tilt"][:])
    
    rot  = [180-45,] # np.array(data_dict["rot"][:])
    tilts = []

    # need to get the general aiming direction to project these into 2D 
    avg_tilt = np.nanmean(tilt)*pi/180
    
    avg_rot =  np.nanmean(rot)*pi/180
    plane_p0 = np.array([0,0,0])
    plane_norm =np.array([
        cos(avg_rot)*cos(avg_tilt),
        sin(avg_rot)*cos(avg_tilt),
        sin(avg_tilt)
    ])

    # we can use the unique xs and ys here to filter these 


    charges = get_analysis_charge(data_dict)

    

    this_data = {
        "points":[],
        "det_eff":[],
        "gain":[],
        "avg_charge":[]
    }

    # we have a series of points we want to evaluate the data at. 
    # the scanpoint meta-data is discarded during the extraction, so we just use unique xy points
    for ix, point in tqdm(enumerate(unique_points)):

        xval = point[0]
        yval = point[1]
        
        conditional = np.logical_and(xs==xval, ys==yval )
        if len(xs[conditional])==0:
            continue
        this_point = np.array([
            xval - PMT_X, 
            yval - PMT_Y, 
            np.mean(zs[conditional]) - PMT_Z_CENTER
        ])

        if any([np.isnan(entry) for entry in this_point]):
            raise ValueError("nan in point: {}".format(this_point))
        


        tilts.append(np.mean(tilt[conditional]))
        projected_point = project_point_to_plane(this_point, plane_p0, plane_norm)

        if any([np.isnan(entry) for entry in projected_point]):
            print(this_point)
            print(plane_p0)
            print(plane_norm)
            raise ValueError("nan in point: {}".format(projected_point))
        


        det_eff = sum(data_dict["n_pass"][conditional])

        these_charges = charges[conditional]
        binned_charges = np.histogram(these_charges, CHARGE_BINNING)[0]/BIN_WIDTHS
        fitped, q1_fit = fit_binned_data(binned_charges)
        if ix%500==0 and DEBUG:
            fine_xs = np.linspace(min(CHARGE_BINNING), max(CHARGE_BINNING), len(CHARGE_BINNING)*3) 
            fine_ys = gaus(fine_xs, q1_fit) # + gaus_mu(fine_xs, fitped)
            plt.stairs(binned_charges , CHARGE_BINNING, fill=False, label="Data")
            plt.plot(fine_xs, fine_ys, label="Fit")
            plt.ylim([0,100])
            plt.xlabel("Charge [ADC]", size=14)
            plt.ylabel("Counts", size=14)

            plt.legend()
            plt.show()

        this_data["gain"].append(q1_fit[1])
        this_data["det_eff"].append(det_eff)
        this_data["points"].append(projected_point)
        this_data["avg_charge"].append(-1*np.mean(data_dict["charge_sum"][conditional])/PTF_SCALE)

    return this_data


def extract_timing(data_dict):
    """
        Extracts the timing information given a full set of data. 
        This requires _all_ of the information. 
    """
    xs = np.array(data_dict["pmt0"]["x"][:])
    ys = np.array(data_dict["pmt0"]["y"][:])
    zs = np.array(data_dict["pmt0"]["y"][:])

    unique_points = np.unique(np.array([xs, ys, zs]).T , axis=0)

    tilt = np.array(data_dict["pmt0"]["tilt"][:])
    rot  = [180-45,] #np.array(data_dict["pmt0"]["rot"][:])

    avg_tilt = np.nanmean(tilt)*pi/180

    plt.plot(range(len(tilt)), tilt)
    plt.show()
    plt.clf()
    print("Observed tilt of {}".format(avg_tilt*180/pi))
    avg_rot = np.nanmean(rot)*pi/180
    print("Observed rot of {}".format(avg_rot*180/pi))


    keep =  np.array(data_dict["pmt0"]["passing"])
    main_mean =  np.array(data_dict["pmt0"]["pulse_times"])

    # not really the monitor, just copying the naming from another script 
    monitor_mean = np.array(data_dict["timing_data"]["pulse_times"])[keep]
    
    dif_bins = np.arange(230, 310,0.5*PTF_SAMPLE_WIDTH)
    time_centers = 0.5*(dif_bins[1:] + dif_bins[:-1])

    diff = (main_mean+290) - monitor_mean

    plane_p0 = np.array([0,0,0])
    plane_norm =np.array([
        cos(avg_rot)*cos(avg_tilt),
        sin(avg_rot)*cos(avg_tilt),
        sin(avg_tilt)
    ])

    this_data = {
        "points":[],
        "times":[],
        "tts":[],
    }

    for ix, point in tqdm(enumerate(unique_points)):

        xval = point[0]
        yval = point[1]

        
        conditional = np.logical_and(xs==xval , np.logical_and(ys==yval , keep))
                    
        if len(xs[conditional])==0:
            continue

        this_point = np.array([
            xval - PMT_X, 
            yval - PMT_Y, 
            PMT_Z_CENTER - np.mean(zs[conditional])
        ])

        spatial_cond = np.logical_and(xs[keep]==xval, ys[keep]==yval)
        projected_point = project_point_to_plane(this_point, plane_p0, plane_norm)

        diff_histo = np.histogram(diff[spatial_cond], bins=dif_bins)[0]
        transit_time, spread = extract_values(time_centers, diff_histo)

        this_data["points"].append(projected_point)
        this_data["times"].append(transit_time)
        this_data["tts"].append(spread)

    return this_data


def main(filename):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    data_dict = h5.File(filename)

    run_number = np.int64(data_dict["run_no"])

    
    sk_pmt = process_dataset_charge(data_dict["pmt0"])
    mon_pmt = process_dataset_charge(data_dict["monitor"])
    timing = extract_timing(data_dict)

    # first let's make charge plots 
    time_keys = ["times", "tts"]
    for tk in time_keys:
        sk_pmt[tk] = timing[tk]
    

    charge_keys = ["det_eff", "gain", "avg_charge"] + time_keys

    titles={
        "det_eff":"Detection Eff.", 
        "gain":"Q1 Gain",
        "times":"Transit Time",
        "tts":r"TTS (1$\sigma$)",
        "avg_charge":"Avg. Charge"
    }
    
    if RATIO:
        bounds = np.array([
            [0.5, 1.5],
            [1.5,4],
            [0, 600],
            [283, 303],
            [0,10],
        ])
    else:
        bounds = np.array([
            [0, 0.5],
            [0, 800],
            [0, 600],
            [283, 303],
            [0,10],
        ])
    
    on_pmt = np.array(sk_pmt["det_eff"])>0.05 

    
    for ik, key in enumerate(charge_keys):
        
        xs = []
        ys = []
        value = []
        is_time = key=="times" or key=="tts"

        for i in range(len(sk_pmt["points"])):
            if RATIO and (not is_time):
                value.append(np.array(sk_pmt[key][i])/np.array(mon_pmt[key][i]))
            else:
                value.append(np.array(sk_pmt[key][i]))
            xs.append(sk_pmt["points"][i][0])
            ys.append(sk_pmt["points"][i][1])
        
        print(np.mean(np.array(sk_pmt[key])[on_pmt]))

        xs = -1*np.array(xs)
        
        colorval = (value - bounds[ik][0])/(bounds[ik][1] - bounds[ik][0])
        
        #plt.imshow([[]], vmin=bounds[ik][0], vmax=bounds[ik][1], cmap="viridis")
        plt.clf()
        plt.pcolormesh([-10,-9],[-10,-9], [[0]],vmin=bounds[ik][0], vmax=bounds[ik][1], cmap="inferno")
        plt.colorbar()
        plt.scatter(xs, ys, color=get_color(colorval, 1,"inferno"))
        plt.xlabel("Projected x [m]",size=12)
        plt.ylabel("Projected y [m]",size=12)
        plt.xlim([min(xs), max(xs)])
        plt.ylim([min(ys), max(ys)])
        plt.title(titles[key], size=14)
        
        
        
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        if not os.path.exists("./plots/{}".format(run_number)):
            os.mkdir("./plots/{}".format(run_number))
        plt.savefig("./plots/{}/{}_{}.png".format(run_number,key, run_number), dpi=400)
        
        plt.show()


if __name__=="__main__":
    import sys 
    main(sys.argv[1])