import json
import os 

import h5py as h5 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, basinhopping
from math import pi, sqrt, log
from scipy.special import gamma, loggamma, erf
from tqdm import tqdm

import matplotlib.pyplot as plt 

from pyPTF.constants import PTF_SCALE
from pyPTF.utils import make_bin_edges, abs_prob


def get_centers(bin_edges):
    return 0.5*(bin_edges[:-1] + bin_edges[1:])


pmt_x = 0.417
pmt_y = 0.297 

pmt_radius =0.508/2


DEBUG = False
RATIO = False

CHARGE_BINNING = np.linspace(-100, 1500, 800)
BIN_CENTERS = 0.5*(CHARGE_BINNING[:-1] + CHARGE_BINNING[1:])
BIN_WIDTHS  = CHARGE_BINNING[1:] - CHARGE_BINNING[:-1]

rtwo=sqrt(2)
rpi = sqrt(2*pi)
def get_analysis_charge(analysis, debug=False, just_height=False):
    #return -1*np.array(analysis["charge_sum"][:])/PTF_SCALE
    amplitude = np.array(analysis["amplitudes"][:])
    
    if just_height:
        return amplitude/PTF_SCALE

    if debug:
        plt.clf()
        plt.hist(amplitude/PTF_SCALE, bins=np.linspace(0, 600, 100))
        plt.xlabel("Charge [ADC]",size=14)
        plt.ylabel("Counts",size=14)
        plt.yscale('log')
        plt.savefig("pulse_height.png",dpi=400)
        plt.show()
    
    sigma= np.array(analysis["sigmas"][:])
    return rpi*np.abs(amplitude*sigma/2)/PTF_SCALE

def get_other_charge(analysis):
    array = np.array(analysis["charges"])/PTF_SCALE
    return array 

def _fitfun(x, params):
    """
        0 - w 
        1 - sigma 0
        2 - Q0
        3 - alpha 
        4 - mu 
        5 - sigma1 
        6 - q1 
        7 - norm

    """
    
    norm0 = params[0]
    sigma0 = 10**params[1] 
    q0 =  params[2]
    sigma1 = 10**params[3]
    q1= params[4]
    norm = params[5]

    ped_peak = norm0*np.exp(-0.5*((x-q0)/sigma0)**2)

    q1_peak = norm*np.exp(-0.5*((x-q1)/sigma1)**2)

    return ped_peak + q1_peak

def poisson_llh(x, y, params, function=_fitfun):
    """
    Poisson likelihood function
    Metric used is -log-likelihood 
    """


    exp = function(x, params)

#  
    metric = -1*(y*np.log(exp) - exp - loggamma(y+1))
    metric[np.isnan(metric)] = 1e5
    metric[np.isinf(metric)] = 1e5
    return np.sum(metric)



def gaus(x, params):
        return params[0]*np.exp(-0.5*((x - params[1])/params[2])**2)
def gaus_mu(x, params):
        return params[0]*np.exp(-0.5*((x - params[1])/params[2])**2)*np.exp(-params[3])


_ped_file = "../../data/pulse_series_fit_run5763.hdf5"
_ped_data = h5.File(_ped_file, 'r')
width = CHARGE_BINNING[1:] - CHARGE_BINNING[:-1]
_ped_20 = np.histogram(get_analysis_charge(_ped_data["pmt0"]), CHARGE_BINNING)[0]/(width) 
_ped_mon = np.histogram(get_analysis_charge(_ped_data["monitor"]), CHARGE_BINNING)[0]/(width) 

def fit_pedestal(data, mask, monitor=False):
    base_exp = _ped_mon[mask] if monitor else _ped_20[mask]

    def new_metric(params):
        exp = base_exp*params[0]
        metric = -1*(data*np.log(exp) - exp - loggamma(data+1))
        metric[np.isnan(metric)] = 1e5
        metric[np.isinf(metric)] = 1e5
        return np.sum(metric)
    
    res = minimize(
        new_metric, 
        x0=[1.0], 
        bounds=[(0, np.inf)]
    )
    return res.x



def fit_gaussian(xs, data, x0, bounds, func=gaus):
        
    def callf(params):
        return poisson_llh(xs, data, params, func)
    
    options = {
        "gtol":1e-20,
        "ftol":1e-20,
        "eps":0.0000001
    }
    result = minimize(callf, x0, bounds=bounds,options=options)
    return result.x # norm, 

def fit_bellamy(x, params):
        """
            0 - w 
            1 - sigma 0
            2 - Q0
            3 - alpha 
            4 - mu 
            5 - sigma1 
            6 - q1 
            7 - norm
            8 - underamplification 

        """
        
        w = params[0]
        sigma0 = 10**params[1] 
        q0 =  params[2]
        alpha = 10**params[3]
        mu = params[4]
        sigma1 = 10**params[5]
        q1= params[6]
        norm = params[7]
        underamp_x = params[8]
        underamp_sig = params[9]
        underamp_amp = params[10]
        

        retval = w*alpha*np.exp(-alpha*x)
        retval+=(1-w)*( np.exp(-0.5*((x-q0)/sigma0)**2) + alpha*np.exp(-alpha*(x-q0)))*np.exp(-mu)
        for _i in range(3):
            n = _i+1
            sigman = sqrt(n)*sigma1 
            qn = n*q1

            gauss= sqrt(2/pi)/(sigman*(1 + erf(qn/(sqrt(2)*sigman))))
            gauss_main=gauss*np.exp(-((x-qn)/(sqrt(2)*sigman))**2 )

            retval += w*(mu**n)*np.exp(-mu)*gauss_main/gamma(1+n)

        return retval*norm #+ underamp_amp*np.exp(-0.5*((underamp_x - x)/underamp_sig)**2)

def charge_fit_bellamy(data, bin_centers):
    
    
    def poisson_llh(x, y, params, function=_fitfun):
        """
        Poisson likelihood function
        Metric used is -log-likelihood 
        """


        exp = function(x, params)
        metric= 0.5*(np.log10(exp) - np.log10(y))**2

        #metric[np.isnan(metric)] = 1e10
    #  
        metric = -1*(y*np.log(exp) - exp - loggamma(y+1))
        #metric[np.isnan(metric)] = 1e5
        #metric[np.isinf(metric)] = 1e5
        return np.sum(metric) 
    
    """    x0 = [
        0.5, # w
        9, # sigma0
        0.9,  #Q0
        -3.812, # alpha
        -3.213e-2, # mu 
        1.2, # sigma1
        600, # Q1     
        np.sum(data),# scale 
        0.2,
        0.1
    ]"""

    x0 = [0.5,  1.02119177e+01,  3.78131765e-01, -1.39201042e+00,
        7.80014926e-01,  2.45973207e+00,  5.29935686e+02,  9.28514982e+05, 126, 10, 95]
    bounds = [
        (0, 1), # w
        (-15,15), # sigma0 
        (-50, 50 ), # Q0 
        (-15,0), # alpha 
        (0, 1200), # mu 
        (-10, 15), # sigma1
        (250, 2000) , #Q1 
        (0, np.inf),
        (0.0001, 200),# underamp charge 
        (0.0001, 200), # underamp width 
        (0, np.inf) # underamp height
    ]

    options = {
        "gtol":1e-20,
        "ftol":1e-20,
        "eps":1e-6
    }

    def call_check(params):
        return poisson_llh(bin_centers, data, params, fit_bellamy) 

    min_res = basinhopping(call_check , x0, minimizer_kwargs={"bounds":bounds, "options":options}, niter=1)

    return min_res.x

def fit_binned_data(data, bin_centers=BIN_CENTERS, monitor=False):
    """
        Fits the charge distribution data and returns the fit parameters
    """
    if monitor:
        pedestal_cut = 50
    else:
        pedestal_cut = 75

    other_cut = 900
    

    # first fit the pedestal peak
    if False:
        pedestal_res = fit_gaussian(
            xs=bin_centers[bin_centers<pedestal_cut],
            data=data[bin_centers<pedestal_cut],
            x0=[np.sum(data[bin_centers<pedestal_cut]), 10, 50, 50],
            bounds=[(1, np.inf),( 0, 100),(1, 200),(0, 10)],
            func= gaus_mu
        ) 
        sub_dat = data - gaus_mu(bin_centers, pedestal_res)
    else:   
        pedestal_res = fit_pedestal(
            data[np.logical_and(bin_centers>pedestal_cut, bin_centers<other_cut) ],
            np.logical_and(bin_centers>pedestal_cut, bin_centers<other_cut),
            monitor
        )
        if monitor:
            sub_dat = data - pedestal_res[0]*_ped_mon
        else:
            sub_dat = data - pedestal_res[0]*_ped_20
    
    #sub_dat[sub_dat<0]=0

    q1_res = fit_gaussian(
        xs=bin_centers[np.logical_and(bin_centers>pedestal_cut, bin_centers<other_cut) ],
        data= sub_dat[np.logical_and(bin_centers>pedestal_cut, bin_centers<other_cut) ],
        x0=[np.sum(sub_dat), 550, 300],
        bounds=[(0, np.inf),( 10, 2000),( 1, 1000)])
    if False:
        q1_res = fit_gaussian(
            xs=bin_centers,
            data=sub_dat,
            x0=[np.sum(sub_dat), 700, 300],
            bounds=[(0, np.inf),( 10, 2000),( 1, 1000)])
    
    # sometimes q1 snaps to 1 for some dumb reason...
    # so we grab the point where it's the biggest but after the pedestal cut
    return pedestal_res, q1_res

def process_analysis(data_dict:dict, is_monitor=False):
    """
        We prepare a dictionary of the analysis results where we bin data in a 2D grid. 
        At each point some quantity is evaluated and added here. 

        We look at :
            average charge (literally the mean)
            Q1 
            Sigma1
            mu 
            n good fits?
            hq1pemu??
    """
    xs = np.array(data_dict["x"][:])
    ys = np.array(data_dict["y"][:])
    zs = np.array(data_dict["z"][:])
    all_tilts = np.array(data_dict["tilt"][:])
    all_rots = np.array(data_dict["rot"][:])

    x_edges = make_bin_edges(xs)
    y_edges = make_bin_edges(ys)
    n_x = len(x_edges)-1
    n_y = len(y_edges)-1

    charges = get_analysis_charge(data_dict)

    amps =np.array(data_dict['amplitudes'])/PTF_SCALE
    sigs = np.array(data_dict["sigmas"])/PTF_SCALE
    height_bins = np.linspace(0,200, 201)
    
    if is_monitor:
        charge_bins = np.linspace(0, 1000, len(CHARGE_BINNING))
    else:
        charge_bins = CHARGE_BINNING

    charge_bin_centers = 0.5*(charge_bins[:-1] + charge_bins[1:])
    charge_bin_widths = charge_bins[1:] - charge_bins[:-1]
    sample = np.transpose([xs,ys,charges])
    binned_charges = np.histogramdd(
        sample, bins=(x_edges, y_edges, charge_bins)
    )[0]

    amp_bins = np.linspace(0,200,201)
    binned_amps = np.histogramdd(
        np.transpose([xs,ys,amps]), bins=(x_edges, y_edges,amp_bins)
    )[0]

    x_centers = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centers = 0.5*(y_edges[:-1] + y_edges[1:])

    print("binning charges")
    
    
    det_eff = np.histogram2d(xs, ys, bins=(x_edges,y_edges), weights=data_dict["n_pass"])[0]

    #error = np.histogram2d(xs, ys, bins=(x_edges,y_edges), weights=data_dict["pp_pass"])[0] - np.histogram2d(xs, ys, bins=(x_edges,y_edges), weights=data_dict["np_pass"])[0]
    #error /= 10
    #error /= det_eff
    keep = np.array(data_dict["passing"])
    charge_pass = np.array(data_dict["charge_sum"])[keep]
    counts = np.histogram2d(xs, ys, bins=(x_edges,y_edges))[0]
    kcounts = np.histogram2d(xs[keep], ys[keep], bins=(x_edges,y_edges))[0]
    avg_charge = np.histogram2d(xs, ys, bins=(x_edges,y_edges), weights=-1*np.array(data_dict["charge_sum"])/PTF_SCALE)[0]/counts
    avg_z = np.histogram2d(xs, ys, bins=(x_edges, y_edges), weights=zs)[0]/counts 
    avg_rot = np.histogram2d(xs, ys, bins=(x_edges, y_edges), weights=all_rots)[0]/counts 
    avg_tilt = np.histogram2d(xs, ys, bins=(x_edges, y_edges), weights=all_tilts)[0]/counts 

    gain = np.histogram2d(xs[keep], ys[keep], bins=(x_edges,y_edges), weights=-1*charge_pass/PTF_SCALE)[0]/kcounts

    bfield = np.histogram2d(xs, ys, bins=(x_edges,y_edges), weights=data_dict["bfield"])[0]/counts

    results = {
        "avg_charge":np.zeros((n_x,n_y)),
        "Q_1":np.zeros((n_x,n_y)),
        "Sigma_1":np.zeros((n_x,n_y)),
        "mu":np.zeros((n_x,n_y)),    
        "det_eff":np.zeros((n_x,n_y)),
        "bfield":np.zeros((n_x,n_y)),
        "hq1pemu":np.zeros((n_x,n_y)),
        "error":np.zeros((n_x, n_y)),
        "xs":x_edges,
        "ys":y_edges,
        "tilt":avg_tilt, 
        "zs":avg_z,
        "rot":avg_rot
    }
    plt.clf()
    for ix in tqdm(range(n_x)):
        for jy in range(n_y):
            #fitparams = np.zeros(8)
            fitped, q1_fit = fit_binned_data(binned_charges[ix][jy]/charge_bin_widths, charge_bin_centers, is_monitor)
            if DEBUG and ((abs(x_centers[ix]-0.4)<5e-3 and abs(y_centers[jy]-0.275)<1e-2) or (abs(x_centers[ix]-0.2)<5e-3 and abs(y_centers[jy]-0.275)<1e-2) or (abs(x_centers[ix]-0.45)<5e-3 and abs(y_centers[jy]-0.275)<1e-2)):
                center = abs(x_centers[ix]-0.4)<5e-3 and abs(y_centers[jy]-0.275)<1e-2
                edge = abs(x_centers[ix]-0.2)<5e-3 and abs(y_centers[jy]-0.275)<1e-2
            

                plt.title("Charge Distribution")
                #plt.stairs(binned_charges[ix][jy]/charge_bin_widths, charge_bins)
                color = "blue" if center else "orange"
                if not (center or edge):
                    color="green"
                plt.stairs(binned_charges[ix][jy], CHARGE_BINNING, color=color)
                #plt.plot(charge_bin_centers, gaus(charge_bin_centers, q1_fit) +gaus_mu(charge_bin_centers, fitped))
                plt.xlabel("Charge [ADC]", size=14)
                plt.ylabel("Arb. Un", size=14)
                plt.yscale('log')
            

            results["avg_charge"][ix][jy] = avg_charge[ix][jy]
            #results["Q_1"][ix][jy] = results["avg_charge"][ix][jy]*det_eff[ix][jy]

            results["Q_1"][ix][jy] = gain[ix][jy]
            results["Sigma_1"][ix][jy] = q1_fit[2]
            #results["mu"][ix][jy] = 10**fitparams[4]
            
            results["hq1pemu"][ix][jy] = q1_fit[1]*results["mu"][ix][jy]
    if DEBUG:
        plt.plot( [], [],color="orange", label="Edge")
        plt.plot([], [],color="blue", label="Center")
        plt.plot([], [],color="green", label="Right")
        plt.legend()
        plt.savefig("amp_dist_{}.png".format( "mon" if is_monitor else "20in"),dpi=400)
        plt.show()
        plt.clf()
    #results["error"] = error
    results["det_eff"] = det_eff
    results["mu"]=-1*np.log(1-results["det_eff"])
    results["bfield"] = bfield

    return results


def main_new(filename):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    name, ext = os.path.splitext(filename)
    data = h5.File(filename,'r')

    run_no = np.int64(data["run_no"])
    monitor = process_analysis(data["monitor"], True)
    pmt_20in_res = process_analysis(data["pmt0"])
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "plots","{}".format(run_no))):
        os.mkdir(os.path.join(os.path.dirname(__file__),"plots","{}".format(run_no)))


    #monitor_res = process_analysis(data["ptfanalysis02"]) 

    out_name = os.path.join(os.path.dirname(__file__),"results", "charge_results_{}.json".format(run_no))
    parse_data = {
        "pmt0":{},
        "monitor":{}
    }
    for key in pmt_20in_res:
        parse_data["pmt0"][key] = pmt_20in_res[key].tolist()
        parse_data["monitor"][key] = monitor[key].tolist()
    
    _obj = open(out_name,'wt')
    json.dump(parse_data, _obj, indent=4)
    _obj.close()

    
    skip_keys =[
        "rot","tilt", "x", "y","z", "run_no"
    ]
    ratio = {}

    ratio_bonds={
        "Q_1":[0,10],
        "Sigma_1":[0,5],
        "mu":[0,3],
        "hq1pemu":[0,7],
        #"avg_charge":[0,0.15],
        "det_eff":[0,2]
    }

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

    bounds = ratio_bonds if RATIO else bounds_dict

    title_keys = {key:key for key in pmt_20in_res.keys()}
    title_keys["Q_1"] = r"$Q_{1}$ Gain [ADC]"
    title_keys["det_eff"] = "Detection Efficiency"
    title_keys["avg_charge"] = "Avg. Charge [ADC]"
    title_keys["error"]=r"d$\delta$/dADC"

    for key in pmt_20in_res.keys():
        if key in skip_keys:
            continue
    
        ratio[key] =pmt_20in_res[key]/monitor[key] if RATIO else pmt_20in_res[key]
        if len(np.shape(ratio[key])) !=2:
            print(np.shape(ratio[key]))
            print("skipping {}".format(key))
            continue
        plt.clf()
        print(key)

        xmesh, ymesh = np.meshgrid(get_centers(pmt_20in_res["xs"]), get_centers(pmt_20in_res["ys"]))

        exclude = (xmesh - pmt_x)**2 + (ymesh-pmt_y)**2 < pmt_radius**2
        

        if True:
                #ratio[key][np.logical_not(exclude).T] = None
                if  key=="det_eff":
                    #print("Mean Detection: {}".format(np.mean(ratio[key][exclude.T])))
                    absorb = abs_prob(get_centers(pmt_20in_res["xs"]), get_centers(pmt_20in_res["ys"]))

                    plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(ratio[key]), cmap='inferno', vmin=bounds[key][0], vmax=bounds[key][1])
                
                elif key in bounds_dict:
                    if  key!="bfield" and key!="avg_charge":
                        #ratio[key][np.logical_not(exclude).T] = None
                        cmap = "inferno"
                    else:
                        cmap ="RdBu"
                    if key=="error":
                        cmap="RdBu"
                    if key=="avg_charge":
                        cmap="inferno"
                    plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(ratio[key]), vmin=bounds[key][0], vmax=bounds[key][1], cmap=cmap)
                else:    
                    
                    plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(ratio[key]), cmap='inferno') #, vmin=-1, vmax=5)
        else:
            plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(ratio[key]))
        cbar = plt.colorbar()
        if RATIO:
            cbar.set_label("20in/Monitor")
        else:
            cbar.set_label("20in {}".format(title_keys[key]))
        plt.xlabel("X [m]",size=14)
        plt.ylabel("Y [m]",size=14)
        plt.gca().set_aspect('equal')
        plt.title(title_keys[key], size=14)
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "{}".format(run_no),"{}_{}.png".format(key,run_no)), dpi=400)

    plt.clf() 
    plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(monitor["avg_charge"]), cmap='inferno', vmin=0, vmax=300)
    print("{} - {}".format(np.mean(monitor["det_eff"]), np.std(monitor["det_eff"])))
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]",size=14)
    plt.gca().set_aspect('equal')
    plt.title(title_keys["avg_charge"], size=14)
    cbar = plt.colorbar()
    cbar.set_label("Monitor")
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "{}".format(run_no),"{}_{}_monitor.png".format("avg_charge",run_no)), dpi=400)

    plt.clf() 
    plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(monitor["error"]), cmap='RdBu', vmin=-0.01, vmax=0.01)
    print("{} - {}".format(np.mean(monitor["error"]), np.std(monitor["error"])))
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]",size=14)
    plt.gca().set_aspect('equal')
    plt.title(title_keys["error"], size=14)
    cbar = plt.colorbar()
    cbar.set_label("Monitor")
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "{}".format(run_no),"{}_{}_monitor.png".format("error",run_no)), dpi=400)

def main(filename):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    name, ext = os.path.splitext(filename)

    print("... loading ")
    data_dict = h5.File(filename, 'r')
    run_no = np.int64(data_dict["run_no"])

    charge = get_analysis_charge(data_dict["pmt0"])


    width = CHARGE_BINNING[1:] - CHARGE_BINNING[:-1]
    b_center = 0.5*(CHARGE_BINNING[:-1] + CHARGE_BINNING[1:])
    binned_data = np.histogram(charge, CHARGE_BINNING)[0]/(width)   
    

    # let's fit the pedestal first 

    ped, q1_res = fit_binned_data(binned_data, b_center, monitor=False)
    print("Gain : {}".format(q1_res[1]))
    fine_xs = np.linspace(min(CHARGE_BINNING), max(CHARGE_BINNING), len(CHARGE_BINNING)-1)
    plt.stairs(binned_data , CHARGE_BINNING, fill=False, label="Data")
    plt.stairs(binned_data -_ped_20*ped[0], CHARGE_BINNING, fill=False, label="Data - Ped")
    print(ped)
    #fine_ys = gaus(fine_xs, q1_res) +gaus_mu(fine_xs, ped)
    fine_ys = gaus(fine_xs, q1_res) + _ped_20*ped[0]
    #plt.xlim([0, 640])
    plt.plot(fine_xs, fine_ys, label="Fit")
    #plt.stairs(_ped_20*ped[0], CHARGE_BINNING, label="Pedestal")
    #plt.ylim([0, 500])
    #plt.yscale('log')
    plt.legend()
    
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots","{}".format(run_no), "charge_dist_all_{}.png".format(np.int64(data_dict["run_no"]))))
    plt.show()

if __name__=="__main__":
    import sys
    #main(sys.argv[1])
    main_new(sys.argv[1])

