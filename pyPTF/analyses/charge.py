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
from pyPTF.utils import make_bin_edges

CHARGE_BINNING = np.linspace(0, 5000, 200)

rtwo=sqrt(2)
rpi = sqrt(2*pi)
def get_analysis_charge(analysis):

    amplitude = np.array(analysis["amplitudes"][:])
    sigma= np.array(analysis["sigmas"][:])
    return rpi*np.abs(amplitude*sigma/2)/PTF_SCALE
def get_other_charge(analysis):
    print(analysis.keys())
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
    alpha = 10**params[3]
    w = params[0]
    mu = params[4]

    sigma0 = 10**params[1] 
    sigma1 = 10**params[5]
    q0 =  params[2]
    q1= params[6]
    norm = params[7]

    retval = w*alpha*np.exp(-alpha*x)

    retval+=(1-w)*( np.exp(-0.5*((x-q0)/sigma0)**2) + alpha*np.exp(-alpha*(x-q0)))*np.exp(-mu)

    for _i in range(10):
        n = _i+1
        sigman = sqrt(n)*sigma1 
        qn = n*q1

        gauss= sqrt(2/pi)/(sigman*(1 + erf(qn/(sqrt(2)*sigman))))
        gauss*=np.exp(-((x-qn)/(sqrt(2)*sigman))**2 )

        retval += (1-w)*(mu**n)*np.exp(-mu)*gauss/gamma(1+n)

    return retval*norm
#gaussian( x[0], npe*q1, sqrt( npe )*s1 )
    # retval += (1-w)*poisson_term( mu, npe ) * gaussian( x[0], npe*q1, sqrt( npe )*s1 )

def _pedestal(x, params):
    sigma0 = 10**params[0] 
    q0 =  params[1]
    norm = params[2]

    rval= norm*np.exp(-0.5*((x-q0)/sigma0)**2)
    return rval

def poisson_llh(x, y, params, function=_fitfun):
    """
    Poisson likelihood function
    Metric used is -log-likelihood 
    """


    exp = function(x, params)
    metric= 0.5*(np.log10(exp) - np.log10(y))**2

    metric[np.isnan(metric)] = 1e10
#  
    metric = -1*(y*np.log(exp) - exp - loggamma(y+1))
    metric[np.isnan(metric)] = 1e5
    metric[np.isinf(metric)] = 1e5
    return np.sum(metric)


def fit_binned_data(data, bin_centers):
    """
        Fits the charge distribution data and returns the fit parameters
    """

    pedestal_cut = 120
    def call_check(params):
        return poisson_llh(bin_centers[bin_centers<2500], data[bin_centers<2500], params, _fitfun)
    def pedestal_call(params):
        return poisson_llh(bin_centers[bin_centers<pedestal_cut], data[bin_centers<pedestal_cut], params, _pedestal)
    #min_res = basinhopping(call_check, x0, minimizer_kwargs={"bounds":bounds}, stepsize=2)

    
    x0_ped = [1.4, 2, np.sum(data[bin_centers<pedestal_cut])]
    bounds_ped = [(-15,15), (-15,5000), (-np.inf, np.inf)]
    options = {
        "gtol":1e-20,
        "ftol":1e-20,
        "eps":0.0000001
    }
    min_res_ped = minimize( pedestal_call, x0_ped, bounds=bounds_ped, options=options)
    # sigma then q0

    x0 = [
        0.75, # w
        min_res_ped.x[0], # sigma0
        min_res_ped.x[1],  #Q0
        -3, # alpha
        2, # mu 
        2, # sigma1
        700, # Q1    
        min_res_ped.x[2]# scale 
    ]
    bounds = [
        (0, 1), # w
        (-15,15), # sigma0 
        (min_res_ped.x[1],min_res_ped.x[1]), # Q0 
        (-15,-1), # alpha 
        (-50, 50), # mu 
        (-15, 15), # sigma1
        (-15,5000) , #Q1 
        (0, np.inf) # scale
    ]


    min_res = minimize(call_check , x0, bounds=bounds, options=options)
    if False:
        min_res = basinhopping(call_check, min_res.x, minimizer_kwargs={
            "bounds":bounds, "options":options
        }, niter=100, stepsize=2)
    return min_res.x

def process_analysis(data_dict:dict):
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

    x_edges = make_bin_edges(xs)
    y_edges = make_bin_edges(ys)
    n_x = len(x_edges)-1
    n_y = len(y_edges)-1

    charges = get_analysis_charge(data_dict)

    charge_edges = np.linspace(0, 5000, 200)
    charge_bin_centers = 0.5*(charge_edges[:-1] + charge_edges[1:])
    charge_bin_widths = charge_edges[1:] - charge_edges[:-1]
    sample = np.transpose([xs,ys,charges])
    binned_charges = np.histogramdd(
        sample, bins=(x_edges, y_edges, charge_edges)
    )[0]

    #n_fits = np.histogram2d(xs, ys, bins=(x_edges,y_edges), weights=data_dict["n_fits"])[0]

    results = {
        "avg_charge":np.zeros((n_x,n_y)),
        "Q_1":np.zeros((n_x,n_y)),
        "Sigma_1":np.zeros((n_x,n_y)),
        "mu":np.zeros((n_x,n_y)),    
        "n_good_fits":np.zeros((n_x,n_y)),
        "hq1pemu":np.zeros((n_x,n_y)),
        "xs":x_edges,
        "ys":y_edges
    }

    for ix in tqdm(range(n_x)):
        for jy in range(n_y):
            fitparams = fit_binned_data(binned_charges[ix][jy]/charge_bin_widths, charge_bin_centers)

            results["avg_charge"][ix][jy] = np.mean(binned_charges[ix][jy])
            results["Q_1"][ix][jy] = fitparams[2]
            results["Sigma_1"][ix][jy] = 10**fitparams[1]
            results["mu"][ix][jy] = fitparams[4]
            results["n_good_fits"][ix][jy] = 0 #n_fits[ix][jy]
            results["hq1pemu"][ix][jy] = fitparams[2]*fitparams[4]
    
    return results



def main_new(filename):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    name, ext = os.path.splitext(filename)
    _obj = open(filename, 'r')
    data = json.load(_obj)
    _obj.close()

    pmt_20in_res = process_analysis(data["pmt0"])
    #charge = get_analysis_charge(data_dict["pmt0"])


    #monitor_res = process_analysis(data["ptfanalysis02"]) 

    
    skip_keys = ["xs", "ys"]
    ratio = {}

    for key in pmt_20in_res.keys():
        if key in skip_keys:
            continue
        ratio[key] =pmt_20in_res[key]
        plt.clf()

        if True:
            if key=="Q_1":
                plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(ratio[key]), vmin=20, vmax=30)
            elif key=="Sigma_1":
                plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(ratio[key]), vmin=0, vmax=100)
            elif key=="mu":
                plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(ratio[key]), vmin=0, vmax=0.5)
            else:
                plt.pcolormesh(pmt_20in_res["xs"], pmt_20in_res["ys"], np.transpose(ratio[key])) #, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.xlabel("X [m]",size=14)
        plt.ylabel("Y [m]",size=14)
        plt.title(key, size=14)
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "{}.png".format(key)), dpi=400)
        

def main(filename):
    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    name, ext = os.path.splitext(filename)

    obj = open(filename, 'r')
    data_dict = json.load(obj)
    obj.close()


    charge = get_analysis_charge(data_dict["pmt0"])


    width = CHARGE_BINNING[1:] - CHARGE_BINNING[:-1]
    b_center = 0.5*(CHARGE_BINNING[:-1] + CHARGE_BINNING[1:])
    binned_data = np.histogram(charge, CHARGE_BINNING)[0]/(width)
    plt.stairs(binned_data, CHARGE_BINNING, fill=False, label="Data")

    # let's fit the pedestal first 

    fit_params = fit_binned_data(binned_data, b_center)
    fine_xs = np.linspace(min(CHARGE_BINNING), max(CHARGE_BINNING), len(CHARGE_BINNING)*3)
    fine_ys = _fitfun(fine_xs, fit_params)
    #plt.xlim([0, 640])
    plt.plot(fine_xs, fine_ys, label="Fit")
    plt.ylim([1e-4, 2e5])
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "charge_dist_all_{}.png".format(data_dict["run_no"])))


if __name__=="__main__":
    import sys
    main_new(sys.argv[1])

