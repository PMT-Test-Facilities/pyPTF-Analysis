import numpy  as np
from pyPTF.enums import PMT

from math import sqrt, pi, log
import pandas as pd 
import os 

from scipy.signal import peak_widths
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from pyPTF.constants import PTF_SAMPLE_WIDTH, PTF_TS, PTF_SCALE


from scipy.interpolate import griddata, RectBivariateSpline
import numpy as np 
from math import log10,sqrt 

DEBUG  = True

pmt_x = 0.417
pmt_y = 0.297 

pmt_radius =0.508/2


optical_file = os.path.join(
    os.path.dirname(__file__), "..", "data","optical_data.csv"
)
_optical_data = pd.read_csv(optical_file,header=0, delimiter="\s+")
xs = np.unique(_optical_data["ray_x"]).tolist()
ys = np.unique(_optical_data["ray_y"]).tolist()

xs = np.round(np.arange(-max(xs), max(xs)+0.01, 0.01),2).tolist()


print(xs)
data = np.zeros((len(xs), len(ys)))
for i in range(len(_optical_data["abs_prob"])):
    ix = xs.index(_optical_data["ray_x"][i])
    iy = ys.index(_optical_data["ray_y"][i])
    data[ix][iy] = _optical_data["abs_prob"][i]
    
    ix = xs.index(-1*_optical_data["ray_x"][i])
    data[ix][iy] = _optical_data["abs_prob"][i]


optical_terpo = RectBivariateSpline(xs, ys, data)
def abs_prob(x, y):
    return optical_terpo(x -pmt_x, y-pmt_y)

import matplotlib.pyplot as plt
def get_color(n, colormax=3.0, cmap="viridis"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)

fwhm_scaler= 2*sqrt(2*log(2))


root_pi = sqrt(2*pi)

def get_centers(bin_edges):
    return 0.5*(bin_edges[:-1] + bin_edges[1:])


class PointScan:
    """
        Holds all of the processed data for a fit-scan
        Does not hold raw waveforms! 

        Then this can later on get the main stuff
    """
    def __init__(self, x, y, z, rot, tilt, which_pmt):
        self._x = x
        self._y = y 
        self._z = z

        self._rot = rot
        self._tilt = tilt
        self._passing = []

        self._amplitudes = []
        self._widths = []
        self._peds = []
        self._means = []
        self._pp_pass = []
        self._np_pass = []
        self._pulse_times = None
        self._npass = 0

        self._which_pmt = which_pmt
    @property
    def passing(self):
        return self._passing.tolist()
    @property
    def np_pass(self):
        return self._np_pass
    @property
    def pp_pass(self):
        return self._pp_pass
    
    @property
    def pulse_times(self):
        return self._pulse_times.tolist()

    @property
    def npass(self):
        return self._npass

    @property 
    def amplitudes(self):
        return self._amplitudes.tolist()
    @property
    def sigmas(self):
        return self._widths.tolist()
    @property
    def pedestals(self):
        return self._peds
    @property
    def charge_sum(self):
        return self._charges.tolist()
    @property
    def means(self):
        return self._means.tolist()

    def __len__(self):
        return len(self._passing)

    def pack(self)->dict:
        return {
            "x":self._x,
            "y":self._y,
            "z":self._z,
            "rot":self._rot,
            "tilt":self._tilt,
            "amplitudes":np.array(self._amplitudes).tolist(),
            "sigmas":np.array(self._widths).tolist(),
            "means":np.array(self._means).tolist(),
            "peds":np.array(self._peds).tolist(),
            "pmt":self._which_pmt
        }
    
    @classmethod
    def unpack(self, data:dict)->'PointScan':
        new = PointScan(
            data["x"],
            data["y"],
            data["z"],
            data["rot"],
            data["tilt"],
            data["pmt"]
        )
        new._amplitudes=data["amplitudes"]
        new._widths=data["sigmas"]
        new._means = data["means"]
        new._peds = data["peds"]
        return new

    def x(self):
        return self._x
    def y(self):
        return self._y 
    def z(self):
        return self._z
    
    @property
    def pass_fract(self):
        return self._npass
    
    def extract_values_fit(self, waveform):
        peds = []
        amps = []
        times = []
        means = []
        widths = []
        pass_status = []
        nbins = 20 

        def dofit(this_waveform):
            def metric(params):
                return np.sum((this_waveform - params[0]*np.exp(-0.5*((PTF_TS - params[1])/params[2])**2))**2)

            x0 = (max(this_waveform), PTF_TS[np.argmax(this_waveform)], 1)
            bounds = [
                (0, np.inf),
                (10, 400),
                (0.1, 100)
            ]

            passing = max(this_waveform)>30

            if passing:
                res = minimize(metric, x0, bounds=bounds).x
                xfit = np.linspace(min(PTF_TS), max(PTF_TS), 1000)
                yfit =res[0]*np.exp(-0.5*((xfit - res[1])/res[2])**2)

                these_cross =  np.argwhere(np.diff(np.sign(yfit - 30))).flatten().tolist()
                if len(these_cross)!=2:
                    passing=False 
                    this_pulse_time = None
                else:
                    this_pulse_time = xfit[these_cross[0]]
            else:
                this_pulse_time = None
                res = (max(this_waveform), PTF_TS[np.argmax(this_waveform)], peak_widths(this_waveform, [np.argmax(this_waveform),] )[0][0]*PTF_SAMPLE_WIDTH/fwhm_scaler) 
            
            if DEBUG and max(this_waveform)>30:
                plt.bar(PTF_TS, this_waveform, width=PTF_TS[1]-PTF_TS[0], color='blue', label="Data")
                
                plt.plot(xfit, yfit, label="Fit", color='orange')
                plt.xlabel("Time [ns]",size=14)
                plt.ylabel("ADC", size=14)
                plt.show()


            return res[0], res[1], res[2], this_pulse_time, passing

        for wave in waveform:
            peds.append( np.mean(wave[:nbins]))    
            scaled = peds[-1] - wave 

            amp, mean, width, pulse_time, passing = dofit(scaled)
            amps.append(amp)
            means.append(mean)
            widths.append(width)
            if passing:
                times.append(pulse_time)
            pass_status.append(passing)

        self._amplitudes = np.array(amps)*PTF_SCALE
        self._means = np.array(means)
        self._widths = np.array(widths)
        self._peds = np.array(peds)
        self._pulse_times = np.array(times)
        self._npass = len(times)/len(amps)
        self._passing = np.array(pass_status)

    def extract_timing_signal(self, waveform):
        inverted = waveform*-1 


        self._passing = np.ones(len(waveform)).astype(bool)
        self._pulse_times = []
        for wave in waveform:
            crossing = np.argwhere(np.diff(np.sign(8200 - wave - 2000)))[0][0]
            self._pulse_times.append( PTF_TS[crossing] )
        self._pulse_times = np.array(self._pulse_times)
    def extract_values(self, waveform):
        """
            Use neato numpy stuff to find the amplitudes for all of the waveforms simultaneously

            Then we flatten the waveform into one long one
            and use scipy peak_widths to find the pulse widths 

            Should be easy enough to modify this to do cuts on the waveforms 

            Cool! 
        """
        nbins = 20

        self._peds = np.mean(waveform[:,:nbins], axis=1)

        self._amplitudes = self._peds-np.min(waveform,axis=1)
        self._charges = np.sum(waveform, axis=1) - self._peds*len(waveform[0])

        amplitude_cut = self._amplitudes > 30
        pamp_cut = self._amplitudes > 35
        namp_cut = self._amplitudes > 25

        # we need the indices of the peaks in the flattened coordinates 
        # so the `range` part is used as an offset 
        # ie the first is offset 0
        # the second is offset 1*(waveform size)
        # etc etc 
        flat_peaks = np.argmin( waveform, axis=1) + np.array(range(len(waveform)))*len(waveform[0])

        expanded_pedestals = np.repeat(self._peds, len(waveform[0]))



        self._means = PTF_TS[np.argmin(waveform, axis=1)] 
        self._widths = peak_widths(expanded_pedestals.flatten()-1*waveform.flatten(), flat_peaks)[0]*PTF_SAMPLE_WIDTH/fwhm_scaler

        # to get the pulse times, we use our fit mean and width
        # to determine the pulse shape at a much more granular scale 
        # 49 -60 for monitor 
        # 23-45 for pmt0
        if self._which_pmt==PMT.Hamamatsu_R3600_PMT.value:
            # 270 291
            min_cut = 110
            max_cut  = 130
        elif self._which_pmt==PMT.PTF_Monitor_PMT.value:
            # 60, 49
            min_cut  = 105
            max_cut = 120
        else:
            min_cut  = 100
            max_cut = 140

        t_cut = np.logical_and(self._means > min_cut, self._means<max_cut)
        passing = np.logical_and(amplitude_cut, t_cut)
        plpass = np.logical_and(pamp_cut, t_cut)
        napass = np.logical_and(namp_cut, t_cut)

        if self._which_pmt!=PMT.PTF_Monitor_PMT.value:
            rescale_amt = 8 # scale factor for granularity 
            super_TS = np.linspace(min(PTF_TS), max(PTF_TS), rescale_amt*len(PTF_TS), endpoint=True)
            amp_mesh, time_mesh = np.meshgrid(self._amplitudes, super_TS)
            mean_mesh, time_mesh = np.meshgrid(self._means, super_TS)
            width_mesh, time_mesh = np.meshgrid(self._widths, super_TS)
            fits = np.transpose(amp_mesh*np.exp(-0.5*((time_mesh - mean_mesh)/(width_mesh))**2 ))
                    
            # without this, spurious signals poke through
            fits[np.logical_not(passing)]*=0
            #waveform[np.logical_not(passing)]*=0
            idxs = []

            # for it, this_fit in enumerate(waveforms):
            for it, this_fit in enumerate(fits):
                these_cross =  np.argwhere(np.diff(np.sign(this_fit - 30))).flatten().tolist()
                #these_cross = np.argwhere(np.diff(np.sign(self._peds[it] - waveform[it] - 30))).flatten().tolist()
                if len(these_cross)==2:
                    idxs+=[these_cross[0]]
                else:
                    passing[it] = False
            # get where the fit pulses cross the threshold 

            self._pulse_times = np.array(super_TS[idxs])
            #self._pulse_times = np.array(PTF_TS[idxs])
        else:
            self._pulse_times = self._means
        
        self._np_pass = np.sum(napass.astype(int))/len(napass) 
        self._pp_pass = np.sum(plpass.astype(int))/len(plpass) 
        self._charges = self._charges*PTF_SCALE
        self._amplitudes = self._amplitudes*PTF_SCALE
        self._means = self._means
        self._widths = self._widths
        self._peds = self._peds*PTF_SCALE
        self._npass = np.sum(passing.astype(int))/len(passing)
        self._passing = passing
        #print("({:.3f},{:.3f}) - {}".format(self._x, self._y, len(self._amplitudes)))
    
    def calculate_charge(self)->np.ndarray:
        """
            Calculates the charges for all events for all of these events 
        """

        return root_pi*np.abs(self._amplitudes*self._widths/2)
    

def make_bin_edges(series):

    parsed = np.unique(series.round(decimals=3))
    sorted_values = list(sorted(parsed))
    if len(sorted_values)==0:
        print(sorted_values)
        raise ValueError()
        return np.array([])
    if len(sorted_values)==1:
        return np.array([sorted_values[0]-0.1, sorted_values[0]+0.1])

    width = sorted_values[1] - sorted_values[0]
    return np.arange(sorted_values[0] - 0.5*width, sorted_values[-1] + 1.5*width, width)
    return np.linspace(sorted_values[0] - 0.5*width, sorted_values[-2] + 0.5*width, len(sorted_values)+1)

def get_loc(x:float, domain:list,closest=False):
    """
    Returns the indices of the entries in domain that border 'x' 
    Raises exception if x is outside the range of domain 
    Assumes 'domain' is sorted!! And this _only_ works if the domain is length 2 or above 
    This is made for finding bin numbers on a list of bin edges 
    """

    if len(domain)<=1:
        raise ValueError("get_loc function only works on domains of length>1. This is length {}".format(len(domain)))


    # I think this is a binary search
    min_abs = 0
    max_abs = len(domain)-1

    lower_bin = int(abs(max_abs-min_abs)/2)
    upper_bin = lower_bin+1

    while not (domain[lower_bin]<=x and domain[upper_bin]>=x):
        if abs(max_abs-min_abs)<=1:
            print("{} in {}".format(x, domain))
            raise Exception("Uh Oh")

        if x<domain[lower_bin]:
            max_abs = lower_bin
        if x>domain[upper_bin]:
            min_abs = upper_bin

        # now choose a new middle point for the upper and lower things
        lower_bin = min_abs + int(abs(max_abs-min_abs)/2)
        upper_bin = lower_bin + 1
    
    assert(x>=domain[lower_bin] and x<=domain[upper_bin])
    if closest:
        return( lower_bin if abs(domain[lower_bin]-x)<abs(domain[upper_bin]-x) else upper_bin )
    else:
        return(lower_bin, upper_bin)
    

def get_closest(x, domain, mapped):
    """
    We imagine some function maps from "domain" to "mapped"

    We have several points evaluated for this function
        domain - list-like of floats. 
        mapped - list-like of floats. Entries in domain, evaluated by the function

    The user provides a value "x," and then we interpolate the mapped value on either side of 'x' to approximate the mapped value of 'x' 
    
    This is really just a linear interpolator 
    """
    if not isinstance(domain, (tuple,list,np.ndarray)):
        raise TypeError("'domain' has unrecognized type {}, try {}".format(type(domain), list))
    if not isinstance(mapped, (tuple,list,np.ndarray)):
        print(mapped)
        raise TypeError("'mapped' has unrecognized type {}, try {}".format(type(mapped), list))
    if not isinstance(x, (float,int)):
        raise TypeError("'x' should be number-like, not {}".format(type(x)))

    if len(domain)!=len(mapped):
        raise ValueError("'domain' and 'mapped' should have same length, got len(domain)={}, len(mapped)={}".format(len(domain), len(mapped)))
    
    lower_bin, upper_bin = get_loc(x, domain)
    
    # linear interp time
    x1 = domain[lower_bin]
    x2 = domain[upper_bin]
    y1 = mapped[lower_bin]
    y2 = mapped[upper_bin]

    slope = (y2-y1)/(x2-x1)
    value = (x*slope + y2 -x2*slope)

#    print("({}, {}) to ({}, {}) gave {}".format(x1,y1,x2,y2, value))
    
    return(value)

def point_plane_distance(point:np.ndarray, plane_p0:np.ndarray, plane_norm:np.ndarray):
    """
        Returns the distance of `point` from the plane defined by an origin point `plane_p0`
        and a plane normal vector `plane_norm`
        each entry should be a length-3 numpy array 
    """

    # we don't assume that the user has actually... normalized... the plane norm 
    # this falls out of a fun little vector calc / geometry problem 
    return (np.dot(point, plane_norm) - np.dot(plane_p0, plane_norm)) / np.dot(plane_norm, plane_norm)

def project_point_to_plane(point:np.ndarray, plane_p0:np.ndarray, plane_norm:np.ndarray):
    """
        each entry should be a length-3 numpy array 
    """
    distance = point_plane_distance(point, plane_p0, plane_norm)

    return point - plane_p0 - plane_norm*distance


class Irregular2DInterpolator:
    """
        This is used to make a 2D interpolator given a set of data that do not lie perfectly on a grid.
        This is done using scipy griddata and scipy RectBivariateSpline 
        interpolation can be `linear` or `cubic` 
        if linear_x/y, then the interpolation is done in linear space. Otherwise, it's done in log space
            setting this to False is helpful if your x/y values span many orders of magnitude 
        if linear_values, then the values are calculated in linear space. Otherwise they'll be evaluated in log space- but returned in linear space 
            setting this to False is helpful if your data values span many orders of magnitude 
        By default, nans are replaced with zeros. 
    """
    def __init__(self, xdata:np.ndarray, 
                 ydata:np.ndarray,
                   values:np.ndarray, linear_x = True, linear_y = True, linear_values=True,
                   replace_nans_with= 0.0, interpolation='linear'):

        self._nomesh_x = xdata
        self._nomesh_y = ydata 
        self._values = values if linear_values else np.log10(values)
        self._linear_values = linear_values
        if linear_x:
            self._xfine = np.linspace(min(self._nomesh_x), 
                                      max(self._nomesh_x), 
                                      int(sqrt(len(self._nomesh_x)))*2, endpoint=True)
        else:
            self._xfine = np.logspace(log10(min(self._nomesh_x)), 
                                      log10(max(self._nomesh_x)), 
                                      int(sqrt(len(self._nomesh_x)))*2, endpoint=True)

        
        if linear_y:
            self._yfine = np.linspace(min(self._nomesh_y), 
                                      max(self._nomesh_y), 
                                      int(sqrt(len(self._nomesh_y)))*2+1, endpoint=True)
        else:
            self._yfine = np.logspace(log10(min(self._nomesh_y)), 
                                      log10(max(self._nomesh_y)), 
                                      int(sqrt(len(self._nomesh_y)))*2+1, endpoint=True)


        mesh_x, mesh_y = np.meshgrid(self._xfine, self._yfine)

        # usee grideval to evaluate a grid of points 
        grid_eval = griddata(
            points=np.transpose([self._nomesh_x, self._nomesh_y]),
            values=self._values, 
            xi=(mesh_x, mesh_y),
            method=interpolation,
            fill_value=1.0
        )
        
        # if there are any nans, scipy 
        if np.any(np.isnan(grid_eval)):
            print("Warning! Nans were found in the evaluation of griddata - we're replacing those with zeros")
        grid_eval[np.isnan(grid_eval)] = replace_nans_with

        # and then prepare an interpolator 
        self._data_int = RectBivariateSpline(
            self._xfine, 
            self._yfine, 
            grid_eval.T
        )

    def __call__(self, xs, ys, grid=False):
        if self._linear_values:
            return self._data_int( xs, ys ,grid=grid)
        else:
            return 10**self._data_int( xs, ys ,grid=grid)