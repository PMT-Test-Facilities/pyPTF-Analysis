import numpy  as np
from pyPTF.enums import PMT

from math import sqrt, pi, log


from scipy.signal import peak_widths
from pyPTF.constants import PTF_SAMPLE_WIDTH, PTF_TS, PTF_SCALE

KEEP_ALL = True

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

        self._amplitudes = []
        self._widths = []
        self._peds = []
        self._means = []
        self._npass = 0

        self._which_pmt = which_pmt

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
    def means(self):
        return self._means.tolist()

    def __len__(self):
        return len(self._amplitudes)

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
        

        amplitude_cut = self._amplitudes > 30*PTF_SCALE


        #all_pass = np.logical_and(time_cut, amplitude_cut)
        if KEEP_ALL:
            all_pass = np.logical_not(np.isnan(self._amplitudes))
        else:
            all_pass = amplitude_cut

        # we need the indices of the peaks in the flattened coordinates 
        # so the `range` part is used as an offset 
        # ie the first is offset 0
        # the second is offset 1*(waveform size)
        # etc etc 
        flat_peaks = np.argmin( waveform, axis=1) + np.array(range(len(waveform)))*len(waveform[0])

        expanded_pedestals = np.repeat(self._peds, len(waveform[0]))


        threshs =self._amplitudes / 30*PTF_SCALE
        threshs[threshs>1.0] = 0.999

        rising_edges =peak_widths(expanded_pedestals.flatten()-1*waveform.flatten(), flat_peaks)[0]*PTF_SAMPLE_WIDTH
        
        self._means = PTF_TS[np.argmin(waveform, axis=1)] #- 0.5*rising_edges
        self._widths = peak_widths(expanded_pedestals.flatten()-1*waveform.flatten(), flat_peaks)[0]*PTF_SAMPLE_WIDTH/fwhm_scaler

        self._amplitudes = self._amplitudes[all_pass]
        self._means = self._means[all_pass]
        self._widths = self._widths[all_pass]
        self._peds = self._peds[all_pass]
        self._npass = np.sum(amplitude_cut.astype(int))/len(amplitude_cut)

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
    return np.linspace(sorted_values[0] - 0.5*width, sorted_values[-1] + 0.5*width, len(sorted_values)+1)

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

