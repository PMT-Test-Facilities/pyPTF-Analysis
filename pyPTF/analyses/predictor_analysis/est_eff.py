
import numpy as np 
import matplotlib.pyplot as plt 
from math import pi, sqrt, cos, sin, exp 

from pyPTF.analyses.tabulate import build_spline
from pyPTF.analyses.get_efficiency import get_tilt_factor

baseline = get_tilt_factor(-90)
spline_interpolator = build_spline()

def sample_azimuth():
    while True:
        azi = np.random.rand()*pi
        other_no = np.random.rand()*1.1

        evaluated = (sin(azi*4)**2 )*exp(-azi/2) + 0.5*exp(-azi)*0.8
        
        if evaluated > other_no:
            return azi


def get_efficienfy(field_zenith, field_azi, this_field):
    """
        zenith of -90 is straight-on
        zenith of 0 is from the side 

        "field" units should be mG
    """

    field_effect= spline_interpolator(field_zenith, field_azi, grid=False)
    print("{} - {}".format(np.nanmin(field_effect), np.nanmax(field_effect)))
    field_strength = np.sqrt(np.sum(this_field**2, axis=1))
    print("{} - {}".format(np.nanmin(field_strength), np.nanmax(field_strength)))
    power = field_strength / 100
    rval = field_effect**power

    return rval

def get_area_effects(positions):
    xs = positions[0]
    ys = positions[1] 
    zs = positions[2] 

    zenith = np.arctan2(zs, np.sqrt(xs**2 +ys**2))

    effects = [get_tilt_factor(entry)/baseline for entry in zenith]
    return np.array(effects)

def get_field_value(positions):
    xs = positions[0]
    ys = positions[1] 
    zs = positions[2] 
    # make it... like... 250 in the middle of the barrel? 

    noisy = np.random.rand(len(xs), 3)
    noisy /= np.reshape(np.repeat(np.sqrt(np.sum(noisy**2, axis=1)),3), (len(xs), 3))
    noisy*=50


    field_mag = 250*(1 - np.exp(-1/((zs/4)**4)))
    a = 0.1
    b = 0.1
    c = 1.0
    field = np.array([a*field_mag, b*field_mag, c*field_mag])/sqrt(a**2 + b**2 + c**2)

    return field.T + noisy

def pmt_positions_orientations()->'tuple[np.ndarray, np.ndarray]':
    xs = []
    ys = []
    zs = []

    dx = []
    dy = []
    dz = []


    n_pmt = 11146

    radius = 33.8/2 
    height = 36.2

    pmt_per_sqm = n_pmt/(pi*height*2*radius + 2*pi*radius**2)
    packing = sqrt(pmt_per_sqm)**-1
    print("PMT distance {}".format(packing))
    # packing / rad = ang_stepi
    for angle in np.arange(0, 2*pi*radius, packing):
        for z_cord in np.arange(0.5*packing, height-0.52*packing, packing):
            xs.append( cos(angle)*radius )
            ys.append( sin(angle)*radius )
            zs.append( z_cord - 0.5*height )

            dz.append( 0 )
            dx.append(-sin(angle))
            dy.append(cos(angle))

    for xpos in np.arange(-radius, radius, packing):
        for ypos in np.arange(-radius, radius, packing):
            if sqrt(xpos**2 + ypos**2)>radius:
                continue
            xs.append(xpos)
            ys.append(ypos)
            zs.append(-0.5*height) 
            dx.append(0)
            dy.append(0)
            dz.append(1)

            xs.append(xpos)
            ys.append(ypos)
            zs.append(0.5*height) 
            dx.append(0)
            dy.append(0)
            dz.append(-1)

    
    print("{} PMTs".format(len(xs)))
    return np.array([xs,ys,zs]).T, np.array([dx, dy, dz]).T



if __name__=="__main__":
    dynodes = np.random.rand(1000)*2*pi

    positions, orientations=  pmt_positions_orientations()

    field = get_field_value(positions.T)
    field_mag = np.sqrt(np.sum(field**2, axis=1))
    
    
    # dot product, divide by magnitudes, then arccos 
    zenthith_angle = np.arccos(np.sum(orientations*field, axis=1)/(np.sqrt(np.sum(orientations, axis=1))*field_mag))
    # this makes the perpendicular zenith pi/2, but we want that to be zero
    zenthith_angle = (pi/2) - zenthith_angle # now goes from [-pi/2 to pi/2]
    
    field_perp = np.cos(zenthith_angle)*field_mag
    print("field total : {} - {}".format(np.nanmin(field_mag), np.nanmax(field_mag)))
    random_azi = np.array([sample_azimuth() for i in range(len(zenthith_angle))])
    field_effect = get_efficienfy(zenthith_angle, random_azi, field) + np.random.rand(len(field_perp))*0.05 -0.025
    area_effect = get_area_effects(positions)


    b_bins = np.linspace(-10, 350, 100)
    ce_bins = np.linspace(0.3 ,1.3, 101)

    binned = np.histogram2d(field_perp, field_effect, bins=(b_bins, ce_bins))[0]
    plt.pcolormesh(b_bins, ce_bins, binned.T , cmap="jet",vmax=60, vmin=0)
    plt.colorbar()
    plt.xlabel("B Parallel [mG]", size=14)
    plt.ylabel("Relative CE",size=14)
    plt.show()
