"""
    Gets the integrated overall efficiency of a pmt from a `charge_results` scan 
"""

import numpy as np
import matplotlib.pyplot as plt 
from math import cos, sin, acos, asin, pi, sqrt, atan2

a = 0.254564663
b = 0.254110205
c = 0.186002389


def analytic_norm(theta, phi):
    norm_y = 1.0/np.sqrt(
        1 + ((b/a)*np.cos(phi)/np.sin(phi))**2 + ((b/c)*(np.tan(theta)/np.sin(phi)))**2
    )

    norm_y = np.ones_like(phi)
    
    if True:
        if isinstance(phi, np.ndarray):
            ones = np.ones_like(phi)
            ones[phi<0]*=-1
            norm_y*=-ones
        else:
            if phi<0:
                norm_y*=-1

    norm_z = norm_y*(b/c)*np.tan(theta)/np.sin(phi)
    norm_x = norm_y*(b/a)*(np.cos(phi)/np.sin(phi))

    mag = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
    norm_x/=mag
    norm_y/=mag
    norm_z/=mag

    return norm_x, norm_y, norm_z # make sure we always aim up 

def is_point_visible(x_pos, y_pos, zenith_angle, azimuth_angle, meshes=False)->np.ndarray:
    """
    
    """
    n_x = np.cos(azimuth_angle)*np.cos(zenith_angle)
    n_y = np.sin(azimuth_angle)*np.cos(zenith_angle)
    n_z = np.sin(zenith_angle)

    if meshes:
        xmesh = x_pos
        ymesh = y_pos
    else:    
        xmesh, ymesh= np.meshgrid(x_pos, y_pos)
    z_pos = c*((1 - (xmesh/a)**2 - (ymesh/b)**2)**0.5)
    phi = np.arctan2(ymesh, xmesh)

    theta = np.arcsin(z_pos/c)

    pmt_nx, pmt_ny, pmt_nz = analytic_norm(theta, phi)

    visible = (n_x*pmt_nx + n_y*pmt_ny + n_z*pmt_nz) 
    visible[np.isnan(visible)]=0
    return visible

def get_tilt_factor(_zenith):
    xs = np.arange(-a, a, 0.001)
    ys = np.arange(-a, a, 0.001)

    deg = pi/180

    zenith = _zenith*deg
    azimuth = 232*deg 

    visible = is_point_visible(xs,ys, zenith, azimuth)
    visible = visible>0
    visible = visible.astype(int)*1.0

    mx, my = np.meshgrid(xs, ys)
    visible[ np.sqrt(mx**2 + my**2) > a  ] = None
    
    return np.sum(visible)

if __name__=="__main__":

    baseline = -1 

    for _zenith in np.linspace(-90, 0, 5, endpoint=True):

        xs = np.arange(-a, a, 0.001)
        ys = np.arange(-a, a, 0.001)

        deg = pi/180

        zenith = _zenith*deg
        azimuth = 232*deg 

        visible = is_point_visible(xs,ys, zenith, azimuth)
        visible = visible>0
        visible = visible.astype(int)*1.0

        

        mx, my = np.meshgrid(xs, ys)
        visible[ np.sqrt(mx**2 + my**2) > a  ] = None

        if baseline<0:
            baseline = np.nansum(visible)
        print("Eff {}%".format(100*np.nansum(visible)/baseline))

        arrow_length = 0.1*cos(zenith)

        plt.pcolormesh(xs, ys, visible, vmin=-0.5, vmax=1.5, cmap='RdBu')
        plt.arrow(0, 0, arrow_length*cos(azimuth), arrow_length*sin(azimuth), head_width=0.01)
        plt.colorbar()
        plt.title("Theta: {} deg".format(_zenith))

        plt.gca().set_aspect('equal')
        plt.savefig("./shadowed_{}deg.png".format(_zenith), dpi=400)
        plt.show()