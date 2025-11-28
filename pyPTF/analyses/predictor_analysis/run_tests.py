import numpy as np 
import matplotlib.pyplot as plt 

import json 
import os 
from math import pi 
from mpl_toolkits.mplot3d import axes3d

from pyPTF.utils import set_axes_equal
from effective_area import build_response_spline , area_scale, get_intersection, convert_to_cartesian, get_differential, construct_offset

N_GRID = 30

pmt_a = 0.5*(0.254564663 + 0.254110205)
pmt_b = pmt_a
pmt_c = 0.186002389


def check_intersection(light_zenith:np.ndarray, light_azimuth:np.ndarray):
    """
    Let's just throw a bunch of points at a screen to draw the ellipse
    Just copying these from below...
    """

    nx = np.cos(light_azimuth)*np.cos(light_zenith)
    ny = np.sin(light_azimuth)*np.cos(light_zenith)
    nz = np.sin(light_zenith)

    source_point = np.array([
        -0.3*nx+0.2, -0.3*ny, -0.3*nz+0.1
    ])

    test_zenith_edges = np.linspace(0, 1, N_GRID)
    test_zeniths = 0.5*(test_zenith_edges[1:] + test_zenith_edges[:-1])


    test_azimuth_edges = np.linspace(-pi, pi, N_GRID+1)
    test_azimuths = 0.5*(test_azimuth_edges[1:] + test_azimuth_edges[:-1])
    
    zenith_mesh, azimuth_mesh = np.meshgrid(test_zeniths, test_azimuths)

    # pmt zenith is determined from the angle as measured from 0 
    xval, yval, zval = convert_to_cartesian(np.arccos(zenith_mesh), azimuth_mesh)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(xval, yval, zval, c="gray")

    point_zen, point_azi = get_intersection(light_zenith, light_azimuth, source_point)

    point_z = np.sin(point_zen)*pmt_c
    point_x =np.sqrt((1 - (point_z/pmt_c)**2)/((1/pmt_a**2) + (np.tan(point_azi)/pmt_b)**2))
    if point_azi>pi/2 or point_azi<-pi/2:
        point_x*=-1
    point_y= np.tan(point_azi)*point_x

    ax.quiver(source_point[0], source_point[1], source_point[2], nx*0.2,ny*0.2,nz*0.2)
    ax.scatter([point_x,], [point_y,], [point_z,], color='red')
    set_axes_equal(ax)
    plt.show()

def check_area_scalers(light_zenith, light_azimuth):
    cos_zen = np.linspace(0, 1, N_GRID)

    test_zenith_edges = np.linspace(0, 1, N_GRID)
    test_zeniths = 0.5*(test_zenith_edges[1:] + test_zenith_edges[:-1])


    test_azimuth_edges = np.linspace(-pi, pi, N_GRID+1)
    test_azimuths = 0.5*(test_azimuth_edges[1:] + test_azimuth_edges[:-1])
    
    zenith_mesh, azimuth_mesh = np.meshgrid(test_zeniths, test_azimuths)

    refactored_scale = area_scale(np.arccos(zenith_mesh), azimuth_mesh, light_zenith, light_azimuth)
    # pmt zenith is determined from the angle as measured from 0 
    
    xval, yval, zval = convert_to_cartesian(np.arccos(zenith_mesh), azimuth_mesh)

    #plt.scatter(xval.flatten(), yval.flatten(), c=new_scale.flatten(), cmap="inferno")
    
    nx = np.cos(light_azimuth)*np.cos(light_zenith)
    ny = np.sin(light_azimuth)*np.cos(light_zenith)
    nz = np.sin(light_zenith)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(xval, yval, zval, c=refactored_scale.flatten(), cmap="inferno", vmin=0, vmax=2)
    
    if False:
        ax.quiver(
            0,0,0.5,
            0,0,-0.5
        ) 
    else:
        ax.quiver(
            -nx*0.5, 
            -ny*0.5, 
            -nz*0.5,
            nx*0.25,
            ny*0.25,
            nz*0.25
        )
    set_axes_equal(ax)
    plt.show()

def plot_azi_zen(filename):
    this_spline = build_response_spline(filename)

    test_zenith_edges = np.linspace(0, 1, N_GRID)
    test_zeniths = 0.5*(test_zenith_edges[1:] + test_zenith_edges[:-1])


    test_azimuth_edges = np.linspace(-pi, pi, N_GRID+1)
    test_azimuths = 0.5*(test_azimuth_edges[1:] + test_azimuth_edges[:-1])

    evaluated = this_spline(test_zeniths, test_azimuths, True)

    plt.pcolormesh(test_zenith_edges, test_azimuth_edges, evaluated.T, vmin=0)
    plt.xlabel("Zenith")
    plt.ylabel("Azimuth")
    plt.colorbar()
    plt.show()

def plot_3d_azi_zen(filename, add_area_effect=False, light_zenith=0, light_azimuth=0):
    this_spline = build_response_spline(filename)

    test_zenith_edges = np.linspace(0, 1, N_GRID)
    test_zeniths = 0.5*(test_zenith_edges[1:] + test_zenith_edges[:-1])


    test_azimuth_edges = np.linspace(-pi, pi, N_GRID+1)
    test_azimuths = 0.5*(test_azimuth_edges[1:] + test_azimuth_edges[:-1])

    zenith_mesh, azimuth_mesh = np.meshgrid(test_zeniths, test_azimuths)

    xval, yval, zval = convert_to_cartesian(np.arccos(zenith_mesh), azimuth_mesh)    

    evaluated = this_spline(xval, yval)
    if add_area_effect:
        refactored_scale = area_scale(np.arccos(zenith_mesh), azimuth_mesh, light_zenith, light_azimuth)
        evaluated*= refactored_scale

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(xval.flatten(), yval.flatten(), zval.flatten(),s=15, c=evaluated.flatten(), cmap="inferno", vmin=0, vmax=0.5)
    set_axes_equal(ax)
    plt.show()

def check_visible(test_file, light_zenith, light_azimuth):
    #raise NotImplementedError("Fix the get_differential return to send all of those points back!")
    this_spline = build_response_spline(test_file)
    rmin = 0.01
    rmax = 0.25
    theta_min = 0
    theta_max = 2*pi 

    nx = np.cos(light_azimuth)*np.cos(light_zenith)
    ny = np.sin(light_azimuth)*np.cos(light_zenith)
    nz = np.sin(light_zenith)

    source_points = np.array([
        -1*nx*0.5,
        -1*ny*0.5,
        -1*nz*0.5
    ])  
    rsq = np.linspace(rmin, rmax, N_GRID)

    radii = rsq
    angles = np.linspace(theta_min, theta_max, N_GRID+1)
    rad_mesh, angle_mesh = np.meshgrid(radii, angles)

    all_ofset = np.zeros((N_GRID, N_GRID+1))
    points = []
    sources = []
    for i in range(len(radii)):
        for j in range(len(angles)):

            offset = construct_offset(light_zenith, light_azimuth, angles[j], radii[i])
            new_source = source_points + offset 
            pmt_zen, pmt_azi = get_intersection(light_zenith, light_azimuth, new_source)
            xval, yval, zval = convert_to_cartesian(pmt_zen, pmt_azi)

            all_ofset[i][j]= get_differential(this_spline, radii[i], angles[j], light_zenith, light_azimuth)
            points.append([xval, yval, zval])
            sources.append(new_source)
    #all_offset = get_differential(this_spline, rad_mesh, angle_mesh, light_zenith, light_azimuth)/radii

    
    points = np.transpose(points)
    xval = points[0]
    yval = points[1]
    zval = points[2]

    sources = np.transpose(sources)
    sourcex = sources[0]
    sourcey = sources[1]
    sourcez = sources[2]

    

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    print("sum: {}".format(np.sum(all_ofset.flatten())))
    ax.scatter(xval.flatten(), yval.flatten(), zval.flatten(),s=15, c=all_ofset.flatten(), cmap="inferno", vmin=0, vmax=0.5)
    ax.scatter(sourcex.flatten(), sourcey.flatten(), sourcez.flatten(),s=15)
    ax.quiver(
            sourcex, 
            sourcey, 
            sourcez,
            nx*0.1*np.ones_like(sourcex),
            ny*0.1*np.ones_like(sourcex),
            nz*0.1*np.ones_like(sourcex)
        )
    set_axes_equal(ax)
    plt.show()



if __name__=="__main__":
        
    root_folder = os.path.join(os.path.dirname(__file__), "..","results")
    template = "charge_results_{}.json"

    test_file = os.path.join(
        root_folder,
        template.format(5745)
    )

    #plot_azi_zen(test_file)
    #plot_3d_azi_zen(test_file, True, -pi/2, 5*pi/4)

    #check_area_scalers(-3*pi/4, 5*pi/4)

    #check_intersection(-7.5*pi/8, 3.75*pi/4)

    check_visible(test_file, -pi/2, 5*pi/4)