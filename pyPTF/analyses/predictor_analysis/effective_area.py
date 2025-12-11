import numpy as np 
from math import pi 
from scipy.integrate import dblquad
from pyPTF.utils import Irregular2DInterpolator
import json 
import os 
import matplotlib.pyplot as plt 
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

"""
    We need to determine a scaling factor for converting between vertical injection and light from other directions

    Need PMT normal direction given ellipsoid parameters


    Fro a given direction of light we need to integrate over the 
"""

pmt_x = 0.417
pmt_y = 0.297 # or should these be fit? 

pmt_a = 0.5*(0.254564663 + 0.254110205)
pmt_b = pmt_a
pmt_c = 0.186002389

PMT_RAD =0.508/2
RSQ = PMT_RAD**2

CHARGE = False 

pmt_a2 = pmt_a**2
pmt_b2 = pmt_b**2 
pmt_c2 = pmt_c**2 

def convert_to_cartesian(_pmt_zeniths, pmt_azimuths):
    
    zval = np.sin(_pmt_zeniths)*pmt_c 
    xval =np.sqrt((1 - (zval/pmt_c)**2)/((1/pmt_a**2) + (np.tan(pmt_azimuths)/pmt_b)**2))
    

    make_minus = np.logical_or(
        pmt_azimuths>pi/2 , pmt_azimuths < -pi/2
    )
    if isinstance(xval, np.ndarray):
        xval[make_minus]*=-1
        
    else:
        if make_minus:
            xval*=-1
    yval = np.tan(pmt_azimuths)*xval 
    return xval, yval, zval


def ellipsoid_normal(azimuth, zenith):
    """
    Determine the normal vector on the surface of an ellipsoid 
    at the point determined by azimuth and zenith 
    """

    phi = azimuth % (2*pi)
    norm_y = 1.0/np.sqrt(
        1 + ((pmt_b/pmt_a)*np.cos(phi)/np.sin(phi))**2 + ((pmt_b/pmt_c)*(np.tan(zenith)/np.sin(phi)))**2
    )

    norm_y = np.ones_like(phi)

    if True:
        if isinstance(phi, np.ndarray):
            ones = np.ones_like(phi)
            ones[phi>pi]*=-1
            norm_y*=ones
        else:
            if phi>pi:
                norm_y*=-1

    norm_z = norm_y*(pmt_b/pmt_c)*np.tan(zenith)/np.sin(phi)
    norm_x = norm_y*(pmt_b/pmt_a)*(np.cos(phi)/np.sin(phi))

    mag = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
    norm_x/=mag
    norm_y/=mag
    norm_z/=mag

    return norm_x, norm_y, norm_z 


def get_intersection(light_zenith:np.ndarray, light_azimuth:np.ndarray, source_point:np.ndarray):
    """
    Determines the point of intersection on a PMT surface in PMT coordinates

        light_zenith and azimuth are used for the directions of light travel
        source point is a (3,N) numpy array where N is the dimensionality of the light vectors

    We determine the intersection by looking for simultaneous solutions 
    to the line equation and ellipse equation 

    The normal of light propagation direction must be towards the PMT given the source_point!!
    """
    nx = np.cos(light_azimuth)*np.cos(light_zenith)
    ny = np.sin(light_azimuth)*np.cos(light_zenith)
    nz = np.sin(light_zenith)

    dot_product= nx*source_point[0] + ny*source_point[1] + nz*source_point[2]
    is_numpy = False
    if isinstance(dot_product, np.ndarray):
        is_numpy = True
        if (dot_product>0).any():
            raise ValueError("Light direction must be towards PMT ")
    else:
        if dot_product>0:
            raise ValueError("Light direction must be towards PMT ")

    quad_A = pmt_b2*pmt_c2*nx**2 + pmt_a2*pmt_c2*ny**2 + pmt_a2*pmt_b2*nz**2 
    quad_B = 2*pmt_b2*pmt_c2*nx*source_point[0] + 2*pmt_a2*pmt_c2*ny*source_point[1] + 2*pmt_a2*pmt_b2*nz*source_point[2]
    quad_C = pmt_b2*pmt_c2*source_point[0]**2 + pmt_a2*pmt_c2*source_point[1]**2 + pmt_a2*pmt_b2*source_point[2]**2 - pmt_a2*pmt_b2*pmt_c2
    
    pre_root = quad_B**2 - 4*quad_A*quad_C    
    has_soln = pre_root>0 

    if is_numpy:
        pre_root[pre_root>0 ] = np.sqrt(pre_root[pre_root>0 ])
        pre_root[pre_root<=0] = np.nan
    else:
        if not has_soln:
            return np.nan, np.nan
        else:
            pre_root = np.sqrt(pre_root)

    # these are solutions to the parametrization of the line.
    # we want solutions where this is the smallest (ie, closest to source_point). 
    # since pre_root is always positive, we are always going to choose the -root solution
    solution = (-quad_B - pre_root)/(2*quad_A)

    soln_x = solution*nx + source_point[0]
    soln_y = solution*ny + source_point[1]
    soln_z = solution*nz + source_point[2]
    
    # now convert these to PMT azimuth and zenith 
    pmt_azi = np.arctan2(soln_y, soln_x)
    pmt_zen = np.arcsin(soln_z/pmt_c)

    return pmt_zen, pmt_azi
         
_raw_absorption_data = np.loadtxt(os.path.join(os.path.dirname(__file__), "glass_simulation.dat"), delimiter=",").T
_raw_absorption_data[1][_raw_absorption_data[1]<0]= 0.0
ABS_SPLINE = CubicSpline(_raw_absorption_data[0]*pi/180, _raw_absorption_data[1])

def get_absorption(theta_outer):
    """
        Given an angle of incidence on the PMT surface of theta, find the 
        Can we use the small angle approximation? No, this does not work
    """

    N_GLASS = 1.4
    GLASS_THICK = 5e-3 
    RADIUS = 0.2 # approximate this like a sphere for "close enough" approximation
    theta_glass = np.arcsin(np.sin(theta_outer)/N_GLASS)

    # in the big-PMT approximation, this returns to the glass-plane case 
    
    angle_of_incidence = theta_glass + np.arctan(np.tan(theta_glass)*GLASS_THICK/RADIUS)

    return ABS_SPLINE(angle_of_incidence)

def area_scale(pmt_zen, pmt_azi, light_zenith, light_azimuth, perfect=False):
    scales = np.ones_like(pmt_zen)
    scales[np.isnan(pmt_zen)] = 0

    # light paramers 
    nx = np.cos(light_azimuth)*np.cos(light_zenith)
    ny = np.sin(light_azimuth)*np.cos(light_zenith)
    nz = np.sin(light_zenith) 

    # normal vector needs calculation 
    norm_x, norm_y, norm_z = ellipsoid_normal(pmt_azi, pmt_zen)

    vertical_scale = np.abs(norm_z*-1)
    injection_scale = np.abs(nx*norm_x + ny*norm_y + nz*norm_z)

    if perfect:
        absorption_scaler = get_absorption(np.arccos(injection_scale)) / get_absorption(np.arccos(vertical_scale))
    else:
        absorption_scaler = get_absorption(np.arccos(injection_scale)) / get_absorption(np.arccos(vertical_scale))

    if perfect:
        return np.ones_like(absorption_scaler)* (injection_scale / vertical_scale)
    else:
        return absorption_scaler * (injection_scale / vertical_scale)
    


def alt_construct_offset(light_zenith, light_azimuth, theta, radius):
    # Base offset: shape (3, N, M)
    offset = np.stack([
        np.zeros_like(radius),
        radius * np.cos(theta),
        radius * np.sin(theta)
    ], axis=0)

    # Rotation matrix around Y (light_zenith)
    R_y = np.array([
        [ np.cos(light_zenith), 0, -np.sin(light_zenith)],
        [ 0,                   1,  0                  ],
        [ np.sin(light_zenith), 0,  np.cos(light_zenith)]
    ])

    # Rotation matrix around Z (light_azimuth)
    R_z = np.array([
        [ np.cos(light_azimuth), -np.sin(light_azimuth), 0],
        [ np.sin(light_azimuth),  np.cos(light_azimuth), 0],
        [ 0,                      0,                     1]
    ])

    # Apply R_y: (3,3) × (3,N,M) → (3,N,M)
    offset = np.einsum('ij,jkl->ikl', R_y, offset)

    # Apply R_z: (3,3) × (3,N,M) → (3,N,M)
    offset = np.einsum('ij,jkl->ikl', R_z, offset)

    return offset

def construct_offset(light_zenith, light_azimuth, theta, radius):
    offset = np.array([
        np.zeros_like(radius),
        radius*np.cos(theta),
        radius*np.sin(theta),
        
    ])

    if type(theta)==np.ndarray:
        return alt_construct_offset(light_zenith, light_azimuth, theta, radius)
    
    # rotate about the Y axis by light_zenith 
    offset = np.matmul(
        np.array([
            [np.cos(light_zenith), 0, -np.sin(light_zenith)],
            [0, 1 , 0], 
            [np.sin(light_zenith), 0, np.cos(light_zenith)]
        ]),
        offset
    )

    # rotate about the Z axis by light_azimuth
    offset = np.matmul(
        np.array([
            [np.cos(light_azimuth), -np.sin(light_azimuth), 0],
            [np.sin(light_azimuth),  np.cos(light_azimuth), 0],
            [0,0,1]
        ]),
        offset
    )
    return offset


def get_differential(pmt_resonse_spline, radius, theta, light_zenith, light_azimuth, perfect=False):
    """
        This is the function called by quad. 
            First, this is a polar coordinate intetral, so we use the jacobian
            Then we need to multiply by the 
    """

    numpy_mode = type(radius)==np.ndarray


    nx = np.cos(light_azimuth)*np.cos(light_zenith)
    ny = np.sin(light_azimuth)*np.cos(light_zenith)
    nz = np.sin(light_zenith)
    if numpy_mode:
        nx = nx * np.ones_like(radius)
        ny = ny * np.ones_like(radius)
        nz = nz * np.ones_like(radius)

    # step far away in the opposite direction to get source point(s)
    source_points = np.array([
        -1*nx*0.5,
        -1*ny*0.5,
        -1*nz*0.5
    ])   

    offset = construct_offset(
        light_zenith, light_azimuth, theta, radius
    )

    source_points= source_points + offset 
    
    pmt_zen, pmt_azi = get_intersection(
        light_zenith, light_azimuth, source_points
    )

    xval, yval, zval = convert_to_cartesian(pmt_zen, pmt_azi)

    
    area_term = area_scale( pmt_zen, pmt_azi, light_zenith, light_azimuth, perfect)

    # using the intersection, we can determine how we rescale the value according to differential areas
    # we also throw in the radius as part of the jacobian for polar coords
    #  pmt_response  *area_scale( pi/2-pmt_zen, pmt_azi, light_zenith, light_azimuth)
    if numpy_mode:
        pmt_zen[pmt_zen<0] = 0
        area_term[np.isnan(area_term)] = 0
    else:
        if pmt_zen<0:
            return 0
        if np.isnan(area_term):
            return 0

    if perfect:
        return radius*area_term
    else:
        pmt_response = pmt_resonse_spline(xval, yval)   
        if numpy_mode:
            pmt_response[np.isnan(pmt_response)] = 0
        else:
            if np.isnan(pmt_response):
                return 0
            
        return (radius*pmt_response*area_term) #, [xval, yval, zval], source_points


def build_response_spline(filename):
    """
    takes the path to a... json file. 

    then it'll load that data in, convert the points to (zenith/azimuth)
    """
    
    _obj = open(filename, 'r')
    data = json.load(_obj)
    _obj.close()

    _xs = np.array(data["pmt0"]["xs"])
    _ys = np.array(data["pmt0"]["ys"])

    _xs = 0.5*(_xs[1:] + _xs[:-1])
    _ys = 0.5*(_ys[1:] + _ys[:-1])
    xs, ys = np.meshgrid(_xs, _ys )
    
    key = "avg_charge" if CHARGE else "det_eff"

    det_eff = np.array(data["pmt0"][key]).T # /(np.array(data["monitor"][key]).T)
    det_eff[np.isnan(det_eff)]=None
    det_eff[np.isinf(det_eff)]=None

        # fit the circle! 
    def metric(params):
        fit_x = params[0]
        fit_y = params[1] 

        included = ((xs-fit_x)**2 + (ys-fit_y)**2 < pmt_a**2).astype(float)
        metric = -1*np.sum(det_eff*included)
        return metric
    
    result = minimize(metric, [0.15, 0.15], bounds=((0.1, 0.55), (0.1, 0.55)), options={
        "eps":1e-2,
    })

    print("PMT Fit Prefers {}/{}".format(result.x[0], result.x[1]))
    xs -= result.x[0]
    ys -= result.x[1]        

    zs = 1 - (xs/pmt_a)**2 - (ys/pmt_b)**2

    
    pmt_zenith  = np.arcsin(zs/pmt_c) 
    pmt_azimuth = np.arctan2(ys, xs)

    print("Build with zenith range {} - {}".format(np.nanmin(pmt_zenith), np.nanmax(pmt_zenith)))
    print("Build with azimuth range {} - {}".format(pmt_azimuth.min(), pmt_azimuth.max()))

    bad_angles = np.logical_or(np.isnan(pmt_zenith), np.isnan(pmt_azimuth))


    mask = (xs)**2 + (ys)**2 < RSQ
    mask = np.logical_and(np.logical_not(np.isnan(det_eff)), mask)
    #mask = np.logical_and(np.logical_not(bad_angles), mask)
    flatmask = mask.flatten() 

    return Irregular2DInterpolator(xs.flatten()[flatmask], ys.flatten()[flatmask], det_eff.flatten()[flatmask])


def get_efficiency(pmt_response_spline, light_zenith, light_azimuth):
    """ 
        Light parameters are in the global coordinate system (zenith, azi)
            and these refer to a direction that the light is propagating 
            This is assumed to be a PLANE wave 
    
        The PMT response spline should be a RectBivariate  Spline (or w/e)
            and should be in terms of the PMT coordinate system 

        We then integrate over PMT coordinates to determine the 
    """

    rmin = 0
    rmax = 0.3 
    theta_min = 0
    theta_max = 2*pi 

    def integrand(radius, angle):
        return get_differential(pmt_response_spline, radius, angle, light_zenith, light_azimuth)
        
    def area_integrand(radius, angle):
        return get_differential(pmt_response_spline, radius, angle, light_zenith, light_azimuth, perfect=True)
        


    # now, integrate over the 
    
    result = dblquad(integrand,rmin, rmax, theta_min, theta_max)
    integrated_efficiency = result[0]
    normalization = dblquad(area_integrand,rmin, rmax, theta_min, theta_max)[0]

    return integrated_efficiency/normalization

def direct_sum(pmt_response_spline, light_zenith, light_azimuth):

    N_GRID = 150
    rad_edge = np.linspace(0, 0.3, N_GRID)
    rad_values = 0.5*(rad_edge[1:] + rad_edge[:-1])
    rad_widths = rad_edge[1] - rad_edge[0]

    theta_edge = np.linspace(0, 2*pi, N_GRID+1)
    theta_values = 0.5*(rad_edge[1:] + rad_edge[:-1])
    theta_widths = theta_edge[1]-theta_edge[0]

    rad_mesh, theta_mesh = np.meshgrid(rad_values, theta_values)

    test= get_differential(pmt_response_spline, rad_mesh, theta_mesh, light_zenith, light_azimuth )
    ratio = get_differential(pmt_response_spline, rad_mesh, theta_mesh, light_zenith, light_azimuth , True)

    return np.sum(rad_widths*test*theta_widths)/np.sum(rad_widths*ratio*theta_widths) 

if __name__=="__main__":

    datafiles = [
        [5745, "0mG"],
        [5769, "-80mG in z"],
        [5751, "-100mG in z"],
        [5752, "100mG in z"],
        [5774, "-100mG in z"],
        [5773, "250mG in z"],
        [5766, "80mG in x"],
        [5749, "250mG in x"],
        [5750, "40mG in y"],
        [5771, "100mG in y"],
        [5776, "-100mG in y"],
    #    [5755, "80mG in y"],
        [5748, "250mG in y"],
        [5777, "100mG in (-Y+X)"],
        [5780, "100mG in (+Y+X)"]
    ]
    for datafile in datafiles:
        #run_number = 5745 # 0mG 
        #run_number = 5843
        run_number = datafile[0]
        print("Working on {}".format(run_number))

        root_folder = os.path.join(os.path.dirname(__file__), "..","results")
        template = "charge_results_{}.json"

        from tqdm import tqdm

        test_file = os.path.join(
            root_folder,
            template.format(run_number)
        )
        spline = build_response_spline(test_file)

        N_GRID = 50
        zeniths = -1*np.arccos(np.linspace(0, 1, N_GRID))
        azimuths = np.linspace(-pi, pi, N_GRID+1)

        results = np.zeros((N_GRID, N_GRID+1))
        for i,zen in tqdm(enumerate(zeniths)):
            for j,azi in enumerate(azimuths):
                results[i][j] = direct_sum(spline, zen, azi)

        plt.figure(figsize=(4,3))
        plt.pcolormesh(np.cos(zeniths), azimuths, results.T, cmap="inferno", vmin=0, vmax=0.5)
        plt.colorbar()
        plt.xlabel("-cos(Zenith)", size=14)
        plt.ylabel("Azimuth", size=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(os.path.dirname(__file__), "plots","angular_response_{}.png".format(run_number)),
            dpi=400
        )
        plt.close()

        data_raw = {
            "zeniths":zeniths.tolist(),
            "azimuths":azimuths.tolist(),
            "efficiency":results.tolist(),
            "ptf_run_number":run_number,
            "ptf_run_file":test_file,
        }
        import json 
        _obj = open(os.path.join(os.path.dirname(__file__),"output_files","run_{}_angular_efficiency.json".format(run_number)),'wt')
        json.dump(data_raw, _obj, indent=4)
        _obj.close()
