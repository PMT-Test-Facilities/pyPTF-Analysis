import os 
import h5py as h5 

from charge import get_analysis_charge, CHARGE_BINNING, fit_binned_data, gaus, gaus_mu, PTF_SCALE, _ped_20
import numpy as np 

import matplotlib.pyplot as plt 



def main(filename, monitor=False, label=""):

    if not os.path.exists(filename):
        raise IOError("File not found {}".format(filename))
    
    name, ext = os.path.splitext(filename)

    print("... loading ")
    data_dict = h5.File(filename, 'r')


    if monitor:
        charge_1pe = get_analysis_charge(data_dict["monitor"], just_height=False) # [amps>30]
    else:
        charge_1pe = get_analysis_charge(data_dict["pmt0"], just_height=False) # [amps>30]
    
    width = CHARGE_BINNING[1:] - CHARGE_BINNING[:-1]

    binned_data = np.histogram(charge_1pe, CHARGE_BINNING)[0]/(width)   
        #ped, q1_res = fit_binned_data(binned_data, b_center)

    plt.stairs(binned_data , CHARGE_BINNING, fill=False, label="Monitor PMT" if monitor else "20in PMT")

    #plt.xlim([0, 640])
    #plt.plot(fine_xs, fine_ys, label="Fit")
    #plt.ylim([1e-4, 2e5])
    
    plt.yscale('log')

    plt.xlabel("Charge [ADC]", size=14)
    plt.ylabel("Arb", size=14)
    plt.legend()

    
    

if __name__=="__main__":
    import sys
    main(sys.argv[1], )
    #main(sys.argv[1], True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "charge_dist.png"))
    plt.show()