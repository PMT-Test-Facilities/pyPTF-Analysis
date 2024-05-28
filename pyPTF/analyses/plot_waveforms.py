import h5py as h5 
import os 
import numpy as np
from pyPTF.constants import PTF_SCALE, PTF_TS, PTF_CAEN_V1730_SAMPLE_RATE
import matplotlib.pyplot as plt 

SAMPLESIZE = 70
PTF_TS = np.linspace(0, float(SAMPLESIZE)*1000/PTF_CAEN_V1730_SAMPLE_RATE, SAMPLESIZE)


data = h5.File(os.path.join(
    os.path.dirname(__file__), "..","..","data","convert_V1730_wave0.h5"
), 'r')

meta = h5.File(os.path.join(
    os.path.dirname(__file__), "..","..","data","out_run05622.hdf5"
), 'r')

xcord = meta["gantry0_x"][:]
ycord = meta["gantry0_y"][:]

for j, scankey in enumerate(data.keys()):
    if j%5000!=0: # plot every hundredth
        continue

    i = int(scankey.split("_")[1])
    these_waveforms = np.array(data[scankey], dtype=float)

    cut = np.array(range(len(these_waveforms)))%10 != 0
    
    these_waveforms = these_waveforms[cut]

    for wave in these_waveforms:
        alpha = (8140 - np.min(wave))/60
        alpha = max([min([0, alpha]),0.5])


        plt.plot(PTF_TS[PTF_TS<550], wave[PTF_TS<550], alpha=alpha, color='k')
        plt.xlabel("time [ns]", size=14)
        plt.ylabel("ADC Counts" ,size=14)
        plt.title("{:.3f}-{:.3f}".format(xcord[i], ycord[i]))
        plt.show()
   
    plt.savefig("./waveforms/wave_{}.png".format(i))
    if i==109:
        plt.show()
    plt.clf()
    print("./waveforms/wave_{}.png".format(i))