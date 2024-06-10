import h5py as h5 
import os 
import numpy as np
from pyPTF.constants import PTF_SCALE, PTF_TS, PTF_CAEN_V1730_SAMPLE_RATE
import matplotlib.pyplot as plt 

SAMPLESIZE = 140
PTF_TS = np.linspace(0, float(SAMPLESIZE)*1000/PTF_CAEN_V1730_SAMPLE_RATE, SAMPLESIZE)


data = h5.File(os.path.join(
    os.path.dirname(__file__), "..","..","data","convert_V1730_wave1.h5"
), 'r')

meta = h5.File(os.path.join(
    os.path.dirname(__file__), "..","..","data","out_run05653.hdf5"
), 'r')

xcord = meta["gantry0_x"][:]
ycord = meta["gantry0_y"][:]

for j, scankey in enumerate(data.keys()):


    i = int(scankey.split("_")[1])
    if i%50!=0:
        continue
    these_waveforms = np.array(data[scankey], dtype=float)

    cut = np.array(range(len(these_waveforms)))%10 != 0
    
    these_waveforms = these_waveforms[cut]
    print(len(these_waveforms))
    for iw, wave in enumerate(these_waveforms):
        plt.clf()
        alpha = 0.1*abs(8140 - np.min(wave))/240
        alpha = max([min([0, alpha]),0.5])

        plt.plot(range(140), wave, alpha=alpha, color='k')
        #plt.plot(49 + np.array(range(len(wave))), wave, alpha=alpha, color='k')
        #plt.ylim([-5, 150])
        plt.xlabel("Time [ns]", size=14)
        plt.ylabel("ADC Counts" ,size=14)
        plt.title("{:.3f}-{:.3f}".format(xcord[i], ycord[i]))
        plt.show(block=False)   
        plt.pause(.005)
#plt.savefig("./waveforms/wave_{}.png".format(i))

    
    print("./waveforms/wave_{}.png".format(i))