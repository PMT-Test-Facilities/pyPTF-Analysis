import h5py as h5 
import os 
import numpy as np
from pyPTF.constants import PTF_SCALE, PTF_TS, PTF_CAEN_V1730_SAMPLE_RATE
import matplotlib.pyplot as plt 

SAMPLESIZE = 70
PTF_TS = np.linspace(0, float(SAMPLESIZE)*1000/PTF_CAEN_V1730_SAMPLE_RATE, SAMPLESIZE)


data = h5.File(os.path.join(
    os.path.dirname(__file__), "..","..","data","convert_V1730_wave1.h5"
), 'r')

#meta = h5.File(os.path.join(
#    os.path.dirname(__file__), "..","..","data","out_run05658.hdf5"
#), 'r')

#xcord = meta["gantry0_x"][:]
#ycord = meta["gantry0_y"][:]

for j, scankey in enumerate(data.keys()):


    i = int(scankey.split("_")[1])
    if i%1000!=0:
        continue
    print("scan {}".format(scankey))
    these_waveforms = np.array(data[scankey], dtype=float)

    cut = np.array(range(len(these_waveforms)))%10 != 0
    
    these_waveforms = these_waveforms[cut]
    for iw, wave in enumerate(these_waveforms):
        plt.clf()
        alpha = 0.1*abs(8140 - np.min(wave))/240
        alpha = 1.0 # max([min([0, alpha]),0.5])
        crossing = range(140)[np.argwhere(np.diff(np.sign(np.mean(wave[:20]) - wave -30)))[0][0]]


        plt.plot(range(SAMPLESIZE), np.mean(wave[:20])-wave, alpha=alpha, color='k')
        plt.vlines(crossing, 0, 8000, color='red')
        #plt.plot(49 + np.array(range(len(wave))), wave, alpha=alpha, color='k')
        #plt.ylim([0, 1000])
        plt.xlabel("Time [ns]", size=14)
        plt.ylabel("ADC Counts" ,size=14)
#        plt.title("{:.3f}-{:.3f}".format(xcord[i], ycord[i]))
        plt.show(block=True)   
        plt.pause(.005)
#plt.savefig("./waveforms/wave_{}.png".format(i))

    
    print("./waveforms/wave_{}.png".format(i))
