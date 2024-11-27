
import numpy as np 
import h5py as h5 
import matplotlib.pyplot as plt 
import os 

from pyPTF.utils import get_color

def main(filenamme):
    data = h5.File(filenamme,'r')

    plot_dir = os.path.join(os.path.dirname(__file__), "plots")

    fig, axes = plt.subplots(1,1)
    #twax = axes.twinx()
    for _i in range(6):
        i = _i + 1

        these_voltages = np.array(data["meta"]["U_coil{}".format(i)][:])
        mean_v = np.mean(these_voltages)
        these_voltages = 100*(these_voltages - mean_v)/mean_v

        these_currents = np.array(data["meta"]["I_coil{}".format(i)][:])
        mean_c = np.mean(these_currents)
        these_currents = 100*(these_currents -mean_c)/mean_c 
        these_currents = these_currents[1:]

        #axes.plot(range(len(these_voltages)), these_voltages, ls='-', color=get_color(i, 6))
        axes.plot(range(len(these_currents)), these_currents, ls='--', color=get_color(i, 6))
        axes.plot([], [], ls='-', color=get_color(i, 6), label="Coil {}".format(i))
    #axes.plot([], [], color='k',ls='-', label="Voltages")
    axes.plot([], [], color='k',ls='--', label="Currents")

    #axes.set_ylabel("Amp change [%] + 0.1", size=14)
    #axes.set_ylim([0.0, 0.2])
    axes.set_ylim([-2,2])
    axes.set_ylabel("Current Change [%]", size=14)
    axes.set_xlabel("Scan Point",size=14)
    axes.legend()
    plt.show()

    plt.clf()

    timestamps = range(len(np.array(data["meta"]["timestamp"    ][1:])))
    mon_current = np.array(data["meta"]["U_monitor0"][1:])
    mon_current = 100*(mon_current - np.mean(mon_current))/mon_current
    pmt_current = np.array(data["meta"]["U_receiver0"][1:])
    pmt_current = 100*(pmt_current - np.mean(pmt_current))/pmt_current

    plt.plot(timestamps, mon_current, label="Monitor PMT" )
    plt.plot(timestamps, pmt_current, label="20\" PMT")
    plt.ylim([-0.05, 0.05])
    plt.legend()
    plt.xlabel("Scanpoint", size=14)
    plt.ylabel("Voltage \% Change", size=14)
    plt.tight_layout()

    plt.show()
    plt.clf()

    return
    temperature = np.array(data["meta"]["phidg0_temperature"][1:])
    humidity = np.array(data["meta"]["phidg0_humidity"][1:])

    fig, ax = plt.subplots(1,1)
    ax.plot(range(len(temperature)), temperature, label="Temperature", color='red')
    ax.plot([], [], color='blue', label="Humidity")
    ax.set_ylabel("Temp [C]", size=14)
    twax = ax.twinx()
    twax.plot(range(len(humidity)), humidity, color='blue')
    twax.set_ylabel("Humidity [%]", size=14)
    ax.set_xlabel("Scanpoint", size=14)
    plt.show()



if __name__=="__main__":
    import sys
    main(sys.argv[1])