
import h5py as h5

import numpy as np
import uproot
import os 
from tqdm import tqdm

import json 
from pyPTF.enums import PMT
from pyPTF.process_raw import process_into_fitseries

KEEP_WAVEFORMS =  False

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

def parse_waveform(full_form, name):
    """
        Reads the waveforms for a scanpoint and writes them into an hdf5 file
    """
    test = h5.File(os.path.join(DATA_FOLDER, "convert_{}.h5".format(name)), 'w')
    #full_form = wf_object.array(library='np')
    for i_scanpoint in range(len(full_form))[:]:
                        
        test.create_dataset("scanpoint_{}_waveforms".format(i_scanpoint), data=np.array(full_form[i_scanpoint], dtype=int), dtype='i4')
        
    test.close()
    
    group = None
    entry = None
    test = None
    return 


def root_to_hdf5(root_file_path, hdf5_file_path):
    """
        Uses uproot to rid ourselves of root.
    """
    # Open the ROOT file using uproot
    out_file = h5.File(hdf5_file_path, 'w')
    with uproot.open(root_file_path) as root_file:
        # Create an HDF5 file
        # Loop over the keys in the ROOT file
        tree = root_file["scan_tree;1"]

        for key in root_file.keys():
                if key=="scan_tree;2":
                    continue
                # For simplicity, this example assumes that each key in the ROOT file
                # corresponds to a tree with branches that can be converted to numpy arrays.
                # Complex structures may need more sophisticated handling.
                tree = root_file[key]

                skip_keys = [
                    "window_width",
                    "start_time",
                    "phidgACC_A".lower(),
                    "opticalBox0_A".lower(),
                    "opticalBox0_B".lower(),
                    "opticalBox0_tilt".lower(),
                    "opticalBox0_temp".lower(),
                    "opticalBox0_hum".lower(),
                    "opticalBox".lower(),
                    "opticalBox1".lower(),
                    "thermocouple",
                    "phid",
                    "evt_timestamp"
                ]

                mean_key =[
                    "phidg0_tilt",
                    "phidg0_Btot",
                    "I_coil1",
                    "I_coil2",
                    "I_coil3",
                    "I_coil4",
                    "I_coil5",
                    "I_coil6",
                    "U_coil1",
                    "U_coil2",
                    "U_coil3",
                    "U_coil4",
                    "U_coil5",
                    "U_coil6",
                    "U_receiver0",
                    "U_monitor0",
                    "I_receiver0",
                    "I_monitor0",
                    "phidg0_temperature",
                    "phidg0_humidity"
                ]

                # Loop over the branches in the tree
                for branch_name in tree.keys():
                    if "V1730" in str(branch_name):
                        print("branch", branch_name)
                        #group = hdf5_file.create_group(branch_name)
                        channel = branch_name[-1]
                        #if channel!="2":
                        #    continue
                        test = tree[branch_name].array(library='np')
                        
                        parse_waveform(test, branch_name)
                        del test
                        continue
                    elif branch_name in mean_key:
                        branch_data = np.array([np.mean(entry) for entry in tree[branch_name].array(library="np")])
                        out_file.create_dataset(branch_name, data=branch_data)
                    elif any([test in branch_name.lower() for test in skip_keys]):
                        continue
                    
                    else:
                        # Convert branch to numpy array
                        branch_data = tree[branch_name].array(library="np")
                        # Create a dataset in HDF5 file from the numpy array
                        out_file.create_dataset(branch_name, data=branch_data)
    
    out_file.close()


def analyze_waveforms(run_number):
    """
        This runs a simple parser on the monitor and the Hammamatsu PMT data 

        It then converts the data into an analysis-friendly format and saves it to disk 
    """
    main_file = os.path.join(
        DATA_FOLDER,
        "out_run0{}.hdf5".format(run_number)
    )


    __raw_meta_data = h5.File(main_file, 'r')
    data = {
        "data_folder":DATA_FOLDER
    }
    data["run_nu"] = run_number
    for key in __raw_meta_data.keys():
        data[key] = np.array(__raw_meta_data[key][:])
    __raw_meta_data.close()

    timing_data = process_into_fitseries(
        meta_data=data, 
        which_pmt=PMT.Timing_NotAPMT
    )

    # this thing kinda does most of the magic
    main_fitseries = process_into_fitseries(
            meta_data=data,
            which_pmt= PMT.Hamamatsu_R3600_PMT,
        )


    secondary_fitseries = process_into_fitseries(
        meta_data=data,
        which_pmt=PMT.PTF_Monitor_PMT,
    )
    


    out_file = os.path.join(
        DATA_FOLDER,
        "pulse_series_fit_run{}.hdf5".format(run_number)
        )
    
    out_data = {
        "pmt0":main_fitseries,
        "monitor":secondary_fitseries,
        "timing_data":timing_data,
        "data_folder" : data["data_folder"],
        "meta":data,
        "run_no":run_number
    }
    print("saving hdf5 file")
    _obj = h5.File(out_file, 'w')
    for pmt_key in ["pmt0", "monitor", "timing_data"]:
        for key in out_data[pmt_key].keys():
            _obj.create_dataset("{}/{}".format(pmt_key, key), data=out_data[pmt_key][key])
    for key in data.keys():
        _obj.create_dataset("meta/{}".format(key), data=data[key])
    _obj.create_dataset("run_no", data=out_data["run_no"])
    _obj.create_dataset("data_folder", data=out_data["data_folder"])
    #_obj.create_dataset("meta", data=out_data["meta"])
    _obj.close()

    print("Cleaning up")
    if not KEEP_WAVEFORMS:
        os.remove(main_file)
        os.remove(os.path.join(DATA_FOLDER, "convert_V1730_wave0.h5"))
        os.remove(os.path.join(DATA_FOLDER, "convert_V1730_wave1.h5"))
        os.remove(os.path.join(DATA_FOLDER, "convert_V1730_wave2.h5"))



# Example usage
if __name__=="__main__":
    import sys
    if len(sys.argv)!=3:
        print("usage: ")
        print("extract_pulses.py  data_file.root run_number")
        sys.exit() 

    outfile = os.path.splitext(sys.argv[1])[0] + ".hdf5"
    try:
        run_number = int(sys.argv[2])
    except ValueError as e:
        print("could not read run number {}".format(sys.argv[2]))
    
    print("... extracting root File")
    root_to_hdf5(sys.argv[1], outfile)
    analyze_waveforms(run_number)
