
import h5py as h5

import numpy as np
import uproot
import os 
from tqdm import tqdm

import json 
from pyPTF.enums import PMT
from pyPTF.process_raw import process_into_fitseries


DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

def parse_waveform(full_form, name):
    """
        Reads the waveforms for a scanpoint and writes them into an hdf5 file
    """
    test = h5.File(os.path.join(DATA_FOLDER, "convert_{}.h5".format(name)), 'w')
    #full_form = wf_object.array(library='np')
    for i_scanpoint in range(len(full_form))[:]:
        max_count = 3000
        if len(full_form[i_scanpoint])>max_count:
            continue 

                        
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

                # Loop over the branches in the tree
                for branch_name in tree.keys():
                    if "V1730" in str(branch_name):
                        print("branch", branch_name)
                        #group = hdf5_file.create_group(branch_name)
                        channel = branch_name[-1]
                        if channel!="0" and channel!="2":
                            continue
                        test = tree[branch_name].array(library='np')
                        
                        parse_waveform(test, branch_name)
                        del test
                        continue
                     
                    elif any([test in branch_name.lower() for test in skip_keys]):
                        continue
                    else:
                        # Convert branch to numpy array
                        print("branch", branch_name)
                        branch_data = tree[branch_name].array(library="np")
                        # Create a dataset in HDF5 file from the numpy array
                        out_file.create_dataset(branch_name, data=branch_data)
    
    out_file.close()


def reconstruct_json_as_dict(data):
    """
        takes a pulse fit series and constructs an analysis-friendly dictionary for the data
    """

    keys = [
        "x","y","z","tilt","rot",
        "amplitudes","sigmas","means","peds"
    ]

    outdata = {
        "pmt0":{key:[] for key in keys},
        "monitor":{key:[] for key in keys},
        "run_no":data["run_no"]
    }
    #pmt = "pmt0"
    for pmt in ["pmt0","monitor"]:
       
        for scanpoint in data[pmt].keys():
            
            for i_wave in range(len(data[pmt][scanpoint]["amplitudes"])):
                outdata[pmt]["x"].append(data[pmt][scanpoint]["x"])
                outdata[pmt]["y"].append(data[pmt][scanpoint]["y"])

            outdata[pmt]["amplitudes"]+= data[pmt][scanpoint]["amplitudes"]
            outdata[pmt]["sigmas"] += data[pmt][scanpoint]["sigmas"]

    return outdata

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

    # this thing kinda does most of the magic
    main_fitseries = process_into_fitseries(
            meta_data=data,
            which_pmt= PMT.Hamamatsu_R3600_PMT,
        )


    secondary_fitseries = process_into_fitseries(
        meta_data=data,
        which_pmt=PMT.PTF_Monitor_PMT,
    )

    os.remove(os.path.join(DATA_FOLDER, "convert_V1730_wave0.h5"))
    os.remove(os.path.join(DATA_FOLDER, "convert_V1730_wave2.h5"))

    out_file = os.path.join(
        DATA_FOLDER,
        "pulse_series_fit_run{}.json".format(run_number)
        )
    
    out_data = {
        "pmt0":main_fitseries,
        "monitor":secondary_fitseries,
        "data_folder" : data["data_folder"],
        "run_no":run_number
    }


    _obj = open(out_file, 'wt')
    json.dump(out_data, _obj, indent=4)
    _obj.close()


# Example usage
if __name__=="__main__":
    import sys
    if len(sys.argv)!=3:
        print("usage: ")
        print("extract_pulses.py  data_file.root run_number")

    outfile = os.path.splitext(sys.argv[1])[0] + ".hdf5"
    try:
        run_number = int(sys.argv[2])
    except ValueError():
        print("could not read run number {}".format(sys.argv[2]))
    
    root_to_hdf5(sys.argv[1], outfile)

    analyze_waveforms(run_number)
