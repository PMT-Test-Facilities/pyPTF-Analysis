# Setting the environment

First, you'll need to append the folder containing this one to your python path. 
You can run this 
```
    export PYTHONPATH=$PYTHONPATH:$PWD
```
from this folder, or add a line like 
```
    export PYTHONPATH=$PYTHONPATH:/path/to/here
```
to your `.bash_profile` or `.bashrc` file.

# Running the code 

This assumes you've already run the `midas2root_ptf` code and processed the midas files. 

You can probably run this on the ptf computer, but you can alternatively run it locally. 

For the case of the latter, download one of the root files.  
Make a `data` folder next to  the `pyPTF` folder, `cd` into it, and download the file you want to run the analysis on.
Just run 
```
scp user@host.triumf.ca:~online/data/out_run05653.root .
```
after changing the `user` and `host` for the PTF computer. 


Then you can use `extract_pulses.py` in `pyPTF` on one of the run files to process them into hdf5 files.

For run 5653, you'd run 
```
    python extract_pulses.py ../data/out_run05653.root 5653
```

That makes a `pulse_series_fit` file with the information for pulses found in each of the waveforms. 
You can run a charge or timing analysis using the files in the `analysis` folder. 

```
python charge.py ../../data/pulse_series_fit_run5653.hdf5
python timing.py ../../data/pulse_series_fit_run5653.hdf5
``` 



