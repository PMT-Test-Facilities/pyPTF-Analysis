#!/bin/bash

# redownloads this data and reprocesses it! 

reproc=(5630 5645 5646 5647 5650 5651 5652 5653 5654)
# reproc=(5645 5646 5647)
for run in ${reproc[@]}; do
    cd ../data/
    scp midptf@midptf01.triumf.ca:/home/midptf/online/data/out_run0${run}.root .
    cd ../pyPTF
    python extract_pulses ../data/out_run0${run}.root ${run}
done

