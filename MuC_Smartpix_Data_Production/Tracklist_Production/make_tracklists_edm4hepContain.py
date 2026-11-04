"""
Author: Daniel Abadjiev
Date: October 30, 2026
Description: Attempts to replicate make_tracklists but on edm4hep instead of slcio files that runs in the container

Run with 
```
apptainer run /cvmfs/unpacked.cern.ch/ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24\:v3.1-amd64
source /opt/setup_mucoll.sh
python make_tracklists_edm4hepContain.py
```
"""

import numpy as np
# import pyLCIO
import ROOT
# import uproot
from math import *
import argparse
import os
import random
from plothelper import *
from datetime import datetime
# import pandas as pd
import edm4hep
import podio
import pandas as pd
import edm4hepToDict
import edm4hepTracksUtils

file_list = ["./edmTestFiles/sim_output.edm4hep.root","./edmTestFiles/sim_output2.edm4hep.root","./edmTestFiles/sim_output3.edm4hep.root","./edmTestFiles/digi_real.edm4hep.root",]#"./edmTestFiles/gen_output.edm4hep.root",]
# file_list = ["./edmTestFiles/digi_real.edm4hep.root",]
track_count=0
break_loop=False
count = 0
fileCountStart = 0
fileCountLimit = 1000
filterSecondaries=True
addMCParticle=True
float_precision=5
for file_path in file_list:
    if break_loop:
            break
    count +=1
    if count<fileCountStart:
        continue
    print("count", count, " ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if count>fileCountLimit:
        print("count>fileCountLimit, so breaking")
        break
    # Get the full path to the file
    if not file_path.endswith("edm4hep.root"):
        continue
    print("Processing file: ", file_path)


        #  for 
    # reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    # reader.open(file_path)
    reader = podio.reading.get_reader(file_path)
    eventDfDicts = []
    for ievt,event in enumerate(reader.get("events")):
        
        print("Processing event %i."%ievt)
        
        # Print all the collection names in the event
        # collection_names = event.getCollectionNames()
        eventDfDict = edm4hepToDict.eventToDfs(event,ievt)
        eventDfDict["tracklistDf"] = edm4hepTracksUtils.hitDftoTracklistDf(eventDfDict["vtxBarrelSimHitsDf"])
        eventDfDicts.append(eventDfDict)

    concatEventDfs = {key: pd.concat([eventDf[key] for eventDf in eventDfDicts], ignore_index=True) for key in eventDfDicts[0]}

    # print(concatEventDfs)
    for key in concatEventDfs.keys():
        savePath = file_path+f".{key}.csv"
        if key == "tracklistDf":
            concatEventDfs[key].to_csv(savePath,sep=" ",index=False,float_format=f"%.{float_precision}f");
        else:
            concatEventDfs[key].to_csv(savePath);
        print(f"saved {len(concatEventDfs[key])} hits of {key} to {savePath}")
        

        
        