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

file_list = ["./edmTestFiles/sim_output.edm4hep.root","./edmTestFiles/sim_output2.edm4hep.root","./edmTestFiles/sim_output3.edm4hep.root","./edmTestFiles/digi_real.edm4hep.root",]#"./edmTestFiles/gen_output.edm4hep.root",]
file_list = ["./edmTestFiles/digi_real.edm4hep.root",]
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

    for ievt,event in enumerate(reader.get("events")):
        
        print("Processing event %i."%ievt)
        
        # Print all the collection names in the event
        # collection_names = event.getCollectionNames()
        collection_names = event.getAvailableCollections()
        print(collection_names)

        # Get vertex barrel hits
        truthParticles = event.get("MCParticles")
        vtxBarrelSimHits = event.get("VertexBarrelCollection")
        print("vtxBarrelSimHits")
        print(len(vtxBarrelSimHits))
        print(vtxBarrelSimHits[0]) #SimTrackerHit <cppyy.gbl.edm4hep.SimTrackerHit object at 0x556a308b48f0>
        print(dir(vtxBarrelSimHits[0]))
        vtxClusterToSimHit = event.get("VXDBarrelRealHitsRelations")
        print("vtxClusterToSimHit")
        print(len(vtxClusterToSimHit))
        print(vtxClusterToSimHit[0]) #<cppyy.gbl.podio.LinkT<edm4hep::TrackerHit,edm4hep::SimTrackerHit,false> object at 0x556a3098b950>
        print(dir(vtxClusterToSimHit[0]))
        vtxClusters = event.get("VXDBarrelRealHits")
        print("vtxClusters")
        print(len(vtxClusters))
        print(vtxClusters[0])#TrackerHitPlane  <cppyy.gbl.edm4hep.TrackerHitPlane object at 0x55f1af4d9e00> #note TrackerHit is an interface, TrackerHitPlane is one type of it
        print(dir(vtxClusters[0]))
        print(vtxClusters[0].id().index)
        print(vtxClusters[0].id().collectionID)
        vtxPixelsToClusters = event.get("VXDBarrelRealRawHitRelations")
        print("vtxPixelsToClusters")
        print(len(vtxPixelsToClusters))
        print(vtxPixelsToClusters[0]) #<cppyy.gbl.podio.LinkT<edm4hep::TrackerHit,edm4hep::SimTrackerHit,false> object at 0x55f1afa82770>
        print(dir(vtxPixelsToClusters[0]))
        print(vtxPixelsToClusters[0].getFrom())
        print(vtxPixelsToClusters[0].getFrom().id().index)
        print(vtxPixelsToClusters[0].getFrom().id().collectionID)
        print(vtxPixelsToClusters[0].getTo())
        print(vtxPixelsToClusters[0].getTo().id().index)
        print(vtxPixelsToClusters[0].getTo().id().collectionID)
        vtxPixels = event.get("VXDBarrelRealPixels")
        print("vtxPixels")
        print(len(vtxPixels))
        print(vtxPixels[0]) #SimTrackerHit <cppyy.gbl.edm4hep.SimTrackerHit object at 0x55f1afb70190>
        print(dir(vtxPixels[0]))
        print(vtxPixels[0].id().index)
        print(vtxPixels[0].id().collectionID)
        
        break

        for hit in vtxBarrelSimHits: 

            position = hit.getPosition()
            print(position)
            print(hit.getParticle())
            print(hit)

        
        