"""
Author: Daniel Abadjiev
Date: October 26, 2026
Description: Attempts to replicate make_tracklists but on edm4hep instead of slcio files

"""

import numpy as np
# import pyLCIO
# import ROOT
import uproot
from math import *
import argparse
import os
import random
from plothelper import *
from datetime import datetime
import pandas as pd

def collectionDfFromTree(f: uproot.reading.ReadOnlyDirectory,collectionBranch = "VertexBarrelCollection",collectionBranchIndex=""):
    assert isinstance(f,uproot.reading.ReadOnlyDirectory)
    assert "events;1" in f.keys()
    if collectionBranchIndex=="":
        collectionBranchIndex = "_"+collectionBranch+"_particle"
    assert collectionBranch in f["events"].keys()
    assert collectionBranchIndex in f["events"].keys()
    df1 = f["events"][collectionBranch].arrays(library="pd")
    df2 = f["events"][collectionBranchIndex].arrays(library="pd")
    eventsDf = df1.merge(df2,how='outer',left_index=True,right_index=True)
    # print(eventsDf)
    # for idx,row in eventsDf.iterrows():
    for idx in eventsDf.index:
        row=eventsDf.loc[idx]
        numHits = [len(row[key]) for key in row.keys()]
        assert len(set(numHits)) == 1
    return eventsDf

def collectionDfToFlatHits(eventsDf):
    hitRows = []
    for idx in eventsDf.index:
        row=eventsDf.loc[idx]
        firstKey = row.keys()[0]
        for ihit in range(len(row[firstKey])):
            hitRow = {".".join(key.split(".")[1:]):row[key][ihit] for key in row.keys()}
            hitRows.append(hitRow)
    hitDf = pd.DataFrame(hitRows)
    return hitRows,hitDf


file_list = ["./edmTestFiles/sim_output.edm4hep.root","./edmTestFiles/sim_output2.edm4hep.root","./edmTestFiles/sim_output3.edm4hep.root","./edmTestFiles/digi_real.edm4hep.root",]#"./edmTestFiles/gen_output.edm4hep.root",]
track_count=0
break_loop=False
count = 0
fileCountStart = 0
fileCountLimit = 1000
filterSecondaries=True
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
    with uproot.open(file_path) as f:
        # print(f.keys())
        # print(f["events"]["VertexBarrelCollection"])
        # print(f["events"]["VertexBarrelCollection"].keys())
        # vertexEventsDF = f["events"]["VertexBarrelCollection"].arrays(library="pd")
        # print(vertexEventsDF.keys())
        vertexEventsDf = collectionDfFromTree(f)
        _,hitDf = collectionDfToFlatHits(vertexEventsDf)
        if hitDf.empty:
            print("this file empty of hits")
            continue
        if filterSecondaries:
            hitDf.filter(cond=hitDf["index"]==0)
        savePath = file_path+".simhits.csv"
        hitDf.to_csv(savePath)
        print(f"saved {len(hitDf)} hits to {savePath}")

        #  for 
    # reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    # reader.open(file_path)

    # for ievt,event in enumerate(reader):
        
    #     print("Processing event %i."%ievt)