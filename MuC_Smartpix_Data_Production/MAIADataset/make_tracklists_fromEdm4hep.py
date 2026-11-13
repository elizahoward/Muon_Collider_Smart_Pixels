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
# from plothelper import *
from datetime import datetime
import pandas as pd
import edm4hepTracksUtils


def collectionDfFromTree(f: uproot.reading.ReadOnlyDirectory,collectionBranch = "VertexBarrelCollection",collectionBranchIndex="",mcBranch="MCParticles",getMcDf=True):
    assert isinstance(f,uproot.reading.ReadOnlyDirectory)
    assert ("events;1" in f.keys()) or ("events;2" in f.keys()) or ("events;3" in f.keys())
    if collectionBranchIndex=="":
        collectionBranchIndex = "_"+collectionBranch+"_particle"
    assert collectionBranch in f["events"].keys()
    assert collectionBranchIndex in f["events"].keys()
    assert mcBranch in f["events"].keys()
    df1 = f["events"][collectionBranch].arrays(library="pd")
    df2 = f["events"][collectionBranchIndex].arrays(library="pd")
    # print(df2.keys())
    df2 = df2.rename(lambda a: a.replace("_particle.","_particle.particle_"),axis="columns")
    # print(df2.keys())
    assert len(df1)==len(df2)
    eventHitDf = df1.merge(df2,how='outer',left_index=True,right_index=True)
    # print(eventHitDf.keys())
    for ievent in eventHitDf.index:#works for empty rows (e.g. neutrino gun, photon gun, etc.)
        row=eventHitDf.loc[ievent]
        numHits = [len(row[key]) for key in row.keys()]
        assert len(set(numHits)) == 1
    if getMcDf:
        eventMcDf = f["events"][mcBranch].arrays(library="pd")
        assert len(eventHitDf)==len(eventMcDf)
    else:
        eventMcDf=pd.DataFrame()
    return eventHitDf,eventMcDf

def collectionDfToFlatHits(eventsDf:pd.DataFrame):
    hitRows = []
    for ievent in eventsDf.index:#works for empty rows (e.g. neutrino gun, photon gun, etc.)
        row=eventsDf.loc[ievent]
        firstKey = row.keys()[0]
        for ihit in range(len(row[firstKey])):
            hitRow = {".".join(key.split(".")[1:]):row[key][ihit] for key in row.keys()}
            hitRow["ievent"] =ievent;
            hitRows.append(hitRow)
    hitDf = pd.DataFrame(hitRows)
    return hitRows,hitDf

# #rewritten by claude with explode
# def collectionDfToFlatHits(eventsDf):
#     df = eventsDf.copy()
#     df.columns = [".".join(k.split(".")[1:]) for k in df.columns]   # strip collection prefix
#     cols = list(df.columns)
#     df = df[df[cols[0]].map(len) > 0]            # drop events with no hits
#     df["ievent"] = df.index
#     flat = df.explode(cols).reset_index(drop=True)
#     return None,flat.infer_objects()
def addMcToHitsPersonal(hitDf,eventMcDf):
    rows=[]
    for ihit,hitRow in hitDf.iterrows():
        ievent=int(hitRow["ievent"])
        # mcParticleIndex=int(hitRow["index"])
        mcParticleIndex=int(hitRow["particle_index"])
        mcParticleCols = [eventMcDf.iloc[ievent][key][mcParticleIndex] for key in eventMcDf.keys()]
        rows.append(mcParticleCols)
    out = hitDf.copy()
    out[eventMcDf.keys()] = pd.DataFrame(rows, index=out.index, columns=eventMcDf.keys())
    return out
#rewritten by cluade has some more checks
def addMcToHits(hitDf:pd.DataFrame, eventMcDf:pd.DataFrame):
    mcKeys = list(eventMcDf.columns)
    newCols = mcKeys
    rows = []
    # for e, i in zip(hitDf["ievent"], hitDf["index"]):
    for e, i in zip(hitDf["ievent"], hitDf["particle_index"]):
        e=int(e)
        i=int(i)
        n = len(eventMcDf[mcKeys[0]].iloc[e])          # particles in this event
        if 0 <= i < n:
            rows.append([eventMcDf[k].iloc[e][i] for k in mcKeys])
        else:
            rows.append([None] * len(mcKeys))          # invalid or out-of-range reference
    out = hitDf.copy()
    out[newCols] = pd.DataFrame(rows, index=out.index, columns=newCols)
    out["mc_valid"] = out[mcKeys[0]].notna()
    return out

def hitDftoTracklistDf(hitDf,flp=0):
    raise ValueError("USE THE VERSION IN THE UTILS CLASS")

def processEdmFileToTrack(file_path):
    with uproot.open(file_path) as f:
        # print( f.keys())
        vertexEventsDf,eventMcDf = collectionDfFromTree(f)
        # print(vertexEventsDf)
        _,hitDf = collectionDfToFlatHits(vertexEventsDf)
        edm4hepTracksUtils.addDecodedCellId(hitDf)
        if hitDf.empty:
            print("this file empty of hits")
            return None,None
        if filterSecondaries:
            # hitDf.filter(cond=hitDf["index"]==0)
            hitDf.filter(cond=hitDf["particle_index"]==0)
        if addMCParticle:
            hitDf = addMcToHits(hitDf,eventMcDf)
        tracklistDf = edm4hepTracksUtils.hitDftoTracklistDf(hitDf)

        #####saving
        savePath = file_path+".simhits.csv"
        hitDf.to_csv(savePath)
        print(f"saved {len(hitDf)} hits to {savePath}")
        
        savePath = file_path+".tracks.csv"
        tracklistDf.to_csv(savePath,sep=" ",index=False,float_format=f"%.{float_precision}f")
        print(f"saved {len(tracklistDf)} hits to {savePath}")
        return hitDf,tracklistDf


file_list = ["./edmTestFiles/sim_output.edm4hep.root","./edmTestFiles/sim_output2.edm4hep.root","./edmTestFiles/sim_output3.edm4hep.root","./edmTestFiles/digi_real.edm4hep.root","./edmTestFiles/sim_output_mu2.edm4hep.root","./edmTestFiles/digi_output_mu2.edm4hep.root","./edmTestFiles/sim_output_mu1.edm4hep.root","./edmTestFiles/sim_output_nu1.edm4hep.root","./edmTestFiles/sim_output_H1.edm4hep.root",]#"./edmTestFiles/gen_output.edm4hep.root",]
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
    processEdmFileToTrack(file_path)


        #  for 
    # reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    # reader.open(file_path)

    # for ievt,event in enumerate(reader):
        
    #     print("Processing event %i."%ievt)