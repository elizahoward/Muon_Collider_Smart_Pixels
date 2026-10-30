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
# from make_tracklists import getYlocalAndGamma
######################
sensorAngles = np.arange(-np.pi-np.pi/8,np.pi+2*np.pi/8,np.pi/8)

def getYlocalAndGamma(x,y):
    # Get exact angle gamma of the hit position
    gammaP=np.arctan2(y,x)

    # Get two sensor angles that are closest to the exact angle
    diff = np.abs(sensorAngles-gammaP)
    index1 = np.argmin(diff)
    gamma1=sensorAngles[index1]
    diff[index1]=3*np.pi
    index2 = np.argmin(diff)
    gamma2=sensorAngles[index2]

    # Rotate x coordinate of the point by each option for gamma
    x1=x*np.cos(-gamma1)-y*np.sin(-gamma1)
    y1=y*np.cos(-gamma1)+x*np.sin(-gamma1)
    x2=x*np.cos(-gamma2)-y*np.sin(-gamma2)
    y2=y*np.cos(-gamma2)+x*np.sin(-gamma2)

    # Determine which x is closest to expected value
    xTrue=30.16475324197002

    diff1=abs(x1-xTrue)
    diff2=abs(x2-xTrue)
    
    # If both x1 and x2 are really close to the ex
    if diff1 < 0.5 and diff2 < 0.5:
        if y1>8.5 or y1<-4.5:
            index=index2
        else:
            index=index1
            
    elif diff1<diff2:
        index=index1
    else:
        index=index2

    if index==index1:
        yentry=y1
    else:
        yentry=y2
    
    ylocal=-round(yentry/25e-3)*25e-3
    # at some point, add limits to possible ROIs

    gamma=sensorAngles[index]
    # Shift range of gamma to 0 to 2 pi
    if gamma<0:
        gamma+=2*np.pi
    
    return ylocal, gamma
##########################

def collectionDfFromTree(f: uproot.reading.ReadOnlyDirectory,collectionBranch = "VertexBarrelCollection",collectionBranchIndex="",mcBranch="MCParticles",getMcDf=True):
    assert isinstance(f,uproot.reading.ReadOnlyDirectory)
    assert "events;1" in f.keys()
    if collectionBranchIndex=="":
        collectionBranchIndex = "_"+collectionBranch+"_particle"
    assert collectionBranch in f["events"].keys()
    assert collectionBranchIndex in f["events"].keys()
    assert mcBranch in f["events"].keys()
    df1 = f["events"][collectionBranch].arrays(library="pd")
    df2 = f["events"][collectionBranchIndex].arrays(library="pd")
    eventHitDf = df1.merge(df2,how='outer',left_index=True,right_index=True)
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
        mcParticleIndex=int(hitRow["index"])
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
    for e, i in zip(hitDf["ievent"], hitDf["index"]):
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
    """
    convert to a df whos rows match Eliza's track = [cota, cotb, p, ops.flp, ylocal, zglobal, pt, t, hit_pdg]
    """
    hitDf["p"] = np.sqrt(hitDf["momentum.x"]**2 + hitDf["momentum.y"]**2 + hitDf["momentum.z"]**2 )
    hitDf["pt"] = np.sqrt(hitDf["momentum.x"]**2 + hitDf["momentum.y"]**2)
    hitDf["t"] = hitDf["time"]
    hitDf["hit_pdg"] = hitDf["MCParticles.PDG"]
    hitDf["zglobal"] = round(hitDf["position.z"]/25e-3)*25e-3 # round to nearest pixel
    out = [getYlocalAndGamma(x, y) for x, y in zip(hitDf["position.x"], hitDf["position.y"])]
    hitDf[["ylocal", "gamma0"]] = out
    hitDf["cota"] = hitDf["momentum.z"]/(hitDf["momentum.x"]*np.cos(hitDf["gamma0"])+hitDf["momentum.y"]*np.sin(hitDf["gamma0"]))
    hitDf["cotb"] = (hitDf["momentum.x"]*np.sin(hitDf["gamma0"])-hitDf["momentum.y"]*np.cos(hitDf["gamma0"]))/(hitDf["momentum.x"]*np.cos(hitDf["gamma0"])+hitDf["momentum.y"]*np.sin(hitDf["gamma0"]))
    if flp == 0:
        hitDf["ylocal"] *= -1
        hitDf["cota"] *= -1
    hitDf["flp"] = flp  
    tracklistDf = hitDf[["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg"]].copy()
    return tracklistDf

def processEdmFileToTrack(file_path):
    with uproot.open(file_path) as f:
        vertexEventsDf,eventMcDf = collectionDfFromTree(f)
        # print(vertexEventsDf)
        _,hitDf = collectionDfToFlatHits(vertexEventsDf)
        if hitDf.empty:
            print("this file empty of hits")
            return None,None
        if filterSecondaries:
            hitDf.filter(cond=hitDf["index"]==0)
        if addMCParticle:
            hitDf = addMcToHits(hitDf,eventMcDf)
        tracklistDf = hitDftoTracklistDf(hitDf)

        #####saving
        savePath = file_path+".simhits.csv"
        hitDf.to_csv(savePath)
        print(f"saved {len(hitDf)} hits to {savePath}")
        
        savePath = file_path+".tracks.csv"
        tracklistDf.to_csv(savePath,sep=" ",index=False,float_format=f"%.{float_precision}f")
        print(f"saved {len(tracklistDf)} hits to {savePath}")
        return hitDf,tracklistDf


file_list = ["./edmTestFiles/sim_output.edm4hep.root","./edmTestFiles/sim_output2.edm4hep.root","./edmTestFiles/sim_output3.edm4hep.root","./edmTestFiles/digi_real.edm4hep.root",]#"./edmTestFiles/gen_output.edm4hep.root",]
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