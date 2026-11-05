"""
Author: Daniel Abadjiev
Date: November 4, 2026
Description: Utils functions moved out of make_tracklists_fromEdm4hep.py to be reused
"""

import pandas as pd
import numpy as np
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
def hitDftoTracklistDf(hitDf: pd.DataFrame,flp=0):
    """
    convert to a df whos rows match Eliza's track = [cota, cotb, p, ops.flp, ylocal, zglobal, pt, t, hit_pdg]
    """
    if hitDf.empty:
        print("hit df empty, so empty tracklist df")
        return pd.DataFrame()
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