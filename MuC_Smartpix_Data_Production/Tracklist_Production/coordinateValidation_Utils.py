"""
Author: Daniel Abadjiev and Eliza Howard
Date: January 26, 2026
Description: Helper functions for the coordinate Validation notebook, as well as the getYlocalAndGamma from make_tracklists
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.rcParams["figure.dpi"] = 150
import ipywidgets as widgets
from particle import PDGID
import sys
sys.path.append("../../../Muon_Collider_Smart_Pixels/daniel/validationPlots")
from plotUtils import *
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes



def plotHits(sig,uniqueGammas,pltStandalone=True,pltShow=True,alpha = 1):
    if pltStandalone:
        fig,ax=plt.subplots()
    else:
        ax = plt.gca()
    colorsList = ["r","b","g","c","m","y","aquamarine","fuchsia",
                "sienna","deepskyblue","springgreen","royalblue",
                "darkorange","indigo","navy","chartreuse"]
    # colorsList = ["r","darkorange","y","greenyellow","springgreen","g","aquamarine","c","deepskyblue","royalblue","b","navy","indigo","m","fuchsia",
    #             "sienna"]
    # uniqueGammas = np.unique(sig['gamma'])
    colors = sig['gamma'].apply(lambda gamma: colorsList[list(uniqueGammas).index(gamma)])
    # colors = sig['moduleID'].apply(lambda ID: colorsList[list(np.arange(1,17,1)).index(ID)])
    ax.scatter(sig['hit_x'],sig["hit_y"],s=1, c=colors, label="Signal hits",alpha = alpha)
    plt.gca().set_aspect('equal')
    circle=plt.Circle((0,0), 30, fill=0, color='k', label="Barrel")
    ax.add_patch(circle)
    ax.set_title("Hits in the barrel xy-plane")
    ax.set_xlabel("x-global [mm]")
    ax.set_ylabel("y-global [mm]")
    ax.set_xlim(-40,40)
    ax.set_ylim(-40,40)
    if pltShow:
        plt.show()






sensorAngles = np.arange(-np.pi-np.pi/8,np.pi+2*np.pi/8,np.pi/8)
moduleIDs = np.concatenate((np.arange(8,17,1), np.arange(1,11,1)))
# CHECK THAT THIS IS CORRECT!

def getYlocalAndGamma(x,y):
    # Get exact angle gamma of the hit position
    gammaP=np.arctan2(y,x)

    # Get two sensor angles that are closest to the exact angle
    diff = np.abs(sensorAngles-gammaP)
    index1 = np.argmin(diff)
    gamma1=sensorAngles[index1]
    diff[index1]=3*np.pi # reassign the previous min to something much larger than the other values
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
    
    # If both x1 and x2 are really close to the correct value
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
    
    ylocal=round(yentry/25e-3)*25e-3
    # at some point, add limits to possible ROIs

    gamma=sensorAngles[index]
    moduleID=moduleIDs[index]
    # Shift range of gamma to 0 to 2 pi
    if gamma<0:
        gamma+=2*np.pi
    
    return ylocal, gamma, moduleID
def getYlocalAndGamma_tester():
    xList = []
    yList = []
    ylocalList = []
    gammaList = []
    moduleIDList = []
    assert len(xList)==len(yList)
    assert len(xList)==len(ylocalList)
    assert len(xList)==len(gammaList)
    assert len(xList)==len(moduleIDList)
    for idx,x in enumerate(xList):
        ylocal, gamma, moduleID = getYlocalAndGamma(xList[idx],yList[idx])
        assert ylocal == ylocalList[idx]
        assert gamma == gammaList[idx]
        assert moduleID == moduleIDList[idx]



