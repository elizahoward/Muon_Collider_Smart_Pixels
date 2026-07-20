"""
Author: Daniel Abadjiev with code from Eliza Howard
Date: July 16, 2026
Description: Take the coordinateValidation.ipynb notebooks that make module layout plots and reproduce them for the paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append("../MuC_Smartpix_Data_Production/Tracklist_Production")
sys.path.append("../daniel/validationPlots")

# from coordinateValidation_Utils import * #instead of importing, copied function over
# def plotHits(sig,uniqueGammas,pltStandalone=True,pltShow=True,alpha = 1):
def plotHitsXY(sig,pltStandalone=True,pltShow=True,alpha = 1):
    if pltStandalone:
        fig,ax=plt.subplots(figsize=(5,4.8))
    else:
        ax = plt.gca()
    colorsList = ["r","b","g","c","m","y","aquamarine","fuchsia",
                "sienna","deepskyblue","springgreen","royalblue",
                "darkorange","indigo","maroon","chartreuse"]
    # colorsList = ["r","darkorange","y","greenyellow","springgreen","g","aquamarine","c","deepskyblue","royalblue","b","navy","indigo","m","fuchsia",
    #             "sienna"]
    # uniqueGammas = np.unique(sig['gamma'])
    # colors = sig['gamma'].apply(lambda gamma: colorsList[list(uniqueGammas).index(gamma)])
    colors = sig['moduleID'].apply(lambda ID: colorsList[list(np.arange(1,17,1)).index(ID)])
    ax.scatter(sig['hit_x'],sig["hit_y"],s=1, c=colors, label="Signal hits",alpha = alpha)
    plt.gca().set_aspect('equal')
    circle=plt.Circle((0,0), 30, fill=0, color='k', label="Barrel")
    ax.add_patch(circle)
    ax.set_title("Hits in the barrel xy-plane")
    ax.set_xlabel("Z-Global [mm]")
    ax.set_ylabel("Y-Global [mm]")
    ax.set_xlim(-40,40)
    ax.set_ylim(-40,40)
    plt.tight_layout()
    if pltShow:
        plt.show()

def plotRZPlane(sig):
    print(sig)
    print(sig.keys())
    plt.figure(figsize=(5,2))

    colorsList = ["r","b","g","c","m","y","aquamarine","fuchsia",
                "sienna","deepskyblue","springgreen","royalblue",
                "darkorange","indigo","maroon","chartreuse"]
    sig['nModule'] = np.floor(sig['zglobal']/13)
    colors = sig['nModule'].apply(lambda ID: colorsList[list(np.arange(1,17,1)).index(ID+7)])

    sig['hit_r'] = np.sqrt( sig["hit_y"]**2 + sig["hit_x"]**2)
    # plt.subplot(211)
    plt.scatter(sig['zglobal'],sig["hit_r"],marker="s",s=2,c=colors, label="Signal hits")
    # plt.ylim([0,60])
    # plt.subplot(212)
    # plt.scatter(sig['hit_z'],sig["hit_r"],s=1,c=colors, label="Signal hits") #to confirm that hit_z is the same as z-global
    plt.ylim([0,60])
    plt.xlabel("Z-Global [mm]")
    plt.ylabel("Radius [mm]")

    plt.tight_layout()
    plt.savefig("ModuleLayoutRZ.png")
    plt.close()

def main():
    sig = pd.read_csv('/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_flp_0/Track_Lists/signal_tracks_extra_info_0.txt', sep=' ')
    print(len(sig))
    plt.close()
    plotHitsXY(sig)
    plt.title("")
    plt.savefig("./ModuleLayoutXY.png")
    plt.close()
    plotRZPlane(sig)

if __name__=="__main__":
    main()