'''
Author: Daniel Abadjiev, with lots of code borrowed from Eliza Howard and Eric You
Date: November 18, 2025
Description: Utilities file for making all the plots to validate dataset
'''

import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt                               
import os 
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
matplotlib.rcParams["figure.dpi"] = 150
from particle import PDGID



#I'll only look at parquets

def load_parquet_pairs(directory, skip_range=None):
    # truth_bib_mm, recon_bib_mm = [], []
    # truth_bib_mp, recon_bib_mp = [], []
    # truth_sig, recon_sig = [], []
    truthDFs, reconDFs = [], []

    for file in os.listdir(directory):
        if skip_range and any(f"{i}" in file for i in skip_range):
            continue
        if "labels" in file:
            filepath = os.path.join(directory, file)
            df = pd.read_parquet(filepath)

            paired_file = os.path.join(directory, file.replace("labels", "recon2D"))
            recon_df = pd.read_parquet(paired_file)

            # if "bib" in file:
            if "mm" in file:
                df["source"] = "bib_mm"
                recon_df["source"] = "bib_mm"
                # truth_bib_mm.append(df)
                # recon_bib_mm.append(recon_df)
            if "mp" in file:
                df["source"] = "bib_mp"
                recon_df["source"] = "bib_mp"
                # truth_bib_mp.append(df)
                # recon_bib_mp.append(recon_df)
            elif "sig" in file:
                df["source"] = "sig"
                recon_df["source"] = "sig"
                # truth_sig.append(df)
                # recon_sig.append(recon_df)
            truthDFs.append(df)
            reconDFs.append(recon_df)
    print("done with the for loop, now concatenating the dataframes")
    return (
        # pd.concat(truth_bib_mm) if truth_bib_mm else pd.DataFrame(),
        # pd.concat(recon_bib_mm) if recon_bib_mm else pd.DataFrame(),
        # pd.concat(truth_bib_mp) if truth_bib_mp else pd.DataFrame(),
        # pd.concat(recon_bib_mp) if recon_bib_mp else pd.DataFrame(),
        # pd.concat(truth_sig) if truth_sig else pd.DataFrame(),
        # pd.concat(recon_sig) if recon_sig else pd.DataFrame(),
        pd.concat(truthDFs) if truthDFs else pd.DataFrame(),
        pd.concat(reconDFs) if reconDFs else pd.DataFrame(),
    )

def reshapeCluster(recon2d__):
    print("I might have broken this funciton, might need to be called on each dataframe subtype")
    return recon2d__.to_numpy().reshape(recon2d__.shape[0],13,21)

def plotHisto(arr,bins=None,postScale=1,title="",pltStandalone=True,pltLabel="",showNums=True):
    if pltStandalone:
        plt.figure(figsize=(4,1))
    if bins is None:
        hist, bin_edges = np.histogram(arr)
    else:
        hist, bin_edges = np.histogram(arr,bins=bins)
    plt.stairs(hist*postScale,bin_edges,label=pltLabel)
    if showNums:
        for i in range(len(hist)):
            # Calculate the x-coordinate for the center of the bar
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            # Place the text slightly above the top of the bar
            plt.text(bin_center, hist[i], int(hist[i]), ha='center', va='bottom')
    plt.title(title)
    if pltStandalone:
        plt.show()
    return hist,bin_edges

def plotManyHisto(arrs,bins=None,postScale=1,title="",pltLabels=["1","2","3"],showNums=False,figsize=(7,3),yscale="linear"):
    assert len(arrs) == len(pltLabels)
    plt.figure(figsize=figsize)
    # plotHisto(arrs,bins=bins,postScale=postScale,title=title,pltStandalone=False,pltLabel=pltLabels,showNums=showNums)
    for i in range(len(arrs)):
        plotHisto(arrs[i],bins=bins,postScale=postScale,title=title,pltStandalone=False,pltLabel=pltLabels[i],showNums=showNums)
    plt.legend()
    plt.yscale(yscale)
    plt.show()
def countBibSig(truthDF,doPrint=False):
    truthSig = truthDF.query('source == "sig"')
    truthBib_mm = truthDF.query('source == "bib_mm"')
    truthBib_mp = truthDF.query('source == "bib_mp"')
    truthBib = truthDF.query('source == "bib_mm" or source == "bib_mp"')
    numSig = len(truthSig)
    numMM = len(truthBib_mm)
    numMP = len(truthBib_mp)
    numBib = len(truthBib)
    numTotal = numBib+numSig
    assert numBib==numMP+numMM
    assert numTotal==len(truthDF)
    fracMM = numMM/numTotal
    fracMP = numMP/numTotal
    fracBib = numBib/numTotal
    fracSig = numSig/numTotal
    if doPrint:
        print(f"len truthSig: {numSig}")
        print(f"len truthBib: {numBib}")
        print(f"len truthBib_mm: {numMM}")
        print(f"len truthBib_mp: {numMP}")    
        print(f"fraction of total that are MM: {fracMM}")
        print(f"fraction of total that are MP: {fracMP}")
        print(f"fraction of total that are Bib: {fracBib}")
        print(f"fraction of total that are Sig: {fracSig}")
    return fracBib, fracSig, fracMM, fracMP,numSig,numBib,truthSig,truthBib_mm,truthBib_mp,truthBib


# === Load Data ===
skip_indices = list(range(1730 - 124+87, 1769))  # 1606+87 [hand-tuned the 87] to 1768




