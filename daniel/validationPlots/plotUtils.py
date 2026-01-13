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
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
matplotlib.rcParams["figure.dpi"] = 150
from particle import PDGID
# import pickle



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
            # else:
                # print("GAAAHHHHH")
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
    # print("I might have broken this funciton, might need to be called on each dataframe subtype")
    return recon2d__.to_numpy().reshape(recon2d__.shape[0],13,21)

def plotHisto(arr,bins=None,postScale=1,title="",pltStandalone=True,pltLabel="",showNums=True):
    if pltStandalone:
        plt.figure(figsize=(4,1))
    if bins is None:
        hist, bin_edges = np.histogram(arr)
    elif len(bins) ==1:
        hist, bin_edges = np.histogram(arr,bins=bins)
    else:
        hist, bin_edges = np.histogram(np.clip(arr,bins[0],bins[-1]),bins=bins)
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

def plotManyHisto(arrs,bins=None,postScale=1,title="",pltLabels=["1","2","3"],pltStandalone=True,showNums=False,figsize=(7,3),yscale="linear"):
    assert len(arrs) == len(pltLabels)
    if pltStandalone:
        plt.figure(figsize=figsize)
    # plotHisto(arrs,bins=bins,postScale=postScale,title=title,pltStandalone=False,pltLabel=pltLabels,showNums=showNums)
    for i in range(len(arrs)):
        plotHisto(arrs[i],bins=bins,postScale=postScale,title=title,pltStandalone=False,pltLabel=pltLabels[i],showNums=showNums)
    plt.legend()
    plt.yscale(yscale)
    if pltStandalone:
        plt.show()

#old version of the function
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

#new version that also does stuff with reconDF
def processReconBibSig(truthDF,reconDF,doPrint=False):
    fracBib, fracSig, fracMM, fracMP,numSig,numBib,truthSig,truthBib_mm,truthBib_mp,truthBib = countBibSig(truthDF,doPrint=doPrint)

    reconSig = reconDF.query('source == "sig"')
    reconBib_mm = reconDF.query('source == "bib_mm"')
    reconBib_mp = reconDF.query('source == "bib_mp"')
    reconBib = reconDF.query('source == "bib_mm" or source == "bib_mp"')


    clustersSig = reconSig.drop("source",axis=1)
    clustersSig = reshapeCluster(clustersSig)

    clustersBib = reconBib.drop("source",axis=1)
    clustersBib = reshapeCluster(clustersBib)

    xSizesSig = np.count_nonzero(clustersSig, axis=2).max(axis=1)
    xSizesBib = np.count_nonzero(clustersBib, axis=2).max(axis=1)

    ySizesSig = np.count_nonzero(clustersSig, axis=1).max(axis=1)
    ySizesBib = np.count_nonzero(clustersBib, axis=1).max(axis=1)

    nPixelsSig = np.count_nonzero(clustersSig, axis=(1,2))
    nPixelsBib = np.count_nonzero(clustersBib, axis=(1,2))

    assert numSig == len(reconSig)
    assert numBib == len(reconBib)
    assert len(truthSig) == len(xSizesSig)
    assert len(truthSig) == len(ySizesSig)
    assert len(truthSig) == len(nPixelsSig)
    assert len(truthBib) == len(xSizesBib)
    assert len(truthBib) == len(ySizesBib)
    assert len(truthBib) == len(nPixelsBib)
    truthSig.loc[:,"xSize"] = list(xSizesSig)
    truthSig.loc[:,"ySize"] = list(ySizesSig)
    truthSig.loc[:,"nPix"]  = list(nPixelsSig)
    truthBib.loc[:,"xSize"] = list(xSizesBib)
    truthBib.loc[:,"ySize"] = list(ySizesBib)
    truthBib.loc[:,"nPix"]  = list(nPixelsBib)

    #from Eliza's code, the average yprofiles in each section
    print("getting average cluster profile in each regime")
    avgClustDictBib = getProfiles(truthBib, clustersBib)
    avgClustDictSig = getProfiles(truthSig, clustersSig)
    print("finished getting average cluster profile in each regime")
    return truthSig, truthBib, reconSig,reconBib_mm,reconBib_mp,reconBib,clustersSig,clustersBib, xSizesSig, xSizesBib,ySizesSig, ySizesBib,nPixelsSig,nPixelsBib,avgClustDictBib,avgClustDictSig

#Tighter version of Eliza's processing cuts
def getProfiles(truth__, clusters__):
    clusters__LowPtPosLowYl  = clusters__[np.where((truth__['q']>0) & (truth__['pt'] <5 ) & (truth__['y-local']>-4.5) & (truth__['y-local']<-2 ))]
    clusters__HighPtPosLowYl = clusters__[np.where((truth__['q']>0) & (truth__['pt'] >95) & (truth__['y-local']>-4.5) & (truth__['y-local']<-2 ))]
    clusters__LowPtNegLowYl  = clusters__[np.where((truth__['q']<0) & (truth__['pt'] <5 ) & (truth__['y-local']>-4.5) & (truth__['y-local']<-2 ))]
    clusters__HighPtNegLowYl = clusters__[np.where((truth__['q']<0) & (truth__['pt'] >95) & (truth__['y-local']>-4.5) & (truth__['y-local']<-2 ))]
    clusters__LowYl          = clusters__[np.where(                                         (truth__['y-local']>-4.5) & (truth__['y-local']<-2 ))]
    clusters__LowPtPosMidYl =  clusters__[np.where((truth__['q']>0) & (truth__['pt'] <5 ) & (truth__['y-local']>0) & (truth__['y-local']<2 ))]
    clusters__HighPtPosMidYl = clusters__[np.where((truth__['q']>0) & (truth__['pt'] >95) & (truth__['y-local']>0) & (truth__['y-local']<2 ))]
    clusters__LowPtNegMidYl =  clusters__[np.where((truth__['q']<0) & (truth__['pt'] <5 ) & (truth__['y-local']>0) & (truth__['y-local']<2 ))]
    clusters__HighPtNegMidYl = clusters__[np.where((truth__['q']<0) & (truth__['pt'] >95) & (truth__['y-local']>0) & (truth__['y-local']<2 ))]
    clusters__MidYl          = clusters__[np.where(                                         (truth__['y-local']>0) & (truth__['y-local']<2 ))]
    clusters__LowPtPosHighYl =  clusters__[np.where((truth__['q']>0) & (truth__['pt'] <5 ) & (truth__['y-local']>6) & (truth__['y-local']<8.5 ))]
    clusters__HighPtPosHighYl = clusters__[np.where((truth__['q']>0) & (truth__['pt'] >95) & (truth__['y-local']>6) & (truth__['y-local']<8.5 ))]
    clusters__LowPtNegHighYl =  clusters__[np.where((truth__['q']<0) & (truth__['pt'] <5 ) & (truth__['y-local']>6) & (truth__['y-local']<8.5 ))]
    clusters__HighPtNegHighYl = clusters__[np.where((truth__['q']<0) & (truth__['pt'] >95) & (truth__['y-local']>6) & (truth__['y-local']<8.5 ))]
    clusters__HighYl          = clusters__[np.where(                                         (truth__['y-local']>6) & (truth__['y-local']<8.5 ))]
    
    clusters__LowPtPosLowZgl =  clusters__[np.where((truth__['q']>0) & (truth__['pt'] <5 ) & (truth__['z-global']>0) & (truth__['z-global']<20 ))]
    clusters__HighPtPosLowZgl = clusters__[np.where((truth__['q']>0) & (truth__['pt'] >95) & (truth__['z-global']>0) & (truth__['z-global']<20 ))]
    clusters__LowPtNegLowZgl =  clusters__[np.where((truth__['q']<0) & (truth__['pt'] <5 ) & (truth__['z-global']>0) & (truth__['z-global']<20 ))]
    clusters__HighPtNegLowZgl = clusters__[np.where((truth__['q']<0) & (truth__['pt'] >95) & (truth__['z-global']>0) & (truth__['z-global']<20 ))]
    clusters__LowZgl          = clusters__[np.where(                                         (truth__['z-global']>0) & (truth__['z-global']<20 ))]
    clusters__LowPtPosMidZgl =  clusters__[np.where((truth__['q']>0) & (truth__['pt'] <5 ) & (truth__['z-global']>20) & (truth__['z-global']<40 ))]
    clusters__HighPtPosMidZgl = clusters__[np.where((truth__['q']>0) & (truth__['pt'] >95) & (truth__['z-global']>20) & (truth__['z-global']<40 ))]
    clusters__LowPtNegMidZgl =  clusters__[np.where((truth__['q']<0) & (truth__['pt'] <5 ) & (truth__['z-global']>20) & (truth__['z-global']<40 ))]
    clusters__HighPtNegMidZgl = clusters__[np.where((truth__['q']<0) & (truth__['pt'] >95) & (truth__['z-global']>20) & (truth__['z-global']<40 ))]
    clusters__MidZgl          = clusters__[np.where(                                         (truth__['z-global']>20) & (truth__['z-global']<50 ))]
    clusters__LowPtPosHighZgl =  clusters__[np.where((truth__['q']>0) & (truth__['pt'] <5 ) & (truth__['z-global']>40) & (truth__['z-global']<65 ))]
    clusters__HighPtPosHighZgl = clusters__[np.where((truth__['q']>0) & (truth__['pt'] >95) & (truth__['z-global']>40) & (truth__['z-global']<65 ))]
    clusters__LowPtNegHighZgl =  clusters__[np.where((truth__['q']<0) & (truth__['pt'] <5 ) & (truth__['z-global']>40) & (truth__['z-global']<65 ))]
    clusters__HighPtNegHighZgl = clusters__[np.where((truth__['q']<0) & (truth__['pt'] >95) & (truth__['z-global']>40) & (truth__['z-global']<65 ))]
    clusters__HighZgl          = clusters__[np.where(                                         (truth__['z-global']>40) & (truth__['z-global']<65 ))]
    yProfile__LowPtPosLowYl = getAverageYProfile(clusters__LowPtPosLowYl)
    yProfile__HighPtPosLowYl = getAverageYProfile(clusters__HighPtPosLowYl)
    yProfile__LowPtNegLowYl = getAverageYProfile(clusters__LowPtNegLowYl)
    yProfile__HighPtNegLowYl = getAverageYProfile(clusters__HighPtNegLowYl)
    yProfile__LowPtPosMidYl = getAverageYProfile(clusters__LowPtPosMidYl)
    yProfile__HighPtPosMidYl = getAverageYProfile(clusters__HighPtPosMidYl)
    yProfile__LowPtNegMidYl = getAverageYProfile(clusters__LowPtNegMidYl)
    yProfile__HighPtNegMidYl = getAverageYProfile(clusters__HighPtNegMidYl)
    yProfile__LowPtPosHighYl = getAverageYProfile(clusters__LowPtPosHighYl)
    yProfile__HighPtPosHighYl = getAverageYProfile(clusters__HighPtPosHighYl)
    yProfile__LowPtNegHighYl = getAverageYProfile(clusters__LowPtNegHighYl)
    yProfile__HighPtNegHighYl = getAverageYProfile(clusters__HighPtNegHighYl)
    xProfile__LowPtPosLowYl = getAverageXProfile(clusters__LowPtPosLowYl)
    xProfile__HighPtPosLowYl = getAverageXProfile(clusters__HighPtPosLowYl)
    xProfile__LowPtNegLowYl = getAverageXProfile(clusters__LowPtNegLowYl)
    xProfile__HighPtNegLowYl = getAverageXProfile(clusters__HighPtNegLowYl)
    xProfile__LowPtPosMidYl = getAverageXProfile(clusters__LowPtPosMidYl)
    xProfile__HighPtPosMidYl = getAverageXProfile(clusters__HighPtPosMidYl)
    xProfile__LowPtNegMidYl = getAverageXProfile(clusters__LowPtNegMidYl)
    xProfile__HighPtNegMidYl = getAverageXProfile(clusters__HighPtNegMidYl)
    xProfile__LowPtPosHighYl = getAverageXProfile(clusters__LowPtPosHighYl)
    xProfile__HighPtPosHighYl = getAverageXProfile(clusters__HighPtPosHighYl)
    xProfile__LowPtNegHighYl = getAverageXProfile(clusters__LowPtNegHighYl)
    xProfile__HighPtNegHighYl = getAverageXProfile(clusters__HighPtNegHighYl)

    yProfile__LowPtPosLowZgl = getAverageYProfile(clusters__LowPtPosLowZgl)
    yProfile__HighPtPosLowZgl = getAverageYProfile(clusters__HighPtPosLowZgl)
    yProfile__LowPtNegLowZgl = getAverageYProfile(clusters__LowPtNegLowZgl)
    yProfile__HighPtNegLowZgl = getAverageYProfile(clusters__HighPtNegLowZgl)
    yProfile__LowPtPosMidZgl = getAverageYProfile(clusters__LowPtPosMidZgl)
    yProfile__HighPtPosMidZgl = getAverageYProfile(clusters__HighPtPosMidZgl)
    yProfile__LowPtNegMidZgl = getAverageYProfile(clusters__LowPtNegMidZgl)
    yProfile__HighPtNegMidZgl = getAverageYProfile(clusters__HighPtNegMidZgl)
    yProfile__LowPtPosHighZgl = getAverageYProfile(clusters__LowPtPosHighZgl)
    yProfile__HighPtPosHighZgl = getAverageYProfile(clusters__HighPtPosHighZgl)
    yProfile__LowPtNegHighZgl = getAverageYProfile(clusters__LowPtNegHighZgl)
    yProfile__HighPtNegHighZgl = getAverageYProfile(clusters__HighPtNegHighZgl)
    xProfile__LowPtPosLowZgl = getAverageXProfile(clusters__LowPtPosLowZgl)
    xProfile__HighPtPosLowZgl = getAverageXProfile(clusters__HighPtPosLowZgl)
    xProfile__LowPtNegLowZgl = getAverageXProfile(clusters__LowPtNegLowZgl)
    xProfile__HighPtNegLowZgl = getAverageXProfile(clusters__HighPtNegLowZgl)
    xProfile__LowPtPosMidZgl = getAverageXProfile(clusters__LowPtPosMidZgl)
    xProfile__HighPtPosMidZgl = getAverageXProfile(clusters__HighPtPosMidZgl)
    xProfile__LowPtNegMidZgl = getAverageXProfile(clusters__LowPtNegMidZgl)
    xProfile__HighPtNegMidZgl = getAverageXProfile(clusters__HighPtNegMidZgl)
    xProfile__LowPtPosHighZgl = getAverageXProfile(clusters__LowPtPosHighZgl)
    xProfile__HighPtPosHighZgl = getAverageXProfile(clusters__HighPtPosHighZgl)
    xProfile__LowPtNegHighZgl = getAverageXProfile(clusters__LowPtNegHighZgl)
    xProfile__HighPtNegHighZgl = getAverageXProfile(clusters__HighPtNegHighZgl)

    yProfile__LowYl = getAverageYProfile(clusters__LowYl)
    yProfile__MidYl = getAverageYProfile(clusters__MidYl)
    yProfile__HighYl = getAverageYProfile(clusters__HighYl)
    yProfile__LowZgl = getAverageYProfile(clusters__LowZgl)
    yProfile__MidZgl = getAverageYProfile(clusters__MidZgl)
    yProfile__HighZgl = getAverageYProfile(clusters__HighZgl)

    xProfile__LowYl = getAverageXProfile(clusters__LowYl)
    xProfile__MidYl = getAverageXProfile(clusters__MidYl)
    xProfile__HighYl = getAverageXProfile(clusters__HighYl)
    xProfile__LowZgl = getAverageXProfile(clusters__LowZgl)
    xProfile__MidZgl = getAverageXProfile(clusters__MidZgl)
    xProfile__HighZgl = getAverageXProfile(clusters__HighZgl)

    yProfile__ = getAverageYProfile(clusters__)
    xProfile__ = getAverageXProfile(clusters__)


    avgClusterDict = {        
        "yProfileLowPtPosLowYl": yProfile__LowPtPosLowYl,
        "yProfileHighPtPosLowYl": yProfile__HighPtPosLowYl,
        "yProfileLowPtNegLowYl": yProfile__LowPtNegLowYl,
        "yProfileHighPtNegLowYl": yProfile__HighPtNegLowYl,
        "yProfileLowPtPosMidYl": yProfile__LowPtPosMidYl,
        "yProfileHighPtPosMidYl": yProfile__HighPtPosMidYl,
        "yProfileLowPtNegMidYl": yProfile__LowPtNegMidYl,
        "yProfileHighPtNegMidYl": yProfile__HighPtNegMidYl,
        "yProfileLowPtPosHighYl": yProfile__LowPtPosHighYl,
        "yProfileHighPtPosHighYl": yProfile__HighPtPosHighYl,
        "yProfileLowPtNegHighYl": yProfile__LowPtNegHighYl,
        "yProfileHighPtNegHighYl": yProfile__HighPtNegHighYl,
        "xProfileLowPtPosLowYl": xProfile__LowPtPosLowYl,
        "xProfileHighPtPosLowYl": xProfile__HighPtPosLowYl,
        "xProfileLowPtNegLowYl": xProfile__LowPtNegLowYl,
        "xProfileHighPtNegLowYl": xProfile__HighPtNegLowYl,
        "xProfileLowPtPosMidYl": xProfile__LowPtPosMidYl,
        "xProfileHighPtPosMidYl": xProfile__HighPtPosMidYl,
        "xProfileLowPtNegMidYl": xProfile__LowPtNegMidYl,
        "xProfileHighPtNegMidYl": xProfile__HighPtNegMidYl,
        "xProfileLowPtPosHighYl": xProfile__LowPtPosHighYl,
        "xProfileHighPtPosHighYl": xProfile__HighPtPosHighYl,
        "xProfileLowPtNegHighYl": xProfile__LowPtNegHighYl,
        "xProfileHighPtNegHighYl": xProfile__HighPtNegHighYl,
                
        "yProfileLowPtPosLowZgl": yProfile__LowPtPosLowZgl,
        "yProfileHighPtPosLowZgl": yProfile__HighPtPosLowZgl,
        "yProfileLowPtNegLowZgl": yProfile__LowPtNegLowZgl,
        "yProfileHighPtNegLowZgl": yProfile__HighPtNegLowZgl,
        "yProfileLowPtPosMidZgl": yProfile__LowPtPosMidZgl,
        "yProfileHighPtPosMidZgl": yProfile__HighPtPosMidZgl,
        "yProfileLowPtNegMidZgl": yProfile__LowPtNegMidZgl,
        "yProfileHighPtNegMidZgl": yProfile__HighPtNegMidZgl,
        "yProfileLowPtPosHighZgl": yProfile__LowPtPosHighZgl,
        "yProfileHighPtPosHighZgl": yProfile__HighPtPosHighZgl,
        "yProfileLowPtNegHighZgl": yProfile__LowPtNegHighZgl,
        "yProfileHighPtNegHighZgl": yProfile__HighPtNegHighZgl,
        "xProfileLowPtPosLowZgl": xProfile__LowPtPosLowZgl,
        "xProfileHighPtPosLowZgl": xProfile__HighPtPosLowZgl,
        "xProfileLowPtNegLowZgl": xProfile__LowPtNegLowZgl,
        "xProfileHighPtNegLowZgl": xProfile__HighPtNegLowZgl,
        "xProfileLowPtPosMidZgl": xProfile__LowPtPosMidZgl,
        "xProfileHighPtPosMidZgl": xProfile__HighPtPosMidZgl,
        "xProfileLowPtNegMidZgl": xProfile__LowPtNegMidZgl,
        "xProfileHighPtNegMidZgl": xProfile__HighPtNegMidZgl,
        "xProfileLowPtPosHighZgl": xProfile__LowPtPosHighZgl,
        "xProfileHighPtPosHighZgl": xProfile__HighPtPosHighZgl,
        "xProfileLowPtNegHighZgl": xProfile__LowPtNegHighZgl,
        "xProfileHighPtNegHighZgl": xProfile__HighPtNegHighZgl,

        "yProfileLowYl": yProfile__LowYl,
        "yProfileMidYl": yProfile__MidYl,
        "yProfileHighYl": yProfile__HighYl,
        "yProfileLowZgl": yProfile__LowZgl,
        "yProfileMidZgl": yProfile__MidZgl,
        "yProfileHighZgl": yProfile__HighZgl,
        "xProfileLowYl": xProfile__LowYl,
        "xProfileMidYl": xProfile__MidYl,
        "xProfileHighYl": xProfile__HighYl,
        "xProfileLowZgl": xProfile__LowZgl,
        "xProfileMidZgl": xProfile__MidZgl,
        "xProfileHighZgl": xProfile__HighZgl,

        "yProfile": yProfile__,
        "xProfile": xProfile__,
    }
    return avgClusterDict

skip_indices = list(range(1730 - 124+87, 1769))  # 1606+87 [hand-tuned the 87] to 1768

#common in every plotting code
def closePlot(PLOT_DIR, interactivePlots, plotName,printOutputDir=True,transparent = False):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, plotName), dpi=300, bbox_inches='tight',transparent=transparent)
    if interactivePlots:
        plt.show()
    else:
        plt.close()
    if printOutputDir:
        print(f"Plot saved as: {os.path.join(PLOT_DIR, plotName)}")
    

def plotPt(truthSig,truthBib_mm,truthBib_mp,truthBib,PLOT_DIR='./plots',interactivePlots=False,doPrint=True):
    plt.figure(figsize=(5, 3))
    key = "pt"
    showNums = False
    plt.subplot(211)
    bins = np.linspace(-0.1,0.1,100)
    plotManyHisto([truthSig[key],truthBib_mm[key],truthBib_mp[key],truthBib[key]],title=f"{key} distribution (low pt)",pltStandalone=False,pltLabels=[f"sig {key}",f"bib mm {key}",f"bib mp {key}",f"bib {key}"],bins=bins,showNums=showNums,figsize=(7,2))
    plt.ylabel("Tracks")
    plt.subplot(212)
    bins = np.linspace(-1,10,100)
    plotManyHisto([truthSig[key],truthBib_mm[key],truthBib_mp[key],truthBib[key]],title="",pltStandalone=False,pltLabels=[f"sig {key}",f"bib mm {key}",f"bib mp {key}",f"bib {key}"],bins=bins,showNums=showNums,figsize=(7,2),yscale='log')
    plt.ylabel("Tracks (log scale)")
    plt.xlabel("momentum pT (GeV)")
    closePlot(PLOT_DIR, interactivePlots,  "bib_signal_pt_lowPtRange.png")
    plt.figure(figsize=(5, 3))
    plt.subplot(211)
    bins = np.logspace(-1,2,100)
    bins = np.concatenate(([-1,0,0.01],bins,[102,109,110,114]))
    plotManyHisto([truthSig[key],truthBib_mm[key],truthBib_mp[key],truthBib[key]],title=f"{key} distribution",pltStandalone=False,pltLabels=[f"sig {key}",f"bib mm {key}",f"bib mp {key}",f"bib {key}"],bins=bins,showNums=showNums,figsize=(7,2),yscale="linear")
    plt.ylabel("Tracks")
    plt.subplot(212)
    # bins = np.logspace(-1,2.1,100)py
    plotManyHisto([truthSig[key],truthBib_mm[key],truthBib_mp[key],truthBib[key]],title="",pltStandalone=False,pltLabels=[f"sig {key}",f"bib mm {key}",f"bib mp {key}",f"bib {key}"],bins=bins,showNums=showNums,figsize=(7,2),yscale="log")
    plt.ylabel("Tracks (log scale)")
    plt.xlabel("momentum pT (GeV)")
    closePlot(PLOT_DIR, interactivePlots,  "bib_signal_pt_fullRange.png")
    if doPrint:
        print(f"BIB pt min: {np.min(truthBib['pt'])} and max: {np.max(truthBib['pt'])}")
        print(f"Signal pt min: {np.min(truthSig['pt'])} and max: {np.max(truthSig['pt'])}")


#From Eric's code 
def create_pastel_red_cmap():
    # Define colors from white to a pastel/soft red
    colors_list = ['#ffffff', '#ffe6e6', '#ffcccc', '#ffb3b3', '#ff9999', '#ff8080', '#ff6666', '#ff4d4d', '#ff3333', '#e62e2e']
    n_bins = 256
    cmap = colors.LinearSegmentedColormap.from_list('pastel_red', colors_list, N=n_bins)
    return cmap

#A function to get the masks together, more sensibly
def getEricsMasks(truthbib, truthsig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,):

    z_global_bib = truthbib['z-global']
    z_global_sig = truthsig['z-global']

    mask_bib_x = ~np.isnan(z_global_bib) & ~np.isnan(xSizesBib)
    mask_bib_x = mask_bib_x & np.isfinite(z_global_bib) & np.isfinite(xSizesBib)
    mask_sig_x = ~np.isnan(z_global_sig) & ~np.isnan(xSizesSig)
    mask_sig_x = mask_sig_x & np.isfinite(z_global_sig) & np.isfinite(xSizesSig)

    mask_bib_y = ~np.isnan(z_global_bib) & ~np.isnan(ySizesBib)
    mask_bib_y = mask_bib_y & np.isfinite(z_global_bib) & np.isfinite(ySizesBib)
    mask_sig_y = ~np.isnan(z_global_sig) & ~np.isnan(ySizesSig)
    mask_sig_y = mask_sig_y & np.isfinite(z_global_sig) & np.isfinite(ySizesSig) 

    mask_bib = mask_bib_x & mask_bib_y
    mask_sig = mask_sig_x & mask_sig_y

    return mask_bib,mask_sig,mask_bib_x,mask_sig_x,mask_bib_y,mask_sig_y,

# --- Plot: Side-by-side comparison of z-global vs x-size ---
def plotZglobalXsize(truthbib, truthsig, xSizesSig, xSizesBib,mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    if not( 'z-global' in truthbib.columns and 'z-global' in truthsig.columns ):
        raise Exception("z-global not present in truthbib or truthsig dataframes")
    assert len(truthbib) == len(xSizesBib)
    assert len(truthsig) == len(xSizesSig)

    pastel_red_cmap = create_pastel_red_cmap()


    #Now fully just Eric's code
    z_global_bib = truthbib['z-global']
    z_global_sig = truthsig['z-global']
    
    # Left panel: BIB
    # mask_bib = ~np.isnan(z_global_bib) & ~np.isnan(xSizesBib)
    # mask_bib = mask_bib & np.isfinite(z_global_bib) & np.isfinite(xSizesBib)
    
    hb = ax[0].hist2d(z_global_bib[mask_bib], xSizesBib[mask_bib], bins=[30, np.arange(0,22,1)], cmap=pastel_red_cmap)
    ax[0].figure.colorbar(hb[3],ax=ax[0])#    ax[0].colorbar()
    ax[0].set_title("BIB", fontsize=20)
    ax[0].set_xlabel('z-global [mm]', fontsize=24)
    ax[0].set_ylabel('x-size (# pixels)', fontsize=24)
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    
    # Right panel: Signal
    # mask_sig = ~np.isnan(z_global_sig) & ~np.isnan(xSizesSig)
    # mask_sig = mask_sig & np.isfinite(z_global_sig) & np.isfinite(xSizesSig)
    
    hb = ax[1].hist2d(z_global_sig[mask_sig], xSizesSig[mask_sig], bins=[30, np.arange(0,22,1)], cmap=pastel_red_cmap)
    ax[1].figure.colorbar(hb[3],ax=ax[1])#    ax[1].colorbar()
    ax[1].set_title("Signal", fontsize=20)
    ax[1].set_xlabel('z-global [mm]', fontsize=24)
    ax[1].set_ylabel('x-size (# pixels)', fontsize=24)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    
    closePlot(PLOT_DIR, interactivePlots,  "bib_signal_zglobal_vs_xsize_comparison.png")
    
    print(f"BIB z-global range: {z_global_bib.min():.2f} to {z_global_bib.max():.2f} mm")
    print(f"Signal z-global range: {z_global_sig.min():.2f} to {z_global_sig.max():.2f} mm")
    if len(truthbib)>0:
        print(f"BIB x-size range: {xSizesBib.min()} to {xSizesBib.max()} pixels")
    else:
        print("no bib, so not printing bib x-size range")
    if len(truthsig)>0:
        print(f"Signal x-size range: {xSizesSig.min()} to {xSizesSig.max()} pixels")
    else:
        print("no signal, so not printing signal x-size range")


# --- Plot 2: Side-by-side comparison of z-global vs y-size ---
def plotZglobalYsize(truthbib, truthsig, ySizesSig, ySizesBib,mask_bib_y,mask_sig_y,PLOT_DIR="./plots",interactivePlots=False):
    if not( 'z-global' in truthbib.columns and 'z-global' in truthsig.columns ):
        raise Exception("z-global not present in truthbib or truthsig dataframes")
    assert len(truthbib) == len(ySizesBib)
    assert len(truthsig) == len(ySizesSig)

    pastel_red_cmap = create_pastel_red_cmap()

    z_global_bib = truthbib['z-global']
    z_global_sig = truthsig['z-global']

    #Now fully just Eric's code
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    # Left panel: BIB
    # mask_bib_y = ~np.isnan(z_global_bib) & ~np.isnan(ySizesBib)
    # mask_bib_y = mask_bib_y & np.isfinite(z_global_bib) & np.isfinite(ySizesBib)
    
    hb = ax[0].hist2d(z_global_bib[mask_bib_y], ySizesBib[mask_bib_y], bins=[30, np.arange(0,14,1)], cmap=pastel_red_cmap)
    ax[0].figure.colorbar(hb[3],ax=ax[0])#    ax[0].colorbar()
    ax[0].set_title("BIB - Y Size", fontsize=63)
    ax[0].set_xlabel('z-global [mm]', fontsize=50)
    ax[0].set_ylabel('y-size (# pixels)', fontsize=50)
    ax[0].tick_params(axis='both', which='major', labelsize=40)
    
    # Right panel: Signal
    # mask_sig_y = ~np.isnan(z_global_sig) & ~np.isnan(ySizesSig)
    # mask_sig_y = mask_sig_y & np.isfinite(z_global_sig) & np.isfinite(ySizesSig)
    
    hb = ax[1].hist2d(z_global_sig[mask_sig_y], ySizesSig[mask_sig_y], bins=[30, np.arange(0,14,1)], cmap=pastel_red_cmap)
    ax[1].figure.colorbar(hb[3],ax=ax[1])#    ax[1].colorbar()
    ax[1].set_title("Signal - Y Size", fontsize=63)
    ax[1].set_xlabel('z-global [mm]', fontsize=50)
    ax[1].set_ylabel('y-size (# pixels)', fontsize=50)
    ax[1].tick_params(axis='both', which='major', labelsize=40)
    
    closePlot(PLOT_DIR, interactivePlots,  "bib_signal_zglobal_vs_ysize_comparison.png")

# --- Plot 3: 2x2 grid showing both x-size and y-size comparisons ---
def plotZglobalXYsize(truthbib, truthsig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    if not( 'z-global' in truthbib.columns and 'z-global' in truthsig.columns ):
        raise Exception("z-global not present in truthbib or truthsig dataframes")
    assert len(truthbib) == len(ySizesBib)
    assert len(truthsig) == len(ySizesSig)

    pastel_red_cmap = create_pastel_red_cmap()

    z_global_bib = truthbib['z-global']
    z_global_sig = truthsig['z-global']

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    #Now fully just Eric's code

    print("TODO!!!! DECIDE WHETHER TO USE SEPARATE X AND Y MASKS")
    mask_bib_y = mask_bib
    mask_sig_y = mask_sig


    # Top row: X-size comparisons
    # Top-left: BIB X-size
    hb = axes[0,0].hist2d(z_global_bib[mask_bib], xSizesBib[mask_bib], bins=[30, np.arange(0,22,1)], cmap=pastel_red_cmap)
    axes[0,0].figure.colorbar(hb[3],ax=axes[0,0])#    axes[0,0].colorbar()
    axes[0,0].set_title("BIB - X Size", fontsize=38)
    axes[0,0].set_xlabel('z-global [mm]', fontsize=31)
    axes[0,0].set_ylabel('x-size (# pixels)', fontsize=31)
    axes[0,0].tick_params(axis='both', which='major', labelsize=23)
    
    # Top-right: Signal X-size
    hb = axes[0,1].hist2d(z_global_sig[mask_sig], xSizesSig[mask_sig], bins=[30, np.arange(0,22,1)], cmap=pastel_red_cmap)
    axes[0,1].figure.colorbar(hb[3],ax=axes[0,1])#    axes[0,1].colorbar()
    axes[0,1].set_title("Signal - X Size", fontsize=38)
    axes[0,1].set_xlabel('z-global [mm]', fontsize=31)
    axes[0,1].set_ylabel('x-size (# pixels)', fontsize=31)
    axes[0,1].tick_params(axis='both', which='major', labelsize=23)
    
    # Bottom row: Y-size comparisons
    # Bottom-left: BIB Y-size
    hb = axes[1,0].hist2d(z_global_bib[mask_bib_y], ySizesBib[mask_bib_y], bins=[30, np.arange(0,14,1)], cmap=pastel_red_cmap)
    axes[1,0].figure.colorbar(hb[3],ax=axes[1,0])#    axes[1,0].colorbar()
    axes[1,0].set_title("BIB - Y Size", fontsize=38)
    axes[1,0].set_xlabel('z-global [mm]', fontsize=31)
    axes[1,0].set_ylabel('y-size (# pixels)', fontsize=31)
    axes[1,0].tick_params(axis='both', which='major', labelsize=23)
    
    # Bottom-right: Signal Y-size
    hb = axes[1,1].hist2d(z_global_sig[mask_sig_y], ySizesSig[mask_sig_y], bins=[30, np.arange(0,14,1)], cmap=pastel_red_cmap)
    axes[1,1].figure.colorbar(hb[3],ax=axes[1,1])#    axes[1,1].colorbar()
    axes[1,1].set_title("Signal - Y Size", fontsize=38)
    axes[1,1].set_xlabel('z-global [mm]', fontsize=31)
    axes[1,1].set_ylabel('y-size (# pixels)', fontsize=31)
    axes[1,1].tick_params(axis='both', which='major', labelsize=23)
    
    closePlot(PLOT_DIR, interactivePlots,  "bib_signal_zglobal_vs_xysize_grid_comparison.png")



def ericsPlotReport(truthbib, truthsig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,PLOT_DIR="./plots"):
    z_global_bib = truthbib['z-global']
    z_global_sig = truthsig['z-global']
    print(f"Plots saved (seems like redundant printing):")
    print(f"1. X-size only: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_xsize_comparison_with_ysize.png')}")
    print("Actually I don't think that one is saved?")
    print(f"2. Y-size only: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_ysize_comparison.png')}")
    print(f"3. 2x2 grid: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_xysize_grid_comparison.png')}")
    
    print(f"\nBIB statistics:")
    if len(truthbib)>0:
        print(f"  z-global range: {z_global_bib.min():.2f} to {z_global_bib.max():.2f} mm")
        print(f"  x-size range: {xSizesBib.min()} to {xSizesBib.max()} pixels")
        print(f"  y-size range: {ySizesBib.min()} to {ySizesBib.max()} pixels")
    else:
        print("no bib, so not printing bib statisticst")
    
    print(f"\nSignal statistics:")
    if len(truthsig)>0:
        print(f"  z-global range: {z_global_sig.min():.2f} to {z_global_sig.max():.2f} mm")
        print(f"  x-size range: {xSizesSig.min()} to {xSizesSig.max()} pixels")
        print(f"  y-size range: {ySizesSig.min()} to {ySizesSig.max()} pixels")
    else:
        print("no signal, so not printing sig statisticst")

#Adapted from Eric's plot_signal_data.py

def genEtaAlphaBetaRq(truthDF):
    if 'z-global' in truthDF.columns:
        theta = np.arctan2(30, truthDF['z-global'])
        truthDF['eta'] = -np.log(np.tan(theta / 2))
  
    # cotAlpha, cotBeta may already be present, but recalculate if not
    if 'cotAlpha' not in truthDF.columns and {'n_x', 'n_z'}.issubset(truthDF.columns):
        truthDF['cotAlpha'] = truthDF['n_x'] / truthDF['n_z']
    if 'cotBeta' not in truthDF.columns and {'n_y', 'n_z'}.issubset(truthDF.columns):
        truthDF['cotBeta'] = truthDF['n_y'] / truthDF['n_z']
    if 'R' not in truthDF.columns:
        #added from Eliza's code
        truthDF['R'] = truthDF['pt']*5.36/(1.60217663*3.57)*1000 # [mm]
    if 'q' not in truthDF.columns:
        truthDF['q'] = truthDF['PID'].apply(lambda pid: PDGID(int(pid)).charge if pd.notnull(pid) else np.nan)
    if 'scalePion' not in truthDF.columns:
        #relative masses of pion and muon/electron, comes from pixelav https://github.com/elizahoward/pixelav/blob/30d7585448f87bcdf10f7f066005a04e4bd34a52/ppixelav2_list_trkpy_n_2f_custom.c#L358
        truthDF['m'] = truthDF['PID'].apply(lambda pid: 105.7 if abs(int(pid))==13 else 0.511 if abs(int(pid))==11 else np.nan) #MeV
        truthDF['scalePion'] = 139.57/truthDF['m']
        truthDF['p_calc1'] = np.sqrt(truthDF['n_x']*truthDF['n_x'] + truthDF['n_y']*truthDF['n_y'] +truthDF['n_z']*truthDF['n_z'])
        truthDF['p_calc1'] = truthDF['p_calc1'] / truthDF['scalePion']

        truthDF['p_calc2'] = truthDF['pt']*np.sqrt(1+ 1/ (truthDF['cotAlpha']*truthDF['cotAlpha'] + truthDF['cotBeta']*truthDF['cotBeta']) )
        truthDF['p_calc3'] = truthDF['pt']*np.sqrt(1+ 1/ (truthDF['cotAlpha']*truthDF['cotAlpha'] + truthDF['cotBeta']*truthDF['cotBeta']) )

    return truthDF

# --- Plot 1: cotAlpha, cotBeta, number_eh_pairs, nPixels ---
def plotEricVarsHistos(truthbib, truthsig,nPixelsSig,nPixelsBib,PLOT_DIR="./plots",interactivePlots=False):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

    ax[0,0].hist(truthsig['cotAlpha'], bins=40, histtype='step', color='g', align='mid', density=True, label="Signal")
    ax[0,0].hist(truthbib['cotAlpha'], bins=40, histtype='step', color='r', align='mid', density=True, label="BIB")
    ax[0,0].set_xlabel('cot(α)')
    ax[0,0].set_ylabel('Track Density')
    ax[0,0].set_xlim(-7.5, 7.5)
    ax[0,0].legend()

    ax[1,0].hist(truthsig['cotBeta'], bins=40, histtype='step', color='g', align='mid', density=True, label="Signal")
    ax[1,0].hist(truthbib['cotBeta'], bins=40, histtype='step', color='r', align='mid', density=True, label="BIB")
    ax[1,0].set_xlabel('cot(β)')
    ax[1,0].set_ylabel('Track Density')
    ax[1,0].set_xlim(-8, 8)
    ax[1,0].legend()

    ax[0,1].hist(truthsig['number_eh_pairs'], bins=40, histtype='step', color='g', align='mid', density=True, label="Signal")
    ax[0,1].hist(truthbib['number_eh_pairs'], bins=40, histtype='step', color='r', align='mid', density=True, label="BIB")
    ax[0,1].set_xlabel('Number of eh pairs')
    ax[0,1].set_ylabel('Track Density')
    ax[0,1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0e}'.format(x)))
    ax[0,1].set_xlim(0, 120000)
    ax[0,1].legend()

    # nPixels: number of nonzero pixels in each cluster
    
    ax[1,1].hist(nPixelsSig, bins=30, histtype='step', color='g', align='mid', density=True, label="Signal")
    ax[1,1].hist(nPixelsBib, bins=30, histtype='step', color='r', align='mid', density=True, label="BIB")
    ax[1,1].set_xlabel('Number of pixels')
    ax[1,1].set_ylabel('Track Density')
    ax[1,1].legend()

    closePlot(PLOT_DIR, interactivePlots, "signal_bib_summary_histograms.png")

# --- Plot 2: 2D histograms of eta vs x-size/y-size ---
def plotEtaXYsize(truthbib, truthsig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.03}, figsize=(10,7))

    # Before each hist2d call, filter out NaN and inf values for both axes
    mask_96 = ~np.isnan(truthsig['eta']) & ~np.isnan(xSizesSig)
    mask_96 = mask_96 & np.isfinite(truthsig['eta']) & np.isfinite(xSizesSig)
    hb = ax[0,1].hist2d(truthsig['eta'][mask_96], xSizesSig[mask_96], bins=[30, np.arange(0,22,1)], cmap='Blues')
    ax[0,1].figure.colorbar(hb[3],ax=ax[0,1])#    ax[0,1].colorbar()
    ax[0,1].set_title("Signal", fontsize=15)

    mask_98 = ~np.isnan(truthsig['eta']) & ~np.isnan(ySizesSig)
    mask_98 = mask_98 & np.isfinite(truthsig['eta']) & np.isfinite(ySizesSig)
    hb = ax[1,1].hist2d(truthsig['eta'][mask_98], ySizesSig[mask_98], bins=[30, np.arange(0,14,1)], cmap='Blues')
    ax[1,1].figure.colorbar(hb[3],ax=ax[1,1])#    ax[1,1].colorbar()
    ax[1,1].set_xlabel('η', fontsize=15)

    ax[0,0].set_ylabel('x-size (# pixels)', fontsize=15)
    ax[1,0].set_ylabel('y-size (# pixels)', fontsize=15)

    #Copying Eric's code but for bib
    # Before each hist2d call, filter out NaN and inf values for both axes
    mask_96 = ~np.isnan(truthbib['eta']) & ~np.isnan(xSizesBib)
    mask_96 = mask_96 & np.isfinite(truthbib['eta']) & np.isfinite(xSizesBib)
    hb = ax[0,0].hist2d(truthbib['eta'][mask_96], xSizesBib[mask_96], bins=[30, np.arange(0,22,1)], cmap='Blues')
    ax[0,0].figure.colorbar(hb[3],ax=ax[0,0])#    ax[0,0].colorbar()
    ax[0,0].set_title("BIB", fontsize=15)

    mask_98 = ~np.isnan(truthbib['eta']) & ~np.isnan(ySizesBib)
    mask_98 = mask_98 & np.isfinite(truthbib['eta']) & np.isfinite(ySizesBib)
    hb = ax[1,0].hist2d(truthbib['eta'][mask_98], ySizesBib[mask_98], bins=[30, np.arange(0,14,1)], cmap='Blues')
    ax[1,0].figure.colorbar(hb[3],ax=ax[1,0])#    ax[1,0].colorbar()
    ax[1,0].set_xlabel('η', fontsize=15)

    closePlot(PLOT_DIR, interactivePlots, "signal_eta_vs_size_2d.png")

# --- Plot 3: 2D histograms of y-local vs x-size/y-size ---
def plotYlocalXYsize(truthbib, truthsig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.03}, figsize=(10,7))

    # Before each hist2d call, filter out NaN and inf values for both axes
    mask_111 = ~np.isnan(truthsig['y-local']) & ~np.isnan(xSizesSig)
    mask_111 = mask_111 & np.isfinite(truthsig['y-local']) & np.isfinite(xSizesSig)
    hb = ax[0,1].hist2d(truthsig['y-local'][mask_111], xSizesSig[mask_111], bins=[30, np.arange(0,22,1)], cmap='Blues')
    ax[0,1].figure.colorbar(hb[3],ax=ax[0,1])#    ax[0,1].colorbar()
    ax[0,1].set_title("Signal", fontsize=15)

    mask_113 = ~np.isnan(truthsig['y-local']) & ~np.isnan(ySizesSig)
    mask_113 = mask_113 & np.isfinite(truthsig['y-local']) & np.isfinite(ySizesSig)
    hb = ax[1,1].hist2d(truthsig['y-local'][mask_113], ySizesSig[mask_113], bins=[30, np.arange(0,14,1)], cmap='Blues')
    ax[1,1].figure.colorbar(hb[3],ax=ax[1,1])#    ax[1,1].colorbar()
    ax[1,1].set_xlabel('y-local [μm]', fontsize=15)

    ax[0,0].set_ylabel('x-size (# pixels)', fontsize=15)
    ax[1,0].set_ylabel('y-size (# pixels)', fontsize=15)

    


    mask_111 = ~np.isnan(truthbib['y-local']) & ~np.isnan(xSizesBib)
    mask_111 = mask_111 & np.isfinite(truthbib['y-local']) & np.isfinite(xSizesBib)
    hb = ax[0,0].hist2d(truthbib['y-local'][mask_111], xSizesBib[mask_111], bins=[30, np.arange(0,22,1)], cmap='Blues')
    ax[0,0].figure.colorbar(hb[3],ax=ax[0,0])#    ax[0,0].colorbar()
    ax[0,0].set_title("BIB", fontsize=15)

    mask_113 = ~np.isnan(truthbib['y-local']) & ~np.isnan(ySizesBib)
    mask_113 = mask_113 & np.isfinite(truthbib['y-local']) & np.isfinite(ySizesBib)
    hb = ax[1,0].hist2d(truthbib['y-local'][mask_113], ySizesBib[mask_113], bins=[30, np.arange(0,14,1)], cmap='Blues')
    ax[1,0].figure.colorbar(hb[3],ax=ax[1,0])#    ax[1,0].colorbar()
    ax[1,0].set_xlabel('y-local [μm]', fontsize=15)


    closePlot(PLOT_DIR, interactivePlots, "signal_ylocal_vs_size_2d.png")

# --- Plot 4: 2D histogram of number_eh_pairs vs pt ---
def plotEhPt(truthbib, truthsig, mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    # Before each hist2d call, filter out NaN and inf values for both axes
    mask_125 = ~np.isnan(truthsig['number_eh_pairs']) & ~np.isnan(truthsig['pt'])
    mask_125 = mask_125 & np.isfinite(truthsig['number_eh_pairs']) & np.isfinite(truthsig['pt'])
    hb = ax[1].hist2d(truthsig['number_eh_pairs'][mask_125], truthsig['pt'][mask_125], bins=30, cmap='Blues')
    ax[1].figure.colorbar(hb[3],ax=ax[1])#    ax[1].colorbar()
    ax[1].set_title("Signal", fontsize=15)
    ax[1].set_ylabel('pt (GeV)', fontsize=15)
    ax[1].set_xlabel('number of electron hole pairs', fontsize=15)

    mask_125 = ~np.isnan(truthbib['number_eh_pairs']) & ~np.isnan(truthbib['pt'])
    mask_125 = mask_125 & np.isfinite(truthbib['number_eh_pairs']) & np.isfinite(truthbib['pt'])
    hb = ax[0].hist2d(truthbib['number_eh_pairs'][mask_125], truthbib['pt'][mask_125], bins=30, cmap='Blues')
    ax[0].figure.colorbar(hb[3],ax=ax[0])#    ax[0].colorbar()
    ax[0].set_title("BIB", fontsize=15)
    ax[0].set_ylabel('pt (GeV)', fontsize=15)
    ax[0].set_xlabel('number of electron hole pairs', fontsize=15)


    closePlot(PLOT_DIR, interactivePlots,  "signal_ehpairs_vs_pt_2d.png")

# --- Plot 5: charge separation for low and high pt ---
def plotPtLowHigh(truthbib, truthsig, mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    # truthsig['q'] = truthsig['PID'].apply(lambda pid: PDGID(int(pid)).charge if pd.notnull(pid) else np.nan)
    # truthbib['q'] = truthbib['PID'].apply(lambda pid: PDGID(int(pid)).charge if pd.notnull(pid) else np.nan)
    # Low pt (<5 GeV) and high pt (>95 GeV)
    truthSigLow = truthsig[truthsig['pt'] < 5].copy()
    truthSigHigh = truthsig[truthsig['pt'] > 95].copy()

    truthSigLowPos = truthSigLow[truthSigLow['q'] > 0]
    truthSigLowNeg = truthSigLow[truthSigLow['q'] < 0]
    truthSigHighPos = truthSigHigh[truthSigHigh['q'] > 0]
    truthSigHighNeg = truthSigHigh[truthSigHigh['q'] < 0]

    truthBibLow = truthbib[truthbib['pt'] < 5].copy()
    truthBibHigh = truthbib[truthbib['pt'] > 95].copy()

    truthBibLowPos = truthBibLow[truthBibLow['q'] > 0]
    truthBibLowNeg = truthBibLow[truthBibLow['q'] < 0]
    truthBibHighPos = truthBibHigh[truthBibHigh['q'] > 0]
    truthBibHighNeg = truthBibHigh[truthBibHigh['q'] < 0]

    # Example: plot pt distributions for low/high pt, positive/negative charge
    fig, ax = plt.subplots(2,2,figsize=(12,5))
    ax[0,1].hist(truthSigLowPos['pt'], bins=30, alpha=0.7, label='Low pt, q>0')
    ax[0,1].hist(truthSigLowNeg['pt'], bins=30, alpha=0.7, label='Low pt, q<0')
    ax[0,1].set_title('Sig Low pt (<5 GeV)')
    ax[0,1].set_xlabel('pt (GeV)')
    ax[0,1].set_ylabel('Tracks')
    ax[0,1].legend()

    ax[1,1].hist(truthSigHighPos['pt'], bins=30, alpha=0.7, label='High pt, q>0')
    ax[1,1].hist(truthSigHighNeg['pt'], bins=30, alpha=0.7, label='High pt, q<0')
    ax[1,1].set_title('Sig High pt (>95 GeV)')
    ax[1,1].set_xlabel('pt (GeV)')
    ax[1,1].set_ylabel('Tracks')
    ax[1,1].legend()

    

    ax[0,0].hist(truthBibLowPos['pt'], bins=30, alpha=0.7, label='Low pt, q>0')
    ax[0,0].hist(truthBibLowNeg['pt'], bins=30, alpha=0.7, label='Low pt, q<0')
    ax[0,0].set_title('Bib Low pt (<5 GeV)')
    ax[0,0].set_xlabel('pt (GeV)')
    ax[0,0].set_ylabel('Tracks')
    ax[0,0].legend()

    ax[1,0].hist(truthBibHighPos['pt'], bins=30, alpha=0.7, label='High pt, q>0')
    ax[1,0].hist(truthBibHighNeg['pt'], bins=30, alpha=0.7, label='High pt, q<0')
    ax[1,0].set_title('Bib High pt (>95 GeV)')
    ax[1,0].set_xlabel('pt (GeV)')
    ax[1,0].set_ylabel('Tracks')
    ax[1,0].legend()



    closePlot(PLOT_DIR, interactivePlots, "signal_pt_charge_separation.png")


#Added plots from Eliza's code (several are also just wrapped into Eric's code)

def plotRadius(truthbib, truthsig,PLOT_DIR="./plots",interactivePlots=False):
    #modified to be on same axis
    plt.figure(figsize=(8,5))
    plt.subplot(211)
    bins = np.arange(0,25,2) #If you use this, need to clip for underflow/overflow
    # bins = 25
    plt.hist(np.clip(truthbib['R'],bins[0],bins[-1]), histtype='step', bins = bins,label="BIB")
    plt.hist(np.clip(truthsig['R'],bins[0],bins[-1]), histtype='step', bins = bins,label="Signal")
    plt.legend()
    plt.title("Radius of track curvature")
    plt.xlabel("Radius [mm]")
    plt.ylabel("Tracks")
    plt.subplot(212)
    plt.plot(truthbib['R'],truthbib['pt'],label="BIB")
    plt.plot(truthsig['R'],truthsig['pt'],label="Signal",alpha=0.7)
    plt.legend()
    plt.title("Radius of track curvature vs. pt")
    plt.xlabel("Radius [mm]")
    plt.ylabel("pT [GeV]")
    closePlot(PLOT_DIR, interactivePlots,  "radiusPlot.png")

#Some of these may be redundant
def getYProfiles(clusters):
    profiles = np.sum(clusters, axis = 2)
    #totalCharge = np.sum(profiles, axis = 1, keepdims=True)
    return profiles

def getAverageYProfile(clusters):
    profiles=getYProfiles(clusters)
    return np.mean(profiles, axis=0)

def getXProfiles(clusters):
    profiles = np.sum(clusters, axis = 1)
    #totalCharge = np.sum(profiles, axis = 1, keepdims=True)
    return profiles

def getAverageXProfile(clusters):
    profiles=getXProfiles(clusters)
    return np.mean(profiles, axis=0)

def getClusterYSizes(clusters):
    profiles=getYProfiles(clusters)
    bool_arr = profiles != 0
    return np.sum(bool_arr, axis = 1)

def getAverageClusterYSize(clusters):
    clusterSizes = getClusterYSizes(clusters)
    return np.mean(clusterSizes)

def getClusterXSizes(clusters):
    profiles=getXProfiles(clusters)
    bool_arr = profiles != 0
    return np.sum(bool_arr, axis = 1)

def getAverageClusterXSize(clusters):
    clusterSizes = getClusterXSizes(clusters)
    return np.mean(clusterSizes)


def plotYprofileYlocalRange(
        # yProfileLowPtPosLowYl,yProfileLowPtNegLowYl,yProfileHighPtPosLowYl,yProfileHighPtNegLowYl,
        # yProfileLowPtPosMidYl,yProfileLowPtNegMidYl,yProfileHighPtPosMidYl,yProfileHighPtNegMidYl,
        # yProfileLowPtPosHighYl,yProfileLowPtNegHighYl,yProfileHighPtPosHighYl,yProfileHighPtNegHighYl,
        avgClustDict,
        titleBibSig = "Signal",
        PLOT_DIR="./plots",interactivePlots=False
        ):
    yaxis=np.arange(1,14,1)
    fig, ax = plt.subplots(1,3, sharey=True, figsize=(18,5))
    ax[0].step(yaxis,avgClustDict["yProfileLowPtPosLowYl"], where="mid", label="Low pT (pos)", c ='r')
    ax[0].step(yaxis,avgClustDict["yProfileLowPtNegLowYl"], where="mid", label="Low pT (neg)", c='b')
    ax[0].step(yaxis,avgClustDict["yProfileHighPtPosLowYl"], where="mid", label="High pT (pos)", c='k')
    ax[0].step(yaxis,avgClustDict["yProfileHighPtNegLowYl"], where="mid", label="High pT (neg)", c='k')
    ax[0].legend()
    ax[0].set_title("-4.5 mm < y-local < -2 mm")
    ax[0].set_ylabel("Average charge in cluster y profile")
    ax[0].set_xlabel("y [pixels]")

    ax[1].step(yaxis,avgClustDict["yProfileLowPtPosMidYl"], where="mid", label="Low pT (pos)", c ='r')
    ax[1].step(yaxis,avgClustDict["yProfileLowPtNegMidYl"], where="mid", label="Low pT (neg)", c='b')
    ax[1].step(yaxis,avgClustDict["yProfileHighPtPosMidYl"], where="mid", label="High pT (pos)", c='k')
    ax[1].step(yaxis,avgClustDict["yProfileHighPtNegMidYl"], where="mid", label="High pT (neg)", c='k')
    ax[1].legend()
    ax[1].set_title("0 mm < y-local < 2 mm")
    ax[1].set_xlabel("y [pixels]")

    ax[2].step(yaxis,avgClustDict["yProfileLowPtPosHighYl"], where="mid", label="Low pT (pos)", c ='r')
    ax[2].step(yaxis,avgClustDict["yProfileLowPtNegHighYl"], where="mid", label="Low pT (neg)", c='b')
    ax[2].step(yaxis,avgClustDict["yProfileHighPtPosHighYl"], where="mid", label="High pT (pos)", c='k')
    ax[2].step(yaxis,avgClustDict["yProfileHighPtNegHighYl"], where="mid", label="High pT (neg)", c='k')
    ax[2].legend()
    ax[2].set_title("6 mm < y-local < 8.5 mm")
    ax[2].set_xlabel("y [pixels]")

    fig.suptitle(f'Average y-profiles for different ranges of y-local {titleBibSig}', fontsize=15)
    closePlot(PLOT_DIR, interactivePlots,  f"{titleBibSig}yprofileVsYlocalRange.png")

def plotYprofileYZRange(
        # yProfileLowZgl,yProfileMidZgl,yProfileHighZgl,yProfileLowYl,yProfileMidYl,yProfileHighYl, 
        avgClustDictBib, avgClustDictSig,       
        PLOT_DIR="./plots",interactivePlots=False
):
    yaxis=np.arange(1,14,1)
    fig, ax = plt.subplots(2,2,figsize=(15,7))
    ax[0,0].step(yaxis,avgClustDictBib["yProfileLowZgl"], where="mid", label="z-global \u2208 [0,20] mm")
    ax[0,0].step(yaxis,avgClustDictBib["yProfileMidZgl"], where="mid", label="z-global \u2208 [20,40] mm")
    ax[0,0].step(yaxis,avgClustDictBib["yProfileHighZgl"], where="mid", label="z-global \u2208 [40,65] mm")
    ax[0,0].legend()
    ax[0,0].set_ylabel("Average charge in cluster y profile")
    ax[0,0].set_xlabel("y [pixels]")
    ax[0,0].set_title("BIB", fontsize=15)


    ax[1,0].step(yaxis,avgClustDictBib["yProfileLowYl"], where="mid", label="y-local \u2208 [-4.5,-2] mm")
    ax[1,0].step(yaxis,avgClustDictBib["yProfileMidYl"], where="mid", label="y-local \u2208 [0,2] mm")
    ax[1,0].step(yaxis,avgClustDictBib["yProfileHighYl"], where="mid", label="y-local \u2208 [6,8.5] mm")
    ax[1,0].legend()
    ax[1,0].set_ylabel("Average charge in cluster y profile")
    ax[1,0].set_xlabel("y [pixels]")


    ax[0,1].step(yaxis,avgClustDictSig["yProfileLowZgl"], where="mid", label="z-global \u2208 [0,20] mm")
    ax[0,1].step(yaxis,avgClustDictSig["yProfileMidZgl"], where="mid", label="z-global \u2208 [20,40] mm")
    ax[0,1].step(yaxis,avgClustDictSig["yProfileHighZgl"], where="mid", label="z-global \u2208 [40,65] mm")
    ax[0,1].legend()
    ax[0,1].set_ylabel("Average charge in cluster y profile")
    ax[0,1].set_xlabel("y [pixels]")
    ax[0,1].set_title("Signal", fontsize=15)


    ax[1,1].step(yaxis,avgClustDictSig["yProfileLowYl"], where="mid", label="y-local \u2208 [-4.5,-2] mm")
    ax[1,1].step(yaxis,avgClustDictSig["yProfileMidYl"], where="mid", label="y-local \u2208 [0,2] mm")
    ax[1,1].step(yaxis,avgClustDictSig["yProfileHighYl"], where="mid", label="y-local \u2208 [6,8.5] mm")
    ax[1,1].legend()
    ax[1,1].set_ylabel("Average charge in cluster y profile")
    ax[1,1].set_xlabel("y [pixels]")

    
    closePlot(PLOT_DIR, interactivePlots,  f"signal_bib_allYprofilesInYlocalZglobalRanges.png")


def clusterYSizeVsYlocal(truth):
    clusterSize=[]
    ylocals=np.arange(-4.5,9,1)
    intervals = np.arange(-5,9.5,1)
    for i in range(len(intervals)-1):
        cut1 = truth['y-local']>=intervals[i]
        cut2 = truth['y-local']<intervals[i+1]
        cut = cut1 & cut2
        clusterThisSize = np.mean(truth[cut]["ySize"])
        clusterSize.append(clusterThisSize)
    return ylocals,clusterSize

def plotClusterYSizes(
        truth_,
        titleBibSig = "Signal",
        PLOT_DIR="./plots",interactivePlots=False):
    ylocals,lowPosSize = clusterYSizeVsYlocal(truth_.query("pt < 5 and q > 0"))
    ylocals,lowNegSize = clusterYSizeVsYlocal(truth_.query("pt < 5 and q < 0"))
    ylocals,highSize = clusterYSizeVsYlocal(truth_.query("pt > 95"))
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(ylocals, lowPosSize, c='r', label="Low pT (pos)")
    ax.scatter(ylocals, lowNegSize, c='b', label="Low pT (neg)")
    ax.scatter(ylocals, highSize, c='k', label="High pT")
    ax.set_xlabel("y-local [mm]")
    ax.set_ylabel("Cluster y sizes [pixels]")
    ax.legend()
    ax.set_title(f"{titleBibSig}")
    closePlot(PLOT_DIR, interactivePlots,  f"{titleBibSig}clusterYSizes.png")

def plotXYProfile(truthBib, truthSig, avgClustDictSig, avgClustDictBib,
        PLOT_DIR="./plots",interactivePlots=False):
    fig, ax=plt.subplots(ncols=2, nrows=2, figsize=(10,8))
    xaxis = np.arange(1,22)
    yaxis=np.arange(1,14,1)
    ax[0,0].step(yaxis,avgClustDictSig["yProfile"], where="mid", label="Signal", c ='g')
    ax[0,0].step(yaxis,avgClustDictBib["yProfile"], where="mid", label="Bib", c ='purple')
    ax[0,0].legend()
    ax[0,0].set_xlabel("y-pixels")
    ax[0,0].set_ylabel("Total charge collected")
    ax[0,0].set_title("Average y-profile Comaprison")

    ax[0,1].step(xaxis,avgClustDictSig["xProfile"], where="mid", label="Signal", c ='g')
    ax[0,1].step(xaxis,avgClustDictBib["xProfile"], where="mid", label="Bib", c ='purple')
    ax[0,1].legend()
    ax[0,1].set_xlabel("x-pixels")
    ax[0,1].set_ylabel("Total charge collected")
    ax[0,1].set_title("Average x-profile Comaprison")

    ax[1,0].hist(truthSig["ySize"], bins=np.arange(0,14,1), histtype='step', density=True, label="Signal", color ='g')
    ax[1,0].hist(truthBib["ySize"], bins=np.arange(0,14,1), histtype='step', density=True, label="Bib", color ='purple')
    ax[1,0].legend()
    ax[1,0].set_xlabel("Cluster y-size [# pixels]")
    ax[1,0].set_ylabel("Track Density")
    ax[1,0].set_title("Cluster y-size Comaprison")

    ax[1,1].hist(truthSig["xSize"], bins=np.arange(0,22,1), histtype='step', density=True, label="Signal", color ='g')
    ax[1,1].hist(truthBib["xSize"], bins=np.arange(0,22,1), histtype='step', density=True, label="Bib", color ='purple')
    ax[1,1].legend()
    ax[1,1].set_xlabel("Cluster x-size [# pixels]")
    ax[1,1].set_ylabel("Track Density")
    ax[1,1].set_title("Cluster x-size Comaprison")

    # fig.tight_layout()

    closePlot(PLOT_DIR, interactivePlots,  f"XYSizeProfiless.png")


########################
# Utilities for plotting tracklists
########################


trackDirBib_mm = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mm/BIB_tracklists/'
trackDirBib_mp = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mp/BIB_tracklists/'
trackDirSig = '/local/d1/smartpixML/bigData/tracklists/signal_tracklists'
#I think these are newest 
trackHeader = ["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg"]
def loadTrackData(directory, trackHeader = trackHeader, bibSigIndic="_"):
    tracks = [
        pd.read_csv(os.path.join(directory, file), sep=' ', header=None, names=trackHeader)
        for file in os.listdir(directory) if ((("_tracklist" in file) or ("_tracks" in file)) and (bibSigIndic in file))
    ]
    if len(tracks)==0:
        print("\nWARNING!!")
        print("There are no tracklists with the indicator ",bibSigIndic)
        print("Warning!!!!\n\n")
        return pd.DataFrame(columns=trackHeader)
    else:
        return pd.concat(tracks)
# def load_log_data(directory,log_header=logHeader):
#     return pd.concat([
#         pd.read_csv(os.path.join(directory, file), sep=' ', header=None, names=log_header)
#         for file in os.listdir(directory) if "_log" in file
#     ])
#can also pass in None for the directories
def loadAllTracks(trackDirBib_mm=trackDirBib_mm,trackDirBib_mp=trackDirBib_mp,trackDirSig=trackDirSig,useBibSigIndic=True):
    if useBibSigIndic:
        tracksBib_mm = loadTrackData(trackDirBib_mm,bibSigIndic="bib_mm")  if trackDirBib_mm else pd.DataFrame(columns=trackHeader)
        tracksBib_mp = loadTrackData(trackDirBib_mp,bibSigIndic="bib_mp")  if trackDirBib_mp else pd.DataFrame(columns=trackHeader)
        tracksSig = loadTrackData(trackDirSig,bibSigIndic="signal") if trackDirSig else pd.DataFrame(columns=trackHeader)
    else:
        tracksBib_mm = loadTrackData(trackDirBib_mm)  if trackDirBib_mm else pd.DataFrame(columns=trackHeader)
        tracksBib_mp = loadTrackData(trackDirBib_mp)  if trackDirBib_mp else pd.DataFrame(columns=trackHeader)
        tracksSig = loadTrackData(trackDirSig) if trackDirSig else pd.DataFrame(columns=trackHeader)
    tracksBib = pd.concat([tracksBib_mm,tracksBib_mp])
    return tracksBib, tracksSig, tracksBib_mp,tracksBib_mm

def plotTrackPPt(tracksBib, tracksSig,binsBib=30,binsSig=30,yscale='log',PLOT_DIR="./plots",interactivePlots=False):
    fig, ax=plt.subplots(ncols=2, nrows=1, figsize=(10,5))
    plt.subplot(121)
    plt.hist(tracksBib["p"],label="p",bins=binsBib)
    plt.hist(tracksBib["pt"], label="pt",alpha=0.7,bins=binsBib)
    print("FIX BINS!")
    plt.title("BIB tracklists, p and pt")
    plt.legend()
    plt.yscale(yscale)
    plt.ylabel("N tracks")
    plt.xlabel("momentum (GeV)")

    plt.subplot(122)
    plt.hist(tracksSig["p"],label="p",bins=binsSig)
    plt.hist(tracksSig["pt"], label="pt",alpha=0.7,bins=binsSig)
    plt.title("Sig tracklists, p and pt")
    plt.legend()
    plt.xlabel("momentum (GeV)")
    plt.ylabel("N tracks")
    plt.yscale(yscale)

    # fig.tight_layout()

    closePlot(PLOT_DIR, interactivePlots,  f"TrackPPt.png")


def plotPtTrackAndParquet(tracksBib, tracksSig,truthBib, truthSig,PLOT_DIR="./plots",interactivePlots=False):
    key = "pt"
    binsBib = 30
    binsSig = 30
    plotKeyTrackParquet(tracksBib, tracksSig,truthBib, truthSig,key,binsBib=binsBib, binsSig=binsSig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots,xlabel="pT (GeV)")

def plotPCalcTrackComparison(tracksDF,bibSigLabel="BIBORSIG",PLOT_DIR="./plots",interactivePlots=False):
    # z = 1./np.sqrt((1.+tracksDF["cotb"]*tracksDF["cotb"]+tracksDF["cota"]*tracksDF["cota"]))
    # x = z*tracksDF["cota"]
    # y = z*tracksDF["cotb"]
    x = tracksDF['x']
    y = tracksDF['y']
    z = tracksDF['z']

    p = tracksDF["pt"] / np.sqrt((z**2 +y**2)/(x**2 +y**2 +z**2 ))

    fig, ax=plt.subplots(ncols=2, nrows=1, figsize=(10,5))
    plt.subplot(121)    
    plt.hist(p,label="p recalculated based on cota, cotb, pt, in the tracklists")
    plt.hist(tracksDF["p"],label="p directly as saved in tracklists",alpha = 0.5)
    plt.yscale('log')
    plt.title(f"{bibSigLabel}")
    plt.legend()
    plt.ylabel("N tracks")
    plt.xlabel("Momentum (GeV)")

    plt.subplot(122)
    plt.hist(p - tracksDF["p"])
    plt.title(f"Difference between p saved in {bibSigLabel} tracklists and \n p recalculated from cota, cotb, pt saved in tracklists")
    plt.yscale('log')
    plt.ylabel("N tracks")
    plt.xlabel('Momentum - momentum = "0" (GeV)')


    closePlot(PLOT_DIR, interactivePlots,  f"TrackPCalcComparison{bibSigLabel}.png")

def calcNxyzTrack(tracksDF,showUnitVerification=False):
    # if len(tracksDF)==0:
    #     print("tracklist is empty, so no calculation")
    #     return tracksDF
    z = 1./np.sqrt((1.+tracksDF["cotb"]*tracksDF["cotb"]+tracksDF["cota"]*tracksDF["cota"])) #locdir[2] https://github.com/elizahoward/pixelav/blob/30d7585448f87bcdf10f7f066005a04e4bd34a52/ppixelav2_list_trkpy_n_2f_custom.c#L341
    flipCoefficient = (np.array(tracksDF["flp"] == 0)*2-1)*-1
    z=flipCoefficient*z
    x = z*tracksDF["cota"] #locdir[0]
    y = z*tracksDF["cotb"] #locdir[1]
    qq = x**2 +y**2 +z**2 

    unitHistBins = np.linspace(0,2,20)
    counts, bins,_ = plt.hist(np.clip(qq,unitHistBins[0],unitHistBins[-1]),bins=unitHistBins)
    plt.title("Magnitude of a vector, should be 1")
    if showUnitVerification:
        print(counts)
        print(bins)
        print(unitHistBins)
        plt.show()
    else:
        plt.close()
    assert np.all(bins==unitHistBins)
    Most0 = counts==0;
    if len(tracksDF)!=0:
        assert not Most0[9]
        Most0[9] = True
        assert np.all(Most0)


    #then pixelAV
    tracksDF['m'] = tracksDF['hit_pdg'].apply(lambda pid: 105.7 if abs(int(pid))==13 else 0.511 if abs(int(pid))==11 else np.nan) #MeV
    if len(tracksDF)!=0:
        print(f"Nan m count: {np.count_nonzero(np.isnan(tracksDF['m']))}")
    tracksDF['scalePion'] = 139.57/tracksDF['m']
    tracksDF['n_x'] = x*tracksDF['p']*tracksDF['scalePion']
    tracksDF['n_y'] = y*tracksDF['p']*tracksDF['scalePion']
    tracksDF['n_z'] = z*tracksDF['p']*tracksDF['scalePion']

    tracksDF['x'] = x
    tracksDF['y'] = y
    tracksDF['z'] = z
    return tracksDF

def plotNxyzTrackParquet(tracksBib, tracksSig,truthBib, truthSig,PLOT_DIR="./plots",interactivePlots=False):
    binsBib = 30
    binsSig = 30
    recalcStr = "(recalculated)"
    print("FIX BINS!")
    fig, ax=plt.subplots(ncols=2, nrows=3, figsize=(10,13))

    key = "n_x"
    plotKeyTrackParquet(tracksBib, tracksSig,truthBib, truthSig,key,binsBib=binsBib, binsSig=binsSig,recalcStr=recalcStr,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots,isSubplot=True,subplots=[321,322],xlabel = "momentum*scalePion (GeV*scalePion)")
    key = "n_y"
    plotKeyTrackParquet(tracksBib, tracksSig,truthBib, truthSig,key,binsBib=binsBib, binsSig=binsSig,recalcStr=recalcStr,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots,isSubplot=True,subplots=[323,324],xlabel = "momentum*scalePion (GeV*scalePion)")
    key = "n_z"
    plotKeyTrackParquet(tracksBib, tracksSig,truthBib, truthSig,key,binsBib=binsBib, binsSig=binsSig,recalcStr=recalcStr,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots,isSubplot=True,subplots=[325,326],xlabel = "momentum*scalePion (GeV*scalePion)")
    closePlot(PLOT_DIR, interactivePlots, "TrackParquet_nxnynz.png")
def plotKeyTrackParquet(tracksBib, tracksSig,truthBib, truthSig,key,binsBib=30, binsSig=30, recalcStr = "",
                        PLOT_DIR="./plots",interactivePlots=False,isSubplot=False,subplots=[],xlabel = ""):
    if isSubplot and len(subplots) ==0:
        raise ValueError("if using this method for subplots, need to have list of subplots")
    if not isSubplot:
        fig, ax=plt.subplots(ncols=2, nrows=1, figsize=(10,5))
        subplots = [121, 122]
    plt.subplot(subplots[0])
    plt.hist(truthBib[key],bins=binsBib,label=f"{key} from parquet")
    plt.hist(tracksBib[key],bins=binsBib,label=f"{key} from track {recalcStr}",alpha=0.7)
    plt.title(f"BIB {key} comparison tracklists to parquets")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("N tracks")
    plt.yscale('log')
    
    plt.subplot(subplots[1])
    plt.hist(truthSig[key],bins=binsSig,label=f"{key} from parquet")
    plt.hist(tracksSig[key],bins=binsSig,label=f"{key} from track {recalcStr}",alpha=0.7)
    plt.title(f"Signal {key} comparison tracklists to parquets")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("N tracks")
    plt.yscale('log')
    if not isSubplot:
        closePlot(PLOT_DIR, interactivePlots,  f"TrackParquet{key}.png")