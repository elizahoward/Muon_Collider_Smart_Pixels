"""
Author: Daniel Abadjiev
Date: July 22, 2026
Description: Script to make plots characterizing model performance for the paper
Inspired by varPredPlotRunner.py, but modified to just get the useful ones
"""
import sys
sys.path.append("../daniel")
sys.path.append("../daniel/validationPlots")
import varPredPlotUtils
from plotUtils import prepHistBins, closePlot,plotManyHisto
import os
import argparse
from pathlib import Path
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

PLOT_DIR = "./evaluationPlots"
interactivePlots = False
styleSheet = "seaborn-v0_8-colorblind"
N_CPU = 1
loadPredVarPkl = True

# if not loadPredVarPkl:
#     import tensorflow as tf 
#     tf.config.set_visible_devices([], 'GPU')

paths = [
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model2.5_fin_results_model2_5_10bit_normalised_selected__model_trial_057.h5",
    ]

def runForPath(path):
    modelID = path[-10:]
    print(path)
    print(modelID)
    pltDir = PLOT_DIR+"/mdl_"+modelID
    Path(pltDir).mkdir(parents=True, exist_ok=True) #moved to runModelPlots so that it can modify based on backround rejection. But for this version want it to be able to run without that
    if loadPredVarPkl:
        try:
            predVarDF = pd.read_pickle(Path(pltDir).joinpath("predVarDF.pkl"))
            with open(Path(pltDir).joinpath("threshVal.pkl"), 'rb') as file:
                threshVal = pickle.load(file)
        except Exception as e:
            print(e)
            raise Exception("You may have to first save pkls before you can read them\nHint: set loadPredVarPkl to False")
            
    else:
        predVarDF, model, predictions, threshVal = varPredPlotUtils.runModelPlots(filepath = path,PLOT_DIR=pltDir, interactivePlots=interactivePlots,extendTitle=path[25:])   
        predVarDF.to_pickle(Path(pltDir).joinpath("predVarDF.pkl"))
        with open(Path(pltDir).joinpath("threshVal.pkl"), 'wb') as file:
            pickle.dump(threshVal, file)

    plotAll1dHists(predVarDF,threshVal,pltDir)


def plotAll1dHists(predVarDF,threshVal,pltDir):
    print(predVarDF)
    print(predVarDF.keys())
    histoKarri(predVarDF,threshVal,pltDir,key="z-global",keyLabel="z-global [mm]",figsize=(5,10),bins=100)
    histoKarri(predVarDF,threshVal,pltDir,key="pt",keyLabel=r"Transverse Momentum $p_T$ [GeV/c]",figsize=(5,10),bins=100)
    histoKarri(predVarDF,threshVal,pltDir,key="y-local",keyLabel="y-local [mm] aaaaah I can't find a good binnning",figsize=(5,10),bins=25)
    histoKarri(predVarDF,threshVal,pltDir,key="xSize",keyLabel="x-Size [# pixels]",bins=np.arange(0,22,1),figsize=(5,10),locLegend="upper right")
    histoKarri(predVarDF,threshVal,pltDir,key="ySize",keyLabel="y-Size [# pixels]",bins=np.arange(0,14,1),figsize=(5,11),locLegend="upper right")
    histoKarri(predVarDF,threshVal,pltDir,key="nModule",keyLabel="Module Number (longitudinally counted)",bins=12,figsize=(5,10))
    histoKarri(predVarDF,threshVal,pltDir,key="nPix",keyLabel="Number of Pixels",bins=np.arange(0,np.max(predVarDF["nPix"]),1),figsize=(5,10),locLegend="upper right")
    print("finished 1d histograms")

def histoKarri(predVarDF,cut,pltDir,key="z-global",keyLabel="",figsize=(5,10),bins="auto",yscale="log",locLegend = "best"):
    configsAll = [
        (predVarDF, "all vectors"),
        (predVarDF.query("trueY == 0"), "all BIB"),
        (predVarDF.query("trueY == 1"), "all Signal"),
        (predVarDF.query("prediction > @cut"), "all vectors accepted by model"),
        (predVarDF.query("trueY == 0 and prediction > @cut"), "all BIB accepted by model"),
        (predVarDF.query("trueY == 1 and prediction > @cut"), "all Signal accepted by model"),
    ]
    configsSig = [
        (predVarDF.query("trueY == 1"), "all Signal"),
        (predVarDF.query("trueY == 1 and prediction > @cut"), "all Signal accepted by model"),
    ]
    configsBib = [
        (predVarDF.query("trueY == 0"), "all BIB"),
        (predVarDF.query("trueY == 0 and prediction > @cut"), "all BIB accepted by model"),
    ]
    plt.figure(figsize=figsize)
    plt.subplot(311)
    #def plotManyHisto(arrs,bins=None,postScale=1,title="",pltLabels=["1","2","3"],pltStandalone=True,showNums=False,
    #               figsize=(7,3),yscale="linear",xlabel="",ylabel="Tracks",
    #               PLOT_DIR=None,interactivePlots=None,saveTitle=None,alphas = None,legendLoc = "best"):
    plotManyHisto(arrs=[df_subset[key] for (df_subset, title) in configsAll],
                  bins=bins,title="All vectors",pltLabels =[title for (df_subset, title) in configsAll],
                  pltStandalone=False,yscale=yscale,xlabel=keyLabel,ylabel="N",legendLoc=locLegend,
                  alphas = [0.7 for _ in configsAll]);
    # for i, (df_subset, title) in enumerate(configsAll, 1):
    #     plt.hist(df_subset[key],label=title,alpha=0.5,histtype="step",bins=bins)
    # plt.legend(loc=locLegend)
    # plt.xlabel(keyLabel)
    # plt.ylabel("N")
    # plt.yscale(yscale)
    # plt.title("All vectors")

    plt.subplot(312)    
    plotManyHisto(arrs=[df_subset[key] for (df_subset, title) in configsBib],
                  bins=bins,title="BIB",pltLabels =[title for (df_subset, title) in configsBib],
                  pltStandalone=False,yscale=yscale,xlabel=keyLabel,ylabel="N",legendLoc=locLegend,
                  alphas = [0.7 for _ in configsBib]);
    # for i, (df_subset, title) in enumerate(configsBib, 1):
    #     plt.hist(df_subset[key],label=title,alpha=0.5,bins=bins)
    # plt.legend(loc=locLegend)
    # plt.xlabel(keyLabel)
    # plt.ylabel("N")
    # plt.yscale(yscale)
    # plt.title("BIB")

    plt.subplot(313)
    plotManyHisto(arrs=[df_subset[key] for (df_subset, title) in configsSig],
                  bins=bins,title="Signal",pltLabels =[title for (df_subset, title) in configsSig],
                  pltStandalone=False,yscale=yscale,xlabel=keyLabel,ylabel="N",legendLoc=locLegend,
                  alphas = [0.7 for _ in configsSig]);
    # for i, (df_subset, title) in enumerate(configsSig, 1):
    #     plt.hist(df_subset[key],label=title,alpha=0.5,bins=bins)
    # plt.legend(loc=locLegend)
    # plt.xlabel(keyLabel)
    # plt.ylabel("N")
    # plt.yscale(yscale)
    # plt.title("Signal")

    closePlot(pltDir, interactivePlots, "karrisHistogram_"+key+"_.png",printOutputDir=True,transparent = False)
    return

def main():
    for path in paths:
        runForPath(path)

def main_multiprocess(nCPU = N_CPU):
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # This catches child processes trying to set it again and lets them pass safely
        pass
    with multiprocessing.Pool(processes=nCPU) as pool:
        pool.map(runForPath,paths)

if __name__ == '__main__':
    main()