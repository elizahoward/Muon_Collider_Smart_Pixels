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
import functools
import matplotlib


PLOT_DIR = "./evaluationPlots"
interactivePlots = False
styleSheet = "seaborn-v0_8-colorblind"
styleSheet = "seaborn-v0_8-poster"
plt.style.use(styleSheet)
N_CPU = 4
loadPredVarPkl = True #if true then load based on saved pkls, if false regenerate the predVarDF and save new pkls
FILTER_TIME = True #add the -0.5 to 15 ns filter

matplotlib.rcParams["figure.dpi"] = 300
plt.rcParams["patch.linewidth"] = 2
# if not loadPredVarPkl:
#     import tensorflow as tf 
#     tf.config.set_visible_devices([], 'GPU')

paths = [
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model2.5_fin_results_model2_5_10bit_normalised_selected__model_trial_057.h5",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model1_fin_results_model1_8bit_normalised_selected__model_trial_1046.h5",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model2.5_fin_results_model2_5_10bit_normalised_selected__model_trial_087.h5",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model3_10bit_normalised_selected_pareto_primary__model_trial_046.h5",
    ]

def newDataRates(predVarDF, cut, pltDir, doPrint=True):
    totalBack = predVarDF.query("trueY == 0 and adjusted_hit_time_30ps_gaussian > -0.5 and adjusted_hit_time_30ps_gaussian < 15")
    modelAccept = totalBack.query("trueY == 0 and prediction > @cut")
    modelAndTimingAccept = totalBack.query("trueY == 0 and prediction > @cut and adjusted_hit_time_30ps_gaussian > -0.09 and adjusted_hit_time_30ps_gaussian < 0.150")
    justTimingAccept = totalBack.query("trueY == 0 and adjusted_hit_time_30ps_gaussian > -0.09 and adjusted_hit_time_30ps_gaussian < 0.150")
    dataBack = np.sum(totalBack["nPix"])
    clusterBack = len(totalBack["nPix"])
    dataModel = np.sum(modelAccept["nPix"])
    clusterModel = len(modelAccept["nPix"])
    dataModelAndTiming = np.sum(modelAndTimingAccept["nPix"])
    clusterModelAndTiming = len(modelAndTimingAccept["nPix"])
    dataTiming = np.sum(justTimingAccept["nPix"])
    clusterTiming = len(justTimingAccept["nPix"])
    dataRateModel = dataModel / dataBack
    dataRateModelAndTiming = dataModelAndTiming / dataBack
    dataRateTiming = dataTiming / dataBack
    clusterRateModel = clusterModel / clusterBack
    clusterRateModelAndTiming = clusterModelAndTiming / clusterBack
    clusterRateTiming = clusterTiming / clusterBack
    #cluster rates are calculated by .evaluate, but for sanity also here.

    dataModelAndTimingOverJustModel = dataModelAndTiming / dataModel
    dataModelAndTimingOverJustModel2 = dataRateModelAndTiming / dataRateModel
    dataModelAndTimingOverJustTiming = dataModelAndTiming / dataTiming
    dataModelAndTimingOverJustTiming2 = dataRateModelAndTiming / dataRateTiming
    if doPrint:
        print(f"dataBack:               {dataBack}")
        print(f"dataModel:              {dataModel}")
        print(f"dataModelAndTiming:     {dataModelAndTiming}")
        print(f"dataTiming:             {dataTiming}")
        print(f"dataRateModel:          {dataRateModel}")
        print(f"1-dataRateModel:          {1-dataRateModel}")
        print(f"dataRateModelAndTiming: {dataRateModelAndTiming}")
        print(f"1-dataRateModelAndTiming: {1-dataRateModelAndTiming}")
        print(f"dataRateTiming:         {dataRateTiming}")
        print(f"1-dataRateTiming:         {1-dataRateTiming}")
        print("=============================================")
        print(f"clusterBack:               {clusterBack}")
        print(f"clusterModel:              {clusterModel}")
        print(f"clusterModelAndTiming:     {clusterModelAndTiming}")
        print(f"clusterTiming:             {clusterTiming}")
        print(f"clusterRateModel:          {clusterRateModel}")
        print(f"1-clusterRateModel:          {1-clusterRateModel}")
        print(f"clusterRateModelAndTiming: {clusterRateModelAndTiming}")
        print(f"1-clusterRateModelAndTiming: {1-clusterRateModelAndTiming}")
        print(f"clusterRateTiming:         {clusterRateTiming}")
        print(f"1-clusterRateTiming:         {1-clusterRateTiming}")
        print("=============================================")
        print(f"dataModelAndTimingOverJustModel:    {dataModelAndTimingOverJustModel}")
        print(f"dataModelAndTimingOverJustModel2:   {dataModelAndTimingOverJustModel2}")
        print(f"dataModelAndTimingOverJustTiming:   {dataModelAndTimingOverJustTiming}")
        print(f"dataModelAndTimingOverJustTiming2:  {dataModelAndTimingOverJustTiming2}")
    with open(pltDir + "/dataRates.txt","w") as f:
        print(f"dataBack:               {dataBack}", file=f)
        print(f"dataModel:              {dataModel}", file=f)
        print(f"dataModelAndTiming:     {dataModelAndTiming}", file=f)
        print(f"dataTiming:             {dataTiming}", file=f)
        print(f"dataRateModel:          {dataRateModel}", file=f)
        print(f"1-dataRateModel:          {1-dataRateModel}", file=f)
        print(f"dataRateModelAndTiming: {dataRateModelAndTiming}", file=f)
        print(f"1-dataRateModelAndTiming: {1-dataRateModelAndTiming}", file=f)
        print(f"dataRateTiming:         {dataRateTiming}", file=f)
        print(f"1-dataRateTiming:         {1-dataRateTiming}", file=f)
        print("=============================================", file=f)
        print(f"clusterBack:               {clusterBack}",file=f)
        print(f"clusterModel:              {clusterModel}",file=f)
        print(f"clusterModelAndTiming:     {clusterModelAndTiming}",file=f)
        print(f"clusterTiming:             {clusterTiming}",file=f)
        print(f"clusterRateModel:          {clusterRateModel}",file=f)
        print(f"1-clusterRateModel:          {1-clusterRateModel}",file=f)
        print(f"clusterRateModelAndTiming: {clusterRateModelAndTiming}",file=f)
        print(f"1-clusterRateModelAndTiming: {1-clusterRateModelAndTiming}",file=f)
        print(f"clusterRateTiming:         {clusterRateTiming}",file=f)
        print(f"1-clusterRateTiming:         {1-clusterRateTiming}",file=f)
        print("=============================================",file=f)
        print(f"dataModelAndTimingOverJustModel:    {dataModelAndTimingOverJustModel}", file=f)
        print(f"dataModelAndTimingOverJustModel2:   {dataModelAndTimingOverJustModel2}", file=f)
        print(f"dataModelAndTimingOverJustTiming:   {dataModelAndTimingOverJustTiming}", file=f)
        print(f"dataModelAndTimingOverJustTiming2:  {dataModelAndTimingOverJustTiming2}",file=f)
    return

def runForPath(path,filterTime=FILTER_TIME):
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
        predVarDF, model, predictions, threshVal = varPredPlotUtils.runModelPlots(filepath = path,PLOT_DIR=pltDir, interactivePlots=interactivePlots,extendTitle=path[25:],filterTime=filterTime)   
        predVarDF.to_pickle(Path(pltDir).joinpath("predVarDF.pkl"))
        with open(Path(pltDir).joinpath("threshVal.pkl"), 'wb') as file:
            pickle.dump(threshVal, file)

    plotAll1dHists(predVarDF,threshVal,pltDir)
    plotNew2by2(predVarDF, threshVal, pltDir)

    newDataRates(predVarDF,threshVal,pltDir)

def plotAll1dHists(predVarDF,threshVal,pltDir):
    print(predVarDF)
    print(predVarDF.keys())
    if "adjusted_hit_time_30ps_gaussian" in predVarDF.keys():
        histoKarri(predVarDF,threshVal,pltDir,key="adjusted_hit_time_30ps_gaussian",keyLabel="Cluster Hit Arrival Time [ns]",figsize=(5,10),bins=100)
        histoKarri(predVarDF,threshVal,pltDir,key="adjusted_hit_time_30ps_gaussian",keyLabel=r"$t_{\mathrm{corr}}$ [ns]",figsize=(6,5),bins=100,plotAll=False,plotSig=False,extendSaveTitle="paperVersion",bibTitle="",ylim=[1e1,4.5e4])
    histoKarri(predVarDF,threshVal,pltDir,key="z-global",keyLabel=r'$z_{\mathrm{global}}$ [mm]',figsize=(6,5),bins=60,plotAll=False,plotSig = False,extendSaveTitle="paperVersion",bibTitle="",ylim=[1e1,4.5e4],locLegend="upper center")
    histoKarri(predVarDF,threshVal,pltDir,key="z-global",keyLabel=r'$z_{\mathrm{global}}$ [mm]',figsize=(5,10),bins=100)
    histoKarri(predVarDF,threshVal,pltDir,key="pt",keyLabel=r"Transverse Momentum $p_T$ [GeV/c]",figsize=(5,10),bins=100)
    histoKarri(predVarDF,threshVal,pltDir,key="y-local",keyLabel="y-local [mm] aaaaah I can't find a good binnning",figsize=(5,10),bins=25)
    histoKarri(predVarDF,threshVal,pltDir,key="xSize",keyLabel=r'$x_{\mathrm{size}}$ [# pixels]',bins=np.arange(0,22,1),figsize=(5,10),locLegend="upper right")
    histoKarri(predVarDF,threshVal,pltDir,key="ySize",keyLabel="y-Size [# pixels]",bins=np.arange(0,14,1),figsize=(5,11),locLegend="upper right")
    histoKarri(predVarDF,threshVal,pltDir,key="nModule",keyLabel="Module Number (longitudinally counted)",bins=12,figsize=(5,10))
    histoKarri(predVarDF,threshVal,pltDir,key="nPix",keyLabel="Number of Pixels",bins=np.arange(0,np.max(predVarDF["nPix"]),1),figsize=(5,10),locLegend="upper right")
    histoKarri(predVarDF,threshVal,pltDir,key="nPix",keyLabel="Number of Pixels",bins=np.arange(0,np.max(predVarDF["nPix"]),1),figsize=(6.6,10),locLegend="upper right",plotAll = False,extendSaveTitle="paperVersion")
    print("finished 1d histograms")

def plotNew2by2(predVarDF, threshVal, pltDir,interactivePlots = False,extendTitle = "",cmap="Blues",figsize=(10,8)):
    plotter = functools.partial(varPredPlotUtils.plotZglobalXsizeJust1,cmap=cmap)
    genTitle = "ZGlobalXSize"
    genTitle = ""
    varPredPlotUtils.plot2by2PredBibSig(predVarDF,plot_func=plotter,genTitle=genTitle,extendTitle = extendTitle,cut=threshVal, PLOT_DIR=pltDir,interactivePlots=interactivePlots,figsize=figsize)


def histoKarri(predVarDF,cut,pltDir,key="z-global",keyLabel="",figsize=(5,10),bins="auto",yscale="log",locLegend = "best",
               plotAll = True, plotBIB = True, plotSig = True,extendSaveTitle="",allTitle="All vectors",bibTitle = "BIB",sigTitle="Signal",
               ylim=None,increaseFontSize=True):
    configsAll = [
        (predVarDF, "all vectors"),
        (predVarDF.query("trueY == 0"), "all BIB"),
        (predVarDF.query("trueY == 1"), "all Signal"),
        (predVarDF.query("prediction > @cut"), "all vectors accepted by model"),
        (predVarDF.query("trueY == 0 and prediction > @cut"), "BIB accepted by model"),
        (predVarDF.query("trueY == 1 and prediction > @cut"), "Signal accepted by model"),
    ]
    configsSig = [
        (predVarDF.query("trueY == 1"), "all Signal"),
        (predVarDF.query("trueY == 1 and prediction > @cut"), "Signal accepted by model"),
        (predVarDF.query("trueY == 1 and prediction > @cut and adjusted_hit_time_30ps_gaussian > -0.09 and adjusted_hit_time_30ps_gaussian < 0.150"), "Signal accepted by model, \n "+ r"$[3\sigma, 5\sigma]$ timing selection"),
    ]
    configsBib = [
        (predVarDF.query("trueY == 0"), "all BIB"),
        (predVarDF.query("trueY == 0 and prediction > @cut"), "BIB accepted by model"),
        (predVarDF.query("trueY == 0 and prediction > @cut and adjusted_hit_time_30ps_gaussian > -0.09 and adjusted_hit_time_30ps_gaussian < 0.150"), "BIB accepted by model, \n "+ r"$[3\sigma, 5\sigma]$ timing selection"),
    ]
    totalPlots = plotAll + plotBIB + plotSig
    plt.figure(figsize=figsize)
    if plotAll:
        plt.subplot(totalPlots,1,1)
        #def plotManyHisto(arrs,bins=None,postScale=1,title="",pltLabels=["1","2","3"],pltStandalone=True,showNums=False,
        #               figsize=(7,3),yscale="linear",xlabel="",ylabel="Tracks",
        #               PLOT_DIR=None,interactivePlots=None,saveTitle=None,alphas = None,legendLoc = "best"):
        plotManyHisto(arrs=[df_subset[key] for (df_subset, title) in configsAll],
                    bins=bins,title=allTitle,pltLabels =[title for (df_subset, title) in configsAll],
                    pltStandalone=False,yscale=yscale,xlabel=keyLabel,ylabel=r"$n_{\mathrm{clusters}}$",legendLoc=locLegend,
                    alphas = [0.6 for _ in configsAll]);
        if ylim is not None:
            plt.ylim(ylim)
        if increaseFontSize:
            plt.gca().yaxis.get_label().set_fontsize(plt.rcParams["axes.labelsize"]+5)
            plt.gca().xaxis.get_label().set_fontsize(plt.rcParams["axes.labelsize"]+5)
    if plotBIB:
        plt.subplot(totalPlots,1,plotAll + plotBIB)    
        plotManyHisto(arrs=[df_subset[key] for (df_subset, title) in configsBib],
                    bins=bins,title=bibTitle,pltLabels =[title for (df_subset, title) in configsBib],
                    pltStandalone=False,yscale=yscale,xlabel=keyLabel,ylabel=r"$n_{\mathrm{clusters}}$",legendLoc=locLegend,
                    alphas = [0.6 for _ in configsBib]);
        if ylim is not None:
            plt.ylim(ylim)
        if increaseFontSize:
            plt.gca().yaxis.get_label().set_fontsize(plt.rcParams["axes.labelsize"]+5)
            plt.gca().xaxis.get_label().set_fontsize(plt.rcParams["axes.labelsize"]+5)
    if plotSig:
        plt.subplot(totalPlots,1,plotAll + plotBIB + plotSig)
        plotManyHisto(arrs=[df_subset[key] for (df_subset, title) in configsSig],
                    bins=bins,title=sigTitle,pltLabels =[title for (df_subset, title) in configsSig],
                    pltStandalone=False,yscale=yscale,xlabel=keyLabel,ylabel=r"$n_{\mathrm{clusters}}$",legendLoc=locLegend,
                    alphas = [0.6 for _ in configsSig]);
        if ylim is not None:
            plt.ylim(ylim)
        if increaseFontSize:
            plt.gca().yaxis.get_label().set_fontsize(plt.rcParams["axes.labelsize"]+5)
            plt.gca().xaxis.get_label().set_fontsize(plt.rcParams["axes.labelsize"]+5)


    closePlot(pltDir, interactivePlots, "karrisHistogram_"+key+extendSaveTitle+"_.png",printOutputDir=True,transparent = False)
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
    main_multiprocess()