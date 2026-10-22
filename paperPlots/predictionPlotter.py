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
from plotUtils import prepHistBins, closePlot,plotManyHisto,plot2dHistFromTruth
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
import json


PLOT_DIR = "./evaluationPlots"
interactivePlots = False
styleSheet = "seaborn-v0_8-colorblind"
styleSheet = "seaborn-v0_8-poster"
plt.style.use(styleSheet)
N_CPU = 4
loadPredVarPkl = False #if true then load based on saved pkls, if false regenerate the predVarDF and save new pkls
FILTER_TIME = True #add the -0.5 to 15 ns filter
TIMING_SMEARING = 30; #ps

matplotlib.rcParams["figure.dpi"] = 300
plt.rcParams["patch.linewidth"] = 2
# if not loadPredVarPkl:
#     import tensorflow as tf 
#     tf.config.set_visible_devices([], 'GPU')


paths = [
    # "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model2.5_fin_results_model2_5_10bit_normalised_selected__model_trial_057.h5",
    # "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model1_fin_results_model1_8bit_normalised_selected__model_trial_1046.h5",
    # "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model2.5_fin_results_model2_5_10bit_normalised_selected__model_trial_087.h5",
    # "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model3_10bit_normalised_selected_pareto_primary__model_trial_046.h5",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/Model1_best_SmallXsizeZglobal.h5",
    ]

def newDataRates(predVarDF, cut, pltDir, doPrint=True,timingVar="adjusted_hit_time_30ps_gaussian",timingStart=-0.09,timingEnd=0.150):
    totalBack = predVarDF.query(f"trueY == 0 and {timingVar} > -0.5 and {timingVar} < 15") #nominal cut
    modelAccept = totalBack.query("trueY == 0 and prediction > @cut")
    modelAndTimingAccept = totalBack.query(f"trueY == 0 and prediction > @cut and {timingVar} > @timingStart and {timingVar} < @timingEnd")
    justTimingAccept = totalBack.query(f"trueY == 0 and {timingVar} > @timingStart and {timingVar} < @timingEnd")
    dataBack = np.sum(totalBack["nPix"])
    clusterBack = len(totalBack["nPix"])
    dataModel = np.sum(modelAccept["nPix"])
    dataModelErr = np.sqrt(np.sum(modelAccept["nPix"])); #Double check logic, but sum of poissons is poisson so should work
    clusterModel = len(modelAccept["nPix"])
    clusterModelErr = np.sqrt(len(modelAccept["nPix"]))
    dataModelAndTiming = np.sum(modelAndTimingAccept["nPix"])
    dataModelAndTimingErr = np.sqrt(np.sum(modelAndTimingAccept["nPix"])); #Double check logic, but sum of poissons is poisson so should work
    clusterModelAndTiming = len(modelAndTimingAccept["nPix"])
    clusterModelAndTimingErr = np.sqrt(len(modelAndTimingAccept["nPix"]))
    dataTiming = np.sum(justTimingAccept["nPix"])
    dataTimingErr = np.sqrt(np.sum(justTimingAccept["nPix"])); #Double check logic, but sum of poissons is poisson so should work
    clusterTiming = len(justTimingAccept["nPix"])
    clusterTimingErr = np.sqrt(len(justTimingAccept["nPix"]))
    dataRateModel = dataModel / dataBack
    dataRateModelErr = dataModelErr / dataBack
    dataRateModelAndTiming = dataModelAndTiming / dataBack
    dataRateModelAndTimingErr = dataModelAndTimingErr / dataBack
    dataRateTiming = dataTiming / dataBack
    dataRateTimingErr = dataTimingErr / dataBack
    clusterRateModel = clusterModel / clusterBack
    clusterRateModelErr = clusterModelErr / clusterBack
    clusterRateModelAndTiming = clusterModelAndTiming / clusterBack
    clusterRateModelAndTimingErr = clusterModelAndTimingErr / clusterBack
    clusterRateTiming = clusterTiming / clusterBack
    clusterRateTimingErr = clusterTimingErr / clusterBack
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
        print(f"dataRateModelErr:          {dataRateModelErr}")
        print(f"1-dataRateModel:          {1-dataRateModel}")
        print(f"dataRateModelAndTiming: {dataRateModelAndTiming}")
        print(f"dataRateModelAndTimingErr: {dataRateModelAndTimingErr}")
        print(f"1-dataRateModelAndTiming: {1-dataRateModelAndTiming}")
        print(f"dataRateTiming:         {dataRateTiming}")
        print(f"dataRateTimingErr:         {dataRateTimingErr}")
        print(f"1-dataRateTiming:         {1-dataRateTiming}")
        print("=============================================")
        print(f"clusterBack:               {clusterBack}")
        print(f"clusterModel:              {clusterModel}")
        print(f"clusterModelAndTiming:     {clusterModelAndTiming}")
        print(f"clusterTiming:             {clusterTiming}")
        print(f"clusterRateModel:          {clusterRateModel}")
        print(f"clusterRateModelErr:          {clusterRateModelErr}")
        print(f"1-clusterRateModel:          {1-clusterRateModel}")
        print(f"clusterRateModelAndTiming: {clusterRateModelAndTiming}")
        print(f"clusterRateModelAndTimingErr: {clusterRateModelAndTimingErr}")
        print(f"1-clusterRateModelAndTiming: {1-clusterRateModelAndTiming}")
        print(f"clusterRateTiming:         {clusterRateTiming}")
        print(f"clusterRateTimingErr:         {clusterRateTimingErr}")
        print(f"1-clusterRateTiming:         {1-clusterRateTiming}")
        print("=============================================")
        print(f"dataModelAndTimingOverJustModel:    {dataModelAndTimingOverJustModel}")
        print(f"dataModelAndTimingOverJustModel2:   {dataModelAndTimingOverJustModel2}")
        print(f"dataModelAndTimingOverJustTiming:   {dataModelAndTimingOverJustTiming}")
        print(f"dataModelAndTimingOverJustTiming2:  {dataModelAndTimingOverJustTiming2}")
    with open(pltDir + f"/dataRates_{timingVar}.txt","w") as f:
        print(f"dataBack:               {dataBack}", file=f)
        print(f"dataModel:              {dataModel}", file=f)
        print(f"dataModelAndTiming:     {dataModelAndTiming}", file=f)
        print(f"dataTiming:             {dataTiming}", file=f)
        print(f"dataRateModel:          {dataRateModel}", file=f)
        print(f"dataRateModelErr:          {dataRateModelErr}", file=f)
        print(f"1-dataRateModel:          {1-dataRateModel}", file=f)
        print(f"dataRateModelAndTiming: {dataRateModelAndTiming}", file=f)
        print(f"dataRateModelAndTimingErr: {dataRateModelAndTimingErr}", file=f)
        print(f"1-dataRateModelAndTiming: {1-dataRateModelAndTiming}", file=f)
        print(f"dataRateTiming:         {dataRateTiming}", file=f)
        print(f"dataRateTimingErr:         {dataRateTimingErr}", file=f)
        print(f"1-dataRateTiming:         {1-dataRateTiming}", file=f)
        print("=============================================", file=f)
        print(f"clusterBack:               {clusterBack}",file=f)
        print(f"clusterModel:              {clusterModel}",file=f)
        print(f"clusterModelAndTiming:     {clusterModelAndTiming}",file=f)
        print(f"clusterTiming:             {clusterTiming}",file=f)
        print(f"clusterRateModel:          {clusterRateModel}",file=f)
        print(f"clusterRateModelErr:          {clusterRateModelErr}",file=f)
        print(f"1-clusterRateModel:          {1-clusterRateModel}",file=f)
        print(f"clusterRateModelAndTiming: {clusterRateModelAndTiming}",file=f)
        print(f"clusterRateModelAndTimingErr: {clusterRateModelAndTimingErr}",file=f)
        print(f"1-clusterRateModelAndTiming: {1-clusterRateModelAndTiming}",file=f)
        print(f"clusterRateTiming:         {clusterRateTiming}",file=f)
        print(f"clusterRateTimingErr:         {clusterRateTimingErr}",file=f)
        print(f"1-clusterRateTiming:         {1-clusterRateTiming}",file=f)
        print("=============================================",file=f)
        print(f"dataModelAndTimingOverJustModel:    {dataModelAndTimingOverJustModel}", file=f)
        print(f"dataModelAndTimingOverJustModel2:   {dataModelAndTimingOverJustModel2}", file=f)
        print(f"dataModelAndTimingOverJustTiming:   {dataModelAndTimingOverJustTiming}", file=f)
        print(f"dataModelAndTimingOverJustTiming2:  {dataModelAndTimingOverJustTiming2}",file=f)
    dataRateResults = {
        "dataBack": dataBack,
        "dataModel": dataModel,
        "dataModelAndTiming": dataModelAndTiming,
        "dataTiming": dataTiming,
        "dataRateModel": dataRateModel,
        "dataRateModelErr": dataRateModelErr,
        "1-dataRateModel": 1-dataRateModel,
        "dataRateModelAndTiming": dataRateModelAndTiming,
        "dataRateModelAndTimingErr": dataRateModelAndTimingErr,
        "1-dataRateModelAndTiming": 1-dataRateModelAndTiming,
        "dataRateTiming": dataRateTiming,
        "dataRateTimingErr": dataRateTimingErr,
        "1-dataRateTiming": 1-dataRateTiming,
        "clusterBack": clusterBack,
        "clusterModel": clusterModel,
        "clusterModelAndTiming": clusterModelAndTiming,
        "clusterTiming": clusterTiming,
        "clusterRateModel": clusterRateModel,
        "clusterRateModelErr": clusterRateModelErr,
        "1-clusterRateModel": 1-clusterRateModel,
        "clusterRateModelAndTiming": clusterRateModelAndTiming,
        "clusterRateModelAndTimingErr": clusterRateModelAndTimingErr,
        "1-clusterRateModelAndTiming": 1-clusterRateModelAndTiming,
        "clusterRateTiming": clusterRateTiming,
        "clusterRateTimingErr": clusterRateTimingErr,
        "1-clusterRateTiming": 1-clusterRateTiming,
        "dataModelAndTimingOverJustModel": dataModelAndTimingOverJustModel,
        "dataModelAndTimingOverJustModel2": dataModelAndTimingOverJustModel2,
        "dataModelAndTimingOverJustTiming": dataModelAndTimingOverJustTiming,
        "dataModelAndTimingOverJustTiming2": dataModelAndTimingOverJustTiming2,
    }
    with open(pltDir + f"/dataRates_{timingVar}.json","w") as f:
        json.dump(varPredPlotUtils.to_python(dataRateResults),f)
    return dataRateResults

def runForPath(path,filterTime=FILTER_TIME,timingSigmaPs=TIMING_SMEARING):
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

    timingVar=f"adjusted_hit_time_{timingSigmaPs}ps_gaussian"
    timingStart=-3e-3*timingSigmaPs;#-0.09
    timingEnd=5e-3*timingSigmaPs;#0.150

    if timingVar != "adjusted_hit_time_30ps_gaussian":
        # df['adjusted_hit_time_30ps_gaussian'] = df['adjusted_hit_time']+np.random.normal(loc=0,scale=30e-3,size=len(df['adjusted_hit_time']))
        predVarDF[timingVar] = predVarDF['adjusted_hit_time']+np.random.normal(loc=0,scale=timingSigmaPs*1e-3,size=len(predVarDF['adjusted_hit_time']))

    assert f"{timingSigmaPs}" in timingVar

    # plotSeveral2by2s(predVarDF, threshVal, pltDir)
    plotAll1dHists(predVarDF,threshVal,pltDir,timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    # plotNew2by2(predVarDF, threshVal, pltDir)

    # newDataRates(predVarDF,threshVal,pltDir,timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)

def plotAll1dHists(predVarDF,threshVal,pltDir,timingVar="adjusted_hit_time_30ps_gaussian",timingStart=-0.09,timingEnd=0.150):
    print(predVarDF)
    print(predVarDF.keys())
    if timingVar in predVarDF.keys():
        histoKarri(predVarDF,threshVal,pltDir,key=timingVar,keyLabel="Cluster Hit Arrival Time [ns]",figsize=(9,16),bins=100,timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
        histoKarri(predVarDF,threshVal,pltDir,key=timingVar,keyLabel=r"$t_{\mathrm{corr}}$ [ns]",figsize=(8,6.2),bins=np.arange(timingStart-.24*2,15,.24),plotAll=False,plotSig=False,extendSaveTitle="paperVersion",bibTitle="",ylim=[1e1,1.1e5],timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd,)
    histoKarri(predVarDF,threshVal,pltDir,key="z-global",keyLabel=r'$z_{\mathrm{global}}$ [mm]',figsize=(8,6.2),bins=60,plotAll=False,plotSig = False,extendSaveTitle="paperVersion",bibTitle="",ylim=[1e1,1.1e5],locLegend="upper center",extraLinebreak="",timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    histoKarri(predVarDF,threshVal,pltDir,key="z-global",keyLabel=r'$z_{\mathrm{global}}$ [mm]',figsize=(9,16),bins=100,timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    histoKarri(predVarDF,threshVal,pltDir,key="pt",keyLabel=r"Transverse Momentum $p_T$ [GeV/c]",figsize=(9,16),bins=100,timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    histoKarri(predVarDF,threshVal,pltDir,key="y-local",keyLabel="y-local [mm]",figsize=(9,16),bins=25,timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    histoKarri(predVarDF,threshVal,pltDir,key="xSize",keyLabel=r'$x_{\mathrm{size}}$ [# pixels]',bins=np.arange(0,22,1),figsize=(9,16),locLegend="upper right",timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    histoKarri(predVarDF,threshVal,pltDir,key="ySize",keyLabel="y-Size [# pixels]",bins=np.arange(0,14,1),figsize=(9,16),locLegend="upper right",timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    histoKarri(predVarDF,threshVal,pltDir,key="nModule",keyLabel="Module Number (longitudinally counted)",bins=12,figsize=(9,16),timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    histoKarri(predVarDF,threshVal,pltDir,key="nPix",keyLabel="Number of Pixels",bins=np.arange(0,np.max(predVarDF["nPix"]),1),figsize=(9,16),locLegend="upper right",timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    histoKarri(predVarDF,threshVal,pltDir,key="nPix",keyLabel="Number of Pixels",bins=np.arange(0,np.max(predVarDF["nPix"]),1),figsize=(6.9,5.6),locLegend="upper right",plotAll = False,plotSig=False,extendSaveTitle="paperVersionBib", bibTitle = "", sigTitle = "",timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd,ylim=[1e1,1.1e5])
    histoKarri(predVarDF,threshVal,pltDir,key="nPix",keyLabel="Number of Pixels",bins=np.arange(0,np.max(predVarDF["nPix"]),1),figsize=(6.9,5.6),locLegend="upper right",plotAll = False,plotBIB=False,extendSaveTitle="paperVersionSig", bibTitle = "", sigTitle = "",timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd,ylim=[1e1,1.1e5])
    histoKarri(predVarDF,threshVal,pltDir,key="cotAlpha",keyLabel=r"cot($\alpha$)",bins=12,figsize=(9,16),timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    histoKarri(predVarDF,threshVal,pltDir,key="cotBeta",keyLabel=r"cot($\beta$)",bins=12,figsize=(9,16),timingVar=timingVar,timingStart=timingStart,timingEnd=timingEnd)
    print("finished 1d histograms")

def plotNew2by2(predVarDF, threshVal, pltDir,interactivePlots = False,extendTitle = "",cmap="Blues",figsize=(10,8), plotter=None,genTitle=""):
    if plotter is None:
        plotter = functools.partial(varPredPlotUtils.plotZglobalXsizeJust1,cmap=cmap)
    # genTitle = "ZGlobalXSize"
    # genTitle = ""
    varPredPlotUtils.plot2by2PredBibSig(predVarDF,plot_func=plotter,genTitle=genTitle,extendTitle = extendTitle,cut=threshVal, PLOT_DIR=pltDir,interactivePlots=interactivePlots,figsize=figsize)

def plotSeveral2by2s(predVarDF, threshVal, pltDir,interactivePlots = False,cmap="Blues",figsize=(10,8), plotter=None):
    binsZGlobal = 30;binsXSize = np.arange(0,22,1);binsYlocal = 30;binsYSize = np.arange(0,14,1);
    title=""# not sure if it should be
    colorLabel=r"$n_{\mathrm{clusters}}$"
    # def plot2dHistFromTruth(truthDF,  keyX, keyY, mask, binsX, binsY, cmap, xlabel,ylabel,title,logColor = True,colorLabel=r"$n_{\mathrm{clusters}}$"):
    keyXs = ["z-global","xSize","ySize","y-local","nPix","pt","cotAlpha","cotBeta"]
    binsXs = [binsZGlobal,binsXSize,binsYSize,binsYlocal,np.arange(0,43,1),100,np.linspace(-10,10,50),np.linspace(-10,10,50)]
    xlabels=[r'$z_{\mathrm{global}}$ [mm]',r'$x_{\mathrm{size}}$ [# pixels]',r'$y_{\mathrm{size}}$ [# pixels]',r'$y_{\mathrm{local}}$ [mm]',r'Number of Pixels',r"Transverse Momentum $p_T$ [GeV/c]",r"cot($\alpha$)",r"cot($\beta$)"]
    nPlotsX=len(keyXs);
    # extendTitles=[keyX+"vsyyyy" for keyX in keyXs]
    # genTitles=[keyX+"vsyyyy" for keyX in keyXs]
    keyYs=["cotAlpha","cotBeta"]
    binsYs=[np.linspace(-10,10,50),np.linspace(-10,10,50)]
    ylabels=[r"cot($\alpha$)",r"cot($\beta$)"]
    allPlotterLists = []
    for idy,keyY in enumerate(keyYs):
        plottersList = []
        for idx in range(nPlotsX):
            plottersList.append(
                functools.partial(plot2dHistFromTruth,  keyX=keyXs[idx],keyY=keyY, mask=None, binsX=binsXs[idx], binsY=binsYs[idy], cmap=cmap, xlabel=xlabels[idx],ylabel=ylabels[idy],logColor = True,colorLabel=colorLabel)
            )
        allPlotterLists.append(plottersList)
    # plotters = [functools.partial(plot2dHistFromTruth,  keyX=keyXs[idx],keyY="xSize", mask=None, binsX=binsXs[idx], binsY=binsXSize, cmap=cmap, xlabel=xlabels[idx],ylabel=r'$x_{\mathrm{size}}$ [# pixels]',logColor = True,colorLabel=colorLabel) for idx in range(nPlots)]
    for idy,plotters in enumerate(allPlotterLists):
        for idx,plotter in enumerate(plotters):
            # print(plotter)
            plotNew2by2(predVarDF,threshVal,pltDir,interactivePlots,"",cmap=None,figsize=figsize,plotter=plotter,genTitle=keyXs[idx]+"vs"+keyYs[idy])

def histoKarri(predVarDF,cut,pltDir,key="z-global",keyLabel="",figsize=(9,16),bins="auto",yscale="log",locLegend = "best",
               plotAll = True, plotBIB = True, plotSig = True,extendSaveTitle="",allTitle="All vectors",bibTitle = "BIB",sigTitle="Signal",lw=4,#lw=plt.rcParams["patch.linewidth"]
               ylim=None,increaseFontSize=True,extraLinebreak="\n",timingVar = "adjusted_hit_time_30ps_gaussian",timingStart=-0.09,timingEnd=0.150,):
    configsAll = [
        (predVarDF, "All vectors"),
        (predVarDF.query("trueY == 0"), "All BIB"),
        (predVarDF.query("trueY == 1"), "All Signal"),
        (predVarDF.query("prediction > @cut"), "Vectors accepted by model"),
        (predVarDF.query("trueY == 0 and prediction > @cut"), "BIB accepted by model"),
        (predVarDF.query("trueY == 1 and prediction > @cut"), "Signal accepted by model"),
    ]
    configsSig = [
        (predVarDF.query("trueY == 1"), "All Signal"),
        (predVarDF.query("trueY == 1 and prediction > @cut"), "Signal accepted by model"),
        (predVarDF.query(f"trueY == 1 and prediction > @cut and {timingVar} > @timingStart and {timingVar} < @timingEnd"), "Signal accepted by model,\n"+ r"$[3\sigma_t, 5\sigma_t]$ timing selection"),
        # (predVarDF.query("trueY == 1 and prediction < @cut"), "Signal rejected by model"),
    ]
    configsBib = [
        (predVarDF.query("trueY == 0"), "All BIB"),
        (predVarDF.query("trueY == 0 and prediction > @cut"), "BIB accepted by model"),
        (predVarDF.query(f"trueY == 0 and prediction > @cut and {timingVar} > @timingStart and {timingVar} < @timingEnd"), f"BIB accepted by model, {extraLinebreak}"+ r"$[3\sigma_t, 5\sigma_t]$ timing selection"),
        # (predVarDF.query(f"trueY == 0 and {timingVar} > @timingStart and {timingVar} < @timingEnd"), f"BIB accepted by {extraLinebreak}"+ r"$[3\sigma_t, 5\sigma_t]$ timing selection"),
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
                    alphas = [0.6 for _ in configsAll],lw=lw);
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
                    alphas = [0.6 for _ in configsBib],lw=lw);
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
                    alphas = [0.6 for _ in configsSig],lw=lw);
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
