"""
Author: Daniel Abadjiev
Date: July 20, 2026
Description: Version of runPlots2.py that makes paper specific plots
"""

import os
import sys
sys.path.append("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/validationPlots")
from SpixPlotter import SmartpixPlotter
import argparse
from pathlib import Path
import matplotlib
import plotUtils
import numpy as np
###########################################################################
########## Defaults for data directories, copied from runPlots2.py
repodir = Path(__file__).resolve().parent.parent
print("Assuming you are starting in .....paperPlots, so git repo dir is ", repodir)
assert repodir.parts[-1] == 'Muon_Collider_Smart_Pixels'
dataDir_all = "Data_Set_2026Feb"
if dataDir_all is not None:
    print("Using a dataset directory with subfolders, following Eliza's 2026 dataset format, ignoring other directories")
    print("Setting plot directory inside dataset folder")
    #either dataDir_all is a string witht the dataset name, in which case join paths from DataFiles and repodir
    #otherwise, dataDir_all should be an absolute path to the dataset
    # if 
    datasetPath = Path(dataDir_all)
    if len(datasetPath.parts) == 1:
        print("looking for the dataset in Data_Files")
        datasetDir = repodir.joinpath("Data_Files").joinpath(datasetPath)
    else:
        print("assuming dataset passed is in an absolute path")
        datasetDir = datasetPath
    parquetDir_all = datasetDir.joinpath("Parquet_Files")
    trackDirBib_mm = datasetDir.joinpath("Track_Lists")
    trackDirBib_mp = datasetDir.joinpath("Track_Lists")
    trackDirSig = datasetDir.joinpath("Track_Lists")
    # PLOT_DIR = datasetDir.joinpath("plots")
###########################################################################

STYLESHEET = "seaborn-v0_8-colorblind"
STYLESHEET = "seaborn-v0_8-poster"

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.pyplot.rcParams["patch.linewidth"] = 2


def main(parquetDir_all = "/local/d1/smartpixML/bigData/allData/",     #this should be not used?          
            #skip_indices = list(range(1730 - 124+87, 1769)),
            trackDirBib_mm = trackDirBib_mm,
            trackDirBib_mp = trackDirBib_mp,
            trackDirSig = trackDirSig,
            processRecon = False,
            interactivePlots = False,
            PLOT_DIR = "./datasetPlots",
            savedPklFromParquet = True,
            processTracks = True,
            processOldTracks = False,
            plotTracklists = True,
            plotParquets = True,
            styleSheet=STYLESHEET,):
    plotter = SmartpixPlotter(
                    #  parquetDir_mm = parquetDir_mm , #Not yet implemented
                    #  parquetDir_mp = parquetDir_mp ,
                    #  parquetDir_sig = parquetDir_sig ,
                    parquetDir_all = parquetDir_all ,
                    skip_indices = None,#list(range(1730 - 124+87, 1769)),
                    trackDirBib_mm = trackDirBib_mm,
                    trackDirBib_mp = trackDirBib_mp,
                    trackDirSig = trackDirSig,
                    processRecon = processRecon,
                    interactivePlots=interactivePlots,
                    PLOT_DIR = PLOT_DIR,# os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                    savedPklFromParquet = savedPklFromParquet,
                    processTracks = processTracks,
                    processOldTracks = processOldTracks,
                    plotTracklists = plotTracklists,
                    plotParquets = plotParquets,
                    styleSheet = styleSheet,
                    )
    plotter.runPlots()
    extraAppendixPlot(plotter)

def extraAppendixPlot(plotter):
    recalcStrs = ["", "", "", "", "", "", "", "", ""]    
    trackHeader = ["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg"]    
    parquetTrackKeys = ["cotAlpha","cotBeta","p_calc1","p_calc1","y-local","z-global","pt","hit_time","PID"]
    xlabels = ["cot(α)", "cot(β)", "Momentum [GeV/c]", "NO", r'$y_{\mathrm{local}}$ [mm]', r'$z_{\mathrm{global}}$ [mm]', r"$p_T$ [GeV/c]", "Raw Hit Time [ns]", "PDG ID"]
    keyLabels = ["cot(α)", "cot(β)", "p", "NO", r"$y_{\mathrm{local}}$", r'$z_{\mathrm{global}}$', r"$p_T$", "Hit Time", "PDG ID"]                    
    plotUtils.plotSomeTracParqVars(plotter.tracksBib, plotter.tracksSig,plotter.truthBib,plotter.truthSig,PLOT_DIR=plotter.PLOT_DIR,interactivePlots=plotter.interactivePlots,
                     recalcStrs=recalcStrs,trackHeader=trackHeader,parquetTrackKeys=parquetTrackKeys,xlabels = xlabels,keyLabels = keyLabels,
                     figsize=(15,25))

    recalcStrs = ["", "",]    
    trackHeader = ["p", "pt"]    
    parquetTrackKeys = ["p_calc1","pt"]
    xlabels = ["Momentum [GeV/c]", r"$p_T$ [GeV/c]"]
    keyLabels = ["p", r"$p_T$"] 
    keyLabels = ["", ""]    
    bibTimeBins = np.linspace(np.min(plotter.tracksBib["t"]),10000,35)
    sigTimeBins = np.linspace(np.min(plotter.tracksSig["t"]),3,35)
    bibPtBins = np.linspace(0,1,30)
    bibPBins = np.linspace(0,2,30)
    binsBibList = [bibPBins,bibPtBins]                
    binsSigList = [None,None]
    plotUtils.plotSomeTracParqVars(plotter.tracksBib, plotter.tracksSig,plotter.truthBib,plotter.truthSig,PLOT_DIR=plotter.PLOT_DIR,interactivePlots=plotter.interactivePlots,
                     recalcStrs=recalcStrs,trackHeader=trackHeader,parquetTrackKeys=parquetTrackKeys,xlabels = xlabels,keyLabels = keyLabels,
                     figsize=(13,12),saveTitle="appendixAllVarspt1.png",binsBibList=binsBibList,binsSigList=binsSigList)

    recalcStrs = ["",]    
    trackHeader = [ "t"]    
    parquetTrackKeys = ["hit_time"]
    xlabels = ["Raw Hit Time [ns]"]
    keyLabels = ["Hit Time"]
    keyLabels = [""]     
    bibTimeBins = np.linspace(np.min(plotter.tracksBib["t"]),10000,35)
    sigTimeBins = np.linspace(np.min(plotter.tracksSig["t"]),3,35)
    binsBibList = [bibTimeBins]                
    binsSigList = [sigTimeBins]                
    plotUtils.plotSomeTracParqVars(plotter.tracksBib, plotter.tracksSig,plotter.truthBib,plotter.truthSig,PLOT_DIR=plotter.PLOT_DIR,interactivePlots=plotter.interactivePlots,
                     recalcStrs=recalcStrs,trackHeader=trackHeader,parquetTrackKeys=parquetTrackKeys,xlabels = xlabels,keyLabels = keyLabels,
                     figsize=(13.5,6.5),saveTitle="appendixAllVarspt2.png",binsBibList=binsBibList,binsSigList=binsSigList)
    
    plotter.tracksBib["alpha"] =np.arctan2(1, plotter.tracksBib["cota"] )
    plotter.tracksSig["alpha"] =np.arctan2(1, plotter.tracksSig["cota"] )
    plotter.truthBib["alpha"] =np.arctan2(1, plotter.truthBib["cotAlpha"] )
    plotter.truthSig["alpha"] =np.arctan2(1, plotter.truthSig["cotAlpha"] )
    plotter.tracksBib["beta"] =np.arctan2(1, plotter.tracksBib["cotb"] )
    plotter.tracksSig["beta"] =np.arctan2(1, plotter.tracksSig["cotb"] )
    plotter.truthBib["beta"] =np.arctan2(1, plotter.truthBib["cotBeta"] )
    plotter.truthSig["beta"] =np.arctan2(1, plotter.truthSig["cotBeta"] )

    recalcStrs = ["", ""]    
    trackHeader = [ "ylocal", "zglobal",]    
    parquetTrackKeys = ["y-local","z-global"]
    xlabels = [r'$y_{\mathrm{local}}$ [mm]', r'$z_{\mathrm{global}}$ [mm]']
    keyLabels = [ r"$y_{\mathrm{local}}$", r'$z_{\mathrm{global}}$']
    keyLabels = ["", ""] 
    ylocalBins = np.arange(np.min(plotter.tracksBib["ylocal"].append(plotter.tracksSig["ylocal"]))-0.25,np.max(plotter.tracksBib["ylocal"])+0.25,0.25)
    # print(ylocalBins)
    # print(np.max(plotter.tracksBib["ylocal"]))
    binsBibList = [ylocalBins,None]
    binsSigList = [ylocalBins,None]
    legendLocs = [["best","best"],["best","best"]]
    plotUtils.plotSomeTracParqVars(plotter.tracksBib, plotter.tracksSig,plotter.truthBib,plotter.truthSig,PLOT_DIR=plotter.PLOT_DIR,interactivePlots=plotter.interactivePlots,
                     recalcStrs=recalcStrs,trackHeader=trackHeader,parquetTrackKeys=parquetTrackKeys,xlabels = xlabels,keyLabels = keyLabels,binsBibList = binsBibList,binsSigList = binsSigList,
                     figsize=(13.5,12),saveTitle="appendixAllVarspt3.png",legendLocs=legendLocs)

    recalcStrs = ["", ""]    
    trackHeader = ["alpha", "beta",]    
    parquetTrackKeys = ["alpha","beta"]
    xlabels = ["α", "β",]
    keyLabels = ["α", "β",] 
    keyLabels = ["", ""]  
    binsBibList = [None,None]
    binsSigList = [None,None]
    legendLocs = [["best","best"],["best","upper left"]]
    plotUtils.plotSomeTracParqVars(plotter.tracksBib, plotter.tracksSig,plotter.truthBib,plotter.truthSig,PLOT_DIR=plotter.PLOT_DIR,interactivePlots=plotter.interactivePlots,
                     recalcStrs=recalcStrs,trackHeader=trackHeader,parquetTrackKeys=parquetTrackKeys,xlabels = xlabels,keyLabels = keyLabels,binsBibList = binsBibList,binsSigList = binsSigList,
                     figsize=(13.5,12),saveTitle="appendixAllVarspt4.png",legendLocs=legendLocs)
    
    # df['adjusted_hit_time'] = df['hit_time']-1e6*np.sqrt(df['z-global']**2+30**2)/299792458
    # df['adjusted_hit_time_30ps_gaussian'] = df['adjusted_hit_time']+np.random.normal(loc=0,scale=30e-3,size=len(df['adjusted_hit_time']))
    # df['adjusted_hit_time_60ps_gaussian'] = df['adjusted_hit_time']+np.random.normal(loc=0,scale=60e-3,size=len(df['adjusted_hit_time']))

    plotter.truthBib["adjusted_hit_time"] = plotter.truthBib['hit_time']-1e6*np.sqrt(plotter.truthBib['z-global']**2+30**2)/299792458
    plotter.truthSig["adjusted_hit_time"] = plotter.truthSig['hit_time']-1e6*np.sqrt(plotter.truthSig['z-global']**2+30**2)/299792458
    plotter.truthBib['adjusted_hit_time_30ps_gaussian'] = plotter.truthBib['adjusted_hit_time']+np.random.normal(loc=0,scale=30e-3,size=len(plotter.truthBib['adjusted_hit_time']))
    plotter.truthSig['adjusted_hit_time_30ps_gaussian'] = plotter.truthSig['adjusted_hit_time']+np.random.normal(loc=0,scale=30e-3,size=len(plotter.truthSig['adjusted_hit_time']))
    
    plotter.tracksBib["adjusted_hit_time"] = plotter.tracksBib['t']-1e6*np.sqrt(plotter.tracksBib['zglobal']**2+30**2)/299792458
    plotter.tracksSig["adjusted_hit_time"] = plotter.tracksSig['t']-1e6*np.sqrt(plotter.tracksSig['zglobal']**2+30**2)/299792458
    plotter.tracksBib['adjusted_hit_time_30ps_gaussian'] = plotter.tracksBib['adjusted_hit_time']+np.random.normal(loc=0,scale=30e-3,size=len(plotter.tracksBib['adjusted_hit_time']))
    plotter.tracksSig['adjusted_hit_time_30ps_gaussian'] = plotter.tracksSig['adjusted_hit_time']+np.random.normal(loc=0,scale=30e-3,size=len(plotter.tracksSig['adjusted_hit_time']))

    recalcStrs = ["", ""]    
    trackHeader = ["adjusted_hit_time", "adjusted_hit_time_30ps_gaussian",]    
    parquetTrackKeys = ["adjusted_hit_time","adjusted_hit_time_30ps_gaussian"]
    xlabels = ["adjusted_hit_time", "adjusted_hit_time_30ps_gaussian",]
    keyLabels = ["adjusted_hit_time", "adjusted_hit_time_30ps_gaussian",] 
    keyLabels = ["", ""]  
    binsBibList = [None,None]
    binsSigList = [None,None]
    legendLocs = [["best","best"],["best","best"]]
    plotUtils.plotSomeTracParqVars(plotter.tracksBib, plotter.tracksSig,plotter.truthBib,plotter.truthSig,PLOT_DIR=plotter.PLOT_DIR,interactivePlots=plotter.interactivePlots,
                     recalcStrs=recalcStrs,trackHeader=trackHeader,parquetTrackKeys=parquetTrackKeys,xlabels = xlabels,keyLabels = keyLabels,binsBibList = binsBibList,binsSigList = binsSigList,
                     figsize=(13.5,12),saveTitle="appendixAllVarspt5.png",legendLocs=legendLocs)

if __name__=="__main__":
    main()