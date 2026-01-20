'''
Author: Daniel Abadjiev, calls code borrowed from Eliza Howard and Eric You
Date: January, 2026
Description: Class for plotting with plotUtils to validate dataset
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
import sys
sys.path.append("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/validationPlots/")
from plotUtils import *
import pickle
from pathlib import Path

class SmartpixPlotter():
    def __init__(self,
                 #flp=0, unused
                #  parquetDir_mm: str = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mm/", Not yet implemented
                #  parquetDir_mp: str = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mp/",
                #  parquetDir_sig: str = "/local/d1/smartpixML/bigData/Simulation_Output_Signal/",
                 parquetDir_all: str = "/local/d1/smartpixML/bigData/allData/",
                 skip_indices: list = list(range(1730 - 124+87, 1769)),
                 trackDirBib_mm: str = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mm/BIB_tracklists/',
                 trackDirBib_mp: str = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mp/BIB_tracklists/',
                 trackDirSig: str = '/local/d1/smartpixML/bigData/tracklists/signal_tracklists',
                 processRecon: bool = True,
                 interactivePlots: bool=False,
                 PLOT_DIR: os.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                 savedPklFromParquet: bool = False,
                 processTracks: bool = True,
                 processOldTracks:bool = False,
                 plotTracklists: bool = True,
                 plotParquets: bool = True,
                 styleSheet: str = 'seaborn-v0_8'
                 ):
        # self.parquetDir_mm = parquetDir_mm 
        # self.parquetDir_mp = parquetDir_mp 
        # self.parquetDir_sig = parquetDir_sig 
        self.parquetDir_all = parquetDir_all 
        self.skip_indices = skip_indices 
        self.trackDirBib_mm = trackDirBib_mm 
        self.trackDirBib_mp = trackDirBib_mp 
        self.trackDirSig = trackDirSig 
        self.processRecon = processRecon 
        self.interactivePlots= interactivePlots
        self.PLOT_DIR = PLOT_DIR 
        self.savedPklFromParquet = savedPklFromParquet 
        self.processTracks = processTracks 
        self.processOldTracks = processOldTracks
        self.plotTracklists = plotTracklists
        self.plotParquets = plotParquets

        self.styleSheet = styleSheet

        os.makedirs(self.PLOT_DIR, exist_ok=True)
        if (not processRecon) and (not savedPklFromParquet):
            raise ValueError("If reprocessing the parquets, must also do processRecon=True")
        
        self.loadParquetData()
        # if self.processTracks:
        self.loadTrackData()
        
        plt.style.use(self.styleSheet)
        return
    def loadParquetData(self):
        print(f"loading data, Currently loading settings: \nprocessRecon: {self.processRecon}"+
              f"\nsavedPklFromParquet: {self.savedPklFromParquet}\ninteractivePlots: {self.interactivePlots}")
        # Dataset with all the stuff
        if not self.savedPklFromParquet:
            self.truthDF, self.reconDF = load_parquet_pairs(self.parquetDir_all, skip_range=self.skip_indices)
            self.truthDF = genEtaAlphaBetaRq(self.truthDF)
            self.truthDF.to_pickle(Path(self.PLOT_DIR).joinpath("dfOfTruth.pkl"))
            if self.processRecon:
                self.reconDF.to_pickle(Path(self.PLOT_DIR).joinpath("dfOfRecon.pkl"))
        else:
            try:
                self.truthDF = pd.read_pickle(Path(self.PLOT_DIR).joinpath("dfOfTruth.pkl"))
            except:
                raise Exception("You may have to first save pkls before you can read them\nHint: set savedPklFromParquet to False")
            if self.processRecon:
                self.reconDF = pd.read_pickle(Path(self.PLOT_DIR).joinpath("dfOfRecon.pkl"))


        self.fracBib, self.fracSig, self.fracMM, self.fracMP,self.numTotalSig,self.numTotalBib,self.truthSig,self.truthBib_mm,self.truthBib_mp,self.truthBib = countBibSig(self.truthDF,doPrint=True)
        if self.processRecon:
            self.truthDF = genEtaAlphaBetaRq(self.truthDF)
            self.truthSig, self.truthBib,self.reconSig,self.reconBib_mm,self.reconBib_mp,self.reconBib,self.clustersSig,self.clustersBib,self.xSizesSig,self.xSizesBib,self.ySizesSig, self.ySizesBib,self.nPixelsSig,self.nPixelsBib,self.avgClustDictBib,self.avgClustDictSig,= processReconBibSig(self.truthDF,self.reconDF,doPrint=True)
            self.truthSig.to_pickle(Path(self.PLOT_DIR).joinpath("dfOfTruthSig.pkl"))
            self.truthBib.to_pickle(Path(self.PLOT_DIR).joinpath("dfOfTruthBib.pkl"))
            with open(Path(self.PLOT_DIR).joinpath("avgProfBib.pkl"),'wb') as handle:
                pickle.dump(self.avgClustDictBib,handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(Path(self.PLOT_DIR).joinpath("avgProfSig.pkl"),'wb') as handle:
                pickle.dump(self.avgClustDictSig,handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.truthSig = pd.read_pickle(Path(self.PLOT_DIR).joinpath("dfOfTruthSig.pkl"))
            self.truthBib = pd.read_pickle(Path(self.PLOT_DIR).joinpath("dfOfTruthBib.pkl"))
            self.xSizesBib=self.truthBib["xSize"]
            self.ySizesBib=self.truthBib["ySize"]
            self.nPixelsBib=self.truthBib["nPix"]
            self.xSizesSig=self.truthSig["xSize"]
            self.ySizesSig=self.truthSig["ySize"]
            self.nPixelsSig=self.truthSig["nPix"]
            with open(Path(self.PLOT_DIR).joinpath("avgProfBib.pkl"),'rb') as handle:
                self.avgClustDictBib = pickle.load(handle)
            with open(Path(self.PLOT_DIR).joinpath("avgProfSig.pkl"),'rb') as handle:
                self.avgClustDictSig = pickle.load(handle)
        print("Finished loading parquet data [not counting tracks], now proceeding to plotting or tracklists")
    def loadTrackData(self):
        if not self.processTracks:
            print("process Tracks is set to false, so not loading tracklists")
            # return
        print("Start loading track data")
        self.tracksBib, self.tracksSig, self.tracksBib_mp,self.tracksBib_mm=loadAllTracks(trackDirBib_mm=self.trackDirBib_mm,trackDirBib_mp=self.trackDirBib_mp,trackDirSig=self.trackDirSig, useBibSigIndic=(not self.processOldTracks))
        self.tracksBib = calcNxyzTrack(self.tracksBib)
        self.tracksSig = calcNxyzTrack(self.tracksSig)
        print("finished loading track data")
    def runPlots(self):
        print("Plotting settings:")
        print(f"interactivePlots: {self.interactivePlots}\nplotTracklists: {self.plotTracklists}\nplotParquets: {self.plotParquets}\n")
        print(f"Plotting directory: {self.PLOT_DIR}")
        if self.plotTracklists:
            if not self.processTracks:
                raise ValueError("Cannot plot tracklists if tracklists are not being processed (so invalid flag combo)")
            plotTrackPPt(self.tracksBib, self.tracksSig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotPtTrackAndParquet(self.tracksBib, self.tracksSig,self.truthBib,self.truthSig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotPCalcTrackComparison(self.tracksBib,bibSigLabel="BIB",PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotPCalcTrackComparison(self.tracksSig,bibSigLabel="Signal",PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)

            plotNxyzTrackParquet(self.tracksBib, self.tracksSig,self.truthBib,self.truthSig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
        if self.plotParquets:
            #Eliza's plots (the unique ones)
            plotCotACotBZY(self.truthBib,self.truthSig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotRadius(self.truthBib,self.truthSig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotYprofileYlocalRange(self.avgClustDictBib,titleBibSig="Bib",PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotYprofileYlocalRange(self.avgClustDictSig,titleBibSig="Signal",PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotYprofileYZRange(self.avgClustDictBib,self.avgClustDictSig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotClusterYSizes(self.truthBib,titleBibSig="Bib",PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotClusterYSizes(self.truthSig,titleBibSig="Signal",PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotXYProfile(self.truthBib, self.truthSig, self.avgClustDictSig, self.avgClustDictBib,
                                PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)

            #Eric's plots
            mask_bib,mask_sig,mask_bib_x,mask_sig_x,mask_bib_y,mask_sig_y, = getEricsMasks(self.truthBib, self.truthSig, self.xSizesSig, self.xSizesBib, self.ySizesSig, self.ySizesBib,)
            plotPt(self.truthSig,self.truthBib_mm,self.truthBib_mp,self.truthBib,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotPtEta(self.truthSig,self.truthBib,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotZglobalXsize(self.truthBib, self.truthSig, self.xSizesSig, self.xSizesBib,mask_bib,mask_sig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            # plotZglobalYsize(self.truthBib, self.truthSig, self.ySizesSig, self.ySizesBib,mask_bib_y,mask_sig_y,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotZglobalYsize(self.truthBib, self.truthSig, self.ySizesSig, self.ySizesBib,mask_bib,mask_sig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotZglobalXYsize(self.truthBib, self.truthSig, self.xSizesSig, self.xSizesBib, self.ySizesSig, self.ySizesBib,mask_bib,mask_sig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            ericsPlotReport(self.truthBib, self.truthSig, self.xSizesSig, self.xSizesBib, self.ySizesSig, self.ySizesBib,PLOT_DIR=self.PLOT_DIR)


            plotEricVarsHistos(self.truthBib, self.truthSig,self.nPixelsSig,self.nPixelsBib,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotEtaXYsize(self.truthBib, self.truthSig, self.xSizesSig, self.xSizesBib, self.ySizesSig, self.ySizesBib,mask_bib,mask_sig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotYlocalXYsize(self.truthBib, self.truthSig, self.xSizesSig, self.xSizesBib, self.ySizesSig, self.ySizesBib,mask_bib,mask_sig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotEhPt(self.truthBib, self.truthSig, mask_bib,mask_sig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
            plotPtLowHigh(self.truthBib, self.truthSig, mask_bib,mask_sig,PLOT_DIR=self.PLOT_DIR,interactivePlots=self.interactivePlots)
        print("Done plotting")
        return
    def plotSingleTrack(self, index):
        return
