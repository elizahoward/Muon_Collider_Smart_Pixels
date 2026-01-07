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
from pathlib import Path
import sys
repodir = Path(__file__).resolve().parent.parent
sys.path.append(f"{repodir}/daniel/validationPlots/")
from plotUtils import *
import pickle


data_main_dir = f"{repodir}/Data_Files/Data_Set_20260107_140951"


flp = 0
# trackHeader = ["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg"]
dataDir_mm = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mm/"
# dataDir_mp = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mp/"
dataDir_sig = f"{data_main_dir}/Parquet_Files"
#dataDir_all = "/local/d1/smartpixML/bigData/allData/"

#skip_indices = list(range(1730 - 124+87, 1769))  # 1606+87 [hand-tuned the 87] to 1768

trackDirBib_mm = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mm/BIB_tracklists/'
# trackDirBib_mp = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mp/BIB_tracklists/'
trackDirSig = f"{data_main_dir}/Track_Lists"

processRecon = True;

interactivePlots=False;
PLOT_DIR = data_main_dir
os.makedirs(PLOT_DIR, exist_ok=True)
savedPklFromParquet = False;

processTracks = True;

print(f"loading data, Currently loading settings: \nprocessRecon: {processRecon}\nsavedPklFromParquet: {savedPklFromParquet}\ninteractivePlots: {interactivePlots}")
# Dataset with all the stuff
if not savedPklFromParquet:
    truthDF, reconDF = load_parquet_pairs(dataDir_sig)
    # truthDF2, reconDF2 = load_parquet_pairs(dataDir_mm)
    # truthDF = pd.concat([truthDF,truthDF2])
    # reconDF = pd.concat([reconDF,reconDF2])
    truthDF = genEtaAlphaBetaRq(truthDF)
    truthDF.to_pickle("dfOfTruth.pkl")
    if processRecon:
        reconDF.to_pickle("dfOfRecon.pkl")
else:
    try:
        truthDF = pd.read_pickle("dfOfTruth.pkl")
    except:
        raise Exception("You may have to first save pkls before you can read them\nHint: set savedPklFromParquet to False")
    if processRecon:
        reconDF = pd.read_pickle("dfOfRecon.pkl")


fracBib, fracSig, fracMM, fracMP,numTotalSig,numTotalBib,truthSig,truthBib_mm,truthBib_mp,truthBib = countBibSig(truthDF,doPrint=True)
if processRecon:
    truthDF = genEtaAlphaBetaRq(truthDF)
    truthSig, truthBib,reconSig,reconBib_mm,reconBib_mp,reconBib,clustersSig,clustersBib,xSizesSig,xSizesBib,ySizesSig, ySizesBib,nPixelsSig,nPixelsBib,avgClustDictBib,avgClustDictSig,= processReconBibSig(truthDF,reconDF,doPrint=True)
    truthSig.to_pickle("dfOfTruthSig.pkl")
    truthBib.to_pickle("dfOfTruthBib.pkl")
    with open("avgProfBib.pkl",'wb') as handle:
        pickle.dump(avgClustDictBib,handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("avgProfSig.pkl",'wb') as handle:
        pickle.dump(avgClustDictSig,handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    truthSig = pd.read_pickle("dfOfTruthSig.pkl")
    truthBib = pd.read_pickle("dfOfTruthBib.pkl")
    xSizesBib=truthBib["xSize"]
    ySizesBib=truthBib["ySize"]
    nPixelsBib=truthBib["nPix"]
    xSizesSig=truthSig["xSize"]
    ySizesSig=truthSig["ySize"]
    nPixelsSig=truthSig["nPix"]
    with open("avgProfBib.pkl",'rb') as handle:
        avgClustDictBib = pickle.load(handle)
    with open("avgProfSig.pkl",'rb') as handle:
        avgClustDictSig = pickle.load(handle)
print("Finished loading data [not counting tracks], now plotting")

if processTracks:
    print("Start loading track data")
    tracksBib, tracksSig, tracksBib_mp,trackDirBib_mm=loadAllTracks(trackDirBib_mm=trackDirBib_mm,trackDirBib_mp=trackDirBib_mp,trackDirSig=trackDirSig)
    tracksBib = calcNxyzTrack(tracksBib)
    tracksSig = calcNxyzTrack(tracksSig)
    print("finished loading track data")
    
    plotTrackPPt(tracksBib, tracksSig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
    plotPtTrackAndParquet(tracksBib, tracksSig,truthBib,truthSig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
    plotPCalcTrackComparison(tracksBib,bibSigLabel="BIB",PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
    plotPCalcTrackComparison(tracksSig,bibSigLabel="Signal",PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)

    plotNxyzTrackParquet(tracksBib, tracksSig,truthBib,truthSig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)


#Eliza's plots (the unique ones)
plotRadius(truthBib,truthSig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotYprofileYlocalRange(avgClustDictBib,titleBibSig="Bib",PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotYprofileYlocalRange(avgClustDictSig,titleBibSig="Signal",PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotYprofileYZRange(avgClustDictBib,avgClustDictSig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotClusterYSizes(truthBib,titleBibSig="Bib",PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotClusterYSizes(truthSig,titleBibSig="Signal",PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotXYProfile(truthBib, truthSig, avgClustDictSig, avgClustDictBib,
                      PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)

#Eric's plots
mask_bib,mask_sig,mask_bib_x,mask_sig_x,mask_bib_y,mask_sig_y, = getEricsMasks(truthBib, truthSig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,)
plotPt(truthSig,truthBib_mm,truthBib_mp,truthBib,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotZglobalXsize(truthBib, truthSig, xSizesSig, xSizesBib,mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
# plotZglobalYsize(truthBib, truthSig, ySizesSig, ySizesBib,mask_bib_y,mask_sig_y,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotZglobalYsize(truthBib, truthSig, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotZglobalXYsize(truthBib, truthSig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
ericsPlotReport(truthBib, truthSig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,PLOT_DIR=PLOT_DIR)


plotEricVarsHistos(truthBib, truthSig,nPixelsSig,nPixelsBib,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotEtaXYsize(truthBib, truthSig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotYlocalXYsize(truthBib, truthSig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotEhPt(truthBib, truthSig, mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotPtLowHigh(truthBib, truthSig, mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)

