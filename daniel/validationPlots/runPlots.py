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


flp = 0
# trackHeader = ["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg"]
dataDir_mm = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mm/"
dataDir_mp = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mp/"
dataDir_sig = "/local/d1/smartpixML/bigData/Simulation_Output_Signal/"
dataDir_all = "/local/d1/smartpixML/bigData/allData/"

<<<<<<< Updated upstream
skip_indices = list(range(1730 - 124+87, 1769))  # 1606+87 [hand-tuned the 87] to 1768

doRecon = True;

interactivePlots=True;
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

=======


skip_indices = list(range(1730 - 124+87, 1769))  # 1606+87 [hand-tuned the 87] to 1768

>>>>>>> Stashed changes

# Dataset with all the stuff
savedPkl = True;
if not savedPkl:
    truthDF, reconDF = load_parquet_pairs(dataDir_all, skip_range=skip_indices)
    truthDF.to_pickle("dfOfTruth.pkl")
<<<<<<< Updated upstream
    if doRecon:
        reconDF.to_pickle("dfOfRecon.pkl")
else:
    truthDF = pd.read_pickle("dfOfTruth.pkl")
    if doRecon:
        reconDF = pd.read_pickle("dfOfRecon.pkl")


fracBib, fracSig, fracMM, fracMP,numTotalSig,numTotalBib,truthSig,truthBib_mm,truthBib_mp,truthBib = countBibSig(truthDF,doPrint=True)
if doRecon:
    reconSig,reconBib_mm,reconBib_mp,reconBib,clustersSig,clustersBib,xSizesSig,xSizesBib,ySizesSig, ySizesBib,= processReconBibSig(truthDF,reconDF,doPrint=True)

mask_bib,mask_sig,mask_bib_x,mask_sig_x,mask_bib_y,mask_sig_y, = getEricsMasks(truthBib, truthSig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,)
plotZglobalXsize(truthBib, truthSig, xSizesSig, xSizesBib,mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
# plotZglobalYsize(truthBib, truthSig, ySizesSig, ySizesBib,mask_bib_y,mask_sig_y,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotZglobalYsize(truthBib, truthSig, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
plotZglobalXYsize(truthBib, truthSig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)
ericsPlotReport(truthBib, truthSig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,PLOT_DIR=PLOT_DIR)
# plt.hist([truthDF["z-global"],truthBib["z-global"],truthSig["z-global"]],stacked=True,histtype='step')
=======
else:
    truthDF = pd.read_pickle("dfOfTruth.pkl")


fracBib, fracSig, fracMM, fracMP,numTotalSig,numTotalBib,truthSig,truthBib_mm,truthBib_mp,truthBib = countBibSig(truthDF,doPrint=True)

plt.hist(truthDF["z-global"])
>>>>>>> Stashed changes
plt.show()

