
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.rcParams["figure.dpi"] = 150
import ipywidgets as widgets
from particle import PDGID
import sys
sys.path.append("../../../Muon_Collider_Smart_Pixels/daniel/validationPlots")
from plotUtils import *
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

from coordinateValidation_Utils import *



datadir = '/home/mwells5/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb_copy_m/'

flp = 0
trackDir = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0716/BIB_tracklists/'
trackDir = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0716_1/BIB_tracklists/'
trackDir = '/home/dabadjiev/smartpixels_ml_dsabadjiev/smartpixML/reGenBIB/MuonColliderSim/Tracklists/BIB_tracklists/'
trackHeader = ["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg"]
# data = pd.read_csv('.BIB_tracklist0.txt',sep=' ',header=None,names=trackHeader)


truthbib = pd.DataFrame()
recon2Dbib = pd.DataFrame()
truthsig = pd.DataFrame()
recon2Dsig = pd.DataFrame()
trackData = pd.DataFrame()

for file in os.listdir(datadir):
    if "labels" in file:
        if "bib" in file: 
            truthbib = pd.concat([truthbib,pd.read_parquet(f"{datadir}{file}")])
            file = file.replace("labels","recon2D")
            recon2Dbib = pd.concat([recon2Dbib,pd.read_parquet(f"{datadir}{file}")])
        elif "sig" in file: 
            truthsig = pd.concat([truthsig,pd.read_parquet(f"{datadir}{file}")])
            file = file.replace("labels","recon2D")
            recon2Dsig = pd.concat([recon2Dsig,pd.read_parquet(f"{datadir}{file}")])
for file in os.listdir(trackDir):
    if "BIB" in file:
        trackData = pd.concat([trackData,pd.read_csv(os.path.join(trackDir,file),sep=' ',header=None,names=trackHeader)])            
            
clustersSig = recon2Dsig.to_numpy().reshape(recon2Dsig.shape[0],13,21)
clustersBib = recon2Dbib.to_numpy().reshape(recon2Dbib.shape[0],13,21)



print(f"# of bib clusters: {len(truthbib)}\n# of sig clusters {len(truthsig)}")
print(f"Total # of clusters: {len(truthbib)+len(truthsig)}")
# print(f"keys bib {recon2Dbib.keys()} ")
print(f"and of truth {truthbib.keys()}")
# print(truthbib.head())
print(f"length of tracklist {len(trackData)}")
print(trackData.head())



def populate_module(data_array):
    cut_array = np.array()
    for idx, val in zip(data_array.index, data_array['z-global']):
        if val in range (0,13):
            cut_array.append(data_df[idx])
    module_array = cut_array.to_numpy().reshape(cut_df.shape[0],520,520)
    return module_array


def plotModule(module_array):
    ##add 2d numpy array
    fig, ax = plt.subplots(figsize=(7,5),dpi=200)

    # Plot charge collected in each pixel
    datamin = module_array.min()
    datamax = module_array.max()
    im = ax.imshow(module_array, vmin=datamin, vmax=datamax, cmap=cmap, interpolation='nearest')
    # ax.hlines(y=centery,xmax=20,xmin=0)
    # ax.vlines(x=centerx,ymax=12, ymin=0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    fig.colorbar(im, cax=cax, location='right',label='Number of eh pairs')
    ax.set_title("Charge collected by 4 ns")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    fig.colorbar(im, cax=cax, location='right',label='Number of eh pairs')

    # Draw grid on both
    # Unit conversiions: modules as 130 x 130 mm or 520 by 520 pixels.

    ax.set_xlim(-0.5,519.5)
    ax.set_ylim(-0.5,519.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5,snap=False)
    
    plt.tight_layout(pad=3.5)
    fig.canvas.draw()