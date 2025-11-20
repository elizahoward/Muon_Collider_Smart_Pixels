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
    # print("I might have broken this funciton, might need to be called on each dataframe subtype")
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
    return truthSig, truthBib, reconSig,reconBib_mm,reconBib_mp,reconBib,clustersSig,clustersBib, xSizesSig, xSizesBib,ySizesSig, ySizesBib,nPixelsSig,nPixelsBib


skip_indices = list(range(1730 - 124+87, 1769))  # 1606+87 [hand-tuned the 87] to 1768

def plotPt(truthSig,truthBib_mm,truthBib_mp,truthBib,PLOT_DIR='./plots',interactivePlots=False):
    plt.figure(figsize=(14, 6))
    key = "pt"
    showNums = False
    plt.subplot(411)
    bins = np.linspace(-0.1,0.1,100)
    plotManyHisto([truthSig[key],truthBib_mm[key],truthBib_mp[key],truthBib[key]],title=f"{key} distribution",pltStandalone=False,pltLabels=[f"sig {key}",f"bib mm {key}",f"bib mp {key}",f"bib {key}"],bins=bins,showNums=showNums,figsize=(7,2))
    plt.ylabel("Tracks")
    plt.subplot(412)
    bins = np.linspace(-1,10,100)
    plotManyHisto([truthSig[key],truthBib_mm[key],truthBib_mp[key],truthBib[key]],title="",pltStandalone=False,pltLabels=[f"sig {key}",f"bib mm {key}",f"bib mp {key}",f"bib {key}"],bins=bins,showNums=showNums,figsize=(7,2),yscale='log')
    plt.ylabel("Tracks")
    plt.subplot(413)
    bins = np.logspace(-1,2.04,100)
    bins = np.concatenate(([-1,0,0.01],bins,[110,114]))
    plotManyHisto([truthSig[key],truthBib_mm[key],truthBib_mp[key],truthBib[key]],title="",pltStandalone=False,pltLabels=[f"sig {key}",f"bib mm {key}",f"bib mp {key}",f"bib {key}"],bins=bins,showNums=showNums,figsize=(7,2),yscale="linear")
    plt.ylabel("Tracks")
    plt.subplot(414)
    # bins = np.logspace(-1,2.1,100)
    plotManyHisto([truthSig[key],truthBib_mm[key],truthBib_mp[key],truthBib[key]],title="",pltStandalone=False,pltLabels=[f"sig {key}",f"bib mm {key}",f"bib mp {key}",f"bib {key}"],bins=bins,showNums=showNums,figsize=(7,2),yscale="log")
    plt.ylabel("Tracks")
    plt.xlabel("momentum p_T")
    plt.savefig(os.path.join(PLOT_DIR, "bib_signal_pt.png"), dpi=300, bbox_inches='tight')
    if interactivePlots:
        plt.show()
    else:
        plt.close()

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
    
    ax[0].hist2d(z_global_bib[mask_bib], xSizesBib[mask_bib], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    ax[0].set_title("BIB", fontsize=20)
    ax[0].set_xlabel('z-global [mm]', fontsize=24)
    ax[0].set_ylabel('x-size (# pixels)', fontsize=24)
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    
    # Right panel: Signal
    # mask_sig = ~np.isnan(z_global_sig) & ~np.isnan(xSizesSig)
    # mask_sig = mask_sig & np.isfinite(z_global_sig) & np.isfinite(xSizesSig)
    
    ax[1].hist2d(z_global_sig[mask_sig], xSizesSig[mask_sig], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    ax[1].set_title("Signal", fontsize=20)
    ax[1].set_xlabel('z-global [mm]', fontsize=24)
    ax[1].set_ylabel('x-size (# pixels)', fontsize=24)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bib_signal_zglobal_vs_xsize_comparison.png"), dpi=300, bbox_inches='tight')
    if interactivePlots:
        plt.show()
    else:
        plt.close()
    
    print(f"Plot saved as: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_xsize_comparison.png')}")
    print(f"BIB z-global range: {z_global_bib.min():.2f} to {z_global_bib.max():.2f} mm")
    print(f"Signal z-global range: {z_global_sig.min():.2f} to {z_global_sig.max():.2f} mm")
    print(f"BIB x-size range: {xSizesBib.min()} to {xSizesBib.max()} pixels")
    print(f"Signal x-size range: {xSizesSig.min()} to {xSizesSig.max()} pixels")


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
    
    ax[0].hist2d(z_global_bib[mask_bib_y], ySizesBib[mask_bib_y], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    ax[0].set_title("BIB - Y Size", fontsize=63)
    ax[0].set_xlabel('z-global [mm]', fontsize=50)
    ax[0].set_ylabel('y-size (# pixels)', fontsize=50)
    ax[0].tick_params(axis='both', which='major', labelsize=40)
    
    # Right panel: Signal
    # mask_sig_y = ~np.isnan(z_global_sig) & ~np.isnan(ySizesSig)
    # mask_sig_y = mask_sig_y & np.isfinite(z_global_sig) & np.isfinite(ySizesSig)
    
    ax[1].hist2d(z_global_sig[mask_sig_y], ySizesSig[mask_sig_y], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    ax[1].set_title("Signal - Y Size", fontsize=63)
    ax[1].set_xlabel('z-global [mm]', fontsize=50)
    ax[1].set_ylabel('y-size (# pixels)', fontsize=50)
    ax[1].tick_params(axis='both', which='major', labelsize=40)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bib_signal_zglobal_vs_ysize_comparison.png"), dpi=300, bbox_inches='tight')
    if interactivePlots:
        plt.show()
    else:
        plt.close()

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
    axes[0,0].hist2d(z_global_bib[mask_bib], xSizesBib[mask_bib], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    axes[0,0].set_title("BIB - X Size", fontsize=38)
    axes[0,0].set_xlabel('z-global [mm]', fontsize=31)
    axes[0,0].set_ylabel('x-size (# pixels)', fontsize=31)
    axes[0,0].tick_params(axis='both', which='major', labelsize=23)
    
    # Top-right: Signal X-size
    axes[0,1].hist2d(z_global_sig[mask_sig], xSizesSig[mask_sig], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    axes[0,1].set_title("Signal - X Size", fontsize=38)
    axes[0,1].set_xlabel('z-global [mm]', fontsize=31)
    axes[0,1].set_ylabel('x-size (# pixels)', fontsize=31)
    axes[0,1].tick_params(axis='both', which='major', labelsize=23)
    
    # Bottom row: Y-size comparisons
    # Bottom-left: BIB Y-size
    axes[1,0].hist2d(z_global_bib[mask_bib_y], ySizesBib[mask_bib_y], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    axes[1,0].set_title("BIB - Y Size", fontsize=38)
    axes[1,0].set_xlabel('z-global [mm]', fontsize=31)
    axes[1,0].set_ylabel('y-size (# pixels)', fontsize=31)
    axes[1,0].tick_params(axis='both', which='major', labelsize=23)
    
    # Bottom-right: Signal Y-size
    axes[1,1].hist2d(z_global_sig[mask_sig_y], ySizesSig[mask_sig_y], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    axes[1,1].set_title("Signal - Y Size", fontsize=38)
    axes[1,1].set_xlabel('z-global [mm]', fontsize=31)
    axes[1,1].set_ylabel('y-size (# pixels)', fontsize=31)
    axes[1,1].tick_params(axis='both', which='major', labelsize=23)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bib_signal_zglobal_vs_xysize_grid_comparison.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, "bib_signal_zglobal_vs_xysize_grid_comparison_transparent.png"), dpi=300, bbox_inches='tight',transparent=True)
    if interactivePlots:
        plt.show()
    else:
        plt.close()



def ericsPlotReport(truthbib, truthsig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,PLOT_DIR="./plots"):
    z_global_bib = truthbib['z-global']
    z_global_sig = truthsig['z-global']
    print(f"Plots saved:")
    print(f"1. X-size only: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_xsize_comparison_with_ysize.png')}")
    print(f"2. Y-size only: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_ysize_comparison.png')}")
    print(f"3. 2x2 grid: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_xysize_grid_comparison.png')}")
    
    print(f"\nBIB statistics:")
    print(f"  z-global range: {z_global_bib.min():.2f} to {z_global_bib.max():.2f} mm")
    print(f"  x-size range: {xSizesBib.min()} to {xSizesBib.max()} pixels")
    print(f"  y-size range: {ySizesBib.min()} to {ySizesBib.max()} pixels")
    
    print(f"\nSignal statistics:")
    print(f"  z-global range: {z_global_sig.min():.2f} to {z_global_sig.max():.2f} mm")
    print(f"  x-size range: {xSizesSig.min()} to {xSizesSig.max()} pixels")
    print(f"  y-size range: {ySizesSig.min()} to {ySizesSig.max()} pixels")

#Adapted from Eric's plot_signal_data.py

def genEtaAlphaBeta(truthDF):
    if 'z-global' in truthDF.columns:
        theta = np.arctan(30 / truthDF['z-global'])
        truthDF['eta'] = -np.log(np.tan(theta / 2))

    # cotAlpha, cotBeta may already be present, but recalculate if not
    if 'cotAlpha' not in truthDF.columns and {'n_x', 'n_z'}.issubset(truthDF.columns):
        truthDF['cotAlpha'] = truthDF['n_x'] / truthDF['n_z']
    if 'cotBeta' not in truthDF.columns and {'n_y', 'n_z'}.issubset(truthDF.columns):
        truthDF['cotBeta'] = truthDF['n_y'] / truthDF['n_z']
    if 'R' not in truthDF.columns:
        #added from Eliza's code
        truthDF['R'] = truthDF['pt']*5.36/(1.60217663*3.57)*1000 # [mm]

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

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "signal_bib_summary_histograms.png"))
    if interactivePlots:
        plt.show()
    else:
        plt.close()

# --- Plot 2: 2D histograms of eta vs x-size/y-size ---
def plotEtaXYsize(truthbib, truthsig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.03}, figsize=(10,7))

    # Before each hist2d call, filter out NaN and inf values for both axes
    mask_96 = ~np.isnan(truthsig['eta']) & ~np.isnan(xSizesSig)
    mask_96 = mask_96 & np.isfinite(truthsig['eta']) & np.isfinite(xSizesSig)
    ax[0,1].hist2d(truthsig['eta'][mask_96], xSizesSig[mask_96], bins=[30, np.arange(0,9,1)], cmap='Blues')
    ax[0,1].set_title("Signal", fontsize=15)

    mask_98 = ~np.isnan(truthsig['eta']) & ~np.isnan(ySizesSig)
    mask_98 = mask_98 & np.isfinite(truthsig['eta']) & np.isfinite(ySizesSig)
    ax[1,1].hist2d(truthsig['eta'][mask_98], ySizesSig[mask_98], bins=[30, np.arange(0,14,1)], cmap='Blues')
    ax[1,1].set_xlabel('η', fontsize=15)

    ax[0,0].set_ylabel('x-size (# pixels)', fontsize=15)
    ax[1,0].set_ylabel('y-size (# pixels)', fontsize=15)

    #Copying Eric's code but for bib
    # Before each hist2d call, filter out NaN and inf values for both axes
    mask_96 = ~np.isnan(truthbib['eta']) & ~np.isnan(xSizesBib)
    mask_96 = mask_96 & np.isfinite(truthbib['eta']) & np.isfinite(xSizesBib)
    ax[0,0].hist2d(truthbib['eta'][mask_96], xSizesBib[mask_96], bins=[30, np.arange(0,9,1)], cmap='Blues')
    ax[0,0].set_title("BIB", fontsize=15)

    mask_98 = ~np.isnan(truthbib['eta']) & ~np.isnan(ySizesBib)
    mask_98 = mask_98 & np.isfinite(truthbib['eta']) & np.isfinite(ySizesBib)
    ax[1,0].hist2d(truthbib['eta'][mask_98], ySizesBib[mask_98], bins=[30, np.arange(0,14,1)], cmap='Blues')
    ax[1,0].set_xlabel('η', fontsize=15)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "signal_eta_vs_size_2d.png"))
    if interactivePlots:
        plt.show()
    else:
        plt.close()

# --- Plot 3: 2D histograms of y-local vs x-size/y-size ---
def plotYlocalXYsize(truthbib, truthsig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.03}, figsize=(10,7))

    # Before each hist2d call, filter out NaN and inf values for both axes
    mask_111 = ~np.isnan(truthsig['y-local']) & ~np.isnan(xSizesSig)
    mask_111 = mask_111 & np.isfinite(truthsig['y-local']) & np.isfinite(xSizesSig)
    ax[0,1].hist2d(truthsig['y-local'][mask_111], xSizesSig[mask_111], bins=[30, np.arange(0,9,1)], cmap='Blues')
    ax[0,1].set_title("Signal", fontsize=15)

    mask_113 = ~np.isnan(truthsig['y-local']) & ~np.isnan(ySizesSig)
    mask_113 = mask_113 & np.isfinite(truthsig['y-local']) & np.isfinite(ySizesSig)
    ax[1,1].hist2d(truthsig['y-local'][mask_113], ySizesSig[mask_113], bins=[30, np.arange(0,14,1)], cmap='Blues')
    ax[1,1].set_xlabel('y-local [μm]', fontsize=15)

    ax[0,0].set_ylabel('x-size (# pixels)', fontsize=15)
    ax[1,0].set_ylabel('y-size (# pixels)', fontsize=15)

    


    mask_111 = ~np.isnan(truthbib['y-local']) & ~np.isnan(xSizesBib)
    mask_111 = mask_111 & np.isfinite(truthbib['y-local']) & np.isfinite(xSizesBib)
    ax[0,0].hist2d(truthbib['y-local'][mask_111], xSizesBib[mask_111], bins=[30, np.arange(0,9,1)], cmap='Blues')
    ax[0,0].set_title("BIB", fontsize=15)

    mask_113 = ~np.isnan(truthbib['y-local']) & ~np.isnan(ySizesBib)
    mask_113 = mask_113 & np.isfinite(truthbib['y-local']) & np.isfinite(ySizesBib)
    ax[1,0].hist2d(truthbib['y-local'][mask_113], ySizesBib[mask_113], bins=[30, np.arange(0,14,1)], cmap='Blues')
    ax[1,0].set_xlabel('y-local [μm]', fontsize=15)


    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "signal_ylocal_vs_size_2d.png"))
    if interactivePlots:
        plt.show()
    else:
        plt.close()

# --- Plot 4: 2D histogram of number_eh_pairs vs pt ---
def plotEhPt(truthbib, truthsig, mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    # Before each hist2d call, filter out NaN and inf values for both axes
    mask_125 = ~np.isnan(truthsig['number_eh_pairs']) & ~np.isnan(truthsig['pt'])
    mask_125 = mask_125 & np.isfinite(truthsig['number_eh_pairs']) & np.isfinite(truthsig['pt'])
    ax[1].hist2d(truthsig['number_eh_pairs'][mask_125], truthsig['pt'][mask_125], bins=30, cmap='Blues')
    ax[1].set_title("Signal", fontsize=15)
    ax[1].set_ylabel('pt (GeV)', fontsize=15)
    ax[1].set_xlabel('number of electron hole pairs', fontsize=15)

    mask_125 = ~np.isnan(truthbib['number_eh_pairs']) & ~np.isnan(truthbib['pt'])
    mask_125 = mask_125 & np.isfinite(truthbib['number_eh_pairs']) & np.isfinite(truthbib['pt'])
    ax[0].hist2d(truthbib['number_eh_pairs'][mask_125], truthbib['pt'][mask_125], bins=30, cmap='Blues')
    ax[0].set_title("BIB", fontsize=15)
    ax[0].set_ylabel('pt (GeV)', fontsize=15)
    ax[0].set_xlabel('number of electron hole pairs', fontsize=15)


    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "signal_ehpairs_vs_pt_2d.png"))
    if interactivePlots:
        plt.show()
    else:
        plt.close()

# --- Plot 5: charge separation for low and high pt ---
def plotPtLowHigh(truthbib, truthsig, mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    truthsig['q'] = truthsig['PID'].apply(lambda pid: PDGID(int(pid)).charge if pd.notnull(pid) else np.nan)
    truthbib['q'] = truthbib['PID'].apply(lambda pid: PDGID(int(pid)).charge if pd.notnull(pid) else np.nan)
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



    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "signal_pt_charge_separation.png"))
    if interactivePlots:
        plt.show()
    else:
        plt.close()

    print(f"Plots saved in {PLOT_DIR}") 

#Added plots from Eliza's code (several are also just wrapped into Eric's code)

