"""
Author: Daniel Abadjiev (based on code by Eliza based on code by smartpixels collaboration)
Date: July 30, 2026
Description: Script to make figure 3 in the paper, the cluster along with x/yprofile and size depicted
uses data from the

Based on notebook plotting_clusters.ipynb by Eliza 

"""

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

plt.style.use("seaborn-v0_8-poster")
# import sys
# sys.path.append('/home/elizahoward/cmspix28-mc-sim/utils')


# def plotClusterWithProfiles(cluster, cmap="Blues",doColorbar=False):
#     fig, ax_main = plt.subplots(figsize=(9, 8))
    
#     im, ax_main = plotSingleCluster(cluster, cmap=cmap, ax=ax_main,doColorbar=False,title="")
    
#     x_profile = np.sum(cluster, axis=0)
#     y_profile = np.sum(cluster, axis=1)

#     divider = make_axes_locatable(ax_main)
    
#     ax_top = divider.append_axes("top", size="18%", pad=0.15, sharex=ax_main)
#     ax_right = divider.append_axes("right", size="18%", pad=0.15, sharey=ax_main)
#     cax = divider.append_axes("right", size="4%", pad=0.4)

#     x_bins = np.arange(len(x_profile))
#     ax_top.step(x_bins, x_profile, where='mid', color='dodgerblue', lw=2)
#     ax_top.set_ylabel("x-size")
#     ax_top.set_title("Charge collected by 4 ns", pad=15)
#     ax_top.tick_params(axis='x', labelbottom=False, bottom=True)

#     y_bins = np.arange(len(y_profile))
#     ax_right.step(y_profile, y_bins, where='mid', color='dodgerblue', lw=2)
#     ax_right.set_xlabel("y-size")
#     ax_right.tick_params(axis='y', labelleft=False, left=True)
#     if doColorbar:
#         fig.colorbar(im, cax=cax, orientation='vertical', label='Number of eh pairs')

#     plt.show()

def get_profile_bounds_fraction(profile):
    # Find all indices where the charge count is greater than zero
    non_zeros = np.where(profile > 0)[0]
    if len(non_zeros) == 0:
        return 0.15, 0.85 # Fallback to standard proportions if empty
    
    first_idx = non_zeros[0]
    last_idx = non_zeros[-1]
    
    # Pixel boundaries for these indices map from (first - 0.5) to (last + 0.5)
    # The complete tracking axis limits are -0.5 to (len(profile) - 0.5)
    axis_min = -0.5
    axis_max = len(profile) - 0.5
    axis_range = axis_max - axis_min
    
    # Convert absolute data boundaries into layout fraction positions (0.0 to 1.0)
    frac_start = ((first_idx - 0.5) - axis_min) / axis_range
    frac_end = ((last_idx + 0.5) - axis_min) / axis_range
    
    return frac_start, frac_end

def plotClusterWithProfiles(cluster,figsize=(10.8, 7)):
    fig, ax_main = plt.subplots(figsize=figsize)
    
    # Call your core function cleanly without creating a ghost colorbar axis
    im, ax_main = plotSingleCluster(cluster, ax=ax_main, doColorbar=False, title="")
    
    x_profile = np.sum(cluster, axis=0)
    y_profile = np.sum(cluster, axis=1)
    # print(x_profile)
    # print(y_profile)

    # Create layout divider for profiles
    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size="18%", pad=0.15, sharex=ax_main)
    ax_right = divider.append_axes("right", size="18%", pad=0.15, sharey=ax_main)

    histOff = 1 # for visualization to see the 0

    # --- Top Panel (X-Profile) ---
    x_bins = np.arange(0,22,1) -0.5
    # ax_top.step(x_bins, x_profile, where='mid',lw=1.5)
    ax_top.stairs( x_profile*10, edges=x_bins,baseline=histOff, lw=1.5,)

    ax_top.tick_params(axis='x', labelbottom=False, bottom=True)
    ax_top.tick_params(axis='y', labelleft=False, left=False)
    ax_top.set_yscale('log')

    # Top Labels & Arrow Placement
    ax_top.text(0.50, 1.2, r'$x_{\mathrm{size}}$', transform=ax_top.transAxes, ha='center', va='bottom',fontsize=plt.rcParams['axes.labelsize'])
    # ax_top.text(0.5, 1.05, r'$\longleftrightarrow$', transform=ax_top.transAxes, ha='center', va='bottom',fontsize=plt.rcParams['axes.labelsize'])
    # ax_top.text(1.01, 0.5, 'x-profile', transform=ax_top.transAxes, ha='left', va='center', rotation=-90,fontsize=plt.rcParams['axes.labelsize'])
    ax_top.text(0.02, 0.3, r'$x_{\mathrm{profile}}$', transform=ax_top.transAxes, ha='left', va='center',fontsize=plt.rcParams['axes.labelsize'])
    x_start, x_end = get_profile_bounds_fraction(x_profile)
    ax_top.annotate('', xy=(x_start, 1.12), xytext=(x_end, 1.12), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='<->', color='black',lw=1.5))
 
    # --- Right Panel (Y-Profile) ---
    y_bins = np.arange(0,14,1) -0.5
    # ax_right.step(y_profile, y_bins, where='mid',  lw=1.5)        
    ax_right.stairs(y_profile, edges=y_bins, baseline=histOff, orientation='horizontal', lw=1.5)    
    ax_right.tick_params(axis='x', labelbottom=False, bottom=False)
    ax_right.tick_params(axis='y', labelleft=False, left=True)
    ax_right.set_xscale('log')

    # Right Labels & Arrow Placement (re-ordered vertically to match top panel format)
    ax_right.text(0.22, 0.75, r'$y_{\mathrm{profile}}$', transform=ax_right.transAxes, ha='center', va='bottom',rotation=-90,fontsize=plt.rcParams['axes.labelsize'])
    ax_right.text(1.2, 0.545, r'$y_{\mathrm{size}}$', transform=ax_right.transAxes, ha='left', va='center', rotation=-90,fontsize=plt.rcParams['axes.labelsize'])
    # ax_right.text(1.05, 0.5, r'$\longleftrightarrow$', transform=ax_right.transAxes, ha='left', va='center',rotation=90,fontsize=plt.rcParams['axes.labelsize'])
    y_start, y_end = get_profile_bounds_fraction(y_profile)
    ax_right.annotate('', xy=(1.15, y_start), xytext=(1.15, y_end), xycoords='axes fraction',
                      arrowprops=dict(arrowstyle='<->', color='black',lw=1.5))


    plt.show()

def plotSingleCluster(cluster,cmap="Blues", ax = plt.gca(),doColorbar=True,title="Charge collected by 4 ns"):
    # fig, ax = plt.subplots(figsize=(7,5),dpi=200)


    # Plot charge collected in each pixel
    datamin = cluster.min()
    dataminPos = cluster[cluster>0].min()
    # print(dataminPos)
    datamax = cluster.max()
    im = ax.imshow(cluster, cmap=cmap, interpolation='nearest',norm=matplotlib.colors.LogNorm(vmin=0.001,vmax=datamax))
    if doColorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        plt.colorbar(im, cax=cax, location='right',label='Number of eh pairs')
    ax.set_title(title)


    # Draw grid on both
    ax.set_xlim(-0.5,20.5)
    ax.set_ylim(-0.5,12.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5,snap=False)

    plt.xlabel(r"$x_{\mathrm{local}}$",fontsize=25)
    plt.ylabel(r"$y_{\mathrm{local}}$",fontsize=25)
    
    plt.tight_layout(pad=3.5)
    return im, ax
    # fig.canvas.draw()

def plotCluster(truth, recon2D, index):
    label = truth.iloc[index]
    cluster = recon2D.iloc[index].to_numpy().reshape(13,21)
    
    fig, ax = plt.subplots(figsize=(7,5),dpi=200)

    # Plot charge collected in each pixel
    datamin = 0 #cluster.min()
    datamax = 8000 #cluster.max()
    im = ax.imshow(cluster, vmin=datamin, vmax=datamax, cmap=cmap,interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    fig.colorbar(im, cax=cax, location='right',label='Number of eh pairs')
    ax.set_title("Charge collected by 4 ns")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    fig.colorbar(im, cax=cax, location='right',label='Number of eh pairs')

    # Draw grid on both
    ax.set_xlim(-0.5,20.5)
    ax.set_ylim(-0.5,12.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5,snap=False)
    
    # Adjust truth values for coordinate system (the origin is in the center of the ROI)
    xentry = label['x-entry']/50 + 21/2
    yentry = label['y-entry']/12.5 + 13/2
    xmid = label['x-midplane']/50 + 21/2
    ymid = label['y-midplane']/12.5 + 13/2
    
    ax.plot(xentry,yentry, 'b.',label='entry point')
    ax.plot(xmid,ymid, 'g.',label='midplane point')
    ax.legend()
    
    plt.tight_layout(pad=3.5)
    fig.canvas.draw()

def main():
    pklPath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb/plots/dfOfTruth.pkl"
    # --- Load data ---
    # print(f"Loading truthDF from {pklPath}")
    # truthDF = pd.read_pickle(pklPath)
    pklPath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb/plots/dfOfRecon.pkl" 
    print(f"Loading reconDF from {pklPath}")
    reconDF = pd.read_pickle(pklPath)
    # clustersSig = recon2Dsig.to_numpy().reshape(recon2Dsig.shape[0],13,21)
    # print(reconDF.iloc[0,0:273].to_numpy())
    index = 373742
    index = 21
    index = 819117
    reconDF = reconDF.query("source == 'bib_mp'")
    nPix = np.count_nonzero(reconDF.iloc[:,0:273].to_numpy(),axis=1)
    print(np.where(nPix>10))
    cluster = reconDF.iloc[index,0:273].to_numpy().reshape(13,21).astype(float)
    print(reconDF.iloc[index,273])
    print(np.unique(reconDF.iloc[:,273]))
    print((reconDF.iloc[:,273]))
    print(len(reconDF))
    # print(cluster)
    plotSingleCluster(cluster)
    plt.savefig("./cluster.png")

    plotClusterWithProfiles(cluster,figsize=(8.1,5.28))
    plt.savefig("./clusterFeatures.png")

    # print(cluster)

    # print(reconDF)

    return


if __name__=="__main__":
    main()