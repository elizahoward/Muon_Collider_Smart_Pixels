
# written by Megan Wells, with input from Ryan Roberts and Eliza Howard 
import numpy as np 
import pandas as pd
import math            
import pickle                   
import os 
import argparse
import gc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
matplotlib.rcParams["figure.dpi"] = 150

# information is from tracks regenerated with moduleID (pre PixelAV data)
# and Daniel's pkls of truthbib labels parquets (post PixelAV data)
trackPath = '/home/mwells5/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb_copy_m/tl_with_moduleID_07_28_2026/'
pklPath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb/plots/dfOfTruth.pkl"
flp = 0

parser = argparse.ArgumentParser(description="Options for what you want your plot to look like.")

parser.add_argument("-source", type='str', help="Choose whether to make from tracks (t) or parquets (p).")
parser.add_argument("-cax", type='str', help=f"Color axis parameter, i.e. pt, hit_time. For full list of options use -source listkeys")
parser.add_argument("-log", help="Sets colorbar norm to logarithmic, and returns a warning if colorbar scale includes negative values.")
parser.add_argument("-zcenter", help="Automatically chooses a zglobal range of one module width at the center of barrel.")
parser.add_argument("-zedge", help="Automatically chooses a zglobal range of one module width at the edge of barrel.")
parser.add_argument("-aht_tight", help="Automatically applies tight adjusted hit time cut (between -0.09 and 0.15 ns).")
parser.add_argument("-aht_loose", help="Automatically applies loose adjusted hit time cut (between -0.15 and 15 ns).")
parser.add_argument("-cut", help="Prompts for a user-entered cut on one or more parameters.")

# if (-cut passed in):
#     print("Index of keys", keys)
#     print("Enter cut parameter from list of keys\n")
#     store entry
#     print("Inclusive minimum bound\n")
#     store entry
#     print("Inclusive maximum bound\n")
#     store entry
#     print("Add another? [y/n]")
#     store entry
#     if y:
#         repeat
#     else: 
#         exit loop


with open(pklPath, 'rb') as file:
    # Reconstruct the Python object
    parquetData = pickle.load(file)

trackData = pd.DataFrame()
trackdata_list = []

trackHeader = ["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg", "moduleID"]

for file in os.listdir(trackPath):
    if "bib_mm" in file:
        trackdata_list.append(pd.read_csv(f"{trackPath}{file}", sep=' ', names=trackHeader))
    elif "bib_mp" in file: 
        trackdata_list.append(pd.read_csv(f"{trackPath}{file}", sep=' ', names=trackHeader))

trackData = pd.concat(trackdata_list)
del trackdata_list
gc.collect()

trackData['adjusted_hit_time'] = trackData['t']-1e6*np.sqrt(trackData['zglobal']**2+30**2)/299792458

