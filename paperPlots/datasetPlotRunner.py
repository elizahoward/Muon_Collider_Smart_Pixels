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

if __name__=="__main__":
    main()