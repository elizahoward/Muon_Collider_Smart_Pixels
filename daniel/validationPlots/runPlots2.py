import os
import sys
sys.path.append("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/validationPlots")
from SpixPlotter import SmartpixPlotter
import argparse
from pathlib import Path



def main(parquetDir_all,               
            #  skip_indices = list(range(1730 - 124+87, 1769)),
            trackDirBib_mm,
            trackDirBib_mp,
            trackDirSig,
            processRecon,
            interactivePlots,
            PLOT_DIR,
            savedPklFromParquet,
            processTracks,
            plotTracklists,
            plotParquets,):
    plotter = SmartpixPlotter(
                    #  parquetDir_mm = parquetDir_mm , #Not yet implemented
                    #  parquetDir_mp = parquetDir_mp ,
                    #  parquetDir_sig = parquetDir_sig ,
                    parquetDir_all = parquetDir_all ,
                    skip_indices = list(range(1730 - 124+87, 1769)),
                    trackDirBib_mm = trackDirBib_mm,
                    trackDirBib_mp = trackDirBib_mp,
                    trackDirSig = trackDirSig,
                    processRecon = processRecon,
                    interactivePlots=interactivePlots,
                    PLOT_DIR = PLOT_DIR,# os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                    savedPklFromParquet = savedPklFromParquet,
                    processTracks = processTracks,
                    plotTracklists = plotTracklists,
                    plotParquets = plotParquets,
                    )
    plotter.runPlots()



parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--ncpu", help="Does nothing", default=35, type=int)
parser.add_argument("-da", "--dataDir_all", help = "Directory with Eliza's format, containing Parquet_Files and Track_Lists subdirectories", default = None, type = str)
parser.add_argument("-dp", "--parquetDir_all", help = "Directory with parquet files (both signal and bib in the same folder)", default = None, type = str)
parser.add_argument("-dmm", "--trackDirBib_mm", help = "Directory with tracklists of bib of type mm", default = None, type = str)
parser.add_argument("-dmp", "--trackDirBib_mp", help = "Directory with tracklists of bib of type mm", default = None, type = str)
parser.add_argument("-ds", "--trackDirSig", help = "Directory with tracklists of signal", default = None, type = str)
parser.add_argument("-dplt", "--PLOT_DIR", help = "Directory for plot outputs", default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"), type = str)
parser.add_argument("-dd","--defaultDirs", action = "store_true")
parser.add_argument("-pr", "--processRecon", action = 'store_true')
parser.add_argument("-pt", "--processTracks", action = 'store_true')
parser.add_argument("-pp", "--processParquets", action = 'store_true')
parser.add_argument("-pltt", "--plotTracklists", action = 'store_true')
parser.add_argument("-pltp", "--plotParquets", action = 'store_true')
parser.add_argument("-plti", "--interactivePlots", action = 'store_true')

ops = parser.parse_args()


dataDir_all = ops.dataDir_all
parquetDir_all = ops.parquetDir_all
trackDirBib_mm = ops.trackDirBib_mm
trackDirBib_mp = ops.trackDirBib_mp
trackDirSig = ops.trackDirSig
PLOT_DIR = ops.PLOT_DIR
defaultDirs = ops.defaultDirs

processRecon = ops.processRecon
processTracks = ops.processTracks
# processParquets = ops.processParquets
savedPklFromParquet = not ops.processParquets
plotTracklists = ops.plotTracklists
plotParquets = ops.plotParquets
interactivePlots = ops.interactivePlots


repodir = Path(__file__).resolve().parent.parent.parent
print("Assuming you are starting in .....daniel/validationPlots, so git repo dir is ", repodir)
assert repodir.parts[-1] == 'Muon_Collider_Smart_Pixels'

if defaultDirs:
    print("Using default directories")
    if (parquetDir_all is not None) or (trackDirBib_mm is not None) or (trackDirBib_mp is not None) or (trackDirSig is not None) or (None is not None):
        raise ValueError("Cannot use custom directories if using default directories flag") 
    parquetDir_mm = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mm/"
    parquetDir_mp = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mp/"
    parquetDir_sig = "/local/d1/smartpixML/bigData/Simulation_Output_Signal/"
    parquetDir_all = "/local/d1/smartpixML/bigData/allData/"
    parquetDir_all = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/Dataset_1215To0108/Parquet_Files"

    skip_indices = list(range(1730 - 124+87, 1769))  # 1606+87 [hand-tuned the 87] to 1768

    trackDirBib_mm = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mm/BIB_tracklists/'
    trackDirBib_mp = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mp/BIB_tracklists/'
    trackDirSig = '/local/d1/smartpixML/bigData/tracklists/signal_tracklists'
    trackDirBib_mm = '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/Dataset_1215To0108/Track_Lists'
    trackDirBib_mp = None
    trackDirSig = None

    print("Repo dir (currently unused)",repodir)

    # PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
if dataDir_all is not None:
    print("Using a dataset directory with subfolders, following Eliza's 2026 dataset format, ignoring other directories")
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
    #Need to implement a combined tracklist directory
    # raise NotImplementedError("Didn't actually implement this yet")

if plotTracklists and not processTracks:
    raise ValueError("Cannot plot tracklists without processing tracklists")

main(parquetDir_all,               
            #  skip_indices = list(range(1730 - 124+87, 1769)),
            trackDirBib_mm,
            trackDirBib_mp,
            trackDirSig,
            processRecon,
            interactivePlots,
            PLOT_DIR,
            savedPklFromParquet,
            processTracks,
            plotTracklists,
            plotParquets,)
