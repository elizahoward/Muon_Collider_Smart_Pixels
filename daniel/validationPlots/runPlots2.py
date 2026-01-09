import os
import sys
sys.path.append("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/validationPlots")
from SpixPlotter import SmartpixPlotter

dataDir_mm = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mm/"
dataDir_mp = "/local/d1/smartpixML/bigData/SimOutput_0730_bigPPt_mp/"
dataDir_sig = "/local/d1/smartpixML/bigData/Simulation_Output_Signal/"
dataDir_all = "/local/d1/smartpixML/bigData/allData/"
dataDir_all = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/Dataset_1215To0108/Parquet_Files"

skip_indices = list(range(1730 - 124+87, 1769))  # 1606+87 [hand-tuned the 87] to 1768

trackDirBib_mm = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mm/BIB_tracklists/'
trackDirBib_mp = '/local/d1/smartpixML/reGenBIB/produceSmartPixMuC/Tracklists0730_mp/BIB_tracklists/'
trackDirSig = '/local/d1/smartpixML/bigData/tracklists/signal_tracklists'
trackDirBib_mm = '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/Dataset_1215To0108/Track_Lists'
trackDirBib_mp = None
trackDirSig = None


plotter = SmartpixPlotter(
                 dataDir_mm = dataDir_mm ,
                 dataDir_mp = dataDir_mp ,
                 dataDir_sig = dataDir_sig ,
                 dataDir_all = dataDir_all ,
                 skip_indices = list(range(1730 - 124+87, 1769)),
                 trackDirBib_mm = trackDirBib_mm,
                 trackDirBib_mp = trackDirBib_mp,
                 trackDirSig = trackDirSig,
                 processRecon = True,
                 interactivePlots=True,
                 PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                 savedPklFromParquet = False,
                 processTracks = True,
                 plotTracklists = True,
                 plotParquets = True,
                 )
plotter.runPlots()