#Author: Daniel Abadjiev
#Date: May 8, 2026
#Description: Runner for varPredPlotUtils.py, that runs the functions over a folder, taken from dataRate.ipynb

import varPredPlotUtils
import os
import argparse
from pathlib import Path
import tensorflow as tf 
import multiprocessing
tf.config.set_visible_devices([], 'GPU')

parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--ncpu", help="Does nothing", default=10, type=int)
parser.add_argument("-d", "--pareto_dir", help = "Directory with h5 files, and script runs over all .h5 files inside", default = None, type = str)
# parser.add_argument("-dp", "--parquetDir_all", help = "Directory with parquet files (both signal and bib in the same folder)", default = None, type = str)
# parser.add_argument("-dmm", "--trackDirBib_mm", help = "Directory with tracklists of bib of type mm", default = None, type = str)
# parser.add_argument("-dmp", "--trackDirBib_mp", help = "Directory with tracklists of bib of type mm", default = None, type = str)
# parser.add_argument("-ds", "--trackDirSig", help = "Directory with tracklists of signal", default = None, type = str)
parser.add_argument("-dplt", "--PLOT_DIR", help = "Directory for plot outputs", default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ratePlots"), type = str)
# parser.add_argument("-dd","--defaultDirs", action = "store_true")
# parser.add_argument("-pr", "--processRecon", action = 'store_true')
# parser.add_argument("-pt", "--processTracks", action = 'store_true')
# parser.add_argument("-po", "--processOldTracks", action = 'store_true')
# parser.add_argument("-pp", "--processParquets", action = 'store_true')
# parser.add_argument("-pltt", "--plotTracklists", action = 'store_true')
# parser.add_argument("-pltp", "--plotParquets", action = 'store_true')
parser.add_argument("-plti", "--interactivePlots", action = 'store_true')
# parser.add_argument("-s","--styleSheet",default="seaborn-v0_8-colorblind",type=str)
parser.add_argument("-s","--styleSheet",default="tableau-colorblind10",type=str) #ggplot, classic, default are also acceptable. Seaborn stylesheets mess things up

ops = parser.parse_args()


paretoDir = ops.pareto_dir
interactivePlots = ops.interactivePlots
PLOT_DIR = ops.PLOT_DIR
nCPU = ops.ncpu

# paths = [(e.path if e.is_file() else "") for e in os.scandir(paretoDir)]
if paretoDir is not None:
    paths = [e.path for e in os.scandir(paretoDir) if e.is_file() and ".h5" in e.path]
else:
    paths = ['/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model3_fin_results/model3_10bit_normalised_selected/pareto_primary/model_trial_084.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model3_fin_results/model3_10bit_normalised_selected/pareto_primary/model_trial_110.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model3_fin_results/model3_8bit_normalised_selected/pareto_primary/model_trial_019.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model3_fin_results/model3_10bit_normalised_selected/pareto_primary/model_trial_016.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model3_fin_results/model3_10bit_normalised_selected/pareto_primary/model_trial_090.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model3_fin_results/model3_10bit_normalised_selected/pareto_primary/model_trial_088.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model3_fin_results/model3_8bit_normalised_selected/pareto_primary/model_trial_100.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model3_fin_results/model3_6bit_normalised_selected/pareto_primary/model_trial_056.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_8bit_normalised_selected/model_trial_063.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_8bit_normalised_selected/model_trial_065.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_8bit_normalised_selected/model_trial_043.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_10bit_normalised_selected/model_trial_087.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_8bit_normalised_selected/model_trial_049.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_10bit_normalised_selected/model_trial_081.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_8bit_normalised_selected/model_trial_029.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_10bit_normalised_selected/model_trial_029.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_10bit_normalised_selected/model_trial_057.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_8bit_normalised_selected/model_trial_088.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_10bit_normalised_selected/model_trial_088.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_6bit_normalised_selected/model_trial_033.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_10bit_normalised_selected/model_trial_033.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_8bit_normalised_selected/model_trial_051.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model2.5_fin_results/model2.5_10bit_normalised_selected/model_trial_051.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/8w0i_i10_sigmoid/hp4q_model_trial_007.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/8w0i_i10_sigmoid/hp4q_model_trial_006.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/10w0i_i12_sigmoid/hp3q_model_trial_005.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/10w0i_i12_sigmoid/hp3q_model_trial_001.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/10w0i_i12_sigmoid/hp3q_model_trial_007.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/8w0i_i10_sigmoid/hp4q_model_trial_008.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/6w0i_i8_sigmoid/hp3q_model_trial_006.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/6w0i_i8_sigmoid/hp4q_model_trial_009.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/6w0i_i8_sigmoid/hp3q_model_trial_000.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/10w0i_i12_sigmoid/hp3q_model_trial_004.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/8w0i_i10_sigmoid/hp3q_model_trial_002.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/8w0i_i10_sigmoid/hp2q_model_trial_006.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/10w0i_i12_sigmoid/hp2q_model_trial_004.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/8w0i_i10_sigmoid/hp2q_model_trial_004.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/6w0i_i8_sigmoid/hp2q_model_trial_004.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/10w0i_i12_sigmoid/hp4q_model_trial_001.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/8w0i_i10_sigmoid/hp2q_model_trial_002.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/8w0i_i10_sigmoid/hp3q_model_trial_005.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/8w0i_i10_sigmoid/hp2q_model_trial_008.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/6w0i_i8_sigmoid/hp2q_model_trial_008.h5', '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Final_Results/model1_fin_results/3w0i_i5_sigmoid/hp2q_model_trial_008.h5']

# for e in os.scandir(paretoDir):
#     if e.is_file():
#         if ".h5" in e.path:

            

def runForPath(path):
    modelID = path[-10:]
    print(path)
    print(modelID)
    pltDir = PLOT_DIR+"/mdl_"+modelID
    # Path(pltDir).mkdir(parents=True, exist_ok=True) #moved to runModelPlots so that it can modify based on backround rejection
    varPredPlotUtils.runModelPlots(filepath = path,PLOT_DIR=pltDir, interactivePlots=interactivePlots,extendTitle=path[25:])

def main():
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # This catches child processes trying to set it again and lets them pass safely
        pass
    with multiprocessing.Pool(processes=nCPU) as pool:
        pool.map(runForPath,paths)
    # for path in paths:
    #     runForPath(path)


if __name__ == '__main__':
    main()