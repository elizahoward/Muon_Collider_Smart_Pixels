import subprocess
import multiprocessing
import numpy as np
import os
import argparse
import sys
import Muon_Collider_Smart_Pixels.subprocess_utils as utils
from datetime import datetime

# user options
parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-j", "--ncpu", help="Number of cores to use", default=35, type=int)
parser.add_argument("-b", "--bin_size", help="Number of tracks per tracklist", default=500, type=int)
parser.add_argument("-t", "--track_total", help="Total number of tracks to simulate (for BIB and signal individually)", default=50000, type=int)
parser.add_argument("-f", "--float_precision", help="Floating point precision", default=5, type=int)
parser.add_argument("-bd", "--benchmark_dir", help="Muon collider simulation benchmark directory", default="/home/karri/mucLLPs/mucoll-benchmarks/", type=str)
ops = parser.parse_args()

# from bin size and total tracks, get number of tracklists
nTracklists = int(np.ceil(ops.track_total / ops.bin_size))

# get absolute path for semiparametric directory
pixelAVdir = os.path.expanduser(ops.pixelAVdir)

# Use date and time to create unique output directory
output_dir = f"Muon_Collider_Smart_Pixels/Data_Files/Data_Set_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
os.makedirs(output_dir)

# create output directories for each intermediate step
output_dir_pgun = f"{output_dir}/Particle_Gun"
output_dir_fluka = f"{output_dir}/FLUKA"
output_dir_detsim = f"{output_dir}/Detector_Sim"
output_dir_tracklists = f"{output_dir}/Track_Lists"
output_dir_pixelav = f"{output_dir}/PixelAV"
output_dir_parquet = f"{output_dir}/Parquet_Files"
os.makedirs(output_dir_pgun)
os.makedirs(output_dir_fluka)
os.makedirs(output_dir_detsim)
os.makedirs(output_dir_tracklists)
os.makedirs(output_dir_pixelav)
os.makedirs(output_dir_parquet)


commands = [] 

# Signal
for run in range(nTracklists): 

    signal_particle_gun = f"{output_dir_pgun}/particle_gun{run}.sclio"

    signal_detetor_sim = f"{output_dir_detsim}/signal_detsim{run}.slcio"

    signal_tracklist = f"{output_dir_tracklists}/signal_tracks{run}.txt"

    signal_pixelav_seed = f"{output_dir_pixelav}/signal_seed{run}"

    signal_pixelav_out = f"{output_dir_pixelav}/signal_pixelav{run}.out"

    signal_parquet = f"{output_dir_parquet}/signal_parquet{run}.parquet"

    # Particle gun
    run_particle_gun = ["python3", f"{ops.benchmark_dir}/generation/pgun/pgun_lcio.py", 
                         "-s", "12345", 
                         "-e", f"{ops.bin_size}", 
                         "--pdg", "13", "-13",
                         "--p", "1", "100", 
                         "--theta", "10", "170", 
                         "--dz", "0", "0", "1.5", 
                         "--d0", "0", "0", "0.0009",
                         f"--{signal_particle_gun}"]
    
    # Run Detector Simulation
    run_detsim = ["ddsim", "--steeringFile", f"{ops.benchmark_dir}/simulation/ilcsoft/steer_baseline.py",
                  "--inputFile", "output_gen.slcio",
                  "--outputFile", "output_sim.slcio"]

    # Make tracklist
    make_tracklist = ["python3",  "Muon_Collider_Smart_Pixels/MuC_Smartpix_Data_Production/Tracklist_Production/make_tracklists.py", 
                      "-i", signal_detetor_sim, 
                      "-o", signal_tracklist, 
                      "-f", str(ops.float_precision), 
                      "-p", "13", 
                      "-flp", "0"]
    
    # Run pixelAV
    run_pixelAV = [pixelAVdir, "Muon_Collider_Smart_Pixels/MuC_Smartpix_Data_Production/PixelAV/bin/ppixelav2_custom.exe", "1", signal_tracklist, signal_pixelav_out, signal_pixelav_seed]

    # Write parquet file
    # Fix input args to datagen
    make_parquet = ["python3", "Muon_Collider_Smart_Pixels/MuC_Smartpix_Data_Production/Data_Processing/datagen.py", 
                    "-i", signal_pixelav_out, 
                    "-o", signal_parquet]

    # commands
    commands.append([(make_tracklist, run_pixelAV, make_parquet,),]) # weird formatting is because pool expects a tuple at input

            
    
utils.run_commands(commands, ops.ncpu)
