import subprocess
import multiprocessing
import numpy as np
import os
import argparse
import sys
from datetime import datetime

parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--ncpu", help="Number of cores to use", default=35, type=int)
parser.add_argument("-bs", "--bin_size", help="Number of tracks per tracklist", default=10, type=int)
parser.add_argument("-t", "--track_total", help="Total number of tracks to simulate (for BIB and signal individually)", default=10, type=int)
parser.add_argument("-f", "--float_precision", help="Floating point precision", default=5, type=int)
parser.add_argument("-flp", "--flp", help="Direction of sensor (1 for FE side out, 0 for FE side down)", default=0, type=int)
parser.add_argument("-bd", "--benchmark_dir", help="Muon collider simulation benchmark directory", default="/home/karri/mucLLPs/mucoll-benchmarks/", type=str)
ops = parser.parse_args()

# from bin size and total tracks, get number of tracklists
nTracklists = 1

def run_executable(executable_path, options):
    command = [executable_path] + options
    subprocess.run(command)

def run_commands(commands):
    for command in commands:
        print(command)
        if "pixelav" in command[0]:
            subprocess.run(command[1:], cwd=command[0])
        
        subprocess.run(command)

def pool_commands(commands, num_cores):
    # Create a pool of processes to run in parallel
    pool = multiprocessing.Pool(num_cores)
    
    # Launch the executable N times in parallel with different options
    # pool.starmap(run_executable, [(path_to_executable, options) for options in options_list])
    #print(commands) 
    pool.starmap(run_commands, commands)
    
    # Close the pool of processes
    pool.close()
    pool.join()


# set up MuColl environment

# Step 1: Source the setup.sh and capture the resulting environment
def get_env_from_setup(script_path):
    # Launch a new bash shell, source the script, and print the environment
    command = f"bash -c 'source {script_path} >/dev/null 2>&1 && env'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to source {script_path}:\n{result.stderr}")

    env = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            env[key] = value

    return env

output_dir = f"Data_Files/Data_Set" 

# create output directories for each intermediate step
output_dir_pgun = f"{output_dir}/Particle_Gun"
output_dir_fluka = f"{output_dir}/FLUKA"
output_dir_detsim = f"{output_dir}/Detector_Sim"
output_dir_tracklists = f"{output_dir}/Track_Lists"
output_dir_pixelav = f"{output_dir}/PixelAV"
output_dir_parquet = f"{output_dir}/Parquet_Files"

# Step 2: Load environment from setup.sh
#env = os.environ.copy()
#env.update(get_env_from_setup('/cvmfs/muoncollider.cern.ch/release/2.8-patch2/setup.sh'))


commands = [] 

print(f"Running {nTracklists} tracklists with {ops.bin_size} tracks each, using {ops.ncpu} cores")

# Signal
for run in range(nTracklists): 

    # Define file names
    signal_particle_gun = f"{output_dir_pgun}/particle_gun{run}.sclio"
    signal_detetor_sim = f"{output_dir_detsim}/signal_detsim{run}.slcio"
    signal_tracklist = f"{output_dir_tracklists}/signal_tracks{run}.txt"
    signal_pixelav_seed = f"{output_dir_pixelav}/signal_seed{run}"
    signal_pixelav_out = f"{output_dir_pixelav}/signal_pixelav{run}.out"
    signal_parquet = f"{output_dir_parquet}/signal_*{run}.parquet" # include * here so it can be easily replaced when the different parquet files are written

    # Particle gun
    run_particle_gun = ["python3", f"{ops.benchmark_dir}/generation/pgun/pgun_lcio.py", 
                         "-s", "12345", 
                         "-e", f"{ops.bin_size}", 
                         "--pdg", "13", "-13",
                         "--p", "1", "100", 
                         "--theta", "10", "170", 
                         "--dz", "0", "0", "1.5", 
                         "--d0", "0", "0", "0.0009",
                        "--", signal_particle_gun]
    
    # Run Detector Simulation
    run_detsim = ["ddsim", "--steeringFile", f"{ops.benchmark_dir}/simulation/ilcsoft/steer_baseline.py",
                  "--inputFile", "output_gen.slcio",
                  "--outputFile", "output_sim.slcio"]

    # Make tracklist
    make_tracklist = ["python3",  "MuC_Smartpix_Data_Production/Tracklist_Production/make_tracklists.py", 
                      "-i", signal_detetor_sim, 
                      "-o", signal_tracklist, 
                      "-f", str(ops.float_precision), 
                      "-p", "13", 
                      "-flp", f"{ops.flp}"]
    
    # Run pixelAV
    run_pixelAV = ["./MuC_Smartpix_Data_Production/PixelAV","./bin/ppixelav2_custom.exe", 
                   "1", 
                   signal_tracklist, 
                   signal_pixelav_out, 
                   signal_pixelav_seed]

    # Write parquet file
    make_parquet = ["python3", "/MuC_Smartpix_Data_Production/Data_Processing/datagen.py", 
                    "-i", signal_pixelav_out, 
                    "-o", signal_parquet]

    # commands
    commands.append([(run_particle_gun,),]) # weird formatting is because pool expects a tuple at input

# Run in parallel
pool_commands(commands, ops.ncpu)
