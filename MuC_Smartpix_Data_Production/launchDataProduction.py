import numpy as np
import os
import argparse
from datetime import datetime
from pathlib import Path
import shutil
#from subprocess_utils import *
import subprocess
import multiprocessing

def run_executable(executable_path, options):
    command = [executable_path] + options
    subprocess.run(command)

def run_commands(commands):
    for command in commands:
        print(command)
        if "PixelAV" in command[0]:
            subprocess.run(command[1:], cwd=command[0])
        elif "ddsim" in command[0] or "pgun_lcio.py" in command[1] or "make_tracklists.py" in command[1]:
            subprocess.run(command, env=env)
        else:
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

def generate_signal():

    # Load evironment for parquet conversion process
    #env2 = os.environ.copy()
    #env2.update(get_env_from_setup(f"{repodir}/env"))

    #subprocess.run("source /cvmfs/muoncollider.cern.ch/release/2.8-patch2/setup.sh", shell=True, executable="/bin/bash")

    print(f"Running {nTracklists} tracklists with {ops.bin_size} tracks each, using {ops.ncpu} cores")

    # 
    # First run particle gun and detector sim, then convert to track lists that pixelAV can use
    # This is not run in parallel because each time you start this step, it takes a lot of time to reload a lot of stuff that takes a while
    #

    commands = [] 

    signal_particle_gun = f"{output_dir_pgun}/particle_gun.slcio"
    signal_detetor_sim = f"{output_dir_detsim}/signal_detsim.slcio"

    # Particle gun
    run_particle_gun = ["python3", f"{ops.benchmark_dir}/generation/pgun/pgun_lcio.py", 
                            "-s", "12345", 
                            "-e", f"{ops.track_total}", 
                            "--pdg", "13", "-13",
                            "--p", "1", "100", 
                            "--theta", "10", "170", 
                            "--dz", "0", "0", "1.5", 
                            "--d0", "0", "0", "0.0009",
                            "--", signal_particle_gun]

    # Run Detector Simulation
    run_detsim = ["ddsim", "--steeringFile", f"{ops.benchmark_dir}/simulation/ilcsoft/steer_baseline.py",
                    "--inputFile", signal_particle_gun,
                    "--outputFile", signal_detetor_sim]

    # Make tracklist
    make_tracklist = ["python3",  f"{repodir}/MuC_Smartpix_Data_Production/Tracklist_Production/make_tracklists.py", 
                        "-i", signal_detetor_sim, 
                        "-odir", output_dir_tracklists, 
                        "-f", str(ops.float_precision), 
                        "-b", str(ops.bin_size),
                        "-flp", f"{ops.flp}", 
                        "-t", str(ops.track_total),
                        "-sig"]
    
    if ops.plot_in_maketracklists:
        make_tracklist.append("-p")

    # Construct tuple of commands based on which step we are starting from
    if START_STEP == 0:
        command_tuple = (run_particle_gun, run_detsim, make_tracklist,)
    elif START_STEP == 1:
        command_tuple = (run_detsim, make_tracklist,)
    elif START_STEP == 2:
        command_tuple = (make_tracklist,)
    else:
        command_tuple = None

    if command_tuple is not None:
        commands.append([command_tuple,])
        # Run in parallel
        pool_commands(commands, ops.ncpu)

    # 
    # Next run pixelAV and convert results to parquets
    # This is run in parallel because pixelAV is an iterative simulation an a single run takes a while but doesn't use 
    # too much computation power, so we can run a bunch in parallel without overwhelming the workstation
    #

    commands = []
                    
    for run in range(nTracklists): 

        # Define file names
        signal_tracklist = f"{output_dir_tracklists}/signal_tracks_{run}.txt"
        signal_pixelav_seed = f"{output_dir_pixelav}/signal_seed_{run}"
        signal_pixelav_out = f"{output_dir_pixelav}/signal_pixelav_{run}.out"
        signal_pixelav_log = f"{output_dir_pixelav}/signal_pixelav_log_{run}.txt"
        signal_parquet = f"{output_dir_parquet}/signal_*_{run}.parquet" # include * here so it can be easily replaced when the different parquet files are written
        
        # Run pixelAV Muon_Collider_Smart_Pixels/MuC_Smartpix_Data_Production/PixelAV
        run_pixelAV = [f"{repodir}/MuC_Smartpix_Data_Production/PixelAV", "./bin/ppixelav2_custom.exe", 
                    "1", 
                    signal_tracklist, 
                    signal_pixelav_out, 
                    signal_pixelav_seed,
                    signal_pixelav_log
                    ]

        # Write parquet file
        make_parquet = ["python3", f"{repodir}/MuC_Smartpix_Data_Production/Data_Processing/datagen.py", 
                        "-i", signal_pixelav_out, 
                        "-o", signal_parquet]

        # Construct tuple of commands based on which step we are starting from
        if START_STEP <= 3:
            command_tuple=(run_pixelAV, make_parquet,)
        elif START_STEP == 4:
            command_tuple=(make_parquet,)
        commands.append([command_tuple,]) # weird formatting is because pool expects a tuple at input

    # Run in parallel
    pool_commands(commands, ops.ncpu)

def generate_bib():
    # Make tracklist
    if START_STEP<=2:
        make_tracklist_bibmm = ["python3",  f"{repodir}/MuC_Smartpix_Data_Production/Tracklist_Production/make_tracklists.py", 
                            "-odir", output_dir_tracklists, 
                            "-f", str(ops.float_precision), 
                            "-b", str(ops.bin_size),
                            "-flp", f"{ops.flp}", 
                            "-t", str(int(ops.track_total/2)),
                            "-bmm"]
        
        make_tracklist_bibmp = ["python3",  f"{repodir}/MuC_Smartpix_Data_Production/Tracklist_Production/make_tracklists.py", 
                            "-odir", output_dir_tracklists, 
                            "-f", str(ops.float_precision), 
                            "-b", str(ops.bin_size),
                            "-flp", f"{ops.flp}", 
                            "-t", str(int(ops.track_total/2)),
                            "-bmp"]
        
        if ops.plot_in_maketracklists:
            make_tracklist_bibmm.append("-p")
            make_tracklist_bibmp.append("-p")
        
        command_tuple=(make_tracklist_bibmp,make_tracklist_bibmm,)

        commands= [[command_tuple,]] # weird formatting is because pool expects a tuple at input

        # Run in parallel
        pool_commands(commands, ops.ncpu)

    # 
    # Next run pixelAV and convert results to parquets
    # This is run in parallel because pixelAV is an iterative simulation an a single run takes a while but doesn't use 
    # too much computation power, so we can run a bunch in parallel without overwhelming the workstation
    #

    commands = []
                    
    for run in range(nTracklists): 

        for bib_type in ["mm","mp"]:

            # Define file names
            bib_tracklist = f"{output_dir_tracklists}/bib_{bib_type}_tracks_{run}.txt"
            bib_pixelav_seed = f"{output_dir_pixelav}/bib_{bib_type}_seed_{run}"
            bib_pixelav_out = f"{output_dir_pixelav}/bib_{bib_type}_pixelav_{run}.out"
            bib_pixelav_log = f"{output_dir_pixelav}/bib_{bib_type}_pixelav_log_{run}.txt"
            bib_parquet = f"{output_dir_parquet}/bib_{bib_type}_*_{run}.parquet" # include * here so it can be easily replaced when the different parquet files are written
            
            # Run pixelAV Muon_Collider_Smart_Pixels/MuC_Smartpix_Data_Production/PixelAV
            run_pixelAV = [f"{repodir}/MuC_Smartpix_Data_Production/PixelAV", "./bin/ppixelav2_custom.exe", 
                        "1", 
                        bib_tracklist, 
                        bib_pixelav_out, 
                        bib_pixelav_seed,
                        bib_pixelav_log
                        ]

            # Write parquet file
            make_parquet = ["python3", f"{repodir}/MuC_Smartpix_Data_Production/Data_Processing/datagen.py", 
                            "-i", bib_pixelav_out, 
                            "-o", bib_parquet]

            
            # Construct tuple of commands based on which step we are starting from
            if START_STEP <= 3:
                command_tuple=(run_pixelAV, make_parquet,)
            elif START_STEP == 4:
                command_tuple=(make_parquet,)
            commands.append([command_tuple,]) # weird formatting is because pool expects a tuple at input

    # Run in parallel
    pool_commands(commands, ops.ncpu)

# user options
parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--ncpu", help="Number of cores to use", default=35, type=int)
parser.add_argument("-bs", "--bin_size", help="Number of tracks per track list", default=10, type=int)
parser.add_argument("-t", "--track_total", help="Total number of tracks to simulate (for BIB and signal individually)", default=10, type=int)
parser.add_argument("-f", "--float_precision", help="Floating point precision", default=5, type=int)
parser.add_argument("-flp", "--flp", help="Direction of sensor (1 for FE side out, 0 for FE side down)", default=0, type=int)
parser.add_argument("-bd", "--benchmark_dir", help="Muon collider simulation benchmark directory", default="/home/karri/mucLLPs/mucoll-benchmarks/", type=str)
parser.add_argument("-p", "--plot_in_maketracklists", help="Make plots during the process of making tracklists", action='store_true')
parser.add_argument("-sig", "--signal", help="Are you generating signal?", action='store_true')
parser.add_argument("-bib", "--bib", help="Are you running bib?", action='store_true')

# If you want to provide your own intermediate files as a starting point, specify their directories with these options
parser.add_argument("-pg", "--pgun_dir", help="Directory containing particle gun output", default=None, type=str)
parser.add_argument("-ds", "--detsim_dir", help="Directory containing detector simulation output", default=None, type=str)
parser.add_argument("-tl", "--tracklist_dir", help="Directory containing track list outputs", default=None, type=str)
parser.add_argument("-pav", "--pixelav_dir", help="Directory containing pixelav outputs", default=None, type=str)

ops = parser.parse_args()

if not ops.signal and not ops.bib:
    raise ValueError("You must include -sig for signal and/or -bib for generating bib")

# from bin size and total tracks, get number of track lists
if ops.tracklist_dir is not None:
    nTracklists = len(os.listdir(ops.tracklist_dir)) # Fix later to count only relevant files
elif ops.pixelav_dir is not None:
    nTracklists = len(os.listdir(ops.pixelav_dir)) # Fix later to count only relevant files
else:
    nTracklists = int(np.ceil(ops.track_total / ops.bin_size))

repodir = Path(__file__).resolve().parent.parent

# determine which step to start from
start_options = [ops.pgun_dir, ops.detsim_dir, ops.tracklist_dir, ops.pixelav_dir]
start_options_bool = [bool(x) for x in start_options]
if sum(start_options_bool) == 0:
    START_STEP = 0  # start from particle gun

elif sum(start_options_bool) != 1:
    raise ValueError("Please provide only one intermediate step directory to start from.")
else:
    START_STEP = start_options_bool.index(True) + 1

# Use date and time to create unique output directory
if START_STEP==0:
    output_dir = f"{repodir}/Data_Files/Data_Set_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
    try:
        os.makedirs(output_dir)
    except:
        raise  FileNotFoundError("You may not have the soft link for Data_Files. Run: ln -s /local/d1/smartpixML/2026Datasets/Data_Files")
else:
    start_dir = start_options[START_STEP - 1]
    output_dir = Path(start_dir).parent

# ADD CHECK TO MAKE SURE TOTAL TRACKS AND BIN SIZE ARE CONSISTENT WITH PROVIDED INTERMEDIATE FILES

# create output directories for each intermediate step
output_dir_pgun = f"{output_dir}/Particle_Gun"
output_dir_detsim = f"{output_dir}/Detector_Sim"
output_dir_tracklists = f"{output_dir}/Track_Lists"
output_dir_pixelav = f"{output_dir}/PixelAV"
output_dir_parquet = f"{output_dir}/Parquet_Files"

#output_dir_fluka = f"{output_dir}/FLUKA"

output_dirs=[output_dir_pgun, output_dir_detsim, output_dir_tracklists, output_dir_pixelav, output_dir_parquet]

if START_STEP==0:
    for dir in output_dirs:
        os.makedirs(dir)
else:
    for dir in output_dirs[START_STEP:]:
        shutil.rmtree(dir, ignore_errors=True) # make sure directory will be empty
        os.makedirs(dir)

# set up MuColl environment
if START_STEP <= 3: 
    # Load environment from relevant setup.sh for running a particle gun, detector simulation, and reading/processing resulting sclio file
    env=os.environ.copy()
    env.update(get_env_from_setup('/cvmfs/muoncollider.cern.ch/release/2.8-patch2/setup.sh'))
else:
    env = None

if ops.signal:
    generate_signal()

if ops.bib:
    generate_bib()