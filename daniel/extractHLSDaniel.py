"""
Author: Daniel Abadjiev
Date: June 25, 2026
Description: script to extract resources from new folder structure
"""

import sys
# sys.path.append("../eric")
sys.path.append("../paperPlots")
from pareto_all_models import *
import re
import os
import glob
from typing import Callable, Any
import pandas as pd
from plotHLSDaniel import *

rtlPath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/oldSyntheses_Pre_20260622_orwhatever/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/rtl.rpt"

def areaScoreFromRtlRpt(rtlPath: str) -> tuple[float, float, float]:
    """
    rtlPath should end with ..../Catapult/myproject.v1/rtl.rpt and then will look for a line that looks like 
    Area Scores
                            Post-Scheduling     Post-DP & FSM   Post-Assignment 
        ----------------- ----------------- ----------------- -----------------
        Total Area Score:   560291.8          588659.5          586221.4        
    and will return these three area scores
    Apparently the areaScorePostAssignment is the one that should be used
    """
    # Enforce that the file must be named exactly 'rtl.rpt'
    fileName = os.path.basename(rtlPath)    
    if fileName != "rtl.rpt":
        raise ValueError(f"Invalid file name: '{fileName}'. The file must be named 'rtl.rpt'.")

    areaScorePostScheduling = -1
    areaScorePostDSP = -1
    areaScorePostAssignment = -1

    pattern = r"Total Area Score:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"

    try:
        with open(rtlPath, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    areaScorePostScheduling = float(match.group(1))
                    areaScorePostDSP = float(match.group(2))
                    areaScorePostAssignment = float(match.group(3))
                    break 
    except FileNotFoundError:
        print(f"Error: The file at {rtlPath} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return areaScorePostScheduling, areaScorePostDSP, areaScorePostAssignment

assert areaScoreFromRtlRpt("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/oldSyntheses_Pre_20260622_orwhatever/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/rtl.rpt") == (560291.8, 588659.5, 586221.4)
assert areaScoreFromRtlRpt("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/oldSyntheses_Pre_20260622_orwhatever/hlsCatapultModel2_20260619_115432/Catapult/myproject.v1/rtl.rpt") == (4290801.6, 4370154.8, 4322300.2)
import re
import os

def latencyFromCycleRpt(cyclPath: str) -> tuple[float, float, int]:
    """
    return the latency from the cycle.rpt, along with clockPeriod and numClocks
    use regex to look at a section that looks like 
    
    Processes/Blocks in Design
    Process                                                                                            Real Operation(s) count Latency Throughput Reset Length II Comments 
    -------------------------------------------------------------------------------------------------- ----------------------- ------- ---------- ------------ -- --------
    /myproject/nnet::hard_sigmoid<output_result_t,result_t,hard_sigmoid_config29>/core                                      24       1          3            0  0          
    ........... 
    /myproject/nnet::linear<input_t,layer3_t,linear_config3>/core                                                          128       1          3            0  0          
    Design Total:                                                                                                         1970      25          3            1  0          
    
    Clock Information
    Clock Signal Edge   Period Sharing Alloc (%) Uncertainty Used by Processes/Blocks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    ------------ ------ ------ ----------------- ----------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    clk          rising 10.000             20.00    0.000000 /myproject/nnet::hard_sigmoid<output_result_t,result_t,hard_sigmoid_config29>/core ...
    """
    # Enforce file name 
    fileName = os.path.basename(cyclPath)
    if fileName != "cycle.rpt":
        raise ValueError(f"Invalid file name: '{fileName}'. The file must be named 'cycle.rpt'.")

    # Initialize return variables
    designLatency = -1.0 #ns
    clockPeriod = -1.0 #ns [according to Guiseppe]
    numClocks = -1 #in clock periods

    # Pattern for 'Design Total:' line to extract Latency (the second integer block)
    # Group 1: Real Operation(s) count, Group 2: Latency, Group 3: Throughput
    latencyPattern = r"Design Total:\s+(\d+)\s+(\d+)\s+(\d+)"
    
    # Pattern for 'clk rising' line to capture the clock Period
    # Group 1 captures the decimal period value (e.g., 10.000)
    clockPattern = r"clk\s+rising\s+([\d.]+)"

    try:
        with open(cyclPath, 'r') as file:
            for line in file:
                # Look for design latency metric
                latencyMatch = re.search(latencyPattern, line)
                if latencyMatch:
                    numClocks = int(latencyMatch.group(2))
                
                # Look for clock period metric
                clockMatch = re.search(clockPattern, line)
                if clockMatch:
                    clockPeriod = float(clockMatch.group(1))

        # Dynamically calculate numClocks if both baseline targets were found
        if numClocks != -1 and clockPeriod != -1.0:
            # Latency metric in Catapult cycle reports represents total execution clock cycles
            designLatency = numClocks * clockPeriod

    except FileNotFoundError:
        print(f"Error: The file at {cyclPath} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return designLatency, clockPeriod, numClocks
assert latencyFromCycleRpt("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/oldSyntheses_Pre_20260622_orwhatever/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/cycle.rpt") == (250.0,10.0,25)
assert latencyFromCycleRpt("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/oldSyntheses_Pre_20260622_orwhatever/hlsCatapultModel2_20260619_115432/Catapult/myproject.v1/cycle.rpt") == (260,10,26)

import os
import re
import glob
from typing import Optional

HLS_PATH_REGEX = re.compile(r"model([\d._]+)_+?(\d+)bit_.*?_+model_trial_(\d+)\.h5")
HLS_UNIVERSAL_REGEX = re.compile(r"m(?:odel)?([\d._]+)(?:_b|\_+)?(\d+)(?:bit)?.*?_+model_trial_(\d+)")

def extractHlsKeyFromPath(pathStr: str) -> tuple:
    """
    Extracts (modelNum, bitWidth, trialNum) from an hlsSummaryCsv path string 
    using the 'h5Path' value. Normalizes underscore decimals (e.g., '2_5' -> '2.5').
    """
    match = HLS_UNIVERSAL_REGEX.search(str(pathStr))
    if not match:
        print("couldn't find model type from hls directory")
        return (None, None, None)
        
    modelNum = match.group(1).replace("_", ".")
    bitWidth = int(match.group(2))
    trialNum = int(match.group(3))
    return (modelNum, bitWidth, trialNum)
def mapSingleHlsToH5File(hlsDirPath: str, h5DirectoryPath: str = "./CrossParetoModels_June2026") -> Optional[str]:
    """
    Takes a single HLS directory path, parses out the model, bit width, and trial number,
    and returns the filename of the matching .h5 file in h5DirectoryPath.
    Returns None if no matching file is found.
    """
    # Compile regex pattern to parse 'hlsDir' (e.g., "m2_5_b10__model_trial_029.h")
    # Group 1: Model number (e.g., "1" or "2_5")
    # Group 2: Bit width number (e.g., "10")
    # Group 3: Trial number (e.g., "029" or "1222")    # UPDATED REGEX: 
    # m([\d.]+) captures digits and literal decimal points (e.g., "1", "2.5", "3")
    # _b(\d+) captures the bit width
    # __model_trial_(\d+) captures the trial number    # FIXED REGEX: 
    # _+ matches one or more underscores, handling both '__model_trial_' and '___model_trial_'
    # hlsRegex = re.compile(r"m([\d.]+)_b(\d+)_+model_trial_(\d+)")
    # match = hlsRegex.search(hlsDirPath)
    modelNum, bitNum, trialNum = extractHlsKeyFromPath(hlsDirPath)
    # print(modelNum,bitNum,trialNum)
    
    # Fetch all .h5 filenames available in the target pool directory
    h5Pattern: str = os.path.join(h5DirectoryPath, "*.h5")
    h5DirectPaths: list[str] = glob.glob(h5Pattern)
    
    for h5FullPath in h5DirectPaths:
        # Isolate just the filename for token verification checks
        h5FileName = os.path.basename(h5FullPath)

        # 1. Match the model identifier (handles "model1" or "model2.5")
        modelMatch: bool = f"model{modelNum}" in h5FileName or f"model{modelNum.replace('.', '_')}" in h5FileName
        
        # 2. Match the precise bit specification (e.g., "_10bit_")
        bitMatch: bool = f"_{bitNum}bit_" in h5FileName
        
        # 3. Extract and match the trial number integer
        trialMatch: bool = False
        trialSearch = re.search(r"trial_(\d+)\.h5", h5FileName)
        if trialSearch and int(trialSearch.group(1)) == trialNum:
            trialMatch = True
            
        # CORRECTED RETURN: Sends back the absolute direct path to the file on disk
        if modelMatch and bitMatch and trialMatch:
            return h5FullPath
        # Pass each target file path into the exact same shared parsing logic
        # fModel, fBit, fTrial = extractHlsKeyFromPath(h5FullPath)
        # print(fModel,fBit,fTrial)
        
        # # Verify if all three physical structural tokens map perfectly
        # if fModel == modelNum and fBit == bitNum and fTrial == trialNum:
        #     return h5FullPath
    print("failed to find ")  
    return None
assert (mapSingleHlsToH5File("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m1_b8__model_trial_1046.h/hlsCatapultModel2_20260625_121940") == "./CrossParetoModels_June2026/model1_fin_results_model1_8bit_normalised_selected__model_trial_1046.h5")
assert (mapSingleHlsToH5File("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m1_b8__model_trial_1046.h/hlsCatapultModel2_20260625_121940/Catapult")== "./CrossParetoModels_June2026/model1_fin_results_model1_8bit_normalised_selected__model_trial_1046.h5")
assert (mapSingleHlsToH5File("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m1_b8__model_trial_1046.h/hlsCatapultModel2_20260625_121940/Catapult/myproject.v1")== "./CrossParetoModels_June2026/model1_fin_results_model1_8bit_normalised_selected__model_trial_1046.h5")
assert (mapSingleHlsToH5File("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m1_b8__model_trial_1031.h/hlsCatapultModel2_20260625_121940/Catapult/myproject.v1")== "./CrossParetoModels_June2026/model1_fin_results_model1_8bit_normalised_selected__model_trial_1031.h5")
assert (mapSingleHlsToH5File("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m3_b10__model_trial_110.h/hlsCatapultModel2_20260625_asdflkadsf/Catapult/myproject.v1")== "./CrossParetoModels_June2026/model3_10bit_normalised_selected_pareto_primary__model_trial_110.h5")
# print(mapSingleHlsToH5File("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m2.5_b6__model_trial_088.h/hlsCatapultModel2_20260625_120405/Catapult/myproject.v1"))
assert (mapSingleHlsToH5File("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m2.5_b6__model_trial_088.h/hlsCatapultModel2_20260625_120405/Catapult/myproject.v1")== "./CrossParetoModels_June2026/model2.5_fin_results_model2_5_6bit_normalised_selected__model_trial_088.h5")
assert (mapSingleHlsToH5File("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m2.5_b6__model_trial_088.h/hlsVitisModel2_blablabla")== "./CrossParetoModels_June2026/model2.5_fin_results_model2_5_6bit_normalised_selected__model_trial_088.h5")
assert (mapSingleHlsToH5File("./hlsVerification/m1_b6___model_trial_836.h/hl...") == "./CrossParetoModels_June2026/model1_fin_results_model1_6bit_normalised_selected__model_trial_836.h5")

def processTargetDirectories(processFunction: Callable[[str], dict], baseDir: str = "./hlsVerification",doPrint = True,typeCatapult = True) -> list[Any]:
    """
    Scans baseDir for trial folders, isolates the newest hlsCatapultModel2 folder,
    executes processFunction passing the absolute path of 'Catapult/myproject.v1',
    and returns a compiled list of all collected outputs.
    """
    # Handles folder structural variations like .h or .h5 using a glob wildcard
    modelDirPattern = os.path.join(baseDir, "m*_b*__model_trial_*")
    modelDirs = glob.glob(modelDirPattern)

    # Initialize a list to hold the outputs from the callback
    allOutputs: list[Any] = []

    for modelDir in modelDirs:
        if not os.path.isdir(modelDir):
            continue

        catapultPattern = os.path.join(modelDir, "hlsCatapultModel*")
        vivadoPattern = os.path.join(modelDir, "hlsVitisModel*")
        if typeCatapult:
            allRuns = glob.glob(catapultPattern)
        else:
            allRuns = glob.glob(vivadoPattern)
        allRunDirs = [d for d in allRuns if os.path.isdir(d)]
        
        if not allRunDirs:
            continue

        # Alphanumeric sorting places the newest ISO-style timestamp folder at the end
        allRunDirs.sort()
        newestRunDir = allRunDirs[-1]

        # Construct the path to the target project directory
        if typeCatapult:
            targetProjectDir = os.path.join(newestRunDir, "Catapult", "myproject.v1")
        else:
            targetProjectDir = newestRunDir

        if os.path.isdir(targetProjectDir):
            # Execute the user-provided callback function and capture its output
            if doPrint:
                print("\nExecuting", processFunction.__name__, "on", targetProjectDir)
            output = processFunction(targetProjectDir,doPrint=doPrint)
            if typeCatapult:
                output["hlsDir"] = targetProjectDir
            else:
                output["hlsDirVitis"] = targetProjectDir
            output["h5Path"] = mapSingleHlsToH5File(targetProjectDir)
            
            # Save the captured result to the master list
            allOutputs.append(output)
        else:
            print(f"Warning: Expected target directory not found: {targetProjectDir}")

    # Return all collected results together
    return allOutputs

def extractProjectMetrics(projectDir: str, doPrint = True) -> dict:
    """
    Callback function that parses both report files inside a given project directory.
    Works for Catapult directories
    """
    # print(f"\nProcessing project run: {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(projectDir))))}")
    if doPrint:
        print(f"Processing project run: {projectDir}")
    
    rtlReportPath = os.path.join(projectDir, "rtl.rpt")
    cycleReportPath = os.path.join(projectDir, "cycle.rpt")
    # Initialize dictionary with default safe fallback values
    metrics: dict[str, Any] = {
        "areaScorePostScheduling": -1.0,
        "areaScorePostDSP": -1.0,
        "areaScorePostAssignment": -1.0,
        "designLatency": -1.0,
        "clockPeriod": -1.0,
        "numClocks": -1
    }

    if os.path.exists(rtlReportPath):
        postSched, postDsp, postAssign = areaScoreFromRtlRpt(rtlReportPath)
        metrics["areaScorePostScheduling"] = postSched
        metrics["areaScorePostDSP"] = postDsp
        metrics["areaScorePostAssignment"] = postAssign

    if os.path.exists(cycleReportPath):
        latency, period, clocks = latencyFromCycleRpt(cycleReportPath)
        metrics["designLatency"] = latency
        metrics["clockPeriod"] = period
        metrics["numClocks"] = clocks
    if doPrint:
        print(metrics)
    return metrics
# assert extractProjectMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/oldSyntheses_Pre_20260622_orwhatever/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/") == ((560291.8, 588659.5, 586221.4), (250.0,10.0,25))
assert extractProjectMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/oldSyntheses_Pre_20260622_orwhatever/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/",doPrint=False) == {'areaScorePostScheduling': 560291.8, 'areaScorePostDSP': 588659.5, 'areaScorePostAssignment': 586221.4, 'designLatency': 250.0, 'clockPeriod': 10.0, 'numClocks': 25}
# Execution example:
# processTargetDirectories("/path/to/your/base/workspace", extractProjectMetrics)

###################################################################################
#Vivado report processing
def find_in_text(pattern, file_path, cast_type=float, default=-1, group=1):
    """Safely extracts a regex match from a text file, defaulting to -1 on failure."""
    if not os.path.exists(file_path):
        return default
    with open(file_path, 'r') as f:
        match = re.search(pattern, f.read())
    return cast_type(match.group(group)) if match else default

def find_latency_row(file_path):
    return find_in_text(
        r'\+ Latency:\s*\n'
        r'(?:[^\r\n]*\r?\n){5}'
        r'([^\r\n]+)',
        file_path,
        str,
        default=""
    )
def check_deterministic_pipeline(rpt_path):
    """Verifies if the pipeline is enabled and min latency equals max latency."""
    row = find_latency_row(rpt_path)
    if not row:
        # raise ValueError("Latency row not found")
        return False
    fields = [x.strip() for x in row.strip().strip('|').split('|')]
    return (
        len(fields) == 7
        and fields[0] == fields[1]
        and fields[6] in ("yes", "function")
    )
def getVsynthMetrics(base_dir: str, doPrint=False):
    rpt_csynth = os.path.join(base_dir, "myproject_prj/solution1/syn/report/myproject_csynth.rpt")
    rpt_vsynth = os.path.join(base_dir, "vivado_synth.rpt")
    latency_row = find_latency_row(rpt_csynth)
    latency_fields = [x.strip() for x in latency_row.strip('|').split('|')] if latency_row else []
    latency_str = (
        latency_fields[4]
        if len(latency_fields) > 4 and re.match(r'[\d.]+\s*[a-zA-Z]+$', latency_fields[4])
        else latency_fields[3]
        if len(latency_fields) > 3 else -1
    )
    # print(latency_fields)
    metrics = {
        "Target Clock (ns)": find_in_text(r'ap_clk\s*\|\s*([\d\.]+)\s*ns', rpt_csynth),
        "Achieved Clock (ns)": find_in_text(r'ap_clk\s*\|\s*[\d\.]+\s*ns\s*\|\s*([\d\.]+)\s*ns', rpt_csynth),
        "Latency (cycles)": int(latency_fields[2]) if len(latency_fields) > 2 else -1,
        "Latency (str)": latency_str,
        "Interval (cycles)": int(latency_fields[6]) if len(latency_fields) > 6 else -1,
        "Post-Synth LUTs": find_in_text(r'Slice LUTs\*\s*\|\s*(\d+)', rpt_vsynth, int),
        "Post-Synth FFs": find_in_text(r'Slice Registers\s*\|\s*(\d+)', rpt_vsynth, int),
        "Post-Synth DSPs": find_in_text(r'DSPs\s*\|\s*(\d+)', rpt_vsynth, int),
        "Post-Synth BRAMs": find_in_text(r'Block RAM Tile\s*\|\s*(\d+)', rpt_vsynth, int),
        "Verified Deterministic": check_deterministic_pipeline(rpt_csynth),
    }
    if doPrint:
        print(metrics)
    return metrics

# print(getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m1_b6___model_trial_836.h/hlsVitisModel2_20260714_050318"))
assert getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m1_b6___model_trial_836.h/hlsVitisModel2_20260714_050318") == {'Target Clock (ns)': 5.0, 'Achieved Clock (ns)': 4.302, 'Latency (cycles)': 23, 'Latency (str)': '0.115 us', 'Interval (cycles)': 1, 'Post-Synth LUTs': 2051, 'Post-Synth FFs': 1762, 'Post-Synth DSPs': 2, "Post-Synth BRAMs":0,'Verified Deterministic': True}
assert getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m2.5_b8___model_trial_063.h/hlsVitisModel2_20260714_062921") == {'Target Clock (ns)': 5.0, 'Achieved Clock (ns)': 4.362, 'Latency (cycles)': 42, 'Latency (str)': '0.210 us', 'Interval (cycles)': 1, 'Post-Synth LUTs': 234454, 'Post-Synth FFs': 223056, 'Post-Synth DSPs': 220, "Post-Synth BRAMs":0,'Verified Deterministic': True}
assert getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m2.5_b8___model_trial_063.h/hlsVitisModel2_20260714_062921") == {'Target Clock (ns)': 5.0, 'Achieved Clock (ns)': 4.362, 'Latency (cycles)': 42, 'Latency (str)': '0.210 us', 'Interval (cycles)': 1, 'Post-Synth LUTs': 234454, 'Post-Synth FFs': 223056, 'Post-Synth DSPs': 220, "Post-Synth BRAMs":0,'Verified Deterministic': True}
# print(getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m3_b10___model_trial_046.h/hlsVitisModel2_20260723_111028"))
assert getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/m3_b10___model_trial_046.h/hlsVitisModel2_20260723_111028") == {'Target Clock (ns)': 10.0, 'Achieved Clock (ns)': 8.745, 'Latency (cycles)': 313, 'Latency (str)': '3.120 us', 'Interval (cycles)': 280, 'Post-Synth LUTs': 492029, 'Post-Synth FFs': 201191, 'Post-Synth DSPs': 0, "Post-Synth BRAMs":0,'Verified Deterministic': False}
###################################################################################

def saveMetrics(allMetrics, savePath = "./hlsComparison/hls_synthesis_metrics.csv"):
    df = pd.DataFrame(allMetrics)
    if "hlsDir" in df.keys():
        df.sort_values(by="hlsDir", inplace=True)
    elif "hlsDirVitis" in df.keys():
        df.sort_values(by="hlsDirVitis", inplace=True)
    else:
        print("No hlsDir or hlsDirVitis key to sort by, so saving unsorted")
    df.to_csv(savePath, index=False)
    # print(df)
    print("Metrics successfully saved to ",savePath)
    return


# Global compiled regex patterns for performance and clarity
PARETO_COLUMN_REGEX = re.compile(r"model(\d+)(?:\_)?(\d*)_(\d+)bit")

# UPDATED GLOBAL REGEX:
# model(\d+) matches the root model number.
# (?:\_)?(\d*) is an optional group.
# _(\d+)(?:bit|w) matches an underscore followed by digits, ending with EITHER "bit" or "w".
PARETO_COLUMN_REGEX = re.compile(r"model([\d.]+)(?:\_)?(\d*)_(\d+)(?:bit|w)")

def extractParetoKeyFromRow(row: pd.Series) -> tuple:
    """
    Extracts (modelNum, bitWidth, trialNum) from a paretoCsv row 
    using 'run_name' and 'trial_id'. Supports both '_Xbit' and '_Xw' formats.
    """
    runName = str(row["run_name"])
    trialId = str(row["trial_id"])
    
    match = PARETO_COLUMN_REGEX.search(runName)
    if not match:
        return (None, None, None)
        
    rawModel = match.group(1)
    if rawModel == "25" or rawModel == "2_5":
        modelNum = "2.5"
    else:
        # Strip decimal padding if it evaluates to an integer (e.g., normalizes '3.0' -> '3')
        modelNum = str(float(rawModel)).rstrip('0').rstrip('.') 
        
    # Group 3 extracts the digits before 'bit' or 'w'
    bitWidth = int(match.group(3))
    trialNum = int(trialId) # Automatically strips leading zeros for clean matching
    
    return (modelNum, bitWidth, trialNum)



def mergeTwoCsvFiles(paretoCsv: str, hlsSummaryCsv: str, outputPath: str = "./hlsComparison/merged_results.csv") -> None:
    """
    Merges the two CSV files together by extracting the model trial, bit number, and model number
    from 'fullPath' in paretoCsv and 'hlsDir' in hlsSummaryCsv.
    """
    # 1. Load the two CSV files into Pandas DataFrames
    dfPareto = pd.read_csv(paretoCsv)
    dfHLS = pd.read_csv(hlsSummaryCsv)
    
    # 2. Add verification check to confirm columns are present before transforming
    assert "fullPath" in dfPareto.columns, f"The paretoCsv '{paretoCsv}' is missing the required 'fullPath' column."
    assert "hlsDir" in dfHLS.columns, f"The hlsSummaryCsv '{hlsSummaryCsv}' is missing the required 'hlsDir' column."
    
    # 3. Create the robust uniform tuple key column in both dataframes using separate functions
    dfPareto["composite_match_key"] = dfPareto.apply(extractParetoKeyFromRow, axis=1)
    dfHLS["composite_match_key"] = dfHLS["hlsDir"].apply(extractHlsKeyFromPath)
    
    # 4. Perform the inner merge using the matching composite tuples
    combinedDf = pd.merge(dfPareto, dfHLS, on="composite_match_key", how="inner")
    
    # 5. Clean up the temporary column
    combinedDf.drop(columns=["composite_match_key"], inplace=True)
    
    # Verify the cross-reference succeeded
    assert len(combinedDf) > 0, "No records matched. Check if your CSV column parsing tokens align."
    
    # 6. Save the finalized comprehensive matched table to disk
    combinedDf.to_csv(outputPath, index=False)
    print(f"Success! Combined {len(combinedDf)} matching rows and saved to '{outputPath}'.")
    return combinedDf


# def plotCombinedDf(combinedDf: pd.DataFrame):
#     """
#     plot the 'luts', 'registers', 'luts_plus_ff', 'dsp', 'bram','areaScorePostScheduling', 'areaScorePostDSP',
#        'areaScorePostAssignment', 'designLatency', 'clockPeriod', 'numClocks' columns for each row in the dataframe
#     """


def main() -> None:
    doPrint = False
    allMetrics = processTargetDirectories(extractProjectMetrics, doPrint=doPrint)
    assert len(allMetrics) > 0, "Assertion failed: No metrics were extracted from the target directories."
    saveMetrics(allMetrics)
    vitisMetrics = processTargetDirectories(getVsynthMetrics, doPrint=doPrint,typeCatapult = False)
    saveMetrics(vitisMetrics,savePath = "./hlsComparison/vsynth_metrics.csv")
    print(f"Successfully processed {len(allMetrics)} project run configurations.")
    print("\n\n\n")
    combinedDf = mergeTwoCsvFiles("../eric/combined_all_models_pareto_newJune2026/pareto_primary.csv","./hlsComparison/hls_synthesis_metrics.csv")

    # print(combinedDf)
    print(combinedDf.keys())
    plotCombinedDf(combinedDf)
    plotCombinedDf2(combinedDf)
    plotCombinedDf2(combinedDf[7:],savePath = "./hlsComparison/hls_variable_correlations_noModel3.png")
    plotCombinedDF3(combinedDf)
    plotCombinedDf3v2(combinedDf)
    plotCombinedDf3v2(combinedDf[7:],savePath = "./hlsComparison/hls_parasite_multi_axis_trendsV2_noModel3.png")
    plotCombinedDf4(combinedDf)
    plotCombinedDf4(combinedDf[7:], savePath = "./hlsComparison/hls_csynthVsAreascore_noModel3.png")

    print("Now processing the parallel synth version")
    print("hls_model_trial_1046")
    getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_shortlist_parallelInput/hls_outputs_vsynth/hls_model_trial_1046",doPrint=True)
    print("hls_model_trial_057")
    getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_shortlist_parallelInput/hls_outputs_vsynth/hls_model_trial_057",doPrint=True)
    print("hls_model_trial_087")
    getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_shortlist_parallelInput/hls_outputs_vsynth/hls_model_trial_087",doPrint=True)
    print("hls_model_trial_046")
    getVsynthMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_shortlist_parallelInput/hls_outputs_vsynth/hls_model_trial_046",doPrint=True)
    
# The proper Python entry point condition
if __name__ == "__main__":
    main()


