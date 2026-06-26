"""
Author: Daniel Abadjiev
Date: June 25, 2026
Description: script to extract resources from new folder structure
"""

import sys
sys.path.append("../eric")
from pareto_all_models import *
from pareto_all_models import _y, RUN_LABELS
import re
import os
import glob
from typing import Callable, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as axisartist

rtlPath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/rtl.rpt"

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

assert areaScoreFromRtlRpt("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/rtl.rpt") == (560291.8, 588659.5, 586221.4)
assert areaScoreFromRtlRpt("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/hlsCatapultModel2_20260619_115432/Catapult/myproject.v1/rtl.rpt") == (4290801.6, 4370154.8, 4322300.2)
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
assert latencyFromCycleRpt("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/cycle.rpt") == (250.0,10.0,25)
assert latencyFromCycleRpt("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/hlsCatapultModel2_20260619_115432/Catapult/myproject.v1/cycle.rpt") == (260,10,26)

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
assert (mapSingleHlsToH5File("./hlsVerification/m1_b6___model_trial_836.h/hl...") == "./CrossParetoModels_June2026/model1_fin_results_model1_6bit_normalised_selected__model_trial_836.h5")

def processTargetDirectories(processFunction: Callable[[str], dict], baseDir: str = "./hlsVerification",doPrint = True) -> list[Any]:
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
        catapultRuns = glob.glob(catapultPattern)
        catapultRunDirs = [d for d in catapultRuns if os.path.isdir(d)]
        
        if not catapultRunDirs:
            continue

        # Alphanumeric sorting places the newest ISO-style timestamp folder at the end
        catapultRunDirs.sort()
        newestCatapultDir = catapultRunDirs[-1]

        # Construct the path to the target project directory
        targetProjectDir = os.path.join(newestCatapultDir, "Catapult", "myproject.v1")

        if os.path.isdir(targetProjectDir):
            # Execute the user-provided callback function and capture its output
            if doPrint:
                print("\nExecuting", processFunction.__name__, "on", targetProjectDir)
            output = processFunction(targetProjectDir,doPrint=doPrint)
            output["hlsDir"] = targetProjectDir
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
# assert extractProjectMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/") == ((560291.8, 588659.5, 586221.4), (250.0,10.0,25))
assert extractProjectMetrics("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/hlsVerification/hlsCatapultModel2_20260622_103255/Catapult/myproject.v1/",doPrint=False) == {'areaScorePostScheduling': 560291.8, 'areaScorePostDSP': 588659.5, 'areaScorePostAssignment': 586221.4, 'designLatency': 250.0, 'clockPeriod': 10.0, 'numClocks': 25}
# Execution example:
# processTargetDirectories("/path/to/your/base/workspace", extractProjectMetrics)

def saveMetrics(allMetrics, savePath = "./hlsComparison/hls_synthesis_metrics.csv"):
    df = pd.DataFrame(allMetrics)
    df.sort_values(by="hlsDir", inplace=True)
    df.to_csv(savePath, index=False)
    print(df)
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

def plotCombinedDf(combinedDf: pd.DataFrame,savePath = "./hlsComparison/hls_model_synthesis_summary_plots.png") -> None:
    """
    Plots structural hardware metrics, synthesis area scores, and latency metrics 
    for each model execution row inside the combined DataFrame.
    """
    # 1. Verification Guard: Check that the DataFrame actually contains data rows
    assert not combinedDf.empty, "Assertion failed: Cannot plot an empty DataFrame."
    
    # 2. Extract or create a clean, descriptive X-axis label for each row configuration
    # If your DataFrame doesn't have an explicit 'hlsDir' short name, we fall back to run_name + trial_id
    plotDf = combinedDf.copy()
    if "hlsDir" in plotDf.columns:
        plotDf["config_label"] = plotDf["hlsDir"].apply(lambda x: str(x).split('/')[0])
    else:
        plotDf["config_label"] = plotDf["run_name"] + "_t" + plotDf["trial_id"].astype(str)

    # Use a clean, scannable plotting style layout
    sns.set_theme(style="whitegrid")
    
    # Create a 3-Row dashboard figure to completely isolate unrelated units of measurement
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    
    # -------------------------------------------------------------------------
    # PLOT 1: HARDWARE LOGIC UTILIZATION (LUTs, Registers, FFs, DSPs)
    # -------------------------------------------------------------------------
    # Melt the specific hardware count columns into a long-form structure for Seaborn
    hardwareCols = ["luts", "registers", "luts_plus_ff", "dsp", "bram"]
    # Check which columns are present to prevent key errors
    activeHardwareCols = [c for c in hardwareCols if c in plotDf.columns]
    
    dfHardware = plotDf.melt(
        id_vars=["config_label"], 
        value_vars=activeHardwareCols, 
        var_name="Metric", 
        value_name="Count"
    )
    
    sns.barplot(data=dfHardware, x="config_label", y="Count", hue="Metric", ax=axes[0])
    axes[0].set_title("RTL Hardware Logic Element Resource Utilization Counts", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Resource Count", fontsize=12)
    axes[0].set_xlabel("")
    
    # -------------------------------------------------------------------------
    # PLOT 2: CATAPULT ESTIMATED AREA SCORES (Scheduling vs DSP vs Assignment)
    # -------------------------------------------------------------------------
    areaCols = ["areaScorePostScheduling", "areaScorePostDSP", "areaScorePostAssignment"]
    activeAreaCols = [c for c in areaCols if c in plotDf.columns]
    
    dfArea = plotDf.melt(
        id_vars=["config_label"], 
        value_vars=activeAreaCols, 
        var_name="Synthesis Stage", 
        value_name="Area Score"
    )
    
    sns.barplot(data=dfArea, x="config_label", y="Area Score", hue="Synthesis Stage", ax=axes[1], palette="viridis")
    axes[1].set_title("Catapult Synthesis Optimization Stage Area Score History", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Estimated Area Score Value", fontsize=12)
    axes[1].set_xlabel("")

    # -------------------------------------------------------------------------
    # PLOT 3: TIMING & LATENCY PERFORMANCES (Clock Cycles & Periods)
    # -------------------------------------------------------------------------
    # We group designLatency and numClocks here since they map to equivalent metric frames
    timingCols = ["designLatency", "numClocks"]
    activeTimingCols = [c for c in timingCols if c in plotDf.columns]
    
    dfTiming = plotDf.melt(
        id_vars=["config_label"], 
        value_vars=activeTimingCols, 
        var_name="Timing Metric", 
        value_name="Cycles"
    )
    
    sns.barplot(data=dfTiming, x="config_label", y="Cycles", hue="Timing Metric", ax=axes[2], palette="magma")
    axes[2].set_title("Hardware Architecture Pipeline Total Latency Execution Cycles", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("Total Clock Cycles", fontsize=12)
    axes[2].set_xlabel("Synthesis Run Configurations", fontsize=12, fontweight="bold")
    
    # Clean up layout presentation details
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    
    # Save a physical file snippet copy directly to your folder directory
    plt.savefig(savePath, dpi=300)
    print(f"Dashboard visualization exported as '{savePath}'")
    plt.show()


def plotCombinedDf2(combinedDf: pd.DataFrame,savePath = "./hlsComparison/hls_variable_correlations.png") -> None:
    """
    Plots a multi-variable line series chart across all rows in combinedDf.
    All specified columns are normalized between 0 and 1. Custom markers 
    and sizes are applied to 'luts_plus_ff', 'areaScorePostAssignment', 
    and 'designLatency' to highlight specific correlation trends.
    """
    # 1. Verification Guard
    assert not combinedDf.empty, "Assertion failed: Cannot plot an empty DataFrame."
    
    # Target columns to trace
    targetColumns = [
        'luts', 'registers', 'luts_plus_ff', 'dsp', 'bram',
        'areaScorePostScheduling', 'areaScorePostDSP', 'areaScorePostAssignment', 
        'designLatency', 'clockPeriod', 'numClocks'
    ]
    
    # Filter to make sure we only plot columns that actually exist in the DataFrame
    activeColumns = [col for col in targetColumns if col in combinedDf.columns]
    assert activeColumns, "Assertion failed: None of the specified target columns were found in the DataFrame."

    # 2. RESTORED CLEAN SHORTHAND X-AXIS LABELLING LOGIC
    plotDf = combinedDf.copy()
    if "hlsDir" in plotDf.columns:
        plotDf["config_label"] = plotDf["hlsDir"].apply(lambda x: str(x).split('/')[2] if '/' in str(x) else str(x))
    else:
        plotDf["config_label"] = plotDf["run_name"] + "_t" + plotDf["trial_id"].astype(str)

    # Reset index to create a clean numeric coordinate line frame [0, 1, 2, ... Row N]
    plotDf = plotDf.reset_index(drop=True)
    
    # 3. Min-Max Normalization step to bring all variable tracks onto an identical scale
    normalizedDf = plotDf.copy()
    for col in activeColumns:
        colMin = plotDf[col].min()
        colMax = plotDf[col].max()
        
        if colMax != colMin:
            normalizedDf[col] = (plotDf[col] - colMin) / (colMax - colMin)
        else:
            normalizedDf[col] = 0.0 # Handles flat columns (like clockPeriod which is always 10.0)

    # 4. Construct the correlation series plot
    plt.figure(figsize=(16, 9))
    sns.set_theme(style="whitegrid")
    
    # Plot each column as an independent, continuous line path spanning across every row index
    for col in activeColumns:
        # --- CONDITIONAL STYLE SWITCHING ---
        if col == "luts_plus_ff":
            currentMarker = '^'      # Triangle pointing up
            currentSize = 10         # Highly prominent marker size
            currentLineWidth = 2.5   # Thicker line path weight
        elif col == "areaScorePostAssignment":
            currentMarker = 's'      # Square marker
            currentSize = 9          # Bigger marker size
            currentLineWidth = 2.2   # Highlighted line weight
        elif col == "designLatency":
            currentMarker = 'h'      # Hexagon marker
            currentSize = 9          # Bigger marker size
            currentLineWidth = 2.2   # Highlighted line weight
        else:
            currentMarker = 'o'      # Default circle marker
            currentSize = 4          # Standard size
            currentLineWidth = 1.5   # Standard line weight

        plt.plot(
            normalizedDf.index, 
            normalizedDf[col], 
            marker=currentMarker, 
            markersize=currentSize, 
            linewidth=currentLineWidth, 
            label=col,
            alpha = 0.8
        )
        
    # Configure graph layout specifications
    plt.title("HLS Variable Correlation Series (Normalized Values Across All Rows)", fontsize=14, fontweight="bold")
    plt.ylabel("Normalized Value Scale (0.0 = Minimum, 1.0 = Maximum)", fontsize=12)
    plt.xlabel("DataFrame Row Configurations (Ordered Sequentially)", fontsize=12)
    
    # Force the X-axis tick placements to correspond precisely to each row index position
    plt.xticks(
        ticks=normalizedDf.index, 
        labels=normalizedDf["config_label"], 
        rotation=45, 
        ha="right", 
        fontsize=9
    )
    
    # Position the legend box completely off to the side so it doesn't block data paths
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.tight_layout()
    
    # Export a permanent png layout file back to disk
    plt.savefig(savePath, dpi=300)
    print(f"Correlation series visualization exported successfully as '{savePath}'")
    plt.show()



def plotCombinedDF3(combinedDf: pd.DataFrame,savePath = "./hlsComparison/hls_parasite_multi_axis_trends.png") -> None:
    """
    Plots a multi-Y-axis parasite plot tracking the actual raw values of 
    'luts_plus_ff', 'areaScorePostAssignment', and 'designLatency' across all rows.
    """
    # 1. Verification Guard
    assert not combinedDf.empty, "Assertion failed: Cannot plot an empty DataFrame."
    
    focusColumns = ['luts_plus_ff', 'areaScorePostAssignment', 'designLatency']
    activeColumns = [col for col in focusColumns if col in combinedDf.columns]
    assert len(activeColumns) == len(focusColumns), (
        f"Assertion failed: Missing required columns. "
        f"Expected {focusColumns}, but found {activeColumns}"
    )

    # 2. FIXED: Restored your exact index extraction rule from plotCombinedDF
    plotDf = combinedDf.copy()
    if "hlsDir" in plotDf.columns:
        plotDf["config_label"] = plotDf["hlsDir"].apply(
            lambda x: str(x).split('/')[2] if '/' in str(x) else str(x)
        )
    else:
        plotDf["config_label"] = plotDf["run_name"] + "_t" + plotDf["trial_id"].astype(str)

    # Reset index to create clean numeric step coordinates [0, 1, 2, ... Row N]
    plotDf = plotDf.reset_index(drop=True)
    rowIndices = plotDf.index.tolist()

    # 3. Initialize the Parasite Axis Architecture matching your working format
    plt.figure(figsize=(16, 9))
    host = host_subplot(111, axes_class=axisartist.Axes)
    plt.subplots_adjust(right=0.70, bottom=0.20) 

    par1 = host.twinx()
    par2 = host.twinx()

    # Apply your exact axis configuration technique
    par1.axis["right"] = par1.new_fixed_axis(loc="right", offset=(0, 0))
    par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(80, 0))

    # 4. Color Definitions matching the structural components
    colorLuts = "#1f77b4"    # Strong Blue
    colorArea = "#2ca02c"    # Strong Green
    colorLatency = "#d62728" # Strong Red

    # 5. Plot Each Metric on its Respective Axis Layer
    
    # Metric 1: luts_plus_ff (Plotted on the primary Host Axis - Left Side)
    p1, = host.plot(
        rowIndices, 
        plotDf["luts_plus_ff"], 
        color=colorLuts, 
        marker='^', 
        markersize=11, 
        linewidth=2.5, 
        label="luts_plus_ff"
    )
    host.set_ylabel("luts_plus_ff (csynth)")
    host.axis["left"].label.set_color(colorLuts)

    # Metric 2: areaScorePostAssignment (Plotted on par1 - Inner Right Side)
    p2, = par1.plot(
        rowIndices, 
        plotDf["areaScorePostAssignment"], 
        color=colorArea, 
        marker='s', 
        markersize=10, 
        linewidth=2.5, 
        label="areaScorePostAssignment"
    )
    par1.set_ylabel("areaScorePostAssignment (Score um^2)")
    par1.axis["right"].label.set_color(colorArea)

    # Metric 3: designLatency (Plotted on par2 - Outer Right Side)
    p3, = par2.plot(
        rowIndices, 
        plotDf["designLatency"], 
        color=colorLatency, 
        marker='h', 
        markersize=10, 
        linewidth=2.5, 
        label="designLatency"
    )
    par2.set_ylabel("designLatency (ns)")
    par2.axis["right"].label.set_color(colorLatency)

    # 6. Configure X-Axis and Labels
    host.set_xlabel("DataFrame Row Configurations (Ordered Sequentially)", fontsize=12, fontweight="bold")
    host.set_title("HLS Multi-Axis Parasite Plot (Raw Parameter Values Tracking)", fontsize=14, fontweight="bold")
    
    # Force the X-axis tick placements to correspond precisely to each row index position
    plt.xticks(
        ticks=plotDf.index, 
        labels=plotDf["config_label"], 
        rotation=45, 
        ha="right", 
        fontsize=9
    )

    # Synchronize limits across the axis structures
    host.set_xlim(-0.5, len(rowIndices) - 0.5)

    # Consolidate labels into a unified legend block
    host.legend(handles=[p1, p2, p3], loc="upper left", fontsize=11)
    
    plt.savefig(savePath, dpi=300)
    print(f"Multi-Y-axis visualization exported successfully as '{savePath}'")
    plt.show()
# import numpy as np
# def _setup_complement_log_y(ax, yseries):
#     ax.set_yscale("log")
#     ax.invert_yaxis()
#     ticks_actual = [t for t in COMP_LOG_TICKS
#                     if t < yseries.max() + 0.05]
#     ticks_comp   = np.clip(1.0 - np.array(ticks_actual, dtype=float), 1e-4, None)
#     ax.set_yticks(ticks_comp)
#     ax.yaxis.set_major_formatter(
#         mticker.FuncFormatter(lambda v, _: f"{1 - v:.2f}"))
#     ax.yaxis.set_minor_formatter(mticker.NullFormatter())
#     y_min = max(1e-4, float(np.min(_y(yseries, True))) * 0.5)
#     y_max = float(np.max(_y(yseries, True))) * 2.0
#     ax.set_ylim(y_max, y_min)
def plotCombinedDf4(combinedDf: pd.DataFrame, savePath = "./hlsComparison/hls_csynthVsAreascore.png") -> None:
    plt.close()
    plt.figure(figsize=(12,20))
    plt.subplot(411)
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["areaScorePostAssignment"]/10e6,label="area score (mm^2)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["areaScorePostAssignment"]/10e6,".",label="area score (mm^2)")
    # plt.plot(combinedDf["luts_plus_ff"],combinedDf["numClocks"]/100,label="numClocks / 100")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["designLatency"]/1000,label="designLatency (us)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["designLatency"]/1000,".",label="designLatency (us)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["parameters"]/1.5e5,label="parameters / 1.5e5")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["bkg_rej_@99%"]/2,label="bkg_rej_@99% / 2")
    # par1.set_ylabel()
    plt.ylabel("areaScorePostAssignment (Score um^2 / 10^6) and numClocks / 100")
    plt.xlabel("luts_plus_ff (csynth)")
    plt.legend()

    plt.subplot(412)
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["areaScorePostAssignment"]/10e6,label="area score (mm^2)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["areaScorePostAssignment"]/10e6,".",label="area score (mm^2)")
    # plt.plot(combinedDf["luts_plus_ff"],combinedDf["numClocks"]/100,label="numClocks / 100")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["designLatency"]/1000,label="designLatency (us)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["designLatency"]/1000,".",label="designLatency (us)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["parameters"]/1.5e5,label="parameters / 1.5e5")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["bkg_rej_@99%"]/2,label="bkg_rej_@99% / 2")
    # par1.set_ylabel()
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("areaScorePostAssignment (Score um^2 / 10^6) and numClocks / 100")
    plt.xlabel("luts_plus_ff (csynth)")
    plt.legend()
    plt.subplot(413)
    plt.plot(combinedDf["areaScorePostAssignment"],combinedDf["bkg_rej_@99%"])
    plt.plot(combinedDf["areaScorePostAssignment"],combinedDf["bkg_rej_@99%"],".")
    # plt.yscale('log')
    ax = plt.gca()
    # _setup_complement_log_y(ax,combinedDf["bkg_rej_@99%"])
    # ax.set_yscale("log")
    # ax.invert_yaxis()
    plt.ylabel("bkg_rej_@99%")
    plt.xlabel("areaScorePostAssignment (Score um^2)")
    plt.subplot(414)
    plt.plot(combinedDf["areaScorePostAssignment"],combinedDf["bkg_rej_@99%"])
    plt.plot(combinedDf["areaScorePostAssignment"],combinedDf["bkg_rej_@99%"],".")
    # plt.yscale('log')
    ax = plt.gca()
    # _setup_complement_log_y(ax,combinedDf["bkg_rej_@99%"])
    # ax.set_yscale("log")
    # ax.invert_yaxis()
    plt.ylabel("bkg_rej_@99%")
    plt.xlabel("areaScorePostAssignment (Score um^2)")
    annotate = True
    if annotate:
        for _, row in combinedDf.iterrows():
            short = (RUN_LABELS.get(row["run_name"], row["run_name"])
                     .replace("Model 1 ", "M1 ")
                     .replace("Model 2.5 ", "M2.5 ")
                     .replace("Model 3 ", "M3 "))
            yval = float(_y(row["bkg_rej_@99%"], False))
            ax.annotate(f"{short}\n{row['trial_id']}",
                        xy=(row["areaScorePostAssignment"], yval),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=6.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow",
                                  alpha=0.8, edgecolor="black", linewidth=0.8),
                        zorder=4)

    plt.savefig(savePath, dpi=300)
    print(f"custom visualization successfully as '{savePath}'")



def main() -> None:
    doPrint = False
    allMetrics = processTargetDirectories(extractProjectMetrics, doPrint=doPrint)
    assert len(allMetrics) > 0, "Assertion failed: No metrics were extracted from the target directories."
    saveMetrics(allMetrics)
    print(f"Successfully processed {len(allMetrics)} project run configurations.")
    print("\n\n\n")
    combinedDf = mergeTwoCsvFiles("../eric/combined_all_models_pareto_newJune2026/pareto_primary.csv","./hlsComparison/hls_synthesis_metrics.csv")

    print(combinedDf)
    print(combinedDf.keys())
    plotCombinedDf(combinedDf)
    plotCombinedDf2(combinedDf)
    plotCombinedDF3(combinedDf)
    plotCombinedDf4(combinedDf)

# The proper Python entry point condition
if __name__ == "__main__":
    main()


