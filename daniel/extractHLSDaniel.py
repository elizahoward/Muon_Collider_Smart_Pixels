"""
Author: Daniel Abadjiev
Date: June 25, 2026
Description: script to extract resources from new folder structure
"""

import sys
sys.path.append("../eric")
from pareto_all_models import *
import re
import os
import glob
from typing import Callable, Any
import pandas as pd
from plotHLSDaniel import *

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
    plotCombinedDf2(combinedDf[5:],savePath = "./hlsComparison/hls_variable_correlations_noModel3.png")
    plotCombinedDF3(combinedDf)
    plotCombinedDf3v2(combinedDf)
    plotCombinedDf3v2(combinedDf[5:],savePath = "./hlsComparison/hls_parasite_multi_axis_trendsV2_noModel3.png")
    plotCombinedDf4(combinedDf)
    plotCombinedDf4(combinedDf[5:], savePath = "./hlsComparison/hls_csynthVsAreascore_noModel3.png")

# The proper Python entry point condition
if __name__ == "__main__":
    main()


