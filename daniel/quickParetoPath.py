#Author: Daniel Abadjiev
#Date: May 14, 2026
#Description: quick script to extract the list of paths to models on the pareto front.
import pandas as pd
import os
import shutil
import datetime

inputParetoCsv = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/combined_all_models_pareto/pareto_primary.csv"
inputParetoCsv = "../eric/combined_all_models_pareto/pareto_primary.csv"
inputParetoCsv = "../eric/combined_all_models_pareto_newJune2026/pareto_primary.csv"


def getPathsFromCsv(inputParetoPath,doPrint = True):
    df = pd.read_csv(inputParetoCsv)

    paretoPaths = df["fullPath"].to_list()
    if doPrint:
        print(paretoPaths)
    return paretoPaths

paretoPaths = getPathsFromCsv(inputParetoCsv)



def resaveToParetoDir(paretoPaths, paretoDir, appendFinished=True):
    """ Takes paths from paretoPaths, modifies them to find the true source files 
        if appendFinished is True, and saves them all to paretoDir.
    """
    # Ensure the main target directory exists
    os.makedirs(paretoDir, exist_ok=True)

    for path_item in paretoPaths:
            # Split into folder path and file name
        parent_dir = os.path.dirname(path_item)
        file_name = os.path.basename(path_item)
        # 1. Determine the actual source path on disk
        if appendFinished and file_name.startswith("model_") and ("model3" in parent_dir) and ("finishedTrials" not in parent_dir):

            
            # Insert 'finishedTrials' into the path string
            actualSource = os.path.join(parent_dir, "finishedTrials", file_name)
        else:
            actualSource = path_item

        # 2. Verify the constructed path exists before copying
        if not os.path.exists(actualSource):
            print(f"Warning: File not found at {actualSource}")
            continue
            
        # 3. Save the file to the target paretoDir
        # newFileName = os.path.basename(actualSource)
        pathParts = actualSource.split(os.sep)
        originalFileName = pathParts[-1]
        
        # Adjust indices dynamically if 'finishedTrials' was added
        if "finishedTrials" in pathParts:
            quantizationDir = pathParts[-3]
            modelDir = pathParts[-4]
        else:
            quantizationDir = pathParts[-2] if pathParts[-2] != "pareto_primary" else pathParts[-3]
            modelDir = pathParts[-4] if pathParts[-2] == "pareto_primary" else pathParts[-3]

        # Construct the new unique filename using camelCase conventions
        newFileName = f"{modelDir}_{quantizationDir}__{originalFileName}"
        # newFileName = f"{quantizationDir}__{originalFileName}"
        #append model number and quantization from path that looks like /home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Results_June2026_99SigEff/model1_fin_results/model1_10bit_normalised_selected/pareto_primary/model_trial_1228.h5
        destination = os.path.join(paretoDir, newFileName)
        
        shutil.copy2(actualSource, destination)
        print(f"Successfully saved: {actualSource} -> {destination}")


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# paretoDir = f"./CrossParetoModels_{timestamp}/"
paretoDir = f"./CrossParetoModels_June2026/"
print(len(paretoPaths))

#I fixed the finishedTrial append in the pareto_all_models.py so don't actually need to appendFinished
resaveToParetoDir(paretoPaths,paretoDir,appendFinished = False)
