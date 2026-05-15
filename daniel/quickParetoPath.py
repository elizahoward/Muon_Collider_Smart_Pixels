#Author: Daniel Abadjiev
#Date: May 14, 2026
#Description: quick script to extract the list of paths to models on the pareto front.
import pandas as pd

inputParetoCsv = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/combined_all_models_pareto/pareto_primary.csv"
inputParetoCsv = "../eric/combined_all_models_pareto/pareto_primary.csv"

def getPathsFromCsv(inputParetoPath,doPrint = True):
    df = pd.read_csv(inputParetoCsv)

    paretoPaths = df["fullPath"].to_list()
    if doPrint:
        print(paretoPaths)
    return paretoPaths

getPathsFromCsv(inputParetoCsv)

