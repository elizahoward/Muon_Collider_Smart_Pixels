#Author: Daniel Abadjiev
#Date: May 14, 2026
#Description: quick script to extract the list of paths to models on the pareto front.
import pandas as pd

inputParetoCsv = "../Muon_Collider_Smart_Pixels/eric/combined_all_models_pareto/pareto_primary.csv"

df = pd.read_csv(inputParetoCsv)

df.head()
