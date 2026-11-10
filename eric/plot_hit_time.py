import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/validationPlots/")
from plotUtils import load_parquet_pairs, countBibSig, plotManyHisto

matplotlib.rcParams["figure.dpi"] = 150

pkl_path = "/local/d1/smartpixML/cutAnalysis/dfOfTruth.pkl"

# --- Load data ---
print(f"Loading truthDF from {pkl_path}")
truthDF = pd.read_pickle(pkl_path)

# --- Split into sig / bib sub-groups ---
fracBib, fracSig, fracMM, fracMP, numTotalSig, numTotalBib, truthSig, truthBib_mm, truthBib_mp, truthBib = countBibSig(truthDF, doPrint=True)

# --- Plot adjusted_hit_time ---
key = "adjusted_hit_time"
bins = np.linspace(-1, 1, 100)

plotManyHisto(
    [truthSig[key], truthBib_mm[key], truthBib_mp[key], truthBib[key]],
    title=f"parquet {key}",
    pltLabels=[f"sig {key}", f"bib mm {key}", f"bib mp {key}", f"bib {key}"],
    bins=bins,
    showNums=False,
    figsize=(7, 2),
    yscale="log",
)
plt.show()
