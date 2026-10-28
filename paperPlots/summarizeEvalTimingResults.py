#GAAA

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
#plot acceptance rates on a log scale.

filepaths=[
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_30ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_60ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_100ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_200ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_500ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_1000ps_gaussian.txt",
    # "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_200ps_gaussian.txt",
    ]
filepaths=[
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_30ps_gaussian.json",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_60ps_gaussian.json",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_100ps_gaussian.json",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_200ps_gaussian.json",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_300ps_gaussian.json",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_400ps_gaussian.json",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_500ps_gaussian.json",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_1000ps_gaussian.json",
    # "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_200ps_gaussian.json",
    ]
PLOT_DIR = "./evaluationPlots"
interactivePlots = False
styleSheet = "seaborn-v0_8-colorblind"
styleSheet = "seaborn-v0_8-poster"
plt.style.use(styleSheet)

matplotlib.rcParams["figure.dpi"] = 300
plt.rcParams["patch.linewidth"] = 2
def readDataRates(path: str) -> dict:
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            key, _, valueStr = line.partition(":")
            key = key.strip()
            valueStr = valueStr.strip()
            try:
                value = int(valueStr)
            except ValueError:
                value = float(valueStr)
            results[key] = value
    return results



# with open("./ahhh.txt","w") as f:
#     for filepath in filepaths:
#         results=readDataRates(filepath)
#         for key in ['1-clusterRateModel','1-clusterRateTiming','1-clusterRateModelAndTiming','1-dataRateModel','1-dataRateTiming','1-dataRateModelAndTiming']:
#             print(key,",",results[key],",",filepath[-18:-13],file=f)
# keys = ['1-clusterRateModel', '1-clusterRateTiming', '1-clusterRateModelAndTiming',
#         '1-dataRateModel', '1-dataRateTiming', '1-dataRateModelAndTiming']
# headers = [r"Cluster Model", r"Cluster Timing", r"Cluster Model+Timing",
#            r"Data Model", r"Data Timing", r"Data Model+Timing"]

# with open("./ahhh.tex", "w") as f:
#     print(r"\begin{table}[htbp]", file=f)
#     print(r"\centering", file=f)
#     print(r"\begin{tabular}{l" + "c" * len(keys) + "}", file=f)
#     print(r"\toprule", file=f)
#     print(r"Config & " + " & ".join(headers) + r" \\", file=f)
#     print(r"\midrule", file=f)
#     for filepath in filepaths:
#         results = readDataRates(filepath)
#         label = filepath[-18:-13].replace("_", r"\_")
#         values = " & ".join(f"{results[key]:.4f}" for key in keys)
#         print(f"{label} & {values} \\\\", file=f)
#     print(r"\bottomrule", file=f)
#     print(r"\end{tabular}", file=f)
#     print(r"\caption{Fraction of clusters and data rejected for each selection.}", file=f)
#     print(r"\label{tab:rejection}", file=f)
#     print(r"\end{table}", file=f)

def plotTimingResDf(allResDf,figsize=(13,5),pltDir:str=os.path.dirname(filepaths[0]),keyX:str="smearingPs",yscale='log',
                    # keyYs=['1-clusterRateModel','1-clusterRateTiming','1-clusterRateModelAndTiming','1-dataRateModel','1-dataRateTiming','1-dataRateModelAndTiming'],
                    keyYs=['clusterRateModel','clusterRateTiming','clusterRateModelAndTiming','dataRateModel','dataRateTiming','dataRateModelAndTiming'],
                    # ylabels=[r'$\mathcal{R}_{\mathrm{BIB,99}}^{\mathrm{cluster}}$',r'$[-3\sigma_t,5\sigma_t]$ rate',r'$\mathcal{R}_{\mathrm{BIB,99}}^{\mathrm{cluster}}$+$[-3\sigma_t,5\sigma_t]$',r'$\mathcal{R}_{\mathrm{BIB,99}}^{\mathrm{data}}$',r'$[-3\sigma,5\sigma]$ data rate',r'$\mathcal{R}_{\mathrm{BIB,99}}^{\mathrm{data}}$+$[-3\sigma,5\sigma]$',],
                    ylabels=["NN selection",r'$[-3\sigma_t,5\sigma_t]$ selection',r'NN + $[-3\sigma_t,5\sigma_t]$ selection',"NN selection",r'$[-3\sigma_t,5\sigma_t]$ selection',r'NN + $[-3\sigma_t,5\sigma_t]$ selection'],
                    ):
    plt.figure(figsize=figsize)
    assert len(ylabels)==len(keyYs)
    # assert len(ylabels)==3
    assert len(keyYs) == 6#currently supported way
    plt.subplot(121)
    for idx in range(3):
        # plt.errorbar(allResDf[keyX],allResDf[keyYs[idx]],yerr=allResDf[keyYs[idx]+"Err"],marker="o",label=ylabels[idx],linestyle='--')
        plt.plot(allResDf[keyX],allResDf[keyYs[idx]],marker="o",label=ylabels[idx],linestyle='--')
        if idx==1:
            plt.hlines(allResDf[keyYs[idx]][0],xmin=0,xmax=1000,linestyle="--",linewidth=1.2,color="gray")
    plt.legend()
    plt.xlabel("Timing Resolution (ps)")
    plt.ylabel("BIB Cluster Efficiency")
    plt.yscale(yscale)
    plt.subplot(122)
    for idx in range(3):
        idx+=3
        # plt.errorbar(allResDf[keyX],allResDf[keyYs[idx]],yerr=allResDf[keyYs[idx][2:]+"Err"],marker="o",label=ylabels[idx],linestyle='--')
        plt.plot(allResDf[keyX],allResDf[keyYs[idx]],marker="o",label=ylabels[idx],linestyle='--')
        if idx==4:
            plt.hlines(allResDf[keyYs[idx]][0],xmin=0,xmax=1000,linestyle="--",linewidth=1.2,color="gray")
            # plt.hlines(0.5,xmin=0,xmax=1000,linestyle="--",linewidth=5,color="gray")
            # plt.text(0,0,"AAHHHHHHHHHHHHHHHHHh")
    plt.legend()
    plt.xlabel("Timing Resolution (ps)")
    plt.ylabel("Relative BIB Data Rate")
    # plt.grid(True,'major')
    plt.yscale(yscale)
    plt.tight_layout()
    
    plt.savefig(os.path.join(pltDir,"timingSmearingStudy.png"))
    print("saved figure to ",os.path.join(pltDir,"timingSmearingStudy.png"))

def main():
    allRes:list[dict] = []
    for filepath in filepaths:
        results = json.load(open(filepath))
        # row = pd.DataFrame(results)
        row = results
        numPs = int(re.search(r"_(\d+)ps_gaussian", filepath)[1])
        row["smearingPs"] = numPs
        # print(row)
        allRes.append(row)
    allResDf = pd.DataFrame(allRes)
    print(allResDf)

    plotTimingResDf(allResDf)




if __name__=="__main__":
    main()