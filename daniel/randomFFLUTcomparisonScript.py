"""
Author: Daniel Abadjiev
Date: August 5, 2026
Description: Quick script to look at some of the FF and LUT numbers

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# print(plt.rcParams.keys())
plt.rcParams["figure.dpi"] = 300

def main():
    df = pd.read_csv("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/combined_all_models_pareto_newJune2026/combined_all_detailed.csv")
    print(df)
    df["color"] = df["model"].map({"model1": "red", "model2_5": "blue","model3":"green"})
    markerList = ["^", "s", "p", "h", "8"] 
    df["markertype"] = df["run_name"].map({"model1_3w5i":    markerList[0], "model25_3bit":  markerList[0],"model3_3bit":markerList[0],
                                            "model1_4w6i":   markerList[1], "model25_4bit":  markerList[1],"model3_4bit":markerList[1],
                                            "model1_6w8i":   markerList[2], "model25_6bit":  markerList[2],"model3_6bit":markerList[2],
                                            "model1_8w10i":  markerList[3], "model25_8bit":  markerList[3],"model3_8bit":markerList[3],
                                            "model1_10w12i": markerList[4], "model25_10bit": markerList[4],"model3_10bit":markerList[4],
                                            })
    print(np.unique(df["markertype"]))
    plt.subplot(311)
    plotLutVsFF(df)
    plt.subplot(312)
    plotLutVsFF(df)
    plt.title("I think it's fine to use ff+luts because it looks like they increase at the same rate")
    plt.xlim([0,1e6])
    plt.subplot(313)
    plotLutVsFF(df,size=5)
    plt.xlim([0,1e5])
    plt.ylim([0,1e4])
    plt.tight_layout()
    plt.savefig("random_luts vs ffs.png")
    return

def plotLutVsFF(df,size=10):
    for i in range(len(df)):                                      
        plt.scatter(df["luts"].iloc[i],df["registers"].iloc[i],color=df["color"].iloc[i],marker=df["markertype"].iloc[i],alpha=0.5,s=size)
    plt.xlabel("luts (csynth)")
    plt.ylabel("ffs (csynth)")


if __name__=="__main__":
    main()