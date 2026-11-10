"""
Author: Daniel Abadjiev
Date: August 5, 2026
Description: Quick script to look at some of the FF and LUT numbers

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
# print(plt.rcParams.keys())
plt.rcParams["figure.dpi"] = 300

def main():
    df = pd.read_csv("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/combined_all_models_pareto_newJune2026/combined_all_detailed.csv")
    # print(df)
    df["color"] = df["model"].map({"model1": "red", "model2_5": "blue","model3":"green"})
    markerList = ["^", "s", "p", "h", "8"] 
    df["markertype"] = df["run_name"].map({"model1_3w5i":    markerList[0], "model25_3bit":  markerList[0],"model3_3bit":markerList[0],
                                            "model1_4w6i":   markerList[1], "model25_4bit":  markerList[1],"model3_4bit":markerList[1],
                                            "model1_6w8i":   markerList[2], "model25_6bit":  markerList[2],"model3_6bit":markerList[2],
                                            "model1_8w10i":  markerList[3], "model25_8bit":  markerList[3],"model3_8bit":markerList[3],
                                            "model1_10w12i": markerList[4], "model25_10bit": markerList[4],"model3_10bit":markerList[4],
                                            })
    # print(np.unique(df["markertype"]))
    dfSubsets = [df, df.query("model == 'model1'"), df.query("model == 'model2_5'"), df.query("model == 'model3'"),df.query("model == 'model1' or model == 'model2_5'"),df.query("luts < 200000"),df.query("luts < 2000000"),df.query("luts < 106400"),]
    totalDFs = len(dfSubsets)
    titles = ["all models", "model 1", "model 2", "model 3", "model 1 and 2","models with luts < 200000", "models with luts < 2000000","models with luts < 106400"]
    assert len(titles) == totalDFs
    plt.figure(figsize=(18*1.5,10*1.5))
    cols = 4
    for i,df in enumerate(dfSubsets):
        plt.subplot(int(np.ceil(totalDFs/cols)),cols,i+1)
        plotLutVsFF(df,title=titles[i])
    plt.tight_layout()
    plt.savefig("random_luts vs ffs.png")
    return

#borrowed from plot_parameters_vs_resources_multi_bit.py
def fit_linear_regression(x, y):
    """
    Fit OLS linear regression and return statistics.
    """
    assert len(x) == len(y)
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    r_squared = r_value ** 2

    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_err": std_err,
        "y_pred": y_pred,
        "residuals": residuals,
        "n_models": len(x)
    }

def plotLutVsFF(df,size=10,title="",keyx="luts",keyy="registers",xlabel="luts (csynth)",ylabel="ffs (csynth)"):
    for i in range(len(df)):                                      
        plt.scatter(df[keyx].iloc[i],df[keyy].iloc[i],color=df["color"].iloc[i],marker=df["markertype"].iloc[i],alpha=0.5,s=size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    addLinearFit(df,keyx,keyy,title)

def addLinearFit(df,keyx,keyy,title,color="black"):
    stats_dict = fit_linear_regression(df[keyx],df[keyy])
    x_line = np.linspace(df[keyx].min(), df[keyx].max(),10)
    y_line = stats_dict["slope"] * x_line + stats_dict["intercept"]
    # print(x_line)
    # print(y_line)
    plt.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=2.0,
        color=color,
        alpha=0.9,
        label=f"linear fit (slope={stats_dict['slope']:.2f}, R²={stats_dict['r_squared']:.3f},\n intercept={stats_dict['intercept']:.2f}, n={int(stats_dict['n_models'])})",
    )
    plt.legend()
    table_text = (
            # f"{title}: "
            f"slope={stats_dict['slope']:.2f}, "
            f"R²={stats_dict['r_squared']:.3f}, n={int(stats_dict['n_models'])}\n"
        )
    # plt.text(
    #     0.5,
    #     0.6,
    #     table_text,
    #     # transform=ax.transAxes,
    #     # fontsize=9,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    #     family="monospace",
    # )
    # slopes_summary.append(
    #     {
    #         "bit_width": bit,
    #         "slope": stats_dict["slope"],
    #         "intercept": stats_dict["intercept"],
    #         "r_squared": stats_dict["r_squared"],
    #         "r_value": stats_dict["r_value"],
    #         "p_value": stats_dict["p_value"],
    #         "std_err": stats_dict["std_err"],
    #         "n_models": len(df),
    #     }
    # )


if __name__=="__main__":
    main()