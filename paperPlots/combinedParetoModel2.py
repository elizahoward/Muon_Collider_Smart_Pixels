"""
Author: Daniel Abadjiev, with integrating some code from Eric You
Date: July 16, 2026
Description: script to plot the model 2s from paretos with different bits onto the same plot
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotParetosTogether(allParetoDfs,saveTitle="./Model2ParetosTogether.png",
                        labels=["3 bit", "4 bit", "6 bit", "8 bit", "10 bit"],
                        colors = ["blue","magenta","green","cyan","purple"]):
    assert len(labels)==len(allParetoDfs)
    assert len(colors)==len(allParetoDfs)
    for idx,paretoDf in enumerate(allParetoDfs):
        primaryDF = paretoDf.query("pareto_type == 'primary'")
        secondaryDF = paretoDf.query("pareto_type == 'secondary'")
        plt.plot(primaryDF["parameters"],primaryDF["bkg_rej_@99%"],"D",markersize=8,label=labels[idx],color=colors[idx],alpha=0.7)
        # plt.plot(secondaryDF["parameters"],secondaryDF["bkg_rej_@99%"],".",label=labels[idx]+" secondary pareto front",color=colors[idx],alpha=0.7)        
        plt.plot(primaryDF["parameters"],primaryDF["bkg_rej_@99%"],"-",color=colors[idx],alpha=0.3)
        # plt.plot(secondaryDF["parameters"],secondaryDF["bkg_rej_@99%"],"--",color=colors[idx],alpha=0.3)
    plt.legend()
    plt.xlabel("Parameters")
    plt.ylabel("Background Rejectiona at 99% Signal Efficiency")
    
    plt.savefig(saveTitle)
def main(paretoCsvPath = "../eric/combined_all_models_pareto_newJune2026/combined_all_detailed.csv"):    
    paretoCsv = pd.read_csv(paretoCsvPath)
    pareto2 = paretoCsv.query("model == 'model2_5'")
    pareto2["pareto_type"] = pareto2["fullPath"].apply(
        lambda x: (
            "secondary"
            if "pareto_secondary" in str(x)
            else ("primary" if "pareto_primary" in str(x) else "unknown")
        )
    )
    print(pareto2)
    bitConfigs = ["model25_3bit","model25_4bit","model25_6bit","model25_8bit","model25_10bit"]
    allPareto2 = [pareto2.query("run_name == @bitConfig") for bitConfig in bitConfigs]
    plotParetosTogether(allPareto2)


if __name__=="__main__":
    main()