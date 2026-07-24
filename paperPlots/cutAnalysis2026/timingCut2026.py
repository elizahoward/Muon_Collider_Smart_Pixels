"""
Author: Daniel Abadjiev
Date: July 13, 2026
Description: Script to redo cutting analysis now since seems easier than trying to use old scripts
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../../MuC_Smartpix_ML")
sys.path.append("../../eric")
from timing_cut_analysis import find_operating_points,TARGET_EFFICIENCIES,plot_roc
#perhaps add 0.9806268960028482 SE to target efficiencies.. 
sys.path.append("../../daniel/validationPlots")
import plotUtils
import matplotlib
# from plot_hit_time import *
###################################################
# sys.path.append("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/validationPlots/")
# from plotUtils import load_parquet_pairs, countBibSig, plotManyHisto
#instead of importing from plot_hit_time, copied and made edits
matplotlib.rcParams["figure.dpi"] = 150

pkl_path = "/local/d1/smartpixML/cutAnalysis/dfOfTruth.pkl"

# --- Load data ---
print(f"Loading truthDF from {pkl_path}")
truthDF = pd.read_pickle(pkl_path)

# --- Split into sig / bib sub-groups ---
fracBib, fracSig, fracMM, fracMP, numTotalSig, numTotalBib, truthSig, truthBib_mm, truthBib_mp, truthBib = plotUtils.countBibSig(truthDF, doPrint=True)

##################################################################

def numEvents(df,key,lower,upper,doPrint=True):
    arr = df[key]
    numTotal = len(df)
    numTrimmed = len(df.query(f"{key} > @lower and {key} < @upper"))
    if doPrint:
        print(f"Total: {numTotal}, trimmed {numTrimmed}")
    return numTrimmed, numTotal

def defaultThresholds(t_max = 15,n_steps = 1000000):
    thresholds = np.linspace(0.0, t_max, n_steps + 1)[1:]  # skip 0
    thresholds = np.linspace(-0.1, t_max, n_steps + 1)
    return thresholds

def numEventsVectorized(
    truthBib=truthBib,
    truthSig=truthSig,
    thresholds=defaultThresholds(),
    key="adjusted_hit_time_30ps_gaussian",
    lower=-0.09
):
    # sig_arr = truthSig[key].values
    # bib_arr = truthBib[key].values

    # # Broadcast comparisons:
    # # shape: (len(thresholds), len(df))
    # sig_mask = (sig_arr > lower) & (sig_arr < thresholds[:, None])
    # bib_mask = (bib_arr > lower) & (bib_arr < thresholds[:, None])

    # # Count along axis 1
    # sig_pass_counts = sig_mask.sum(axis=1)
    # bib_pass_counts = bib_mask.sum(axis=1)
    # Extract and sort once
    sig_vals = np.sort(truthSig[key].values)
    bib_vals = np.sort(truthBib[key].values)

    # Find index where values become > lower
    sig_start = np.searchsorted(sig_vals, lower, side="right")
    bib_start = np.searchsorted(bib_vals, lower, side="right")

    # For each threshold, find index where values become >= threshold
    sig_end = np.searchsorted(sig_vals, thresholds, side="left")
    bib_end = np.searchsorted(bib_vals, thresholds, side="left")

    # Counts = end - start
    sig_pass_counts = sig_end - sig_start
    bib_pass_counts = bib_end - bib_start

    return sig_pass_counts, bib_pass_counts

def computeSweep2(truthBib=truthBib, truthSig=truthSig, thresholds = defaultThresholds(), key = "adjusted_hit_time_30ps_gaussian", lower = -0.09):
    # sig_pass_counts = np.zeros(thresholds.size)
    # bib_pass_counts = np.zeros(thresholds.size)
    # print(sig_pass_counts)
    # for i,threshold in enumerate(thresholds):
    #    _,sig_pass_counts[i] = numEvents(truthSig,key,lower,threshold,doPrint=False)
    #    _,bib_pass_counts[i] = numEvents(truthBib,key,lower,threshold,doPrint=False)

    # sig_pass_counts = np.array([
    #     numEvents(truthSig, key, lower, thr, doPrint=False)[1]
    #     for thr in thresholds
    # ])

    # bib_pass_counts = np.array([
    #     numEvents(truthBib, key, lower, thr, doPrint=False)[1]
    #     for thr in thresholds
    # ])
    sig_pass_counts, bib_pass_counts = numEventsVectorized(truthBib,truthSig,thresholds,key,lower)
    fullTotalSig, n_sig = numEvents(truthSig,key,-0.5,15)
    fullTotalBib, n_bib = numEvents(truthBib,key,-0.5,15)

    sig_eff = sig_pass_counts / n_sig 
    fpr     = bib_pass_counts / n_bib
    bkg_rej = 1.0 - fpr

    return pd.DataFrame({
        "threshold_ns": thresholds,
        "sig_eff": sig_eff,
        "fpr": fpr,
        "bkg_rej": bkg_rej,
    })

def plotHistoWithCuts(key = "adjusted_hit_time_30ps_gaussian",cutLocations = [-0.09,0.15], cutColors = ["green"],
                      PLOT_DIR = ".",interactivePlots = False,saveTitle = "cutHist",figsize = (6.5,3),
                      bins = np.linspace(-0.5, 15, 100),standalone=True,cutLabels = None,customLegendFunc = None):
    if standalone:
        plt.figure(figsize =figsize)
    plotUtils.plotManyHisto(
        [truthSig[key], truthBib[key]],
        title=f"Signal and BIB Timing Distribution (30 ps smearing)",
        pltLabels=[f"Signal",  f"BIB"],
        bins=bins,
        showNums=False,
        # figsize=(7, 2),
        yscale="log",
        xlabel="Time (ns)",
        pltStandalone=False,
        legendLoc = "upper right"
    )
    # plt.ylim([0,1e4])
    if cutLabels is None:
        cutLabels = ["" for _ in cutLocations]
    plt.vlines(cutLocations,0,10e5,color=cutColors,label=cutLabels)
    if customLegendFunc is not None:
        customLegendFunc()
    # plt.legend()
    if standalone:
        plotUtils.closePlot(PLOT_DIR,interactivePlots,f"{saveTitle}.png")



def doEricsSweepAnalysis(sweep_df,outputDir = "."):
    print("\nFinding operating points...")
    bkg_rej_weights = {0.95: 0, 0.98: 0, 0.99: 1}
    op_df = find_operating_points(sweep_df, TARGET_EFFICIENCIES, bkg_rej_weights)

    print("\nOperating points:")
    print(op_df.to_string(index=False))
    weighted = op_df["weighted_bkg_rej"].iloc[0]
    print(f"\n  Weighted background rejection = {weighted:.6f}" if weighted is not None else "\n  Weighted BKG rejection: not reached")

    os.makedirs(outputDir, exist_ok=True)
    sweep_path = os.path.join(outputDir, "timing_cut_sweep.csv")
    op_path    = os.path.join(outputDir, "timing_cut_operating.csv")
    roc_path   = os.path.join(outputDir, "timing_cut_roc.png")

    sweep_df.to_csv(sweep_path, index=False)
    op_df.to_csv(op_path, index=False)

    print("\nGenerating ROC curve...")
    plot_roc(sweep_df, op_df, roc_path,timingCutLabel="Timing cut -0.090 ns < t < T",title="ROC for Timing Cut")

    print(f"\nSaved:")
    print(f"  {sweep_path}")
    print(f"  {op_path}")
    print(f"  {roc_path}")
    print("=" * 65)

def plotHistosTogether(PLOT_DIR = ".",interactivePlots = False,saveTitle = "cutHistTogether",):
    plt.figure(figsize=(7,5))
    plt.subplot(211)
    plotHistoWithCuts(cutColors=["black","purple"],standalone=False)
    plt.subplot(212)
    plotHistoWithCuts(cutLocations=[-0.09,0.15,0.063705,0.142950,0.369645],cutColors=["black","purple","cyan","red","green"],
                      bins = np.linspace(-0.2, 1, 100),standalone=False)    
    plotUtils.closePlot(PLOT_DIR,interactivePlots,f"{saveTitle}.png")

def customPaperLegend(cutLocations=[-0.215,0.15,0.063705,0.142950,0.369645],cutColors=["black","purple","cyan","red","green"],
                      cutNames = ["start of \n cut",r"$5\sigma$ cut","99%  SE cut","98% SE cut","95% SE cut",],
                      cutYs = [1.5e5,5e4,6e5,2e5,6e5],
                      xOff = [0,0.01,0.0035,0.01,0.005]):
    # from matplotlib.lines import Line2D
    # custom_lines = [Line2D([0], [0], color='teal', linestyle='--'),
    #             Line2D([0], [0], color='gold', linestyle='-.')]
    # plt.legend(custom_lines, ['Dashed Line', 'Dash-dot Line'], facecolor='lightgrey', title='Line Styles', fontsize='medium', title_fontsize='large')

    # plt.legend()
    assert len(cutLocations)==len(cutColors)
    assert len(cutLocations)==len(cutNames)
    assert len(cutLocations)==len(cutYs)
    assert len(cutLocations)==len(xOff)
    for idx in range(len(cutLocations)):
        plt.text(cutLocations[idx]+xOff[idx],cutYs[idx],cutNames[idx],color=cutColors[idx])
    # plt.text(-0.2,1e5,,color="black")
    # plt.text(0.15,1e4,,color="purple")
    # plt.text(0.063705,1e6,,color="cyan")
    # plt.text(0.142950,1e5,,color="red")
    # plt.text(0.369645,1e5,color="green")

def main():
    key = "adjusted_hit_time_30ps_gaussian"
    fullTotal, cutTotal = numEvents(truthBib,key,-0.5,15)
    _,cutTight =  numEvents(truthBib,key,-0.09,0.15)
    plotHistoWithCuts(cutColors=["black","purple"])
    sweepDf = computeSweep2()
    print(sweepDf)
    sweepDf.to_csv("sweepDf.csv")
    doEricsSweepAnalysis(sweepDf)
    plotHistoWithCuts(cutLocations=[-0.09,0.15,0.063705,0.142950,0.369645],cutColors=["black","purple","cyan","red","green"],
                      bins = np.linspace(-0.2, 1, 100),saveTitle="cutHistPost_forPaper",
                      cutLabels=["-3 \sigma ","5\sigma","95% SE","98% SE","99% SE",],customLegendFunc=customPaperLegend)
    plotHistosTogether()


if __name__=="__main__":
    main()