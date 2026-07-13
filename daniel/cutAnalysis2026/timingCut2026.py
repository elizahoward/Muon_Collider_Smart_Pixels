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
sys.path.append("../validationPlots")
import plotUtils
from plot_hit_time import *


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
                      PLOT_DIR = ".",interactivePlots = False,saveTitle = "cutHist",
                      bins = np.linspace(-0.5, 15, 100),standalone=True):
    if standalone:
        plt.figure()
    plotUtils.plotManyHisto(
        [truthSig[key], truthBib[key]],
        title=f"Signal and BIB Timing Distribution (30 ps smearing)",
        pltLabels=[f"Signal",  f"BIB"],
        bins=bins,
        showNums=False,
        # figsize=(7, 2),
        yscale="log",
        xlabel="time (ns)",
        pltStandalone=False,
        legendLoc = "upper right"
    )
    # plt.ylim([0,1e4])
    plt.vlines(cutLocations,0,10e5,color=cutColors)
    if standalone:
        plotUtils.closePlot(PLOT_DIR,interactivePlots,f"{saveTitle}.png")

from timing_cut_analysis import find_operating_points,TARGET_EFFICIENCIES,plot_roc

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

def main():
    key = "adjusted_hit_time_30ps_gaussian"
    fullTotal, cutTotal = numEvents(truthBib,key,-0.5,15)
    _,cutTight =  numEvents(truthBib,key,-0.09,0.15)
    plotHistoWithCuts(cutColors=["black","purple"])
    sweepDf = computeSweep2()
    print(sweepDf)
    doEricsSweepAnalysis(sweepDf)
    plotHistoWithCuts(cutLocations=[-0.09,0.15,0.063705,0.142950,0.369645],cutColors=["black","purple","cyan","red","green"],
                      bins = np.linspace(-0.2, 1, 100),saveTitle="cutHistPost")
    plotHistosTogether()


if __name__=="__main__":
    main()