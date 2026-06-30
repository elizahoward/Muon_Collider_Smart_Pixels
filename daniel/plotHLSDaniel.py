"""
Author: Daniel Abadjiev
Date: June 29, 2026
Description: script for plotting functions for the hlsComparison/merged_results.csv which has catpult and vitis csynth results
"""

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as axisartist
import pandas as pd
from pareto_all_models import _y, RUN_LABELS
from typing import Callable, Any


def plotCombinedDf(combinedDf: pd.DataFrame,savePath = "./hlsComparison/hls_model_synthesis_summary_plots.png") -> None:
    """
    Plots structural hardware metrics, synthesis area scores, and latency metrics 
    for each model execution row inside the combined DataFrame.
    """
    # 1. Verification Guard: Check that the DataFrame actually contains data rows
    assert not combinedDf.empty, "Assertion failed: Cannot plot an empty DataFrame."
    
    # 2. Extract or create a clean, descriptive X-axis label for each row configuration
    # If your DataFrame doesn't have an explicit 'hlsDir' short name, we fall back to run_name + trial_id
    plotDf = combinedDf.copy()
    if "hlsDir" in plotDf.columns:
        plotDf["config_label"] = plotDf["hlsDir"].apply(lambda x: str(x).split('/')[0])
    else:
        plotDf["config_label"] = plotDf["run_name"] + "_t" + plotDf["trial_id"].astype(str)

    # Use a clean, scannable plotting style layout
    sns.set_theme(style="whitegrid")
    
    # Create a 3-Row dashboard figure to completely isolate unrelated units of measurement
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    
    # -------------------------------------------------------------------------
    # PLOT 1: HARDWARE LOGIC UTILIZATION (LUTs, Registers, FFs, DSPs)
    # -------------------------------------------------------------------------
    # Melt the specific hardware count columns into a long-form structure for Seaborn
    hardwareCols = ["luts", "registers", "luts_plus_ff", "dsp", "bram"]
    # Check which columns are present to prevent key errors
    activeHardwareCols = [c for c in hardwareCols if c in plotDf.columns]
    
    dfHardware = plotDf.melt(
        id_vars=["config_label"], 
        value_vars=activeHardwareCols, 
        var_name="Metric", 
        value_name="Count"
    )
    
    sns.barplot(data=dfHardware, x="config_label", y="Count", hue="Metric", ax=axes[0])
    axes[0].set_title("RTL Hardware Logic Element Resource Utilization Counts", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Resource Count", fontsize=12)
    axes[0].set_xlabel("")
    
    # -------------------------------------------------------------------------
    # PLOT 2: CATAPULT ESTIMATED AREA SCORES (Scheduling vs DSP vs Assignment)
    # -------------------------------------------------------------------------
    areaCols = ["areaScorePostScheduling", "areaScorePostDSP", "areaScorePostAssignment"]
    activeAreaCols = [c for c in areaCols if c in plotDf.columns]
    
    dfArea = plotDf.melt(
        id_vars=["config_label"], 
        value_vars=activeAreaCols, 
        var_name="Synthesis Stage", 
        value_name="Area Score"
    )
    
    sns.barplot(data=dfArea, x="config_label", y="Area Score", hue="Synthesis Stage", ax=axes[1], palette="viridis")
    axes[1].set_title("Catapult Synthesis Optimization Stage Area Score History", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Estimated Area Score Value", fontsize=12)
    axes[1].set_xlabel("")

    # -------------------------------------------------------------------------
    # PLOT 3: TIMING & LATENCY PERFORMANCES (Clock Cycles & Periods)
    # -------------------------------------------------------------------------
    # We group designLatency and numClocks here since they map to equivalent metric frames
    timingCols = ["designLatency", "numClocks"]
    activeTimingCols = [c for c in timingCols if c in plotDf.columns]
    
    dfTiming = plotDf.melt(
        id_vars=["config_label"], 
        value_vars=activeTimingCols, 
        var_name="Timing Metric", 
        value_name="Cycles"
    )
    
    sns.barplot(data=dfTiming, x="config_label", y="Cycles", hue="Timing Metric", ax=axes[2], palette="magma")
    axes[2].set_title("Hardware Architecture Pipeline Total Latency Execution Cycles", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("Total Clock Cycles", fontsize=12)
    axes[2].set_xlabel("Synthesis Run Configurations", fontsize=12, fontweight="bold")
    
    # Clean up layout presentation details
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    
    # Save a physical file snippet copy directly to your folder directory
    plt.savefig(savePath, dpi=300)
    print(f"Dashboard visualization exported as '{savePath}'")
    plt.show()


def plotCombinedDf2(combinedDf: pd.DataFrame,savePath = "./hlsComparison/hls_variable_correlations.png") -> None:
    """
    Plots a multi-variable line series chart across all rows in combinedDf.
    All specified columns are normalized between 0 and 1. Custom markers 
    and sizes are applied to 'luts_plus_ff', 'areaScorePostAssignment', 
    and 'designLatency' to highlight specific correlation trends.
    """
    # 1. Verification Guard
    assert not combinedDf.empty, "Assertion failed: Cannot plot an empty DataFrame."
    
    # Target columns to trace
    targetColumns = [
        'luts', 'registers', 'luts_plus_ff', 'dsp', 'bram',
        'areaScorePostScheduling', 'areaScorePostDSP', 'areaScorePostAssignment', 
        'designLatency', 'clockPeriod', 'numClocks'
    ]
    
    # Filter to make sure we only plot columns that actually exist in the DataFrame
    activeColumns = [col for col in targetColumns if col in combinedDf.columns]
    assert activeColumns, "Assertion failed: None of the specified target columns were found in the DataFrame."

    # 2. RESTORED CLEAN SHORTHAND X-AXIS LABELLING LOGIC
    plotDf = combinedDf.copy()
    if "hlsDir" in plotDf.columns:
        plotDf["config_label"] = plotDf["hlsDir"].apply(lambda x: str(x).split('/')[2] if '/' in str(x) else str(x))
    else:
        plotDf["config_label"] = plotDf["run_name"] + "_t" + plotDf["trial_id"].astype(str)

    # Reset index to create a clean numeric coordinate line frame [0, 1, 2, ... Row N]
    plotDf = plotDf.reset_index(drop=True)
    
    # 3. Min-Max Normalization step to bring all variable tracks onto an identical scale
    normalizedDf = plotDf.copy()
    for col in activeColumns:
        colMin = plotDf[col].min()
        colMax = plotDf[col].max()
        
        if colMax != colMin:
            normalizedDf[col] = (plotDf[col] - colMin) / (colMax - colMin)
        else:
            normalizedDf[col] = 0.0 # Handles flat columns (like clockPeriod which is always 10.0)

    # 4. Construct the correlation series plot
    plt.figure(figsize=(16, 9))
    sns.set_theme(style="whitegrid")
    
    # Plot each column as an independent, continuous line path spanning across every row index
    for col in activeColumns:
        # --- CONDITIONAL STYLE SWITCHING ---
        if col == "luts_plus_ff":
            currentMarker = '^'      # Triangle pointing up
            currentSize = 10         # Highly prominent marker size
            currentLineWidth = 2.5   # Thicker line path weight
        elif col == "areaScorePostAssignment":
            currentMarker = 's'      # Square marker
            currentSize = 9          # Bigger marker size
            currentLineWidth = 2.2   # Highlighted line weight
        elif col == "designLatency":
            currentMarker = 'h'      # Hexagon marker
            currentSize = 9          # Bigger marker size
            currentLineWidth = 2.2   # Highlighted line weight
        else:
            currentMarker = 'o'      # Default circle marker
            currentSize = 4          # Standard size
            currentLineWidth = 1.5   # Standard line weight

        plt.plot(
            normalizedDf.index, 
            normalizedDf[col], 
            marker=currentMarker, 
            markersize=currentSize, 
            linewidth=currentLineWidth, 
            label=col,
            alpha = 0.8
        )
        
    # Configure graph layout specifications
    plt.title("HLS Variable Correlation Series (Normalized Values Across All Rows)", fontsize=14, fontweight="bold")
    plt.ylabel("Normalized Value Scale (0.0 = Minimum, 1.0 = Maximum)", fontsize=12)
    plt.xlabel("DataFrame Row Configurations (Ordered Sequentially)", fontsize=12)
    
    # Force the X-axis tick placements to correspond precisely to each row index position
    plt.xticks(
        ticks=normalizedDf.index, 
        labels=normalizedDf["config_label"], 
        rotation=45, 
        ha="right", 
        fontsize=9
    )
    
    # Position the legend box completely off to the side so it doesn't block data paths
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.tight_layout()
    
    # Export a permanent png layout file back to disk
    plt.savefig(savePath, dpi=300)
    print(f"Correlation series visualization exported successfully as '{savePath}'")
    plt.show()



def plotCombinedDF3(combinedDf: pd.DataFrame,savePath = "./hlsComparison/hls_parasite_multi_axis_trends.png") -> None:
    """
    Plots a multi-Y-axis parasite plot tracking the actual raw values of 
    'luts_plus_ff', 'areaScorePostAssignment', and 'designLatency' across all rows.
    """
    # 1. Verification Guard
    assert not combinedDf.empty, "Assertion failed: Cannot plot an empty DataFrame."
    
    focusColumns = ['luts_plus_ff', 'areaScorePostAssignment', 'designLatency']
    activeColumns = [col for col in focusColumns if col in combinedDf.columns]
    assert len(activeColumns) == len(focusColumns), (
        f"Assertion failed: Missing required columns. "
        f"Expected {focusColumns}, but found {activeColumns}"
    )

    # 2. FIXED: Restored your exact index extraction rule from plotCombinedDF
    plotDf = combinedDf.copy()
    if "hlsDir" in plotDf.columns:
        plotDf["config_label"] = plotDf["hlsDir"].apply(
            lambda x: str(x).split('/')[2] if '/' in str(x) else str(x)
        )
    else:
        plotDf["config_label"] = plotDf["run_name"] + "_t" + plotDf["trial_id"].astype(str)

    # Reset index to create clean numeric step coordinates [0, 1, 2, ... Row N]
    plotDf = plotDf.reset_index(drop=True)
    rowIndices = plotDf.index.tolist()

    # 3. Initialize the Parasite Axis Architecture matching your working format
    plt.figure(figsize=(16, 9))
    host = host_subplot(111, axes_class=axisartist.Axes)
    plt.subplots_adjust(right=0.70, bottom=0.20) 

    par1 = host.twinx()
    par2 = host.twinx()

    # Apply your exact axis configuration technique
    par1.axis["right"] = par1.new_fixed_axis(loc="right", offset=(0, 0))
    par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(80, 0))

    # 4. Color Definitions matching the structural components
    colorLuts = "#1f77b4"    # Strong Blue
    colorArea = "#2ca02c"    # Strong Green
    colorLatency = "#d62728" # Strong Red

    # 5. Plot Each Metric on its Respective Axis Layer
    
    # Metric 1: luts_plus_ff (Plotted on the primary Host Axis - Left Side)
    p1, = host.plot(
        rowIndices, 
        plotDf["luts_plus_ff"], 
        color=colorLuts, 
        marker='^', 
        markersize=11, 
        linewidth=2.5, 
        label="luts_plus_ff"
    )
    host.set_ylabel("luts_plus_ff (csynth)")
    host.axis["left"].label.set_color(colorLuts)

    # Metric 2: areaScorePostAssignment (Plotted on par1 - Inner Right Side)
    p2, = par1.plot(
        rowIndices, 
        plotDf["areaScorePostAssignment"], 
        color=colorArea, 
        marker='s', 
        markersize=10, 
        linewidth=2.5, 
        label="areaScorePostAssignment"
    )
    par1.set_ylabel("areaScorePostAssignment (Score um^2)")
    par1.axis["right"].label.set_color(colorArea)

    # Metric 3: designLatency (Plotted on par2 - Outer Right Side)
    p3, = par2.plot(
        rowIndices, 
        plotDf["designLatency"], 
        color=colorLatency, 
        marker='h', 
        markersize=10, 
        linewidth=2.5, 
        label="designLatency"
    )
    par2.set_ylabel("designLatency (ns)")
    par2.axis["right"].label.set_color(colorLatency)

    # 6. Configure X-Axis and Labels
    host.set_xlabel("DataFrame Row Configurations (Ordered Sequentially)", fontsize=12, fontweight="bold")
    host.set_title("HLS Multi-Axis Parasite Plot (Raw Parameter Values Tracking)", fontsize=14, fontweight="bold")
    
    # Force the X-axis tick placements to correspond precisely to each row index position
    plt.xticks(
        ticks=plotDf.index, 
        labels=plotDf["config_label"], 
        rotation=45, 
        ha="right", 
        fontsize=9
    )

    # Synchronize limits across the axis structures
    host.set_xlim(-0.5, len(rowIndices) - 0.5)

    # Consolidate labels into a unified legend block
    host.legend(handles=[p1, p2, p3], loc="upper left", fontsize=11)
    
    plt.savefig(savePath, dpi=300)
    print(f"Multi-Y-axis visualization exported successfully as '{savePath}'")
    plt.show()
# import numpy as np
# def _setup_complement_log_y(ax, yseries):
#     ax.set_yscale("log")
#     ax.invert_yaxis()
#     ticks_actual = [t for t in COMP_LOG_TICKS
#                     if t < yseries.max() + 0.05]
#     ticks_comp   = np.clip(1.0 - np.array(ticks_actual, dtype=float), 1e-4, None)
#     ax.set_yticks(ticks_comp)
#     ax.yaxis.set_major_formatter(
#         mticker.FuncFormatter(lambda v, _: f"{1 - v:.2f}"))
#     ax.yaxis.set_minor_formatter(mticker.NullFormatter())
#     y_min = max(1e-4, float(np.min(_y(yseries, True))) * 0.5)
#     y_max = float(np.max(_y(yseries, True))) * 2.0
#     ax.set_ylim(y_max, y_min)
def plotCombinedDf4(combinedDf: pd.DataFrame, savePath = "./hlsComparison/hls_csynthVsAreascore.png") -> None:
    plt.close()
    plt.figure(figsize=(12,20))
    plt.subplot(411)
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["areaScorePostAssignment"]/10e6,label="area score (mm^2)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["areaScorePostAssignment"]/10e6,".",label="area score (mm^2)")
    # plt.plot(combinedDf["luts_plus_ff"],combinedDf["numClocks"]/100,label="numClocks / 100")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["designLatency"]/1000,label="designLatency (us)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["designLatency"]/1000,".",label="designLatency (us)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["parameters"]/1.5e5,label="parameters / 1.5e5")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["bkg_rej_@99%"]/2,label="bkg_rej_@99% / 2")
    # par1.set_ylabel()
    plt.ylabel("areaScorePostAssignment (Score um^2 / 10^6) and numClocks / 100")
    plt.xlabel("luts_plus_ff (csynth)")
    plt.legend()

    plt.subplot(412)
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["areaScorePostAssignment"]/10e6,label="area score (mm^2)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["areaScorePostAssignment"]/10e6,".",label="area score (mm^2)")
    # plt.plot(combinedDf["luts_plus_ff"],combinedDf["numClocks"]/100,label="numClocks / 100")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["designLatency"]/1000,label="designLatency (us)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["designLatency"]/1000,".",label="designLatency (us)")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["parameters"]/1.5e5,label="parameters / 1.5e5")
    plt.plot(combinedDf["luts_plus_ff"],combinedDf["bkg_rej_@99%"]/2,label="bkg_rej_@99% / 2")
    # par1.set_ylabel()
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("areaScorePostAssignment (Score um^2 / 10^6) and numClocks / 100")
    plt.xlabel("luts_plus_ff (csynth)")
    plt.legend()
    plt.subplot(413)
    plt.plot(combinedDf["areaScorePostAssignment"],combinedDf["bkg_rej_@99%"])
    plt.plot(combinedDf["areaScorePostAssignment"],combinedDf["bkg_rej_@99%"],".")
    # plt.yscale('log')
    ax = plt.gca()
    # _setup_complement_log_y(ax,combinedDf["bkg_rej_@99%"])
    # ax.set_yscale("log")
    # ax.invert_yaxis()
    plt.ylabel("bkg_rej_@99%")
    plt.xlabel("areaScorePostAssignment (Score um^2)")
    plt.subplot(414)
    plt.plot(combinedDf["areaScorePostAssignment"],combinedDf["bkg_rej_@99%"])
    plt.plot(combinedDf["areaScorePostAssignment"],combinedDf["bkg_rej_@99%"],".")
    # plt.yscale('log')
    ax = plt.gca()
    # _setup_complement_log_y(ax,combinedDf["bkg_rej_@99%"])
    # ax.set_yscale("log")
    # ax.invert_yaxis()
    plt.ylabel("bkg_rej_@99%")
    plt.xlabel("areaScorePostAssignment (Score um^2)")
    annotate = True
    if annotate:
        for _, row in combinedDf.iterrows():
            short = (RUN_LABELS.get(row["run_name"], row["run_name"])
                     .replace("Model 1 ", "M1 ")
                     .replace("Model 2.5 ", "M2.5 ")
                     .replace("Model 3 ", "M3 "))
            yval = float(_y(row["bkg_rej_@99%"], False))
            ax.annotate(f"{short}\n{row['trial_id']}",
                        xy=(row["areaScorePostAssignment"], yval),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=6.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow",
                                  alpha=0.8, edgecolor="black", linewidth=0.8),
                        zorder=4)

    plt.savefig(savePath, dpi=300)
    print(f"custom visualization successfully as '{savePath}'")

def plotParasitically(xList: list[Any], yList:list[Any], xLabel:str, yLabels:list, colors: list = ["red", "blue", "green"], 
                      offsets:list = [0, 80], legendLabels:list = None, doLegend: bool = False, isSubplot:bool = False, subPlotNum = 0,
                      markers:list = None,title: str = "",figsize = [20,5],yScales:list = None,alphas:list = None):
    """
    plot several dataseries with common axis, using the parasitic stuff dynamically
    xList and yList are the x/y series that is plotted
    colors are the colors for each series
    and offsets are the y-axis offsets for the y-axes on the right, starting at par1.plot(xList[1], yList[1])
    """
    #verify inputs
    if isSubplot or (subPlotNum != 0):
        print("Not yet implemented as part of a subplot")
        return
    numPlots = len(xList)
    assert len(yList) == numPlots
    assert len(yLabels) == numPlots
    assert len(colors) == numPlots
    assert len(offsets) == numPlots-1
    if legendLabels is None:        
        print("Warning: Cannot make a legend without legendLabels, setting all labels to empty")
        legendLabels = ["" for i in range(numPlots)]
    assert len(legendLabels) == numPlots
    if markers is None:
        markers = ["o" for i in range(numPlots)]
    else:
        assert len(markers) == numPlots
    if yScales is None:
        yScales = ['linear' for i in range(numPlots)]
    assert len(yScales) == numPlots
    if alphas is None:
        alphas = [0.8 for i in range(numPlots)]
    assert len(alphas) == numPlots
    
    if not isSubplot:
        plt.figure(figsize = figsize)
        host = host_subplot(111, axes_class = axisartist.Axes)
        plt.subplots_adjust(right = 0.75)
    pars = [host.twinx() for i in range(numPlots -1)]
    #adjust offsets
    for idx, par in enumerate(pars):
        par.axis["right"] = par.new_fixed_axis(loc="right", offset=(offsets[idx],0))
    pars = [host] + pars
    assert len(pars) == numPlots #this should be unnecessary, just a sanity check

    #do the plotttting
    for idx in range(numPlots):
        pars[idx].plot(xList[idx], yList[idx], label = legendLabels[idx], color=colors[idx],marker=markers[idx],markersize=10,linewidth=2.5,alpha=alphas[idx])
        pars[idx].axis["left" if idx==0 else "right"].label.set_color(colors[idx])
        pars[idx].axis["left" if idx==0 else "right"].major_ticklabels.set_color(colors[idx])
        pars[idx].set(ylabel=yLabels[idx],yscale = yScales[idx])
    plt.xlabel(xLabel)
    plt.title(title)
    if doLegend:
        plt.legend()


    return pars

def plotCombinedDf3v2(combinedDf: pd.DataFrame,savePath = "./hlsComparison/hls_parasite_multi_axis_trendsV2.png") -> None:
    #initialization copied from before
    assert not combinedDf.empty, "Assertion failed: Cannot plot an empty DataFrame."
    
    focusColumns = ["bkg_rej_@99%",'luts_plus_ff', 'areaScorePostAssignment', 'designLatency',]
    activeColumns = [col for col in focusColumns if col in combinedDf.columns]
    assert len(activeColumns) == len(focusColumns), (
        f"Assertion failed: Missing required columns. "
        f"Expected {focusColumns}, but found {activeColumns}"
    )

    # 2. FIXED: Restored your exact index extraction rule from plotCombinedDF
    plotDf = combinedDf.copy()
    if "hlsDir" in plotDf.columns:
        plotDf["config_label"] = plotDf["hlsDir"].apply(
            lambda x: str(x).split('/')[2] if '/' in str(x) else str(x)
        )
    else:
        plotDf["config_label"] = plotDf["run_name"] + "_t" + plotDf["trial_id"].astype(str)

    # Reset index to create clean numeric step coordinates [0, 1, 2, ... Row N]
    plotDf = plotDf.reset_index(drop=True)
    rowIndices = plotDf.index.tolist()
    
    pars = plotParasitically([rowIndices for i in range(4)],[plotDf[column] for column in focusColumns],
                      "DataFrame Row Configurations (Ordered Sequentially)",yLabels = ["Background Rejection at 99% Sig Efficiency","luts_plus_ff (csynth)","areaScorePostAssignment (Score um^2)","designLatency (ns)"],
                      markers = ["o","^","h","s",],title = "Summary of hls metrics",doLegend=True, figsize=(20,10),
                      legendLabels = focusColumns,colors=["black",'red','blue','green'],offsets=[0,80,140],
                      yScales=["linear","log","log","linear"])
    # # Force the X-axis tick placements to correspond precisely to each row index position
    # plt.xticks(
    #     ticks=plotDf.index, 
    #     labels=plotDf["config_label"], 
    #     rotation=45, 
    #     ha="right", 
    #     fontsize=9
    # )
    host = pars[0]
    host.set_xticks(plotDf.index)
    host.set_xticklabels(plotDf["config_label"], rotation=45, ha="right", fontsize=9)
    bottom = host.axis["bottom"]
    bottom.major_ticklabels.set_rotation(45)
    bottom.major_ticklabels.set_ha("right")
    bottom.major_ticklabels.set_fontsize(9)
    bottom.label.set_pad(100) 
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, right=0.85)
        
    plt.savefig(savePath, dpi=300)
    print(f"Rebuilt Multi-Y-axis visualization exported successfully as '{savePath}'")
    plt.show()
    



