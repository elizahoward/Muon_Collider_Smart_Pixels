#!/usr/bin/env python3
"""
Cross-model Pareto selection on weighted background rejection vs FF & LUT usage.

This script is a lightweight, analysis-only companion to
`analyze_and_select_pareto_roc.py`. Instead of re-running ROC evaluation, it:

- Loads the existing ROC-based analysis CSVs (with weighted background rejection)
- Loads the HLS resource utilization CSVs (FF/registers, LUTs, etc.)
- Merges these per-trial metrics across four runs/models
- Performs a multi-objective Pareto selection:
    * Maximize:  primary_metric  (weighted background rejection)
    * Minimize:  luts, registers (FF)
- Saves combined CSVs and a quick diagnostic plot.

Assumptions:
- Each run directory already contains:
    - `roc_based_analysis_detailed.csv`
    - `resource_utilization.csv`
- `trial_id` in the ROC CSV matches `model_name` in the resource CSV via:
      model_name == f"model_trial_{trial_id}"

Author: Eric (script mock-up by assistant)
Date: February 2026
"""

import os
import argparse
from dataclasses import dataclass
from typing import List
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class RunConfig:
    name: str          # Short tag for the run/model (e.g. 'model2_5_10bit')
    path: str          # Absolute path to the run directory


def load_run_metrics(run: RunConfig) -> pd.DataFrame:
    """
    Load ROC-based metrics and resource utilization for a single run,
    then merge them into a single DataFrame.
    """
    roc_csv = os.path.join(run.path, "roc_based_analysis_detailed.csv")
    res_csv = os.path.join(run.path, "resource_utilization.csv")

    if not os.path.isfile(roc_csv):
        raise FileNotFoundError(f"ROC analysis CSV not found: {roc_csv}")
    if not os.path.isfile(res_csv):
        raise FileNotFoundError(f"Resource utilization CSV not found: {res_csv}")

    roc_df = pd.read_csv(roc_csv)
    res_df = pd.read_csv(res_csv)

    # Ensure expected columns exist
    required_roc_cols = {"trial_id", "parameters", "auc", "primary_metric"}
    missing_roc = required_roc_cols - set(roc_df.columns)
    if missing_roc:
        raise ValueError(f"{roc_csv} missing required columns: {sorted(missing_roc)}")

    required_res_cols = {"model_name", "luts", "registers"}
    missing_res = required_res_cols - set(res_df.columns)
    if missing_res:
        raise ValueError(f"{res_csv} missing required columns: {sorted(missing_res)}")

    # Normalise trial_id types for a clean merge:
    # - ROC CSV usually has numeric IDs (e.g. 3, 27, 101)
    # - Resource CSV encodes them in model_name (e.g. 'model_trial_003')
    #
    # We convert both to zero-padded 3-character strings so that:
    #   3   -> '003'
    #   27  -> '027'
    #   101 -> '101'
    roc_df = roc_df.copy()
    roc_df["trial_id"] = roc_df["trial_id"].astype(str).str.zfill(3)

    # Create trial_id column in resource DF from model_name = 'model_trial_XXX'
    res_df = res_df.copy()
    res_df["trial_id"] = res_df["model_name"].str.replace("model_trial_", "", regex=False)
    res_df["trial_id"] = res_df["trial_id"].astype(str).str.zfill(3)

    # Merge on trial_id; inner join to keep only trials with both ROC + resources
    merged = pd.merge(roc_df, res_df, on="trial_id", how="inner", suffixes=("", "_res"))

    # Tag with run/model name and create a globally-unique ID
    merged["run_name"] = run.name
    merged["global_id"] = merged["run_name"] + "_" + merged["trial_id"].astype(str)

    return merged


def is_dominated(point: pd.Series,
                 others: pd.DataFrame,
                 maximize_cols: List[str],
                 minimize_cols: List[str]) -> bool:
    """
    Check if a point is dominated by any other point in others.
    A point A is dominated by B if:
      - B is at least as good as A in all objectives, and
      - B is strictly better than A in at least one objective.
    """
    for _, other in others.iterrows():
        better_in_all = True
        strictly_better_in_one = False

        # Maximization objectives (e.g. primary_metric)
        for col in maximize_cols:
            if other[col] < point[col]:
                better_in_all = False
                break
            if other[col] > point[col]:
                strictly_better_in_one = True

        if not better_in_all:
            continue

        # Minimization objectives (e.g. luts, registers)
        for col in minimize_cols:
            if other[col] > point[col]:
                better_in_all = False
                break
            if other[col] < point[col]:
                strictly_better_in_one = True

        if better_in_all and strictly_better_in_one:
            return True

    return False


def find_pareto_front(df: pd.DataFrame,
                      maximize_cols: List[str],
                      minimize_cols: List[str]) -> pd.DataFrame:
    """Return the subset of df that is Pareto optimal."""
    pareto_indices = []

    for idx, row in df.iterrows():
        others = df.drop(index=idx)
        if not is_dominated(row, others, maximize_cols, minimize_cols):
            pareto_indices.append(idx)

    pareto_df = df.loc[pareto_indices].copy()
    # Sort primarily by performance, then by LUTs / registers
    pareto_df = pareto_df.sort_values(
        by=[maximize_cols[0], "luts", "registers"],
        ascending=[False, True, True],
    )
    return pareto_df


def plot_pareto_luts_vs_metric(df: pd.DataFrame,
                               pareto_df: pd.DataFrame,
                               output_dir: str) -> None:
    """
    Create two plots:
    1. Full plot showing all models and Pareto front
    2. Zoomed plot focusing on top-left corner (high metric, low LUT+FF)
    X-axis: LUTs + FF (registers) combined.
    """
    # Color mapping for runs
    run_colors = {
        'model2_5_10bit': 'blue',
        'model2_5_6bit_new': 'green',
        'eric_model_2_5_4bit': 'purple',
        'eric_model_2_5_8bit': 'red',
        'eric_new_smaller_2_23_run': 'orange',
    }
    
    run_labels = {
        'model2_5_10bit': 'Model 2.5 (10-bit)',
        'model2_5_6bit_new': 'Model 2.5 (6-bit)',
        'eric_model_2_5_4bit': 'Model 2.5 (4-bit)',
        'eric_model_2_5_8bit': 'Model 2.5 (8-bit)',
        'eric_new_smaller_2_23_run': 'Model 2.5 (smaller)',
    }
    
    x_col = 'luts_plus_ff'
    
    # ========== FULL PLOT ==========
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot all models by run
    for run_name, color in run_colors.items():
        df_run = df[df['run_name'] == run_name]
        if df_run.empty:
            continue
        ax.scatter(df_run[x_col], df_run['primary_metric'], 
                   alpha=0.4, s=60, c=color, edgecolors='gray', 
                   linewidth=0.5, label=run_labels.get(run_name, run_name), zorder=1)
    
    # Pareto front - color by run
    for run_name, color in run_colors.items():
        pareto_run = pareto_df[pareto_df['run_name'] == run_name]
        if not pareto_run.empty:
            ax.scatter(pareto_run[x_col], pareto_run['primary_metric'], 
                       alpha=0.9, s=150, c=color, edgecolors='black', 
                       linewidth=2, marker='D', zorder=3,
                       label=f'Pareto: {run_labels.get(run_name, run_name)}')
    
    # Pareto line
    pareto_sorted = pareto_df.sort_values(x_col)
    ax.plot(pareto_sorted[x_col], pareto_sorted['primary_metric'],
            'k--', alpha=0.6, linewidth=2, zorder=2, label='Pareto front')
    
    # Annotate Pareto points with trial ID and model label
    for _, row in pareto_df.iterrows():
        trial_id = row['trial_id']
        run_label = run_labels.get(row['run_name'], row['run_name']).replace('Model 2.5 ', 'M2.5 ')
        # ax.annotate(f"{run_label}\n{trial_id}", 
        #            xy=(row[x_col], row['primary_metric']),
        #            xytext=(8, 8), textcoords='offset points',
        #            fontsize=7, color='black', fontweight='bold',
        #            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
        #                     alpha=0.8, edgecolor='black', linewidth=1),
        #            zorder=4)
    
    # Labels and title
    ax.set_xlabel('LUTs + FF (registers)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weighted Background Rejection', fontsize=14, fontweight='bold')
    ax.set_title('Cross-Model Pareto Front: Weighted Bkg Rej vs LUTs + FF', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # Statistics box
    stats_text = (
        f"Total models: {len(df)}\n"
        f"Pareto optimal: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
        f"Metric range: {df['primary_metric'].min():.4f} - {df['primary_metric'].max():.4f}"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'pareto_front_weighted_bkgrej_vs_luts_plus_ff_combined.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()
    
    # ========== ZOOMED PLOT (top-left corner) ==========
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot all models by run
    for run_name, color in run_colors.items():
        df_run = df[df['run_name'] == run_name]
        if df_run.empty:
            continue
        ax.scatter(df_run[x_col], df_run['primary_metric'], 
                   alpha=0.4, s=60, c=color, edgecolors='gray', 
                   linewidth=0.5, label=run_labels.get(run_name, run_name), zorder=1)
    
    # Pareto front - color by run
    for run_name, color in run_colors.items():
        pareto_run = pareto_df[pareto_df['run_name'] == run_name]
        if not pareto_run.empty:
            ax.scatter(pareto_run[x_col], pareto_run['primary_metric'], 
                       alpha=0.9, s=150, c=color, edgecolors='black', 
                       linewidth=2, marker='D', zorder=3,
                       label=f'Pareto: {run_labels.get(run_name, run_name)}')
    
    # Pareto line
    pareto_sorted = pareto_df.sort_values(x_col)
    ax.plot(pareto_sorted[x_col], pareto_sorted['primary_metric'],
            'k--', alpha=0.6, linewidth=2, zorder=2, label='Pareto front')
    
    # Annotate Pareto points with trial ID and model label
    for _, row in pareto_df.iterrows():
        trial_id = row['trial_id']
        run_label = run_labels.get(row['run_name'], row['run_name']).replace('Model 2.5 ', 'M2.5 ')
        # ax.annotate(f"{run_label}\n{trial_id}", 
        #            xy=(row[x_col], row['primary_metric']),
        #            xytext=(8, 8), textcoords='offset points',
        #            fontsize=7, color='black', fontweight='bold',
        #            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
        #                     alpha=0.8, edgecolor='black', linewidth=1),
        #            zorder=4)
    
    # Zoom to specific region: 0.6-1.0 rejection, 0-250k LUT+FF
    ax.set_xlim(0, 0.25e6)  # 0 to 250,000 LUTs + FF
    ax.set_ylim(0.6, 1.0)   # 0.6 to 1.0 weighted background rejection
    
    # Labels and title
    ax.set_xlabel('LUTs + FF (registers)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weighted Background Rejection', fontsize=14, fontweight='bold')
    ax.set_title('Cross-Model Pareto Front (Zoomed): Top-Left Corner', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # Statistics box
    stats_text = (
        f"Total models: {len(df)}\n"
        f"Pareto optimal: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
        f"Zoomed to top performers"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    plot_path_zoom = os.path.join(output_dir, 'pareto_front_weighted_bkgrej_vs_luts_plus_ff_zoomed.png')
    plt.savefig(plot_path_zoom, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path_zoom}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model Pareto selection on weighted background rejection vs FF/LUT usage.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/New_hyperparameter_runs",
        help="Base directory containing the four run subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for combined CSVs and plots "
             "(default: <base_dir>/combined_pareto_ff_lut).",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    if args.output_dir is None:
        output_dir = os.path.join(base_dir, "combined_pareto_ff_lut")
    else:
        output_dir = os.path.abspath(args.output_dir)

    runs = [
        RunConfig(name="model2_5_10bit",
                  path=os.path.join(base_dir, "model2_5_10bit")),
        RunConfig(name="model2_5_6bit_new",
                  path=os.path.join(base_dir, "model2_5_6bit_new")),
        RunConfig(name="eric_model_2_5_4bit",
                  path=os.path.join(base_dir, "eric_model_2_5_4bit")),
        RunConfig(name="eric_model_2_5_8bit",
                  path=os.path.join(base_dir, "model2_5_8_bit")),
    ]

    # Load and concatenate metrics
    all_dfs = []
    for run in runs:
        if not os.path.isdir(run.path):
            print(f"Warning: run directory not found, skipping: {run.path}")
            continue
        print(f"Loading metrics for run: {run.name} ({run.path})")
        df_run = load_run_metrics(run)
        all_dfs.append(df_run)

    if not all_dfs:
        raise RuntimeError("No runs could be loaded; nothing to do.")

    all_df = pd.concat(all_dfs, ignore_index=True)

    # Combined LUT + FF (registers) for x-axis
    all_df["luts_plus_ff"] = all_df["luts"] + all_df["registers"]

    # Basic sanity printout
    print(f"\nTotal models with ROC + resources: {len(all_df)}")
    print("Runs present:", ", ".join(sorted(all_df["run_name"].unique())))
    print(f"Primary metric range: {all_df['primary_metric'].min():.4f} - "
          f"{all_df['primary_metric'].max():.4f}")

    # Multi-objective Pareto:
    #   - maximize: primary_metric (weighted background rejection)
    #   - minimize: luts, registers (FF)
    maximize_cols = ["primary_metric"]
    minimize_cols = ["luts", "registers"]

    print("\n" + "=" * 80)
    print("PARETO SELECTION")
    print("=" * 80)
    
    pareto_df = find_pareto_front(all_df, maximize_cols=maximize_cols, minimize_cols=minimize_cols)

    print(f"\nPareto optimal models (across all runs): {len(pareto_df)}")

    os.makedirs(output_dir, exist_ok=True)

    # Save combined detailed CSV
    combined_csv = os.path.join(output_dir, "combined_roc_resources_detailed.csv")
    all_df.to_csv(combined_csv, index=False)
    print(f"\nSaved combined detailed CSV: {combined_csv}")

    # Save primary Pareto CSV
    primary_csv = os.path.join(output_dir, "pareto_ff_lut_primary.csv")
    cols_to_keep = [
        "run_name",
        "trial_id",
        "global_id",
        "parameters",
        "auc",
        "primary_metric",
        "luts",
        "registers",
        "luts_plus_ff",
        "luts_percent",
        "registers_percent",
        "dsp",
        "dsp_percent",
        "estimated_fmax",
    ]
    existing_cols = [c for c in cols_to_keep if c in pareto_df.columns]
    pareto_df[existing_cols].to_csv(primary_csv, index=False)
    print(f"Saved Pareto optimal CSV: {primary_csv}")

    # Save JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(all_df),
        'pareto_optimal_models': len(pareto_df),
        'primary_metric_range': {
            'min': float(all_df['primary_metric'].min()),
            'max': float(all_df['primary_metric'].max()),
            'mean': float(all_df['primary_metric'].mean())
        },
        'luts_range': {
            'min': int(all_df['luts'].min()),
            'max': int(all_df['luts'].max()),
            'mean': float(all_df['luts'].mean())
        },
        'luts_plus_ff_range': {
            'min': int(all_df['luts_plus_ff'].min()),
            'max': int(all_df['luts_plus_ff'].max()),
            'mean': float(all_df['luts_plus_ff'].mean())
        },
        'registers_range': {
            'min': int(all_df['registers'].min()),
            'max': int(all_df['registers'].max()),
            'mean': float(all_df['registers'].mean())
        }
    }
    
    summary_json = os.path.join(output_dir, 'pareto_ff_lut_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved analysis summary: {summary_json}")

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    plot_pareto_luts_vs_metric(all_df, pareto_df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
