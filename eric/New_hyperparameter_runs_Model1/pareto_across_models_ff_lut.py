#!/usr/bin/env python3
"""
Cross-model Pareto selection on weighted background rejection vs FF & LUT usage.

Companion to pareto_across_models_ff_lut.py in New_hyperparameter_runs_Model2_5,
adapted for the three Model 1 bit-width runs:
  - Model 1 (4-bit)  →  Model1_quantized_4w0i_pareto_roc_selected/
  - Model 1 (6-bit)  →  Model1_quantized_6w0i_pareto_roc_selected/
  - Model 1 (8-bit)  →  Model1_quantized_8w0i_pareto_roc_selected/

Each run directory must already contain:
  - roc_based_analysis_detailed.csv
  - resource_utilization.csv

Author: Eric
Date: April 2026
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
    name: str   # Short tag (e.g. 'model1_4bit')
    path: str   # Absolute path to the run directory


def load_run_metrics(run: RunConfig) -> pd.DataFrame:
    roc_csv = os.path.join(run.path, "roc_based_analysis_detailed.csv")
    res_csv = os.path.join(run.path, "resource_utilization.csv")

    if not os.path.isfile(roc_csv):
        raise FileNotFoundError(f"ROC analysis CSV not found: {roc_csv}")
    if not os.path.isfile(res_csv):
        raise FileNotFoundError(f"Resource utilization CSV not found: {res_csv}")

    roc_df = pd.read_csv(roc_csv)
    res_df = pd.read_csv(res_csv)

    required_roc_cols = {"trial_id", "parameters", "auc", "primary_metric"}
    missing_roc = required_roc_cols - set(roc_df.columns)
    if missing_roc:
        raise ValueError(f"{roc_csv} missing required columns: {sorted(missing_roc)}")

    required_res_cols = {"model_name", "luts", "registers"}
    missing_res = required_res_cols - set(res_df.columns)
    if missing_res:
        raise ValueError(f"{res_csv} missing required columns: {sorted(missing_res)}")

    roc_df = roc_df.copy()
    roc_df["trial_id"] = roc_df["trial_id"].astype(str).str.zfill(3)

    res_df = res_df.copy()
    res_df["trial_id"] = res_df["model_name"].str.replace("model_trial_", "", regex=False)
    res_df["trial_id"] = res_df["trial_id"].astype(str).str.zfill(3)

    merged = pd.merge(roc_df, res_df, on="trial_id", how="inner", suffixes=("", "_res"))
    merged["run_name"] = run.name
    merged["global_id"] = merged["run_name"] + "_" + merged["trial_id"].astype(str)

    return merged


def is_dominated(point: pd.Series,
                 others: pd.DataFrame,
                 maximize_cols: List[str],
                 minimize_cols: List[str]) -> bool:
    for _, other in others.iterrows():
        better_in_all = True
        strictly_better_in_one = False

        for col in maximize_cols:
            if other[col] < point[col]:
                better_in_all = False
                break
            if other[col] > point[col]:
                strictly_better_in_one = True

        if not better_in_all:
            continue

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
    pareto_indices = []
    for idx, row in df.iterrows():
        others = df.drop(index=idx)
        if not is_dominated(row, others, maximize_cols, minimize_cols):
            pareto_indices.append(idx)

    pareto_df = df.loc[pareto_indices].copy()
    pareto_df = pareto_df.sort_values(
        by=[maximize_cols[0], "luts", "registers"],
        ascending=[False, True, True],
    )
    return pareto_df


RUN_COLORS = {
    'model1_4bit': 'purple',
    'model1_6bit': 'green',
    'model1_8bit': 'red',
}

RUN_LABELS = {
    'model1_4bit': 'Model 1 (4-bit)',
    'model1_6bit': 'Model 1 (6-bit)',
    'model1_8bit': 'Model 1 (8-bit)',
}


def plot_pareto_luts_vs_metric(df: pd.DataFrame,
                               pareto_df: pd.DataFrame,
                               output_dir: str) -> None:
    """Full plot and zoomed plot of weighted bkg rejection vs LUTs + FF."""
    x_col = 'luts_plus_ff'

    for zoomed in (False, True):
        fig, ax = plt.subplots(figsize=(14, 9))

        # All models, coloured by bit-width
        for run_name, color in RUN_COLORS.items():
            df_run = df[df['run_name'] == run_name]
            if df_run.empty:
                continue
            ax.scatter(df_run[x_col], df_run['primary_metric'],
                       alpha=0.4, s=60, c=color, edgecolors='gray',
                       linewidth=0.5, label=RUN_LABELS.get(run_name, run_name), zorder=1)

        # Pareto points, coloured by bit-width
        for run_name, color in RUN_COLORS.items():
            pareto_run = pareto_df[pareto_df['run_name'] == run_name]
            if not pareto_run.empty:
                ax.scatter(pareto_run[x_col], pareto_run['primary_metric'],
                           alpha=0.9, s=150, c=color, edgecolors='black',
                           linewidth=2, marker='D', zorder=3,
                           label=f'Pareto: {RUN_LABELS.get(run_name, run_name)}')

        # Pareto connecting line
        pareto_sorted = pareto_df.sort_values(x_col)
        ax.plot(pareto_sorted[x_col], pareto_sorted['primary_metric'],
                'k--', alpha=0.6, linewidth=2, zorder=2, label='Pareto front')

        # Annotations on Pareto points
        for _, row in pareto_df.iterrows():
            short_label = RUN_LABELS.get(row['run_name'], row['run_name']).replace('Model 1 ', 'M1 ')
            ax.annotate(f"{short_label}\n{row['trial_id']}",
                        xy=(row[x_col], row['primary_metric']),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=7, color='black', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                                  alpha=0.8, edgecolor='black', linewidth=1),
                        zorder=4)

        ax.set_xlabel('LUTs + FF (registers)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Weighted Background Rejection', fontsize=14, fontweight='bold')

        if zoomed:
            ax.set_xlim(0, 0.25e6)
            ax.set_ylim(0.6, 1.0)
            ax.set_title('Model 1: Cross-Bit-Width Pareto Front (Zoomed)',
                         fontsize=16, fontweight='bold', pad=20)
            stats_text = (f"Total models: {len(df)}\n"
                          f"Pareto optimal: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
                          f"Zoomed to top performers")
        else:
            ax.set_title('Model 1: Cross-Bit-Width Pareto Front — Weighted Bkg Rej vs LUTs + FF',
                         fontsize=16, fontweight='bold', pad=20)
            stats_text = (f"Total models: {len(df)}\n"
                          f"Pareto optimal: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
                          f"Metric range: {df['primary_metric'].min():.4f} – {df['primary_metric'].max():.4f}")

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        suffix = '_zoomed' if zoomed else '_combined'
        plot_path = os.path.join(
            output_dir,
            f'pareto_front_weighted_bkgrej_vs_luts_plus_ff{suffix}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-bit-width Pareto selection for Model 1: weighted bkg rej vs FF/LUT.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Base directory (default: directory containing this script).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <base_dir>/combined_pareto_ff_lut).",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else \
        os.path.join(base_dir, "combined_pareto_ff_lut")

    runs = [
        RunConfig(name="model1_4bit",
                  path=os.path.join(base_dir, "Model1_quantized_4w0i_pareto_roc_selected")),
        RunConfig(name="model1_6bit",
                  path=os.path.join(base_dir, "Model1_quantized_6w0i_pareto_roc_selected")),
        RunConfig(name="model1_8bit",
                  path=os.path.join(base_dir, "Model1_quantized_8w0i_pareto_roc_selected")),
    ]

    # Load and merge
    all_dfs = []
    for run in runs:
        if not os.path.isdir(run.path):
            print(f"Warning: run directory not found, skipping: {run.path}")
            continue
        print(f"Loading metrics for run: {run.name}  ({run.path})")
        all_dfs.append(load_run_metrics(run))

    if not all_dfs:
        raise RuntimeError("No runs could be loaded; nothing to do.")

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df["luts_plus_ff"] = all_df["luts"] + all_df["registers"]

    print(f"\nTotal models with ROC + resources: {len(all_df)}")
    print("Runs present:", ", ".join(sorted(all_df["run_name"].unique())))
    print(f"Primary metric range: {all_df['primary_metric'].min():.4f} – "
          f"{all_df['primary_metric'].max():.4f}")

    print("\n" + "=" * 80)
    print("PARETO SELECTION")
    print("=" * 80)

    pareto_df = find_pareto_front(all_df,
                                  maximize_cols=["primary_metric"],
                                  minimize_cols=["luts", "registers"])
    print(f"\nPareto optimal models (across all bit-widths): {len(pareto_df)}")

    os.makedirs(output_dir, exist_ok=True)

    # Save CSVs
    combined_csv = os.path.join(output_dir, "combined_roc_resources_detailed.csv")
    all_df.to_csv(combined_csv, index=False)
    print(f"\nSaved combined detailed CSV: {combined_csv}")

    cols_to_keep = [
        "run_name", "trial_id", "global_id", "parameters", "auc",
        "primary_metric", "luts", "registers", "luts_plus_ff",
        "luts_percent", "registers_percent", "dsp", "dsp_percent", "estimated_fmax",
    ]
    existing_cols = [c for c in cols_to_keep if c in pareto_df.columns]
    primary_csv = os.path.join(output_dir, "pareto_ff_lut_primary.csv")
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
            'mean': float(all_df['primary_metric'].mean()),
        },
        'luts_range': {
            'min': int(all_df['luts'].min()),
            'max': int(all_df['luts'].max()),
            'mean': float(all_df['luts'].mean()),
        },
        'luts_plus_ff_range': {
            'min': int(all_df['luts_plus_ff'].min()),
            'max': int(all_df['luts_plus_ff'].max()),
            'mean': float(all_df['luts_plus_ff'].mean()),
        },
        'registers_range': {
            'min': int(all_df['registers'].min()),
            'max': int(all_df['registers'].max()),
            'mean': float(all_df['registers'].mean()),
        },
    }
    summary_json = os.path.join(output_dir, 'pareto_ff_lut_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved analysis summary: {summary_json}")

    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_pareto_luts_vs_metric(all_df, pareto_df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
