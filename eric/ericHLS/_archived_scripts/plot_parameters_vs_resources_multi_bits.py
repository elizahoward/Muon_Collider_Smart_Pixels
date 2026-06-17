#!/usr/bin/env python3
"""
Plot Model Parameters vs Hardware Resources (FF + LUT) for multiple bit-widths
on the same graph.

This script scans the New_hyperparameter_runs directory for bit-width specific
subdirectories (e.g., eric_model_2_5_4bit, model2_5_6bit_new, model2_5_8_bit,
model2_5_10bit), loads their resource and Pareto CSVs, fits a separate linear
regression (OLS) for each bit-width, and overlays all scatters and regression
lines on a single plot so that the slopes can be directly compared.

Usage:
    python plot_parameters_vs_resources_multi_bits.py \
        --runs_dir ../New_hyperparameter_runs

You can also explicitly specify which subdirectories (bit-widths) to include.

Author: Eric (multi-bit overlay helper)
Date: March 2026
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    from scipy import stats
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Error: matplotlib and scipy are required")
    print("Install with: pip install matplotlib scipy")
    sys.exit(1)


def infer_bit_width_from_name(name: str) -> str:
    """
    Try to infer the bit-width from a directory name.
    
    Examples:
        'eric_model_2_5_4bit'  -> '4'
        'model2_5_6bit_new'    -> '6'
        'model2_5_8_bit'       -> '8'
        'model2_5_10bit'       -> '10'
    """
    import re

    # Look for patterns like '4bit', '10bit'
    m = re.search(r'(\d+)\s*bit', name.replace("_", " "))
    if m:
        return m.group(1)

    m = re.search(r'(\d+)bit', name)
    if m:
        return m.group(1)

    return name


def load_and_merge_data(resource_csv, pareto_csv):
    """
    Load resource utilization and parameter data, then merge them.

    This mirrors the logic from plot_parameters_vs_resources.py.
    """
    resource_df = pd.read_csv(resource_csv)
    resource_df["trial_id"] = resource_df["model_name"].str.extract(
        r"model_trial_(\d+)"
    )[0]

    param_df = pd.read_csv(pareto_csv)

    resource_df["trial_id"] = resource_df["trial_id"].astype(str)
    param_df["trial_id"] = param_df["trial_id"].astype(str).str.zfill(3)

    merged_df = pd.merge(
        resource_df,
        param_df[["trial_id", "parameters"]],
        on="trial_id",
        how="inner",
    )

    merged_df["total_resources"] = (
        merged_df["luts_used"] + merged_df["registers_used"]
    )

    return merged_df


def fit_linear_regression(x, y):
    """
    Fit OLS linear regression and return statistics.
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
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
    }


def collect_bit_width_runs(runs_dir, subdirs=None):
    """
    Collect (bit_label, path) pairs for runs directories that contain the
    expected CSV files.
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    candidates = []

    if subdirs:
        # Only use the user-specified subdirectories
        for sd in subdirs:
            p = runs_path / sd
            if p.is_dir():
                candidates.append(p)
    else:
        # Auto-discover: any subdir that has resource_utilization.csv and
        # pareto_optimal_models_roc_combined.csv
        for p in runs_path.iterdir():
            if not p.is_dir():
                continue
            res = p / "resource_utilization.csv"
            par = p / "pareto_optimal_models_roc_combined.csv"
            if res.is_file() and par.is_file():
                candidates.append(p)

    runs = []
    for p in sorted(candidates):
        res = p / "resource_utilization.csv"
        par = p / "pareto_optimal_models_roc_combined.csv"
        if not res.is_file() or not par.is_file():
            continue
        bit_label = infer_bit_width_from_name(p.name)
        runs.append(
            {
                "bit": bit_label,
                "dir": p,
                "resource_csv": res,
                "pareto_csv": par,
            }
        )

    return runs


def plot_multi_bit(runs_info, output_path, slopes_csv_path=None):
    """
    Create a single plot with scatter + regression line for each bit-width.
    Also optionally save a CSV summarizing slopes and fit metrics.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(runs_info)))  # distinct colors

    slopes_summary = []

    for color, info in zip(colors, runs_info):
        bit = info["bit"]
        df = info["df"]
        stats_dict = info["regression"]

        # Scatter
        ax.scatter(
            df["parameters"],
            df["total_resources"],
            alpha=0.6,
            s=60,
            color=color,
            edgecolors="black",
            linewidth=0.7,
            label=f"{bit}-bit data",
        )

        # Regression line over the span of that bit-width's parameters
        x_line = np.array([df["parameters"].min(), df["parameters"].max()])
        y_line = stats_dict["slope"] * x_line + stats_dict["intercept"]
        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=2.0,
            color=color,
            alpha=0.9,
            label=f"{bit}-bit fit (slope={stats_dict['slope']:.2f})",
        )

        slopes_summary.append(
            {
                "bit_width": bit,
                "slope": stats_dict["slope"],
                "intercept": stats_dict["intercept"],
                "r_squared": stats_dict["r_squared"],
                "r_value": stats_dict["r_value"],
                "p_value": stats_dict["p_value"],
                "std_err": stats_dict["std_err"],
                "n_models": len(df),
            }
        )

    ax.set_xlabel("Number of Parameters", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Hardware Resources (FF + LUT)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Model Parameters vs Hardware Resources\n"
        "Multi-bit Comparison with Linear Regression (OLS)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.grid(True, alpha=0.3, linestyle="--", zorder=1)

    # Place legend outside to reduce clutter
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)

    # Small summary table of slopes vs bit-width inside the plot
    slopes_df = pd.DataFrame(slopes_summary).sort_values("bit_width")
    table_text = "Slopes by Bit-width\n"
    table_text += "──────────────────\n"
    for _, row in slopes_df.iterrows():
        table_text += (
            f"{row['bit_width']}-bit: slope={row['slope']:.2f}, "
            f"R²={row['r_squared']:.3f}, n={int(row['n_models'])}\n"
        )

    ax.text(
        0.02,
        0.98,
        table_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Multi-bit plot saved to: {output_path}")
    plt.close()

    if slopes_csv_path is not None:
        slopes_df.to_csv(slopes_csv_path, index=False)
        print(f"✓ Slopes summary saved to: {slopes_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot model parameters vs hardware resources (FF+LUT) for multiple "
            "bit-width runs on a single graph."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover all bit-width runs in the default directory
  python plot_parameters_vs_resources_multi_bits.py \\
      --runs_dir ../New_hyperparameter_runs

  # Specify a subset of subdirectories (bit-widths)
  python plot_parameters_vs_resources_multi_bits.py \\
      --runs_dir ../New_hyperparameter_runs \\
      --subdirs eric_model_2_5_4bit model2_5_6bit_new model2_5_8_bit model2_5_10bit
        """,
    )

    default_runs_dir = (
        Path(__file__).resolve().parent.parent / "New_hyperparameter_runs"
    )

    parser.add_argument(
        "--runs_dir",
        type=str,
        default=str(default_runs_dir),
        help=f"Directory containing bit-width run subdirectories "
        f"(default: {default_runs_dir})",
    )

    parser.add_argument(
        "--subdirs",
        type=str,
        nargs="*",
        help="Optional list of subdirectory names to include (default: auto-discover)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output plot path "
            "(default: parameters_vs_resources_multi_bits.png in runs_dir)"
        ),
    )

    parser.add_argument(
        "--slopes_csv",
        type=str,
        default=None,
        help=(
            "Optional output CSV path to store bit-width slopes and fit "
            "metrics (default: parameters_vs_resources_multi_bits_slopes.csv "
            "in runs_dir)"
        ),
    )

    args = parser.parse_args()

    runs_dir = args.runs_dir

    print("=" * 80)
    print("MULTI-BIT PARAMETERS VS HARDWARE RESOURCES ANALYSIS")
    print("=" * 80)
    print(f"\nRuns directory: {runs_dir}")
    if args.subdirs:
        print(f"Subdirectories (user-specified): {', '.join(args.subdirs)}")
    else:
        print("Subdirectories: auto-discovering based on CSV presence")
    print("-" * 80)

    # Determine output paths
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            runs_dir, "parameters_vs_resources_multi_bits.png"
        )

    if args.slopes_csv:
        slopes_csv_path = args.slopes_csv
    else:
        slopes_csv_path = os.path.join(
            runs_dir, "parameters_vs_resources_multi_bits_slopes.csv"
        )

    # Collect runs
    runs_info = collect_bit_width_runs(runs_dir, args.subdirs)
    if not runs_info:
        print("Error: No valid bit-width runs found with required CSV files.")
        sys.exit(1)

    print("\nFound the following runs:")
    for info in runs_info:
        print(
            f"  - {info['bit']}-bit: {info['dir']} "
            f"(resource_utilization.csv + pareto_optimal_models_roc_combined.csv)"
        )

    # Load, merge, and fit regression for each run
    for info in runs_info:
        print(f"\nProcessing {info['bit']}-bit run in {info['dir']}...")
        df = load_and_merge_data(info["resource_csv"], info["pareto_csv"])
        print(f"  ✓ Merged {len(df)} models")
        regression_stats = fit_linear_regression(
            df["parameters"].values, df["total_resources"].values
        )
        print(
            f"  ✓ Fit slope={regression_stats['slope']:.4f}, "
            f"R²={regression_stats['r_squared']:.4f}"
        )
        info["df"] = df
        info["regression"] = regression_stats

    # Plot and save summary
    print("\nGenerating multi-bit plot...")
    plot_multi_bit(runs_info, output_path, slopes_csv_path=slopes_csv_path)

    print("\n" + "=" * 80)
    print("MULTI-BIT ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - Plot:         {output_path}")
    print(f"  - Slopes CSV:   {slopes_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

