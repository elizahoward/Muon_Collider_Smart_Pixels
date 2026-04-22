#!/usr/bin/env python3
"""
Combined Pareto plot: Model 1 (4/6/8-bit) vs Model 2.5 (4/6/8/10-bit)

Outputs:
  1. ..._full.png                      — linear-linear, joint Pareto
  2. ..._zoomed.png                    — linear-linear, zoomed top-left
  3. ..._subfronts_linear.png          — linear x/y + per-family sub-fronts
  4. ..._subfronts_semilogy.png        — x log, y linear + sub-fronts
  5. ..._subfronts_loglog.png          — log-log + sub-fronts
  6. ..._subfronts_complog_xlinear.png — linear x, y = 1−metric on log scale (inverted)
  7. ..._subfronts_complog_xlog.png    — log x,   y = 1−metric on log scale (inverted)

Plots 6 & 7 spread out the y-axis near the top where differences are tiny,
by plotting (1 − weighted_bkg_rejection) on a log scale with the axis inverted
so high performance still sits at the top. Tick labels show the original metric.

Author: Eric
Date: April 2026
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────────
ERIC = os.path.dirname(os.path.abspath(__file__))

CSV_MODEL1  = os.path.join(ERIC, "New_hyperparameter_runs_Model1",
                            "combined_pareto_ff_lut",
                            "combined_roc_resources_detailed.csv")
CSV_MODEL25 = os.path.join(ERIC, "New_hyperparameter_runs_Model2_5",
                            "combined_pareto_ff_lut",
                            "combined_roc_resources_detailed.csv")
OUTPUT_DIR  = os.path.join(ERIC, "combined_model1_model2_5_pareto")

# ── Style ──────────────────────────────────────────────────────────────────────
RUN_COLORS = {
    "model1_4bit":               "#9b59b6",
    "model1_6bit":               "#2ecc71",
    "model1_8bit":               "#e74c3c",
    "eric_model_2_5_4bit":       "#6c3483",
    "model2_5_6bit_new":         "#1a8a4a",
    "eric_model_2_5_8bit":       "#922b21",
    "model2_5_8_bit":            "#922b21",
    "model2_5_10bit":            "#2980b9",
    "eric_new_smaller_2_23_run": "#e67e22",
}

RUN_LABELS = {
    "model1_4bit":               "Model 1  (4-bit)",
    "model1_6bit":               "Model 1  (6-bit)",
    "model1_8bit":               "Model 1  (8-bit)",
    "eric_model_2_5_4bit":       "Model 2.5 (4-bit)",
    "model2_5_6bit_new":         "Model 2.5 (6-bit)",
    "eric_model_2_5_8bit":       "Model 2.5 (8-bit)",
    "model2_5_8_bit":            "Model 2.5 (8-bit)",
    "model2_5_10bit":            "Model 2.5 (10-bit)",
    "eric_new_smaller_2_23_run": "Model 2.5 (smaller)",
}

FAMILY_LINE = {
    "Model 1":   dict(color="#c0392b", lw=1.4, ls="--",  label="Model 1 Pareto front"),
    "Model 2.5": dict(color="#1a5276", lw=1.4, ls="-.",  label="Model 2.5 Pareto front"),
    "Combined":  dict(color="black",   lw=1.8, ls="-",   label="Combined Pareto front"),
}

# Tick positions for the complement-log y-axis (actual metric values shown as labels)
COMP_LOG_TICKS = [0.9, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.30, 0.10, 0.0]


# ── Pareto helpers ─────────────────────────────────────────────────────────────
def is_dominated(point, others, maximize_cols, minimize_cols):
    for _, other in others.iterrows():
        better_in_all, strictly_better = True, False
        for c in maximize_cols:
            if other[c] < point[c]: better_in_all = False; break
            if other[c] > point[c]: strictly_better = True
        if not better_in_all:
            continue
        for c in minimize_cols:
            if other[c] > point[c]: better_in_all = False; break
            if other[c] < point[c]: strictly_better = True
        if better_in_all and strictly_better:
            return True
    return False


def find_pareto_front(df, maximize_cols=("primary_metric",),
                      minimize_cols=("luts_plus_ff",)):
    idx = [i for i, row in df.iterrows()
           if not is_dominated(row, df.drop(index=i), list(maximize_cols), list(minimize_cols))]
    return df.loc[idx].copy().sort_values(
        by=[maximize_cols[0], "luts_plus_ff"], ascending=[False, True])


# ── Y-axis transform helpers ───────────────────────────────────────────────────
def _y(series_or_val, complement):
    """Return 1 - v (clipped to a small positive) when complement=True."""
    if complement:
        return np.clip(1.0 - np.asarray(series_or_val, dtype=float), 1e-4, None)
    return series_or_val


def _setup_complement_log_y(ax, df):
    """
    Configure y-axis for complement-log display:
    - log scale on (1 - metric)
    - axis inverted so high rejection is at top
    - tick labels showing actual metric values
    """
    ax.set_yscale("log")
    ax.invert_yaxis()

    # Set tick positions in complement space, label with actual metric values
    ticks_actual  = [t for t in COMP_LOG_TICKS if t < df["primary_metric"].max() + 0.05]
    ticks_comp    = np.clip(1.0 - np.array(ticks_actual, dtype=float), 1e-4, None)
    ax.set_yticks(ticks_comp)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{1 - v:.2f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Sensible y limits: from just above the best model down to the worst
    y_min = max(1e-4, float(np.min(_y(df["primary_metric"], True))) * 0.5)
    y_max = float(np.max(_y(df["primary_metric"], True))) * 2.0
    ax.set_ylim(y_max, y_min)   # inverted: large complement (bad) at bottom of view


# ── Core drawing helpers ───────────────────────────────────────────────────────
def _draw_scatter(ax, df, pareto_df, x_col, complement=False, annotate=True):
    for run_name in df["run_name"].unique():
        color  = RUN_COLORS.get(run_name, "gray")
        label  = RUN_LABELS.get(run_name, run_name)
        subset = df[df["run_name"] == run_name]
        ax.scatter(subset[x_col], _y(subset["primary_metric"], complement),
                   alpha=0.30, s=25, c=color, edgecolors="none",
                   label=label, zorder=1)

    for run_name in pareto_df["run_name"].unique():
        color = RUN_COLORS.get(run_name, "gray")
        p_sub = pareto_df[pareto_df["run_name"] == run_name]
        ax.scatter(p_sub[x_col], _y(p_sub["primary_metric"], complement),
                   alpha=0.90, s=60, c=color, edgecolors="black",
                   linewidth=1.2, marker="D", zorder=3)

    ps = pareto_df.sort_values(x_col)
    style = {k: v for k, v in FAMILY_LINE["Combined"].items() if k not in ("label", "lw")}
    ax.plot(ps[x_col], _y(ps["primary_metric"], complement),
            **style, alpha=0.25, zorder=2, lw=1.2)

    if annotate:
        for _, row in pareto_df.iterrows():
            short = RUN_LABELS.get(row["run_name"], row["run_name"])
            short = short.replace("Model 1 ", "M1 ").replace("Model 2.5 ", "M2.5 ")
            yval  = float(_y(row["primary_metric"], complement))
            ax.annotate(f"{short}\n{row['trial_id']}",
                        xy=(row[x_col], yval),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=6.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow",
                                  alpha=0.8, edgecolor="black", linewidth=0.8),
                        zorder=4)


def _draw_subfronts(ax, pareto_m1, pareto_m25, pareto_all, x_col, complement=False):
    for pareto, key, alpha in [(pareto_m1,  "Model 1",   0.85),
                                (pareto_m25, "Model 2.5", 0.85),
                                (pareto_all, "Combined",  0.25)]:
        style = FAMILY_LINE[key]
        ps = pareto.sort_values(x_col)
        ax.plot(ps[x_col], _y(ps["primary_metric"], complement),
                color=style["color"], lw=style["lw"], ls=style["ls"],
                alpha=alpha, zorder=5, label=style["label"])


def _add_legend_and_stats(ax, df, pareto_df, complement=False, extra_handles=None):
    handles, labels = ax.get_legend_handles_labels()
    if extra_handles:
        handles += extra_handles
        labels  += [h.get_label() for h in extra_handles]
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    # Legend placement: lower-left works better when y is complement-inverted
    loc = "upper left" if complement else "lower right"
    ax.legend(h2, l2, loc=loc, fontsize=8, framealpha=0.9, ncol=2, columnspacing=0.8)

    stats = (f"Total models: {len(df)}\n"
             f"Pareto optimal: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
             f"Metric range: {df['primary_metric'].min():.4f} – {df['primary_metric'].max():.4f}")
    ax.text(0.02, 0.02 if complement else 0.98, stats,
            transform=ax.transAxes, fontsize=9,
            va="bottom" if complement else "top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))


def _finalize(ax, title, xscale="linear", complement=False):
    tags = []
    if xscale != "linear": tags.append("x-log")
    if complement:          tags.append("y: 1−metric log")
    tag = f" [{', '.join(tags)}]" if tags else ""
    ax.set_title(title + tag, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("LUTs + FF (registers)", fontsize=12, fontweight="bold")
    ylabel = "Weighted Bkg Rejection  (1−metric, log scale)" if complement \
             else "Weighted Background Rejection"
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.28, linestyle="--", which="both")
    if xscale == "log":
        ax.set_xscale("log")


# ── Public plot functions ──────────────────────────────────────────────────────
def make_plot_simple(df, pareto_df, output_dir, zoomed=False, annotate=True):
    x_col = "luts_plus_ff"
    fig, ax = plt.subplots(figsize=(15, 9))
    _draw_scatter(ax, df, pareto_df, x_col, annotate=annotate)
    if zoomed:
        ax.set_xlim(0, 1.5e5); ax.set_ylim(0.6, 1.0)
        suffix, title = "_zoomed", "Model 1 vs Model 2.5 — Pareto Front (Zoomed)"
    else:
        suffix = "_full"
        title  = "Model 1 vs Model 2.5 — Combined Pareto Front\nWeighted Bkg Rejection vs LUTs + FF"
    _finalize(ax, title)
    _add_legend_and_stats(ax, df, pareto_df)
    plt.tight_layout()
    tag  = "" if annotate else "_nolabels"
    path = os.path.join(output_dir, f"pareto_model1_vs_model2_5{suffix}{tag}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); print(f"  ✓ {path}")
    plt.close()


def make_plot_subfronts(df, pareto_df, pareto_m1, pareto_m25, output_dir,
                        xscale="linear", complement=False, annotate=True):
    """
    xscale: 'linear' or 'log'
    complement: if True, plot 1−metric on log scale (inverted) for better y resolution
    annotate: if True, add trial-ID labels on Pareto points
    """
    x_col = "luts_plus_ff"
    fig, ax = plt.subplots(figsize=(15, 9))
    _draw_scatter(ax, df, pareto_df, x_col, complement=complement, annotate=annotate)
    _draw_subfronts(ax, pareto_m1, pareto_m25, pareto_df, x_col, complement=complement)
    if complement:
        _setup_complement_log_y(ax, df)

    title = ("Model 1 vs Model 2.5 — Sub-fronts + Combined Pareto\n"
             "Weighted Bkg Rejection vs LUTs + FF")
    _finalize(ax, title, xscale=xscale, complement=complement)

    extra = [mlines.Line2D([], [], color=FAMILY_LINE[k]["color"],
                           lw=FAMILY_LINE[k]["lw"], ls=FAMILY_LINE[k]["ls"],
                           label=FAMILY_LINE[k]["label"])
             for k in ("Model 1", "Model 2.5", "Combined")]
    _add_legend_and_stats(ax, df, pareto_df, complement=complement, extra_handles=extra)
    plt.tight_layout()

    parts = ["xlog" if xscale == "log" else "xlinear",
             "ycomplog" if complement else "ylinear"]
    tag   = "" if annotate else "_nolabels"
    fname = "pareto_model1_vs_model2_5_subfronts_" + "_".join(parts) + tag + ".png"
    path  = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight"); print(f"  ✓ {path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading CSVs...")
    df1  = pd.read_csv(CSV_MODEL1)
    df25 = pd.read_csv(CSV_MODEL25)
    df   = pd.concat([df1, df25], ignore_index=True)
    for d in (df, df1, df25):
        d["luts_plus_ff"] = d["luts"] + d["registers"]

    print(f"  Model 1 rows:   {len(df1)}")
    print(f"  Model 2.5 rows: {len(df25)}")
    print(f"  Combined total: {len(df)}")
    print(f"  Primary metric range: {df['primary_metric'].min():.4f} – {df['primary_metric'].max():.4f}")

    print("\nComputing Pareto fronts...")
    pareto_all = find_pareto_front(df)
    pareto_m1  = find_pareto_front(df1)
    pareto_m25 = find_pareto_front(df25)
    print(f"  Joint Pareto:     {len(pareto_all)} models")
    print(f"  Model 1 Pareto:   {len(pareto_m1)} models")
    print(f"  Model 2.5 Pareto: {len(pareto_m25)} models")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save CSVs / JSON
    df.to_csv(os.path.join(OUTPUT_DIR, "combined_all_detailed.csv"), index=False)
    cols = ["run_name","trial_id","global_id","parameters","auc","primary_metric",
            "luts","registers","luts_plus_ff","luts_percent","registers_percent",
            "dsp","dsp_percent","estimated_fmax"]
    pareto_all[[c for c in cols if c in pareto_all.columns]].to_csv(
        os.path.join(OUTPUT_DIR, "pareto_primary.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_models": len(df),
            "joint_pareto_models": len(pareto_all),
            "model1_pareto_models": len(pareto_m1),
            "model25_pareto_models": len(pareto_m25),
            "metric_range": {"min": float(df["primary_metric"].min()),
                             "max": float(df["primary_metric"].max())},
            "luts_plus_ff_range": {"min": int(df["luts_plus_ff"].min()),
                                   "max": int(df["luts_plus_ff"].max())},
        }, f, indent=2)

    print("\nGenerating plots...")

    for annotate in (True, False):
        # Simple views
        make_plot_simple(df, pareto_all, OUTPUT_DIR, zoomed=False, annotate=annotate)
        make_plot_simple(df, pareto_all, OUTPUT_DIR, zoomed=True,  annotate=annotate)

        # Sub-fronts: standard scales
        make_plot_subfronts(df, pareto_all, pareto_m1, pareto_m25, OUTPUT_DIR,
                            xscale="linear", complement=False, annotate=annotate)
        make_plot_subfronts(df, pareto_all, pareto_m1, pareto_m25, OUTPUT_DIR,
                            xscale="log",    complement=False, annotate=annotate)

        # Sub-fronts: complement-log y
        make_plot_subfronts(df, pareto_all, pareto_m1, pareto_m25, OUTPUT_DIR,
                            xscale="linear", complement=True, annotate=annotate)
        make_plot_subfronts(df, pareto_all, pareto_m1, pareto_m25, OUTPUT_DIR,
                            xscale="log",    complement=True, annotate=annotate)

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
