#!/usr/bin/env python3
"""
combine_pareto_resources.py

Accepts a list of run folders, auto-detects their layout type, extracts ROC
metrics and HLS resource data, and produces a combined CSV + Pareto plots.

Usage:
    python combine_pareto_resources.py FOLDER [FOLDER ...] [--output OUTPUT_DIR]

Three supported folder types:
  Type 1 (flat):          roc_based_analysis_detailed.csv + hls_outputs/ in root
  Type 2 (split pareto):  roc_based_analysis_detailed.csv + pareto_primary/ subdir
  Type 3 (legacy/vSynth): resource_utilization.csv pre-extracted (ROC metrics unavailable)

Output directory (default: combined_pareto_<timestamp>/) is created next to this script.
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent

COMP_LOG_TICKS = [0.9, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.30, 0.10, 0.0]

OUTPUT_COLUMNS = [
    "trial_id", "parameters", "auc", "primary_metric",
    "bkg_rej_@95%", "bkg_rej_@98%", "bkg_rej_@99%",
    "model_name", "val_accuracy", "h5_file", "output_dir",
    "luts", "luts_used", "luts_available", "luts_percent",
    "registers", "registers_used", "registers_available", "registers_percent",
    "bram", "bram_used", "bram_available", "bram_percent",
    "dsp", "dsp_used", "dsp_available", "dsp_percent",
    "latency_min", "latency_max", "interval_min", "interval_max", "estimated_fmax",
    "run_name", "global_id", "model_family", "bit_width", "luts_plus_ff",
]

PARETO_SUMMARY_COLUMNS = [
    "run_name", "trial_id", "global_id", "parameters", "auc", "primary_metric",
    "bkg_rej_@95%", "bkg_rej_@98%", "bkg_rej_@99%",
    "luts", "registers", "luts_plus_ff", "luts_percent", "registers_percent",
    "dsp", "dsp_percent", "estimated_fmax", "model_family", "bit_width",
]


# ── Detection and Metadata ────────────────────────────────────────────────────

def detect_folder_type(folder: Path) -> str:
    if (folder / "pareto_primary").is_dir():
        return "type2"
    elif (folder / "resource_utilization.csv").exists():
        return "type3"
    elif (folder / "roc_based_analysis_detailed.csv").exists():
        return "type1"
    else:
        return "unknown"


def extract_metadata(folder: Path) -> dict:
    name = os.path.basename(str(folder)).lower()
    # Check model2_5 before model1/model3 to avoid false matches
    if "model2_5" in name or "model2.5" in name:
        family = "Model 2.5"
    elif "model1" in name or "model_1" in name:
        family = "Model 1"
    elif "model3" in name or "model_3" in name:
        family = "Model 3"
    else:
        family = "Unknown"
    m = re.search(r"(\d+)w\d+i", name) or re.search(r"(\d+)bit", name)
    bit_width = int(m.group(1)) if m else None
    return {
        "run_name":     os.path.basename(str(folder)),
        "model_family": family,
        "bit_width":    bit_width,
    }


# ── Trial-ID Normalization ────────────────────────────────────────────────────

def normalize_trial_id(raw_id) -> str:
    s = str(raw_id).strip()
    for prefix in ("model_trial_", "trial_"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
    return str(int(s)).zfill(3)


# ── csynth.rpt Parser ─────────────────────────────────────────────────────────

def _cell_val(s: str, as_float: bool = False):
    s = s.strip()
    if s in ("-", ""):
        return None
    if s.startswith("~"):
        s = s[1:]
    try:
        return float(s) if as_float else int(s)
    except ValueError:
        return None


# Applied to the Summary subsection only (before "+ Detail:")
_RE_TOTAL = re.compile(
    r"^\|Total\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|",
    re.MULTILINE,
)
_RE_AVAIL = re.compile(
    r"^\|Available\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|",
    re.MULTILINE,
)
_RE_UTIL_PCT = re.compile(
    r"^\|Utilization \(%\)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|\s*(~?\d+)\s*\|",
    re.MULTILINE,
)
_RE_TIMING = re.compile(
    r"\|\s*ap_clk\s*\|\s*[\d.]+\s*ns\s*\|\s*([\d.]+)\s*ns\s*\|"
)
_RE_LATENCY = re.compile(
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*[\d.]+\s*\w+\s*\|\s*[\d.]+\s*\w+\s*\|"
    r"\s*(\d+)\s*\|\s*(\d+)\s*\|",
)


def parse_csynth_rpt(rpt_path: Path) -> dict:
    """Parse a Vitis HLS csynth.rpt and return resource/timing dict."""
    empty = {k: None for k in [
        "luts", "luts_used", "luts_available", "luts_percent",
        "registers", "registers_used", "registers_available", "registers_percent",
        "bram", "bram_used", "bram_available", "bram_percent",
        "dsp", "dsp_used", "dsp_available", "dsp_percent",
        "latency_min", "latency_max", "interval_min", "interval_max", "estimated_fmax",
    ]}
    try:
        text = rpt_path.read_text(errors="replace")
    except Exception as e:
        print(f"    WARNING: cannot read {rpt_path}: {e}")
        return empty

    result = dict(empty)
    try:
        # Timing
        m = _RE_TIMING.search(text)
        if m:
            try:
                result["estimated_fmax"] = round(1000.0 / float(m.group(1)), 4)
            except ZeroDivisionError:
                pass

        # Latency — extract the full Summary section (before + Detail:)
        lat_section = re.search(
            r"\+ Latency:.*?\* Summary:(.*?)(?=\+ Detail:)",
            text, re.DOTALL,
        )
        if lat_section:
            lat_m = _RE_LATENCY.search(lat_section.group(1))
            if lat_m:
                result["latency_min"]  = int(lat_m.group(1))
                result["latency_max"]  = int(lat_m.group(2))
                result["interval_min"] = int(lat_m.group(3))
                result["interval_max"] = int(lat_m.group(4))

        # Utilization — extract Summary block only (before "+ Detail:")
        util_m = re.search(
            r"== Utilization Estimates.*?\* Summary:(.*?)(?=\+ Detail:)",
            text, re.DOTALL,
        )
        if util_m:
            sb = util_m.group(1)
            # Column order in table: BRAM_18K | DSP | FF | LUT | URAM
            t = _RE_TOTAL.search(sb)
            if t:
                result["bram_used"]      = _cell_val(t.group(1))
                result["dsp_used"]       = _cell_val(t.group(2))
                result["registers_used"] = _cell_val(t.group(3))
                result["luts_used"]      = _cell_val(t.group(4))
            a = _RE_AVAIL.search(sb)
            if a:
                result["bram_available"]      = _cell_val(a.group(1))
                result["dsp_available"]       = _cell_val(a.group(2))
                result["registers_available"] = _cell_val(a.group(3))
                result["luts_available"]      = _cell_val(a.group(4))
            u = _RE_UTIL_PCT.search(sb)
            if u:
                result["bram_percent"]      = _cell_val(u.group(1), as_float=True)
                result["dsp_percent"]       = _cell_val(u.group(2), as_float=True)
                result["registers_percent"] = _cell_val(u.group(3), as_float=True)
                result["luts_percent"]      = _cell_val(u.group(4), as_float=True)

        # Schema aliases
        result["luts"]      = result["luts_used"]
        result["registers"] = result["registers_used"]
        result["bram"]      = result["bram_used"]
        result["dsp"]       = result["dsp_used"]

    except Exception as e:
        print(f"    WARNING: parse error in {rpt_path}: {e}")

    return result


# ── Resource Loading ──────────────────────────────────────────────────────────

def find_csynth_rpts(hls_outputs_dir: Path) -> dict:
    """Return {normalized_trial_id: Path} for all csynth.rpt files."""
    pattern = str(
        hls_outputs_dir / "hls_model_trial_*" /
        "myproject_prj" / "solution1" / "syn" / "report" / "myproject_csynth.rpt"
    )
    result = {}
    for rpt in glob.glob(pattern):
        rpt_path = Path(rpt)
        trial_dir = rpt_path.parts[-6]  # hls_model_trial_NNN
        try:
            tid = normalize_trial_id(trial_dir.replace("hls_model_trial_", ""))
            result[tid] = rpt_path
        except (ValueError, IndexError):
            print(f"    WARNING: could not parse trial_id from {trial_dir}")
    return result


def load_resources_from_csynth(hls_outputs_dir: Path) -> pd.DataFrame:
    """Parse all csynth.rpt files under hls_outputs_dir into a DataFrame."""
    rpts = find_csynth_rpts(hls_outputs_dir)
    if not rpts:
        print(f"    WARNING: no csynth.rpt files found under {hls_outputs_dir}")
        return pd.DataFrame()
    rows = []
    for tid, rpt_path in sorted(rpts.items()):
        data = parse_csynth_rpt(rpt_path)
        data["trial_id"] = tid
        rows.append(data)
    return pd.DataFrame(rows)


def load_resources_from_csv(resource_csv: Path) -> pd.DataFrame:
    """Load pre-extracted resource_utilization.csv."""
    df = pd.read_csv(resource_csv)
    if "model_name" in df.columns:
        df["trial_id"] = df["model_name"].apply(
            lambda x: normalize_trial_id(str(x).replace("model_trial_", "").strip())
        )
    return df


# ── ROC CSV Loading ───────────────────────────────────────────────────────────

def load_roc_csv(roc_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(roc_csv)
    df["trial_id"] = df["trial_id"].apply(normalize_trial_id)
    return df


# ── Per-Folder Processing ─────────────────────────────────────────────────────

def process_folder(folder: Path) -> pd.DataFrame:
    folder = Path(folder).resolve()
    ftype  = detect_folder_type(folder)
    meta   = extract_metadata(folder)

    print(f"  [{ftype}] {folder.name}")
    if ftype == "unknown":
        print(f"    WARNING: unrecognized folder layout, skipping.")
        return pd.DataFrame()

    roc_df = pd.DataFrame()
    res_df = pd.DataFrame()

    if ftype in ("type1", "type2"):
        roc_csv = folder / "roc_based_analysis_detailed.csv"
        if roc_csv.exists():
            roc_df = load_roc_csv(roc_csv)
        else:
            print(f"    WARNING: roc_based_analysis_detailed.csv not found")

        hls_dir = (folder / "hls_outputs" if ftype == "type1"
                   else folder / "pareto_primary" / "hls_outputs")
        if hls_dir.exists():
            res_df = load_resources_from_csynth(hls_dir)
        else:
            print(f"    WARNING: hls_outputs not found at {hls_dir}")

        if roc_df.empty or res_df.empty:
            print(f"    WARNING: missing ROC or resource data, skipping.")
            return pd.DataFrame()

        merged = pd.merge(roc_df, res_df, on="trial_id", how="inner")
        if merged.empty:
            print(f"    WARNING: no trial_ids matched between ROC and resource data")
            return pd.DataFrame()
        print(f"    Loaded {len(merged)} models "
              f"(inner join of {len(roc_df)} ROC × {len(res_df)} synthesized)")

    else:  # type3
        res_df = load_resources_from_csv(folder / "resource_utilization.csv")
        if res_df.empty:
            print(f"    WARNING: resource_utilization.csv is empty")
            return pd.DataFrame()
        for col in ["parameters", "auc", "primary_metric",
                    "bkg_rej_@95%", "bkg_rej_@98%", "bkg_rej_@99%"]:
            if col not in res_df.columns:
                res_df[col] = float("nan")
        merged = res_df
        print(f"    Loaded {len(merged)} models (Type 3 — ROC metrics unavailable)")

    merged = merged.copy()
    merged["run_name"]     = meta["run_name"]
    merged["model_family"] = meta["model_family"]
    merged["bit_width"]    = meta["bit_width"]
    merged["global_id"]    = meta["run_name"] + "_" + merged["trial_id"].astype(str)
    merged["luts_plus_ff"] = (
        pd.to_numeric(merged.get("luts"), errors="coerce") +
        pd.to_numeric(merged.get("registers"), errors="coerce")
    )

    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = float("nan")

    return merged[OUTPUT_COLUMNS]


# ── Combined DataFrame ────────────────────────────────────────────────────────

def build_combined_df(folders: list) -> pd.DataFrame:
    dfs = []
    for folder in folders:
        df = process_folder(Path(folder))
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise RuntimeError("No data could be loaded from any supplied folder.")
    combined = pd.concat(dfs, ignore_index=True)
    combined["luts_plus_ff"] = (
        pd.to_numeric(combined["luts"], errors="coerce") +
        pd.to_numeric(combined["registers"], errors="coerce")
    )
    return combined


# ── Pareto Computation (adapted from pareto_model1_vs_model2_5.py) ────────────

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
    valid = df.dropna(subset=list(maximize_cols) + list(minimize_cols))
    n_dropped = len(df) - len(valid)
    if n_dropped > 0:
        print(f"    (Pareto: skipped {n_dropped} rows with NaN objective columns)")
    if valid.empty:
        return pd.DataFrame(columns=df.columns)
    idx = [i for i, row in valid.iterrows()
           if not is_dominated(row, valid.drop(index=i),
                               list(maximize_cols), list(minimize_cols))]
    return valid.loc[idx].copy().sort_values(
        by=[maximize_cols[0], "luts_plus_ff"], ascending=[False, True])


# ── Style Maps ────────────────────────────────────────────────────────────────

def build_style_maps(df: pd.DataFrame):
    unique_runs = sorted(df["run_name"].unique())
    tab10 = cm.get_cmap("tab10")
    run_colors = {
        name: mcolors.to_hex(tab10(i % 10))
        for i, name in enumerate(unique_runs)
    }

    run_labels = {}
    for name in unique_runs:
        row = df[df["run_name"] == name].iloc[0]
        bw = row.get("bit_width")
        if bw is not None and not (isinstance(bw, float) and np.isnan(bw)):
            run_labels[name] = f"{row['model_family']} ({int(bw)}-bit)"
        else:
            run_labels[name] = name

    unique_families = sorted(df["model_family"].unique())
    ls_cycle = ["--", "-.", ":", (0, (3, 1, 1, 1))]
    set1 = cm.get_cmap("Set1")
    family_line = {}
    for i, family in enumerate(unique_families):
        family_line[family] = {
            "color": mcolors.to_hex(set1(i % 9)),
            "lw":    1.4,
            "ls":    ls_cycle[i % len(ls_cycle)],
            "label": f"{family} Pareto front",
        }
    family_line["Combined"] = {
        "color": "black", "lw": 1.8, "ls": "-", "label": "Combined Pareto front",
    }
    return run_colors, run_labels, family_line


# ── Y-axis Helpers (from pareto_model1_vs_model2_5.py) ───────────────────────

def _y(series_or_val, complement):
    if complement:
        return np.clip(1.0 - np.asarray(series_or_val, dtype=float), 1e-4, None)
    return series_or_val


def _setup_complement_log_y(ax, df):
    valid = df["primary_metric"].dropna()
    if valid.empty:
        return
    ax.set_yscale("log")
    ax.invert_yaxis()
    ticks_actual = [t for t in COMP_LOG_TICKS if t < valid.max() + 0.05]
    ticks_comp   = np.clip(1.0 - np.array(ticks_actual, dtype=float), 1e-4, None)
    ax.set_yticks(ticks_comp)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{1 - v:.2f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    y_min = max(1e-4, float(np.min(_y(valid, True))) * 0.5)
    y_max = float(np.max(_y(valid, True))) * 2.0
    ax.set_ylim(y_max, y_min)


# ── Core Drawing Helpers ──────────────────────────────────────────────────────

def _draw_scatter_dynamic(ax, df, pareto_df, x_col, run_colors, run_labels,
                           family_line, complement=False, annotate=True):
    plot_df = df.dropna(subset=[x_col, "primary_metric"])
    for run_name in plot_df["run_name"].unique():
        color  = run_colors.get(run_name, "gray")
        label  = run_labels.get(run_name, run_name)
        subset = plot_df[plot_df["run_name"] == run_name]
        ax.scatter(subset[x_col], _y(subset["primary_metric"], complement),
                   alpha=0.30, s=25, c=color, edgecolors="none",
                   label=label, zorder=1)

    for run_name in pareto_df["run_name"].unique():
        color = run_colors.get(run_name, "gray")
        p_sub = pareto_df[pareto_df["run_name"] == run_name]
        ax.scatter(p_sub[x_col], _y(p_sub["primary_metric"], complement),
                   alpha=0.90, s=60, c=color, edgecolors="black",
                   linewidth=1.2, marker="D", zorder=3)

    if not pareto_df.empty:
        ps    = pareto_df.sort_values(x_col)
        style = {k: v for k, v in family_line["Combined"].items() if k not in ("label", "lw")}
        ax.plot(ps[x_col], _y(ps["primary_metric"], complement),
                **style, alpha=0.25, zorder=2, lw=1.2)

    if annotate:
        for _, row in pareto_df.iterrows():
            short = run_labels.get(row["run_name"], row["run_name"])
            short = (short.replace("Model 2.5 ", "M2.5 ")
                         .replace("Model 1 ", "M1 ")
                         .replace("Model 3 ", "M3 "))
            yval = float(_y(row["primary_metric"], complement))
            ax.annotate(
                f"{short}\n{row['trial_id']}",
                xy=(row[x_col], yval),
                xytext=(7, 7), textcoords="offset points",
                fontsize=6.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow",
                          alpha=0.8, edgecolor="black", linewidth=0.8),
                zorder=4,
            )


def _draw_subfronts_dynamic(ax, family_paretos, pareto_all, x_col,
                             family_line, complement=False):
    for family, pareto in family_paretos.items():
        if pareto.empty:
            continue
        style = family_line.get(family, family_line["Combined"])
        ps = pareto.sort_values(x_col)
        ax.plot(ps[x_col], _y(ps["primary_metric"], complement),
                color=style["color"], lw=style["lw"], ls=style["ls"],
                alpha=0.85, zorder=5, label=style["label"])
    if not pareto_all.empty:
        comb = family_line["Combined"]
        ps   = pareto_all.sort_values(x_col)
        ax.plot(ps[x_col], _y(ps["primary_metric"], complement),
                color=comb["color"], lw=comb["lw"], ls=comb["ls"],
                alpha=0.25, zorder=5, label=comb["label"])


def _add_legend_and_stats_dynamic(ax, df, pareto_df, run_colors, run_labels,
                                   family_line, complement=False, extra_handles=None):
    handles, labels = ax.get_legend_handles_labels()
    if extra_handles:
        handles += extra_handles
        labels  += [h.get_label() for h in extra_handles]
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    loc = "upper left" if complement else "lower right"
    ax.legend(h2, l2, loc=loc, fontsize=8, framealpha=0.9, ncol=2, columnspacing=0.8)

    valid_metric = df["primary_metric"].dropna()
    n_total  = len(df)
    n_pareto = len(pareto_df)
    pct = 100 * n_pareto / n_total if n_total else 0
    metric_str = (f"Metric range: {valid_metric.min():.4f} – {valid_metric.max():.4f}"
                  if not valid_metric.empty else "Metric range: N/A")
    stats = (f"Total models: {n_total}\n"
             f"Pareto optimal: {n_pareto} ({pct:.1f}%)\n"
             f"{metric_str}")
    ax.text(0.02, 0.02 if complement else 0.98, stats,
            transform=ax.transAxes, fontsize=9,
            va="bottom" if complement else "top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))


def _finalize_dynamic(ax, title, xscale="linear", yscale="linear", complement=False):
    tags = []
    if xscale != "linear":            tags.append("x-log")
    if yscale == "log" and not complement: tags.append("y-log")
    if complement:                    tags.append("y: 1−metric log")
    tag = f" [{', '.join(tags)}]" if tags else ""
    ax.set_title(title + tag, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("LUTs + FF (registers)", fontsize=12, fontweight="bold")
    ylabel = ("Weighted Bkg Rejection  (1−metric, log scale)" if complement
              else "Weighted Background Rejection")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.28, linestyle="--", which="both")
    if xscale == "log":
        ax.set_xscale("log")
    if yscale == "log" and not complement:
        ax.set_yscale("log")


# ── Public Plot Functions ─────────────────────────────────────────────────────

def make_combined_plot_simple(df, pareto_df, output_dir, run_colors, run_labels,
                               family_line, zoomed=False, annotate=True):
    x_col = "luts_plus_ff"
    fig, ax = plt.subplots(figsize=(15, 9))
    _draw_scatter_dynamic(ax, df, pareto_df, x_col, run_colors, run_labels,
                          family_line, annotate=annotate)
    if zoomed:
        ax.set_xlim(0, 1.5e5); ax.set_ylim(0.6, 1.0)
        suffix = "_zoomed"
        title  = "Combined Pareto Front (Zoomed)\nWeighted Bkg Rejection vs LUTs + FF"
    else:
        suffix = "_full"
        title  = "Combined Pareto Front\nWeighted Bkg Rejection vs LUTs + FF"
    _finalize_dynamic(ax, title)
    _add_legend_and_stats_dynamic(ax, df, pareto_df, run_colors, run_labels, family_line)
    plt.tight_layout()
    tag  = "" if annotate else "_nolabels"
    path = os.path.join(output_dir, f"combined_pareto{suffix}{tag}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  ✓ {os.path.basename(path)}")
    plt.close()


def make_combined_plot_subfronts(df, pareto_df, family_paretos, output_dir,
                                  run_colors, run_labels, family_line,
                                  xscale="linear", yscale="linear",
                                  complement=False, annotate=True):
    x_col = "luts_plus_ff"
    fig, ax = plt.subplots(figsize=(15, 9))
    _draw_scatter_dynamic(ax, df, pareto_df, x_col, run_colors, run_labels,
                          family_line, complement=complement, annotate=annotate)
    _draw_subfronts_dynamic(ax, family_paretos, pareto_df, x_col,
                            family_line, complement=complement)
    if complement:
        _setup_complement_log_y(ax, df.dropna(subset=["primary_metric"]))

    title = "Combined Pareto — Sub-fronts\nWeighted Bkg Rejection vs LUTs + FF"
    _finalize_dynamic(ax, title, xscale=xscale, yscale=yscale, complement=complement)

    extra = [
        mlines.Line2D([], [], color=family_line[k]["color"],
                      lw=family_line[k]["lw"], ls=family_line[k]["ls"],
                      label=family_line[k]["label"])
        for k in list(family_paretos.keys()) + ["Combined"]
        if k in family_line
    ]
    _add_legend_and_stats_dynamic(ax, df, pareto_df, run_colors, run_labels,
                                   family_line, complement=complement,
                                   extra_handles=extra)
    plt.tight_layout()

    parts = ["xlog" if xscale == "log" else "xlinear"]
    if complement:
        parts.append("ycomplog")
    elif yscale == "log":
        parts.append("ylog")
    else:
        parts.append("ylinear")
    tag  = "" if annotate else "_nolabels"
    fname = "combined_pareto_subfronts_" + "_".join(parts) + tag + ".png"
    path  = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  ✓ {os.path.basename(path)}")
    plt.close()


def generate_all_plots(df, pareto_df, family_paretos, output_dir):
    run_colors, run_labels, family_line = build_style_maps(df)
    for annotate in (True, False):
        make_combined_plot_simple(df, pareto_df, output_dir,
                                   run_colors, run_labels, family_line,
                                   zoomed=False, annotate=annotate)
        make_combined_plot_simple(df, pareto_df, output_dir,
                                   run_colors, run_labels, family_line,
                                   zoomed=True, annotate=annotate)
        make_combined_plot_subfronts(df, pareto_df, family_paretos, output_dir,
                                     run_colors, run_labels, family_line,
                                     xscale="linear", yscale="linear",
                                     complement=False, annotate=annotate)
        make_combined_plot_subfronts(df, pareto_df, family_paretos, output_dir,
                                     run_colors, run_labels, family_line,
                                     xscale="log", yscale="linear",
                                     complement=False, annotate=annotate)
        make_combined_plot_subfronts(df, pareto_df, family_paretos, output_dir,
                                     run_colors, run_labels, family_line,
                                     xscale="log", yscale="log",
                                     complement=False, annotate=annotate)
        make_combined_plot_subfronts(df, pareto_df, family_paretos, output_dir,
                                     run_colors, run_labels, family_line,
                                     xscale="linear", yscale="linear",
                                     complement=True, annotate=annotate)
        make_combined_plot_subfronts(df, pareto_df, family_paretos, output_dir,
                                     run_colors, run_labels, family_line,
                                     xscale="log", yscale="linear",
                                     complement=True, annotate=annotate)


# ── Save Outputs ──────────────────────────────────────────────────────────────

def save_outputs(df, pareto_df, family_paretos, output_dir, input_folders, folder_types):
    df.to_csv(os.path.join(output_dir, "combined_roc_resources_detailed.csv"), index=False)

    cols = [c for c in PARETO_SUMMARY_COLUMNS if c in pareto_df.columns]
    pareto_df[cols].to_csv(os.path.join(output_dir, "pareto_primary.csv"), index=False)

    valid_metric = df["primary_metric"].dropna()
    valid_luts   = df["luts_plus_ff"].dropna()
    summary = {
        "timestamp":                datetime.now().isoformat(),
        "input_folders":            [str(f) for f in input_folders],
        "folder_types":             {str(k): v for k, v in folder_types.items()},
        "total_models":             len(df),
        "joint_pareto_models":      len(pareto_df),
        "per_family_pareto_models": {fam: len(fdf) for fam, fdf in family_paretos.items()},
        "metric_range": {
            "min": float(valid_metric.min()) if not valid_metric.empty else None,
            "max": float(valid_metric.max()) if not valid_metric.empty else None,
        },
        "luts_plus_ff_range": {
            "min": int(valid_luts.min()) if not valid_luts.empty else None,
            "max": int(valid_luts.max()) if not valid_luts.empty else None,
        },
        "runs_included": sorted(df["run_name"].unique().tolist()),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combine pareto-selected HLS result folders into one CSV + Pareto plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folders", nargs="+", metavar="FOLDER",
                        help="One or more result folder paths")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: combined_pareto_<timestamp>/ next to script)")
    args = parser.parse_args()

    # Resolve and validate folder paths
    folder_paths = []
    for f in args.folders:
        p = Path(f)
        if not p.is_dir():
            for base in (Path.cwd(), SCRIPT_DIR):
                candidate = base / f
                if candidate.is_dir():
                    p = candidate
                    break
        if not p.is_dir():
            print(f"ERROR: folder does not exist: {f}")
            sys.exit(1)
        folder_paths.append(p.resolve())

    # Determine output directory
    if args.output is None:
        out_name   = f"combined_pareto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = SCRIPT_DIR / out_name
    else:
        out_path   = Path(args.output)
        output_dir = out_path if out_path.is_absolute() else SCRIPT_DIR / out_path
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Combining {len(folder_paths)} folder(s)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    folder_types = {p: detect_folder_type(p) for p in folder_paths}
    for p, t in folder_types.items():
        print(f"  {t}: {p.name}")
    print()

    print("Loading data...")
    try:
        df = build_combined_df(folder_paths)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"\nTotal models loaded: {len(df)}")
    n_nan = df[["primary_metric", "luts_plus_ff"]].isna().any(axis=1).sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan} rows have NaN primary_metric or luts_plus_ff "
              f"(Type 3 folders) — excluded from Pareto computation, shown as scatter only")

    print("\nComputing Pareto fronts...")
    pareto_all     = find_pareto_front(df)
    family_paretos = {}
    for family in sorted(df["model_family"].unique()):
        fdf = df[df["model_family"] == family]
        fp  = find_pareto_front(fdf)
        family_paretos[family] = fp
        print(f"  {family} Pareto: {len(fp)} models")
    print(f"  Joint Pareto:  {len(pareto_all)} models")

    print("\nSaving CSV and JSON...")
    save_outputs(df, pareto_all, family_paretos, output_dir,
                 folder_paths, {str(k): v for k, v in folder_types.items()})
    print(f"  ✓ combined_roc_resources_detailed.csv ({len(df)} rows)")
    print(f"  ✓ pareto_primary.csv ({len(pareto_all)} rows)")
    print(f"  ✓ summary.json")

    print("\nGenerating plots (14 files)...")
    generate_all_plots(df, pareto_all, family_paretos, output_dir)

    print(f"\n{'='*60}")
    print(f"Done. All outputs in: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
