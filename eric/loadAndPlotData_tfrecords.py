"""
Reproduces the loadAndPlotData.ipynb analysis on the TFRecord dataset used by
timing_cut_analysis.py.

The notebook used parquet/pickle files with a DataFrame containing columns like
'adjusted_hit_time', 'source', etc. This script reads the same features from
TFRecords and produces equivalent plots:

  1. Timing histograms (signal vs BIB)
  2. Sweep of one-sided cut t < T  →  fracSig, fracBib, sigKeptFrac, bibKeptFrac
  3. ROC curve – ML style  (fpr vs tpr)
  4. ROC curve – physics style  (backRej vs sigEffic)
  5. Operating-point report at selected cut values

The TFRecord dataset has no MM/MP sub-split, so BIB is treated as a single class.

Usage
-----
    python loadAndPlotData_tfrecords.py
    python loadAndPlotData_tfrecords.py --data-folder /path/to/TF_Records/... --split val
    python loadAndPlotData_tfrecords.py --feature adjusted_hit_time --split both

Author: Eric (adapted from Daniel Abadjiev's notebook)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_FOLDER = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData_normalized"
)

# Feature names available in the TFRecords
AVAILABLE_FEATURES = [
    "adjusted_hit_time_30ps_gaussian",
    "adjusted_hit_time",
]

# Mirror the notebook's hand-picked operating-point cuts (ns)
NOTEBOOK_CUTS = [5.0, 0.15]  # upper bound of one-sided window
NOTEBOOK_CUT_LOWER = {"5.0": -0.5, "0.15": -0.09}  # lower bound used in the notebook

TARGET_EFFICIENCIES = [0.95, 0.98, 0.99]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce loadAndPlotData.ipynb on TFRecord dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-folder", default=DEFAULT_DATA_FOLDER)
    p.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="val",
    )
    p.add_argument(
        "--feature",
        default="adjusted_hit_time_30ps_gaussian",
        choices=AVAILABLE_FEATURES,
        help="Timing feature to use for cuts",
    )
    p.add_argument("--t-max", type=float, default=10.0,
                   help="Max upper threshold for sweep (ns)")
    p.add_argument("--n-steps", type=int, default=100_000,
                   help="Number of threshold steps in linear sweep")
    p.add_argument("--output-dir", default=".", help="Directory for output PNGs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# TFRecord loading
# ---------------------------------------------------------------------------
def _make_parse_fn(feature_name):
    def _parse_fn(example_proto):
        feature_description = {
            "y": tf.io.FixedLenFeature([], tf.string),
            feature_name: tf.io.FixedLenFeature([], tf.string),
        }
        ex = tf.io.parse_single_example(example_proto, feature_description)
        y = tf.io.parse_tensor(ex["y"], out_type=tf.float32)
        t = tf.io.parse_tensor(ex[feature_name], out_type=tf.float32)
        return t, y
    return _parse_fn


def load_times_and_labels(tfrecord_dirs, feature_name):
    all_t, all_y = [], []
    for d in tfrecord_dirs:
        files = sorted([
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tfrecord")
        ])
        if not files:
            print(f"  WARNING: no .tfrecord files in {d}")
            continue
        print(f"  Loading {len(files)} files from {d}")
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_make_parse_fn(feature_name),
                              num_parallel_calls=tf.data.AUTOTUNE)
        for t_batch, y_batch in dataset:
            all_t.append(t_batch.numpy().ravel())
            all_y.append(y_batch.numpy().ravel())

    times = np.concatenate(all_t)
    labels = np.concatenate(all_y)
    return times, labels


def count_sig_bib(times, labels, query_mask=None):
    """
    Return (fracBib, fracSig, numSig, numBib) for hits passing query_mask.
    fracSig = numSig / (numSig + numBib)  – fraction of the subset that is signal
    """
    if query_mask is not None:
        t = times[query_mask]
        l = labels[query_mask]
    else:
        t = times
        l = labels
    numSig = int((l == 1).sum())
    numBib = int((l == 0).sum())
    total  = numSig + numBib
    fracSig = numSig / total if total > 0 else 0.0
    fracBib = numBib / total if total > 0 else 0.0
    return fracBib, fracSig, numSig, numBib


def plot_timing_histograms(times, labels, feature_name, output_dir):
    sig = times[labels == 1]
    bib = times[labels == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, bins, title_suffix in zip(
        axes,
        [np.linspace(-1, 4, 100), np.linspace(-1, 100, 100)],
        ["zoom: -1 to 4 ns", "wide: -1 to 100 ns"],
    ):
        ax.hist(sig, bins=bins, alpha=0.6, label=f"sig  (n={len(sig):,})",
                color="steelblue", density=True)
        ax.hist(bib, bins=bins, alpha=0.6, label=f"bib  (n={len(bib):,})",
                color="tomato", density=True)
        ax.set_yscale("log")
        ax.set_xlabel(f"{feature_name} (ns)")
        ax.set_ylabel("density")
        ax.set_title(f"{feature_name}\n{title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "timing_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _metrics_at_cut(sig, bib, T, style="sym"):
    """Return (sigKeptFrac, bibKeptFrac, fracSig) for a single cut T."""
    n_sig = len(sig)
    n_bib = len(bib)
    if style == "sym":
        sig_kept = int((np.abs(sig) < T).sum())
        bib_kept = int((np.abs(bib) < T).sum())
    else:
        sig_kept = int((sig < T).sum())
        bib_kept = int((bib < T).sum())
    total_kept = sig_kept + bib_kept
    fracSig = sig_kept / total_kept if total_kept > 0 else 0.0
    return sig_kept / n_sig, bib_kept / n_bib, fracSig


def _single_cut_ax(ax, sig, bib, T, feature_name, x_lo, x_hi, bins=120):
    """Draw histogram + both cut styles + metrics onto ax."""
    b = np.linspace(x_lo, x_hi, bins)
    ax.hist(sig, bins=b, alpha=0.55, color="steelblue", density=True,
            label=f"signal  (n={len(sig):,})")
    ax.hist(bib, bins=b, alpha=0.55, color="tomato",   density=True,
            label=f"BIB  (n={len(bib):,})")
    ax.set_yscale("log")
    ax.set_xlabel(f"{feature_name} (ns)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_xlim(x_lo, x_hi)

    # ---- symmetric cut  |t| < T  (solid lines + blue shade) ----
    sym_sigK, sym_bibK, sym_fracSig = _metrics_at_cut(sig, bib, T, style="sym")
    shade_lo = max(-T, x_lo)
    shade_hi = min( T, x_hi)
    if shade_hi > shade_lo:
        ax.axvspan(shade_lo, shade_hi, alpha=0.08, color="steelblue", zorder=0)
    for x in [-T, T]:
        if x_lo <= x <= x_hi:
            ax.axvline(x, color="steelblue", lw=1.4, linestyle="-",
                       label=f"|t|<T sym" if x == T else None)

    # ---- one-sided cut  t < T  (dashed line + orange shade) ----
    one_sigK, one_bibK, one_fracSig = _metrics_at_cut(sig, bib, T, style="one")
    shade_one_hi = min(T, x_hi)
    if shade_one_hi > x_lo:
        ax.axvspan(x_lo, shade_one_hi, alpha=0.06, color="darkorange", zorder=0)
    if x_lo <= T <= x_hi:
        ax.axvline(T, color="darkorange", lw=1.4, linestyle="--")

    # ---- metrics text box ----
    txt = (
        f"T = {T:.4f} ns\n"
        f"\nsymmetric |t|<T\n"
        f"  sigKept  = {sym_sigK:.4f}\n"
        f"  bibKept  = {sym_bibK:.4f}\n"
        f"  fracSig  = {sym_fracSig:.4f}\n"
        f"\none-sided t<T\n"
        f"  sigKept  = {one_sigK:.4f}\n"
        f"  bibKept  = {one_bibK:.4f}\n"
        f"  fracSig  = {one_fracSig:.4f}"
    )
    ax.text(0.98, 0.97, txt, transform=ax.transAxes,
            fontsize=7, va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    # ---- legend: just hist entries + one entry per cut style ----
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Patch(fc="steelblue", alpha=0.25, label="sym kept region"),
        Patch(fc="darkorange", alpha=0.2,  label="one-sided kept region"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper left")


def plot_cut_histograms(times, labels, feature_name, cut_values, output_dir):
    """
    For each T in cut_values, save a two-panel PNG (zoom + wide view) showing
    the timing histogram with symmetric |t|<T and one-sided t<T cuts annotated.
    Also saves an overview grid with all cuts as subplots.
    """
    subdir = os.path.join(output_dir, "cut_histograms")
    os.makedirs(subdir, exist_ok=True)

    sig = times[labels == 1]
    bib = times[labels == 0]

    cut_values = sorted(cut_values)

    for T in cut_values:
        x_zoom_hi = min(max(T * 1.5, 4.0), 20.0)
        views = [
            ("zoom",  -1.5,  x_zoom_hi),
            ("wide",  -1.5,  100.0),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (view_name, x_lo, x_hi) in zip(axes, views):
            _single_cut_ax(ax, sig, bib, T, feature_name, x_lo, x_hi)
            ax.set_title(f"T = {T:.4f} ns  [{view_name}]", fontsize=10)

        fig.suptitle(f"{feature_name} — cut at T = {T:.4f} ns", fontsize=11)
        fig.tight_layout()
        fname = f"cut_T{T:.4f}ns.png".replace(" ", "")
        path = os.path.join(subdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

    # ---- overview grid: all cuts, zoom view only ----
    n = len(cut_values)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes_flat = np.array(axes).ravel() if n > 1 else [axes]
    for ax, T in zip(axes_flat, cut_values):
        x_hi = min(max(T * 1.5, 4.0), 20.0)
        _single_cut_ax(ax, sig, bib, T, feature_name, -1.5, x_hi)
        ax.set_title(f"T = {T:.4f} ns", fontsize=9)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{feature_name} — all cuts overview", fontsize=12)
    fig.tight_layout()
    path = os.path.join(subdir, "overview_all_cuts.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _metrics_at_window(sig, bib, t_lo, t_hi):
    """Return (sigKeptFrac, bibKeptFrac, fracSig) for asymmetric window t_lo < t < t_hi."""
    n_sig = len(sig)
    n_bib = len(bib)
    sig_kept = int(((sig > t_lo) & (sig < t_hi)).sum())
    bib_kept = int(((bib > t_lo) & (bib < t_hi)).sum())
    total_kept = sig_kept + bib_kept
    fracSig = sig_kept / total_kept if total_kept > 0 else 0.0
    return sig_kept / n_sig, bib_kept / n_bib, fracSig


def plot_asymmetric_cut_histogram(times, labels, feature_name, t_lo, t_hi, output_dir):
    """
    Two-panel histogram (zoom + wide) showing the asymmetric window cut
    t_lo < t < t_hi with shaded kept region, vertical boundary lines, and metrics.
    Saved to output_dir/cut_histograms/cut_asym_{t_lo}_{t_hi}ns.png.
    """
    subdir = os.path.join(output_dir, "cut_histograms")
    os.makedirs(subdir, exist_ok=True)

    sig = times[labels == 1]
    bib = times[labels == 0]

    sigK, bibK, fracSig = _metrics_at_window(sig, bib, t_lo, t_hi)
    backRej = 1.0 - bibK

    txt = (
        f"window: {t_lo*1000:.0f} ps  to  {t_hi*1000:.0f} ps\n"
        f"  sigKeptFrac  = {sigK:.4f}\n"
        f"  bibKeptFrac  = {bibK:.4f}\n"
        f"  backRej      = {backRej:.4f}\n"
        f"  fracSig      = {fracSig:.4f}"
    )

    x_zoom_hi = max(t_hi * 2.0, 0.5)
    views = [
        ("zoom", min(t_lo * 2.0, -0.2), x_zoom_hi),
        ("wide", -1.5, 100.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (view_name, x_lo, x_hi) in zip(axes, views):
        bins = np.linspace(x_lo, x_hi, 120)
        ax.hist(sig, bins=bins, alpha=0.55, color="steelblue", density=True,
                label=f"signal  (n={len(sig):,})")
        ax.hist(bib, bins=bins, alpha=0.55, color="tomato",   density=True,
                label=f"BIB  (n={len(bib):,})")
        ax.set_yscale("log")

        # shaded kept region
        shade_lo = max(t_lo, x_lo)
        shade_hi = min(t_hi, x_hi)
        if shade_hi > shade_lo:
            ax.axvspan(shade_lo, shade_hi, alpha=0.15, color="green", zorder=0,
                       label="kept region")

        # boundary lines
        for x, lbl in [(t_lo, f"{t_lo*1000:.0f} ps"), (t_hi, f"{t_hi*1000:.0f} ps")]:
            if x_lo <= x <= x_hi:
                ax.axvline(x, color="green", lw=1.6, linestyle="-", label=lbl)

        ax.text(0.98, 0.97, txt, transform=ax.transAxes,
                fontsize=7.5, va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

        ax.set_xlabel(f"{feature_name} (ns)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.set_xlim(x_lo, x_hi)
        ax.set_title(f"asymmetric cut  [{t_lo*1000:.0f} ps, {t_hi*1000:.0f} ps]  [{view_name}]",
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"{feature_name} — asymmetric window  {t_lo*1000:.0f} ps to {t_hi*1000:.0f} ps",
        fontsize=11,
    )
    fig.tight_layout()

    lo_tag = f"{t_lo*1000:.0f}ps".replace("-", "neg")
    hi_tag = f"{t_hi*1000:.0f}ps"
    path = os.path.join(subdir, f"cut_asym_{lo_tag}_to_{hi_tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _sweep(sorted_sig, sorted_bib, n_sig_total, n_bib_total, thresholds):
    """Shared core: given pre-sorted arrays, compute rates at each threshold."""
    sig_pass = np.searchsorted(sorted_sig, thresholds, side="left")
    bib_pass = np.searchsorted(sorted_bib, thresholds, side="left")
    total_pass = sig_pass + bib_pass
    fracSig     = np.where(total_pass > 0, sig_pass / total_pass, 0.0)
    fracBib     = np.where(total_pass > 0, bib_pass / total_pass, 0.0)
    sigKeptFrac = sig_pass / n_sig_total
    bibKeptFrac = bib_pass / n_bib_total
    backRej     = 1.0 - bibKeptFrac
    return fracSig, fracBib, sigKeptFrac, bibKeptFrac, backRej, n_sig_total, n_bib_total


def sweep_one_sided(times, labels, thresholds):
    """One-sided cut  t < T  (matches the notebook's `adjusted_hit_time < @cut`)."""
    sig_mask = labels == 1
    bib_mask = labels == 0
    return _sweep(
        np.sort(times[sig_mask]),
        np.sort(times[bib_mask]),
        int(sig_mask.sum()), int(bib_mask.sum()),
        thresholds,
    )


def sweep_symmetric(times, labels, thresholds):
    """Symmetric window cut  |t| < T  (matches timing_cut_analysis.py)."""
    sig_mask = labels == 1
    bib_mask = labels == 0
    return _sweep(
        np.sort(np.abs(times[sig_mask])),
        np.sort(np.abs(times[bib_mask])),
        int(sig_mask.sum()), int(bib_mask.sum()),
        thresholds,
    )


def plot_rates_vs_cut(thresholds, fracSig, fracBib, sigKeptFrac, bibKeptFrac,
                      backRej, feature_name, output_dir, cut_label="t < T",
                      fname_prefix="rates_vs_cut"):
    xlabel = f"Threshold T  [{cut_label}]  (ns)"

    # --- full log-scale view ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, fracSig,      label="fracSig – signal fraction in kept region")
    ax.plot(thresholds, fracBib,      label="fracBib – BIB fraction in kept region")
    ax.plot(thresholds, sigKeptFrac,  label="sigKeptFrac (TPR / sigEffic)")
    ax.plot(thresholds, bibKeptFrac,  label="bibKeptFrac (FPR)")
    ax.plot(thresholds, backRej,      label="backRej = 1 – FPR")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction")
    ax.set_title(f"All rates vs. timing cut — full range (log)\n{feature_name}  [{cut_label}]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{fname_prefix}_log.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # --- linear view, zoomed to important range (mirrors notebook) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, fracSig,      label="fracSig – signal fraction in kept region")
    ax.plot(thresholds, fracBib,      label="fracBib – BIB fraction in kept region")
    ax.plot(thresholds, sigKeptFrac,  label="sigKeptFrac (TPR / sigEffic)")
    ax.plot(thresholds, bibKeptFrac,  label="bibKeptFrac (FPR)")
    ax.plot(thresholds, backRej,      label="backRej = 1 – FPR")
    ax.set_xscale("linear")
    ax.set_xlim([thresholds[0], 1.2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction")
    ax.set_title(f"All rates vs. timing cut — important range (linear)\n{feature_name}  [{cut_label}]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{fname_prefix}_linear.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3 & 4: ROC curves (ML style and physics style)
# ---------------------------------------------------------------------------
def _find_op_points(tpr, fpr, backRej, thresholds, target_efficiencies):
    op_points = []
    for target in sorted(target_efficiencies):
        idx_arr = np.where(tpr >= target)[0]
        if len(idx_arr) == 0:
            op_points.append((target, None, None, None, None))
        else:
            idx = idx_arr[0]
            op_points.append((target, thresholds[idx], tpr[idx], fpr[idx], backRej[idx]))
    return op_points


def plot_roc_curves(sigKeptFrac, bibKeptFrac, backRej, thresholds,
                    feature_name, output_dir, target_efficiencies, cut_label="t < T"):
    tpr = sigKeptFrac
    fpr = bibKeptFrac
    op_points = _find_op_points(tpr, fpr, backRej, thresholds, target_efficiencies)
    colors = {0.95: "tab:orange", 0.98: "tab:green", 0.99: "tab:red"}

    # --- ML style ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=1.5, color="steelblue", label=f"timing cut [{cut_label}]")
    for target, T, sig_e, fp, br in op_points:
        if T is None:
            continue
        ax.scatter(fp, sig_e, color=colors.get(target, "black"), zorder=5, s=60,
                   label=f"{int(target*100)}% sig eff | bkgRej={br:.4f} | T={T:.4f} ns")
    ax.set_xlabel("FPR (background acceptance)")
    ax.set_ylabel("Signal Efficiency (TPR)")
    ax.set_title(f"ROC curve – ML style\n{feature_name}  [{cut_label}]")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(output_dir, "roc_ml_style.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # --- physics style ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(backRej, tpr, lw=1.5, color="steelblue", label=f"timing cut [{cut_label}]")
    for target, T, sig_e, fp, br in op_points:
        if T is None:
            continue
        ax.scatter(br, sig_e, color=colors.get(target, "black"), zorder=5, s=60,
                   label=f"{int(target*100)}% sig eff | bkgRej={br:.4f} | T={T:.4f} ns")
    ax.set_xlabel("Background Rejection (1 – FPR)")
    ax.set_ylabel("Signal Efficiency (TPR)")
    ax.set_title(f"ROC curve – physics style\n{feature_name}  [{cut_label}]")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(output_dir, "roc_physics_style.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    return op_points


def plot_roc_comparison(sym_tpr, sym_fpr, sym_backRej, sym_thresholds,
                        one_tpr, one_fpr, one_backRej, one_thresholds,
                        feature_name, output_dir, target_efficiencies):
    """Overlay symmetric |t|<T and one-sided t<T ROC curves for direct comparison."""
    sym_ops = _find_op_points(sym_tpr, sym_fpr, sym_backRej, sym_thresholds, target_efficiencies)
    one_ops = _find_op_points(one_tpr, one_fpr, one_backRej, one_thresholds, target_efficiencies)
    colors = {0.95: "tab:orange", 0.98: "tab:green", 0.99: "tab:red"}

    for style, xlabel, ylabel, x_sym, y_sym, x_one, y_one, ops_sym, ops_one, fname in [
        ("ML style", "FPR (background acceptance)", "Signal Efficiency (TPR)",
         sym_fpr, sym_tpr, one_fpr, one_tpr, sym_ops, one_ops, "roc_comparison_ml.png"),
        ("physics style", "Background Rejection (1 – FPR)", "Signal Efficiency (TPR)",
         sym_backRej, sym_tpr, one_backRej, one_tpr, sym_ops, one_ops, "roc_comparison_physics.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_sym, y_sym, lw=2.0, color="steelblue",   label="symmetric  |t| < T")
        ax.plot(x_one, y_one, lw=2.0, color="darkorange", linestyle="--", label="one-sided  t < T")

        # Mark operating points for symmetric cut only (to avoid clutter)
        for target, T, sig_e, fp, br in ops_sym:
            if T is None:
                continue
            x_val = fp if style == "ML style" else br
            ax.scatter(x_val, sig_e, color=colors.get(target, "black"),
                       zorder=5, s=70, marker="o",
                       label=f"sym {int(target*100)}%: bkgRej={br:.4f}, T={T:.4f} ns")
        for target, T, sig_e, fp, br in ops_one:
            if T is None:
                continue
            x_val = fp if style == "ML style" else br
            ax.scatter(x_val, sig_e, color=colors.get(target, "black"),
                       zorder=5, s=70, marker="^",
                       label=f"one  {int(target*100)}%: bkgRej={br:.4f}, T={T:.4f} ns")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"ROC comparison – {style}\n{feature_name}")
        ax.legend(fontsize=7, loc="lower right" if style == "ML style" else "lower left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Print operating-point summary (mirrors the notebook's printed block)
# ---------------------------------------------------------------------------
def print_operating_points(times, labels, feature_name, cuts, n_sig_total, n_bib_total):
    hdr = f"{'Cut T (ns)':<15} {'numSig':>10} {'numBib':>10} {'fracSig':>10} {'sigKeptFrac':>13} {'bibKeptFrac':>13}"
    sep = "-" * 75

    _, fracSig0, numSig0, numBib0 = count_sig_bib(times, labels)
    baseline = (f"{'no cut':<15} {numSig0:>10,} {numBib0:>10,} "
                f"{fracSig0:>10.4f} {'1.0000':>13} {'1.0000':>13}")

    for cut_style, mask_fn, title in [
        ("one-sided  t < T",   lambda T: times < T,             "one-sided cut: t < T"),
        ("symmetric  |t| < T", lambda T: np.abs(times) < T,     "symmetric cut: |t| < T"),
    ]:
        print("\n" + "=" * 75)
        print(f"Operating-point summary — {title}")
        print(f"Feature: {feature_name}")
        print("=" * 75)
        print(hdr)
        print(sep)
        print(baseline)
        for T in cuts:
            mask = mask_fn(T)
            _, fracSig, numSig, numBib = count_sig_bib(times, labels, query_mask=mask)
            sigKept = numSig / n_sig_total if n_sig_total > 0 else 0.0
            bibKept = numBib / n_bib_total if n_bib_total > 0 else 0.0
            print(f"{T:<15.4f} {numSig:>10,} {numBib:>10,} "
                  f"{fracSig:>10.4f} {sigKept:>13.6f} {bibKept:>13.6f}")
        print("=" * 75)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    dirs_to_load = []
    if args.split in ("train", "both"):
        dirs_to_load.append(os.path.join(args.data_folder, "tfrecords_train"))
    if args.split in ("val", "both"):
        dirs_to_load.append(os.path.join(args.data_folder, "tfrecords_validation"))

    print("=" * 65)
    print("loadAndPlotData reproduction — TFRecord edition")
    print("=" * 65)
    print(f"  Data folder : {args.data_folder}")
    print(f"  Split       : {args.split}")
    print(f"  Feature     : {args.feature}")
    print(f"  Threshold   : sym 0→{args.t_max} ns | one-sided data_min→{args.t_max} ns  ({args.n_steps} steps)")
    print()

    print("Loading TF records...")
    times, labels = load_times_and_labels(dirs_to_load, args.feature)
    sig_mask = labels == 1
    bib_mask = labels == 0
    n_sig_total = int(sig_mask.sum())
    n_bib_total = int(bib_mask.sum())
    total = len(labels)
    print(f"  Loaded {total:,} hits")
    print(f"  Signal : {n_sig_total:,}  ({100*n_sig_total/total:.2f}%)")
    print(f"  BIB    : {n_bib_total:,}  ({100*n_bib_total/total:.2f}%)")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Histograms
    print("\nPlotting timing histograms...")
    plot_timing_histograms(times, labels, args.feature, args.output_dir)

    # 2. Build threshold grids.
    # Symmetric |t|<T: T must be positive.
    # One-sided t<T: T can go negative (data has negative times).
    print("\nSweeping thresholds...")
    t_min_one = max(-2.0, float(np.min(times)))   # don't go below actual data minimum
    thresholds_log = np.logspace(1, 2, 10)

    thresholds_sym = np.unique(np.concatenate([
        np.linspace(0.0, args.t_max, args.n_steps + 1)[1:],
        thresholds_log,
    ]))
    thresholds_one = np.unique(np.concatenate([
        np.linspace(t_min_one, args.t_max, args.n_steps + 1),
        thresholds_log,
    ]))

    sym_results = sweep_symmetric(times, labels, thresholds_sym)
    one_results = sweep_one_sided(times, labels, thresholds_one)
    sym_fracSig, sym_fracBib, sym_sigKept, sym_bibKept, sym_backRej, _, _ = sym_results
    one_fracSig, one_fracBib, one_sigKept, one_bibKept, one_backRej, _, _ = one_results

    # 3. Rate plots — both cut styles
    print("\nPlotting rates vs. cut...")
    plot_rates_vs_cut(thresholds_sym, sym_fracSig, sym_fracBib, sym_sigKept, sym_bibKept,
                      sym_backRej, args.feature, args.output_dir,
                      cut_label="|t| < T", fname_prefix="rates_vs_cut_sym")
    plot_rates_vs_cut(thresholds_one, one_fracSig, one_fracBib, one_sigKept, one_bibKept,
                      one_backRej, args.feature, args.output_dir,
                      cut_label="t < T", fname_prefix="rates_vs_cut_one")

    # 4. ROC curves (symmetric)
    print("\nPlotting ROC curves (symmetric cut)...")
    sym_op_points = plot_roc_curves(sym_sigKept, sym_bibKept, sym_backRej, thresholds_sym,
                                    args.feature, args.output_dir, TARGET_EFFICIENCIES,
                                    cut_label="|t| < T")

    # 5. Comparison ROC: symmetric vs. one-sided
    print("\nPlotting ROC comparison (symmetric vs. one-sided)...")
    plot_roc_comparison(sym_sigKept, sym_bibKept, sym_backRej, thresholds_sym,
                        one_sigKept, one_bibKept, one_backRej, thresholds_one,
                        args.feature, args.output_dir, TARGET_EFFICIENCIES)

    # 6. Histograms with cut lines
    print("\nPlotting cut histograms...")
    op_thresholds = [T for _, T, _, _, _ in sym_op_points if T is not None]
    all_cut_values = sorted(set(NOTEBOOK_CUTS + op_thresholds))
    plot_cut_histograms(times, labels, args.feature, all_cut_values, args.output_dir)

    # Asymmetric cut: -90 ps to +150 ps  (matching notebook results2)
    plot_asymmetric_cut_histogram(
        times, labels, args.feature,
        t_lo=-0.09, t_hi=0.15,
        output_dir=args.output_dir,
    )

    # 7. Operating-point tables (both cut styles)
    print_operating_points(times, labels, args.feature,
                           NOTEBOOK_CUTS, n_sig_total, n_bib_total)

    # Print sweep operating points for symmetric cut
    print("\nROC operating points — symmetric |t| < T:")
    print(f"{'Target eff':<12} {'Threshold (ns)':>16} {'Actual eff':>12} "
          f"{'FPR':>10} {'BkgRej':>10}")
    print("-" * 65)
    for target, T, sig_e, fp, br in sym_op_points:
        if T is None:
            print(f"{target:<12.2f} {'N/A':>16}")
        else:
            print(f"{target:<12.2f} {T:>16.6f} {sig_e:>12.6f} {fp:>10.6f} {br:>10.6f}")

    print(f"\nAll outputs written to: {os.path.abspath(args.output_dir)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
