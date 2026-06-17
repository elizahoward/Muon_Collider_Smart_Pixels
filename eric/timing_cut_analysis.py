"""
Timing cut analysis using adjusted_hit_time_30ps_gaussian.

Sweeps a symmetric time window cut |t| < T over the validation TF records,
computes signal efficiency and background rejection at each threshold, then
reports operating points at target signal efficiencies with weighted background
rejection:

    weighted_bkg_rej = w95*BR@95% + w98*BR@98% + w99*BR@99%

Outputs two CSVs and a ROC curve PNG:
  - timing_cut_sweep.csv      : full threshold sweep
  - timing_cut_operating.csv  : operating points at target signal efficiencies
  - timing_cut_roc.png        : ROC curve (FPR vs signal efficiency)

Usage
-----
    python timing_cut_analysis.py
    python timing_cut_analysis.py --data-folder /path/to/TF_Records/... --split both
    python timing_cut_analysis.py --t-max 5.0 --n-steps 200000

Author: Eric
Date: 2026
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

DEFAULT_DATA_FOLDER = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData_normalized"
)

FEATURE = "adjusted_hit_time_30ps_gaussian"

TARGET_EFFICIENCIES = [0.95, 0.98, 0.99]
DEFAULT_BKG_REJ_WEIGHTS = {0.95: 0.1, 0.98: 0.7, 0.99: 0.2}


def parse_args():
    p = argparse.ArgumentParser(
        description="Timing window cut analysis on TF records",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-folder", default=DEFAULT_DATA_FOLDER)
    p.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="val",
        help="Which TFRecord split to load",
    )
    p.add_argument("--t-max", type=float, default=10.0, help="Max threshold to sweep (ns)")
    p.add_argument("--n-steps", type=int, default=100000, help="Number of threshold steps")
    p.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write CSV outputs",
    )
    p.add_argument("--w95", type=float, default=DEFAULT_BKG_REJ_WEIGHTS[0.95])
    p.add_argument("--w98", type=float, default=DEFAULT_BKG_REJ_WEIGHTS[0.98])
    p.add_argument("--w99", type=float, default=DEFAULT_BKG_REJ_WEIGHTS[0.99])
    return p.parse_args()


def _parse_fn(example_proto):
    feature_description = {
        "y": tf.io.FixedLenFeature([], tf.string),
        FEATURE: tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    y = tf.io.parse_tensor(example["y"], out_type=tf.float32)
    t = tf.io.parse_tensor(example[FEATURE], out_type=tf.float32)
    return t, y


def load_times_and_labels(tfrecord_dirs):
    """Load all timing values and labels from a list of TFRecord directories."""
    all_t, all_y = [], []
    for d in tfrecord_dirs:
        files = sorted([
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tfrecord")
        ])
        if not files:
            print(f"  WARNING: no .tfrecord files found in {d}")
            continue
        print(f"  Loading {len(files)} files from {d}")
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        for t_batch, y_batch in dataset:
            all_t.append(t_batch.numpy().ravel())
            all_y.append(y_batch.numpy().ravel())

    times = np.concatenate(all_t)
    labels = np.concatenate(all_y)
    return times, labels


def compute_sweep(times, labels, thresholds):
    """
    Vectorized sweep: for each threshold T apply cut |t| < T and compute
    sig_eff, fpr, bkg_rej.  Uses searchsorted on sorted absolute-value arrays
    so cost is O(n log n + k log n) instead of O(n*k).
    """
    signal_mask = labels == 1
    bib_mask = labels == 0

    n_sig = int(signal_mask.sum())
    n_bib = int(bib_mask.sum())

    print(f"  Total hits: {len(labels):,}  |  Signal: {n_sig:,}  |  BIB: {n_bib:,}")

    sig_abs = np.sort(np.abs(times[signal_mask]))
    bib_abs = np.sort(np.abs(times[bib_mask]))

    thresholds = np.asarray(thresholds)

    # searchsorted gives the number of values strictly less than T (left side)
    sig_pass_counts = np.searchsorted(sig_abs, thresholds, side="left")
    bib_pass_counts = np.searchsorted(bib_abs, thresholds, side="left")

    sig_eff = sig_pass_counts / n_sig if n_sig > 0 else np.zeros_like(thresholds, dtype=float)
    fpr     = bib_pass_counts / n_bib if n_bib > 0 else np.zeros_like(thresholds, dtype=float)
    bkg_rej = 1.0 - fpr

    return pd.DataFrame({
        "threshold_ns": thresholds,
        "sig_eff": sig_eff,
        "fpr": fpr,
        "bkg_rej": bkg_rej,
    })


def find_operating_points(sweep_df, target_efficiencies, bkg_rej_weights):
    """
    For each target signal efficiency, find the threshold where sig_eff first
    reaches or exceeds the target (smallest threshold that achieves the target).
    Returns a DataFrame of operating points plus weighted_bkg_rej summary.
    """
    rows = []
    for target in sorted(target_efficiencies):
        mask = sweep_df["sig_eff"] >= target
        if not mask.any():
            rows.append({
                "target_sig_eff": target,
                "threshold_ns": None,
                "actual_sig_eff": None,
                "fpr": None,
                "bkg_rej": None,
            })
        else:
            idx = mask.idxmax()  # first index where condition holds
            row = sweep_df.loc[idx]
            rows.append({
                "target_sig_eff": target,
                "threshold_ns": row["threshold_ns"],
                "actual_sig_eff": row["sig_eff"],
                "fpr": row["fpr"],
                "bkg_rej": row["bkg_rej"],
            })

    op_df = pd.DataFrame(rows)

    # Compute weighted background rejection
    weighted = 0.0
    valid = True
    for target, weight in bkg_rej_weights.items():
        match = op_df[op_df["target_sig_eff"] == target]
        if match.empty or match["bkg_rej"].isna().all():
            valid = False
            break
        weighted += weight * match["bkg_rej"].values[0]

    op_df["weighted_bkg_rej_weights"] = str(bkg_rej_weights)
    op_df["weighted_bkg_rej"] = weighted if valid else None
    return op_df


def plot_roc(sweep_df, op_df, output_path):
    """
    ROC curve: x = FPR (background acceptance), y = signal efficiency.
    Operating points at target signal efficiencies are marked.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(sweep_df["fpr"], sweep_df["sig_eff"], lw=1.5, color="steelblue",
            label="Timing cut |t| < T")

    colors = {0.95: "tab:orange", 0.98: "tab:green", 0.99: "tab:red"}
    for _, row in op_df.iterrows():
        if row["fpr"] is None:
            continue
        target = row["target_sig_eff"]
        label = (
            f"{int(target*100)}% sig. eff.  |  "
            f"bkg rej = {row['bkg_rej']:.4f}  |  "
            f"T = {row['threshold_ns']:.4f} ns"
        )
        ax.scatter(row["fpr"], row["actual_sig_eff"],
                   color=colors.get(target, "black"), zorder=5,
                   s=60, label=label)

    weighted = op_df["weighted_bkg_rej"].iloc[0]
    if weighted is not None:
        ax.text(0.97, 0.05,
                f"Weighted bkg rej = {weighted:.4f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("False Positive Rate (background acceptance)", fontsize=11)
    ax.set_ylabel("Signal Efficiency (TPR)", fontsize=11)
    ax.set_title("ROC curve — adjusted_hit_time_30ps_gaussian window cut", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    bkg_rej_weights = {0.95: args.w95, 0.98: args.w98, 0.99: args.w99}

    dirs_to_load = []
    if args.split in ("train", "both"):
        dirs_to_load.append(os.path.join(args.data_folder, "tfrecords_train"))
    if args.split in ("val", "both"):
        dirs_to_load.append(os.path.join(args.data_folder, "tfrecords_validation"))

    print("=" * 65)
    print("Timing Cut Analysis — adjusted_hit_time_30ps_gaussian")
    print("=" * 65)
    print(f"  Data folder : {args.data_folder}")
    print(f"  Split       : {args.split}")
    print(f"  Threshold   : 0 → {args.t_max} ns  ({args.n_steps} steps)")
    print(f"  BR weights  : 95%={args.w95}  98%={args.w98}  99%={args.w99}")
    print()

    print("Loading TF records...")
    times, labels = load_times_and_labels(dirs_to_load)
    print(f"  Loaded {len(times):,} hits\n")

    print("Sweeping thresholds...")
    thresholds = np.linspace(0.0, args.t_max, args.n_steps + 1)[1:]  # skip 0
    sweep_df = compute_sweep(times, labels, thresholds)

    print("\nFinding operating points...")
    op_df = find_operating_points(sweep_df, TARGET_EFFICIENCIES, bkg_rej_weights)

    print("\nOperating points:")
    print(op_df.to_string(index=False))
    weighted = op_df["weighted_bkg_rej"].iloc[0]
    print(f"\n  Weighted background rejection = {weighted:.6f}" if weighted is not None else "\n  Weighted BKG rejection: not reached")

    os.makedirs(args.output_dir, exist_ok=True)
    sweep_path = os.path.join(args.output_dir, "timing_cut_sweep.csv")
    op_path    = os.path.join(args.output_dir, "timing_cut_operating.csv")
    roc_path   = os.path.join(args.output_dir, "timing_cut_roc.png")

    sweep_df.to_csv(sweep_path, index=False)
    op_df.to_csv(op_path, index=False)

    print("\nGenerating ROC curve...")
    plot_roc(sweep_df, op_df, roc_path)

    print(f"\nSaved:")
    print(f"  {sweep_path}")
    print(f"  {op_path}")
    print(f"  {roc_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
