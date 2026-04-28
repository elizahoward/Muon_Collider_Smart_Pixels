"""
Plot per-position value histograms for all features from TFRecords.

For cluster (13x21=273 pixels), x_profile (21 positions), y_profile (13 positions):
  each position gets its own histogram, all shown together as a 2D heatmap
  (position index on x-axis, value on y-axis, frequency as color).

Scalar features (nModule, x_local, y_local, z_global, x_size, y_size) get
standard 1D histograms.

Output figures:
  - cluster_histogram.png      : per-pixel value heatmap (273 positions)
  - x_profile_histogram.png    : per-position value heatmap (21 positions)
  - y_profile_histogram.png    : per-position value heatmap (13 positions)
  - nModule_histogram.png
  - x_local_histogram.png
  - y_local_histogram.png
  - z_global_histogram.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DEFAULT_DATA_FOLDER = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData_normalized/tfrecords_train"
)

FEATURES = ["cluster", "x_profile", "y_profile", "nModule", "x_local", "y_local", "z_global"]

CLUSTER_W, CLUSTER_H = 21, 13   # x cols, y rows
X_PROF_LEN = CLUSTER_W           # 21
Y_PROF_LEN = CLUSTER_H           # 13


def parse_fn(example_proto):
    feature_description = {f: tf.io.FixedLenFeature([], tf.string) for f in FEATURES}
    example = tf.io.parse_single_example(example_proto, feature_description)
    return {f: tf.io.parse_tensor(example[f], out_type=tf.float32) for f in FEATURES}


def collect_values(tfrecord_files, max_batches):
    # Each TFRecord entry is a batch of examples stored flat.
    # Reshape per record into (batch_size, n_positions), then concatenate.
    cluster_rows   = []
    x_profile_rows = []
    y_profile_rows = []
    nmodule_vals   = []
    x_local_vals   = []
    y_local_vals   = []
    z_global_vals  = []

    dataset = tf.data.TFRecordDataset(tfrecord_files)

    for i, raw in enumerate(dataset):
        if i >= max_batches:
            break
        parsed = parse_fn(raw)

        cluster_flat   = parsed["cluster"].numpy().flatten()
        x_profile_flat = parsed["x_profile"].numpy().flatten()
        y_profile_flat = parsed["y_profile"].numpy().flatten()

        n_examples = cluster_flat.shape[0] // (CLUSTER_H * CLUSTER_W)
        if i == 0:
            print(f"  Examples per record: {n_examples}  "
                  f"cluster px: {CLUSTER_H}x{CLUSTER_W}, "
                  f"x_profile: {X_PROF_LEN}, y_profile: {Y_PROF_LEN}")

        cluster_rows.append(cluster_flat[:n_examples * CLUSTER_H * CLUSTER_W].reshape(n_examples, CLUSTER_H * CLUSTER_W))
        x_profile_rows.append(x_profile_flat[:n_examples * X_PROF_LEN].reshape(n_examples, X_PROF_LEN))
        y_profile_rows.append(y_profile_flat[:n_examples * Y_PROF_LEN].reshape(n_examples, Y_PROF_LEN))
        nmodule_vals.append(parsed["nModule"].numpy().flatten()[:n_examples])
        x_local_vals.append(parsed["x_local"].numpy().flatten()[:n_examples])
        y_local_vals.append(parsed["y_local"].numpy().flatten()[:n_examples])
        z_global_vals.append(parsed["z_global"].numpy().flatten()[:n_examples])

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1} records...")

    return (
        np.concatenate(cluster_rows,   axis=0),   # (N_total, 273)
        np.concatenate(x_profile_rows, axis=0),   # (N_total, 21)
        np.concatenate(y_profile_rows, axis=0),   # (N_total, 13)
        np.concatenate(nmodule_vals),
        np.concatenate(x_local_vals),
        np.concatenate(y_local_vals),
        np.concatenate(z_global_vals),
    )


def plot_values(vals_2d, feature_name, out_path, bins):
    """vals_2d: (N, n_positions) — flattened so each pixel/bin value is one histogram entry."""
    vals = vals_2d.flatten()
    fig, ax = plt.subplots(figsize=(10, 5))

    unique = np.unique(np.round(vals, 4))
    n_bins = 150 if bins == "unique" else int(bins)
    if len(unique) <= 200:
        counts, edges = np.histogram(vals, bins=len(unique))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.85,
               color="steelblue", edgecolor="none", alpha=0.85)
    else:
        ax.hist(vals, bins=n_bins, color="steelblue", edgecolor="none", alpha=0.85)

    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (count)")
    ax.set_title(f"{feature_name} value distribution  ({vals_2d.shape[1]} positions × {vals_2d.shape[0]:,} examples)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_scalar(vals, feature_name, out_path, bins):
    fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = 100 if bins == "unique" else int(bins)
    unique = np.unique(np.round(vals, 6))
    if len(unique) <= 200:
        width = 0.8 * (unique[1] - unique[0]) if len(unique) > 1 else 0.5
        counts = np.array([np.sum(np.round(vals, 6) == u) for u in unique])
        ax.bar(unique, counts, width=width, color="mediumseagreen", edgecolor="none", alpha=0.85)
    else:
        ax.hist(vals, bins=n_bins, color="mediumseagreen", edgecolor="none", alpha=0.85)

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Frequency (count)")
    ax.set_title(f"{feature_name} distribution")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-position value histograms for TFRecord features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_folder", default=DEFAULT_DATA_FOLDER,
                        help="Directory containing .tfrecord files")
    parser.add_argument("--max_batches", type=int, default=200,
                        help="Number of TFRecord examples to read")
    parser.add_argument("--bins", default="80",
                        help="Histogram bins for value axis: 'unique' or an integer")
    parser.add_argument("--out_dir", default=".",
                        help="Output directory for saved figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tfrecord_files = sorted(glob.glob(os.path.join(args.data_folder, "*.tfrecord")))
    if not tfrecord_files:
        raise FileNotFoundError(f"No .tfrecord files found in {args.data_folder}")

    print(f"Found {len(tfrecord_files)} TFRecord files. Reading up to {args.max_batches} examples...")
    (cluster, x_profile, y_profile,
     nmodule_vals, x_local_vals, y_local_vals, z_global_vals) = collect_values(tfrecord_files, args.max_batches)

    for name, arr in [
        ("cluster",   cluster),
        ("x_profile", x_profile),
        ("y_profile", y_profile),
    ]:
        flat = arr.flatten()
        print(f"{name:10s} — shape: {arr.shape}  range: [{flat.min():.4f}, {flat.max():.4f}]")

    for name, vals in [
        ("nModule",  nmodule_vals),
        ("x_local",  x_local_vals),
        ("y_local",  y_local_vals),
        ("z_global", z_global_vals),
    ]:
        print(f"{name:10s} — total: {len(vals):,}  range: [{vals.min():.4f}, {vals.max():.4f}]")

    plot_values(cluster,   "cluster",   os.path.join(args.out_dir, "cluster_histogram.png"),   args.bins)
    plot_values(x_profile, "x_profile", os.path.join(args.out_dir, "x_profile_histogram.png"), args.bins)
    plot_values(y_profile, "y_profile", os.path.join(args.out_dir, "y_profile_histogram.png"), args.bins)

    for name, vals in [
        ("nModule",  nmodule_vals),
        ("x_local",  x_local_vals),
        ("y_local",  y_local_vals),
        ("z_global", z_global_vals),
    ]:
        plot_scalar(vals, name, os.path.join(args.out_dir, f"{name}_histogram.png"), args.bins)


if __name__ == "__main__":
    main()
