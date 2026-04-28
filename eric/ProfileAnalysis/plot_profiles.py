"""
Plot value-frequency histograms for all features from TFRecords.

Output figures:
  - cluster_histogram.png      : frequency of each pixel value across all 13x21 entries
  - xy_profile_histogram.png   : x_profile (len-21) and y_profile (len-13) overlaid
  - nModule_histogram.png      : nModule scalar distribution
  - x_local_histogram.png      : x_local scalar distribution
  - y_local_histogram.png      : y_local scalar distribution
  - z_global_histogram.png     : z_global scalar distribution
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


def parse_fn(example_proto):
    feature_description = {f: tf.io.FixedLenFeature([], tf.string) for f in FEATURES}
    example = tf.io.parse_single_example(example_proto, feature_description)
    return {f: tf.io.parse_tensor(example[f], out_type=tf.float32) for f in FEATURES}


def collect_values(tfrecord_files, max_batches):
    cluster_vals = []
    x_profile_vals = []
    y_profile_vals = []
    nmodule_vals = []
    x_local_vals = []
    y_local_vals = []
    z_global_vals = []

    dataset = tf.data.TFRecordDataset(tfrecord_files)

    for i, raw in enumerate(dataset):
        if i >= max_batches:
            break
        parsed = parse_fn(raw)

        cluster_vals.append(parsed["cluster"].numpy().flatten())
        x_profile_vals.append(parsed["x_profile"].numpy().flatten())
        y_profile_vals.append(parsed["y_profile"].numpy().flatten())
        nmodule_vals.append(parsed["nModule"].numpy().flatten())
        x_local_vals.append(parsed["x_local"].numpy().flatten())
        y_local_vals.append(parsed["y_local"].numpy().flatten())
        z_global_vals.append(parsed["z_global"].numpy().flatten())

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1} batches...")

    return (
        np.concatenate(cluster_vals),
        np.concatenate(x_profile_vals),
        np.concatenate(y_profile_vals),
        np.concatenate(nmodule_vals),
        np.concatenate(x_local_vals),
        np.concatenate(y_local_vals),
        np.concatenate(z_global_vals),
    )


def plot_cluster(cluster_vals, out_path, bins):
    fig, ax = plt.subplots(figsize=(10, 5))

    unique, counts = np.unique(np.round(cluster_vals, 4), return_counts=True)
    if bins == "unique" or len(unique) <= 200:
        ax.bar(unique, counts, width=0.8 * (unique[1] - unique[0]) if len(unique) > 1 else 0.5,
               color="steelblue", edgecolor="none", alpha=0.85)
        ax.set_xlabel("Pixel value (log-compressed charge)")
    else:
        ax.hist(cluster_vals, bins=int(bins), color="steelblue", edgecolor="none", alpha=0.85)
        ax.set_xlabel("Pixel value (log-compressed charge)")

    ax.set_ylabel("Frequency (count)")
    ax.set_title("Cluster pixel value distribution\n(all 13×21 entries pooled)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_xy_profiles(x_vals, y_vals, out_path, bins):
    fig, ax = plt.subplots(figsize=(10, 5))

    def _hist_data(vals, n_bins):
        vmin, vmax = vals.min(), vals.max()
        edges = np.linspace(vmin, vmax, n_bins + 1)
        counts, edges = np.histogram(vals, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts

    n_bins = 150 if bins == "unique" else int(bins)
    xc, xn = _hist_data(x_vals, n_bins)
    yc, yn = _hist_data(y_vals, n_bins)

    ax.bar(xc, xn, width=(xc[1] - xc[0]) * 0.85,
           color="royalblue", alpha=0.6, label="x_profile (len-21)")
    ax.bar(yc, yn, width=(yc[1] - yc[0]) * 0.85,
           color="tomato", alpha=0.6, label="y_profile (len-13)")

    ax.set_xlabel("Profile value (log-compressed summed charge)")
    ax.set_ylabel("Frequency (count)")
    ax.set_title("X- and Y-profile value distributions\n(all entries pooled, overlaid)")
    ax.set_yscale("log")
    ax.legend()
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
        description="Histogram of cluster and profile values from TFRecords",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_folder", default=DEFAULT_DATA_FOLDER,
                        help="Directory containing .tfrecord files")
    parser.add_argument("--max_batches", type=int, default=200,
                        help="Number of TFRecord examples to read")
    parser.add_argument("--bins", default="unique",
                        help="Histogram bins: 'unique' to use exact values, or an integer")
    parser.add_argument("--out_dir", default=".",
                        help="Output directory for saved figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tfrecord_files = sorted(glob.glob(os.path.join(args.data_folder, "*.tfrecord")))
    if not tfrecord_files:
        raise FileNotFoundError(f"No .tfrecord files found in {args.data_folder}")

    print(f"Found {len(tfrecord_files)} TFRecord files. Reading up to {args.max_batches} examples...")
    cluster_vals, x_vals, y_vals, nmodule_vals, x_local_vals, y_local_vals, z_global_vals = \
        collect_values(tfrecord_files, args.max_batches)

    for name, vals in [
        ("cluster",   cluster_vals),
        ("x_profile", x_vals),
        ("y_profile", y_vals),
        ("nModule",   nmodule_vals),
        ("x_local",   x_local_vals),
        ("y_local",   y_local_vals),
        ("z_global",  z_global_vals),
    ]:
        print(f"{name:10s} — total: {len(vals):,}  unique: {len(np.unique(np.round(vals, 4))):,}  range: [{vals.min():.4f}, {vals.max():.4f}]")

    plot_cluster(cluster_vals, os.path.join(args.out_dir, "cluster_histogram.png"), args.bins)
    plot_xy_profiles(x_vals, y_vals, os.path.join(args.out_dir, "xy_profile_histogram.png"), args.bins)

    for name, vals in [
        ("nModule",  nmodule_vals),
        ("x_local",  x_local_vals),
        ("y_local",  y_local_vals),
        ("z_global", z_global_vals),
    ]:
        out_path = os.path.join(args.out_dir, f"{name}_histogram.png")
        plot_scalar(vals, name, out_path, args.bins)


if __name__ == "__main__":
    main()
