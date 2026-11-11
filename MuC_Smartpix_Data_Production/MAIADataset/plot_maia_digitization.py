#!/usr/bin/env python3
"""Plot the available MAIA BIB cluster tables and the small signal-only pixel sample.

The BIB ROOT file is not read. Its committed Parquet tables supply cluster and
sim-hit links; the committed digi_real CSVs supply example fired-pixel maps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_LINKED_STUB = REPO / "local_only/maia_linked/digi_output_mu2.edm4hep.root"
DEFAULT_PIXEL_STUB = HERE / "edmTestFiles/digi_real.edm4hep.root"
DEFAULT_PIXEL_LINKS = HERE / "edmTestFiles/digi_output_mu2.edm4hep.root.vtxPixelsToClustersDf.parquet"
DEFAULT_OUTPUT = REPO / "local_only/maia_visualizations"


def read_clusters(stub: Path) -> pd.DataFrame:
    columns = [
        "ievent", "index", "collectionID", "cellID", "position.x", "position.y",
        "position.z", "eDep", "n_sim_hits", "n_signal_sim_hits",
        "n_overlay_sim_hits", "max_dist_to_sim_mm",
    ]
    path = Path(f"{stub}.vtxClustersDfLinked.parquet")
    clusters = pd.read_parquet(path, columns=columns)
    if clusters.empty or clusters.duplicated(["ievent", "index"]).any():
        raise ValueError("cluster table is empty or has duplicate (ievent, index) keys")
    return clusters


def count_pixels(clusters: pd.DataFrame, link_path: Path) -> np.ndarray:
    """Count pixel links by cluster, scanning Parquet batches instead of loading all links."""
    stride = 1_000_000_000  # same event/index key convention as linkEdmHits.py
    event = clusters["ievent"].to_numpy(np.int64)
    index = clusters["index"].to_numpy(np.int64)
    if (event < 0).any() or (index < 0).any() or (index >= stride).any():
        raise ValueError("cluster event/index cannot be encoded with the linker key stride")
    keys = event * stride + index
    order = np.argsort(keys)
    sorted_keys = keys[order]
    ids = clusters["collectionID"].to_numpy(np.int64)
    counts = np.zeros(len(clusters), dtype=np.int64)
    parquet = pq.ParquetFile(link_path)

    for batch in parquet.iter_batches(
        batch_size=500_000,
        columns=["ievent", "from_collectionID", "from_index"],
    ):
        event_b = batch.column("ievent").to_numpy().astype(np.int64, copy=False)
        index_b = batch.column("from_index").to_numpy().astype(np.int64, copy=False)
        id_b = batch.column("from_collectionID").to_numpy().astype(np.int64, copy=False)
        if (event_b < 0).any() or (index_b < 0).any() or (index_b >= stride).any():
            raise ValueError("pixel link has an invalid event or cluster index")
        lookup = np.searchsorted(sorted_keys, event_b * stride + index_b)
        if (lookup >= len(sorted_keys)).any():
            raise ValueError("pixel link points beyond the cluster table")
        row = order[lookup]
        if not np.array_equal(keys[row], event_b * stride + index_b) or not np.array_equal(ids[row], id_b):
            raise ValueError("pixel link points to a missing or different cluster")
        counts += np.bincount(row, minlength=len(clusters))

    if int(counts.sum()) != parquet.metadata.num_rows:
        raise ValueError("pixel-link count changed during processing")
    return counts


def save_occupancy(clusters: pd.DataFrame, signal: np.ndarray, output: Path) -> None:
    bib = clusters.loc[~signal]
    sig = clusters.loc[signal]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
    views = [
        (bib["position.z"], np.hypot(bib["position.x"], bib["position.y"]),
         sig["position.z"], np.hypot(sig["position.x"], sig["position.y"]),
         np.linspace(-70, 70, 281), np.linspace(25, 110, 171), "z (mm)", "radius (mm)", "Barrel occupancy: z–r"),
        (bib["position.x"], bib["position.y"], sig["position.x"], sig["position.y"],
         np.linspace(-110, 110, 221), np.linspace(-110, 110, 221),
         "x (mm)", "y (mm)", "Barrel occupancy: x–y"),
    ]
    for ax, (x, y, sx, sy, xedges, yedges, xlabel, ylabel, title) in zip(axes, views):
        hist, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        image = ax.pcolormesh(
            xedges, yedges, np.ma.masked_equal(hist.T, 0),
            cmap="magma", norm=LogNorm(vmin=1, vmax=max(1, hist.max())), shading="auto",
        )
        ax.scatter(sx, sy, c="#31d5e8", edgecolors="black", linewidths=0.7,
                   s=75, marker="o", label=f"signal-linked clusters ({len(sig)})", zorder=4)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
        bin_width = xedges[1] - xedges[0]
        fig.colorbar(image, ax=ax, label=f"Overlay-only clusters / {bin_width:g} × {yedges[1] - yedges[0]:g} mm bin")
    axes[1].set_aspect("equal", adjustable="box")
    fig.suptitle("Vertex barrel reconstructed clusters · two overlaid events", fontsize=11)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_features(clusters: pd.DataFrame, signal: np.ndarray, pixel_counts: np.ndarray, output: Path) -> None:
    same_sensor = clusters["cellID"].isin(clusters.loc[signal, "cellID"])
    bib = ~signal
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    colors = {"all": "#67798a", "matched": "#2877a8", "signal": "#dc513f"}
    energy = clusters["eDep"].to_numpy(float)
    positive = energy[energy > 0]
    bins = np.geomspace(max(positive.min() * 0.9, 1e-9), positive.max() * 1.05, 90)
    ax = axes[0]
    ax.hist(energy[bib], bins=bins, histtype="step", linewidth=1.4,
            color=colors["all"], label=f"overlay-only ({bib.sum():,})")
    ax.hist(energy[bib & same_sensor], bins=bins, histtype="step", linewidth=1.5,
            color=colors["matched"], label=f"overlay-only, same sensors ({(bib & same_sensor).sum():,})")
    ax.plot(energy[signal], np.full(signal.sum(), 0.7), "|", markersize=12,
            markeredgewidth=1.5, color=colors["signal"], label=f"signal-linked ({signal.sum()})")
    ax.set(xscale="log", yscale="log", ylim=(0.5, None), xlabel="Cluster eDep (GeV)",
           ylabel="Clusters per log-spaced bin", title="Reconstructed cluster energy")
    ax.legend(fontsize=8)

    # Pixel counts are integers. The final category keeps every cluster while
    # avoiding the artificial spikes caused by logarithmic bins over integers.
    ax = axes[1]
    x = np.arange(1, 22)
    all_counts = np.bincount(np.minimum(pixel_counts[bib], 21), minlength=22)[1:]
    sensor_counts = np.bincount(np.minimum(pixel_counts[bib & same_sensor], 21), minlength=22)[1:]
    ax.bar(x, all_counts, width=0.82, color=colors["all"], alpha=0.7,
           label=f"overlay-only ({bib.sum():,})")
    ax.bar(x, sensor_counts, width=0.44, color=colors["matched"],
           label=f"overlay-only, same sensors ({(bib & same_sensor).sum():,})")
    ax.plot(pixel_counts[signal], np.full(signal.sum(), 0.7), "|", markersize=12,
            markeredgewidth=1.5, color=colors["signal"], label=f"signal-linked ({signal.sum()})")
    ax.axvline(20.5, color="0.5", linewidth=0.8)
    ax.set(yscale="log", ylim=(0.5, None), xlim=(0.5, 21.5),
           xlabel="Pixel links per cluster (21+ is an overflow bin)",
           ylabel="Clusters per count", title="Pixel links per reconstructed cluster")
    ax.set_xticks([1, 2, 3, 4, 5, 7, 10, 15, 20, 21],
                  ["1", "2", "3", "4", "5", "7", "10", "15", "20", "21+"])
    ax.legend(fontsize=8)
    fig.suptitle("Two overlaid events · signal marks show individual clusters", fontsize=11)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_link_quality(link_path: Path, output: Path) -> dict:
    links = pd.read_parquet(link_path, columns=["dist_mm", "sim_overlay", "cl_cellID", "sim_cellID"])
    dist = links["dist_mm"].to_numpy(float)
    overlay = links["sim_overlay"].to_numpy(bool)
    mismatch = links["cl_cellID"].to_numpy() != links["sim_cellID"].to_numpy()
    finite = np.isfinite(dist) & (dist > 0)
    bins = np.geomspace(max(dist[finite].min() * 0.9, 1e-7), dist[finite].max() * 1.05, 90)
    all_hist, _ = np.histogram(dist[finite], bins=bins)
    mismatch_hist, _ = np.histogram(dist[finite & mismatch], bins=bins)
    centres = np.sqrt(bins[:-1] * bins[1:])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].stairs(all_hist, bins, color="#67798a", label="all links")
    axes[0].plot(dist[~overlay & finite], np.full((~overlay & finite).sum(), 0.7), "|",
                 markersize=12, color="#dc513f", label="signal sim-hit links")
    axes[0].axvline(10, color="black", linestyle="--", linewidth=1, label="10 mm review threshold")
    axes[0].set(xscale="log", yscale="log", ylim=(0.5, None), xlabel="Cluster–sim hit distance (mm)",
                ylabel="Links per log-spaced bin", title="Link distances")
    axes[0].legend(fontsize=8)

    valid = all_hist >= 30
    axes[1].plot(centres[valid], mismatch_hist[valid] / all_hist[valid], ".-", color="#2877a8")
    axes[1].axvline(10, color="black", linestyle="--", linewidth=1)
    axes[1].set(xscale="log", ylim=(0, 1), xlabel="Cluster–sim hit distance (mm)",
                ylabel="Fraction with different cellID", title="Cell ID mismatch (≥30 links per bin)")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"links": len(links), "links_over_10_mm": int((dist > 10).sum()),
            "links_with_different_cellID": int(mismatch.sum())}


def save_pixel_examples(stub: Path, output: Path) -> dict:
    links = pd.read_csv(f"{stub}.vtxPixelsToClustersDf.csv", usecols=[
        "ievent", "from_collectionID", "from_index", "to_collectionID", "to_index"])
    pixels = pd.read_csv(f"{stub}.vtxPixelsDf.csv", usecols=[
        "ievent", "collectionID", "index", "cellID", "position.x", "position.y", "eDep"])
    joined = links.merge(
        pixels, left_on=["ievent", "to_collectionID", "to_index"],
        right_on=["ievent", "collectionID", "index"], how="left", validate="one_to_one",
    )
    if len(joined) != len(links) or joined["eDep"].isna().any():
        raise ValueError("some signal-only pixels could not be linked to a cluster")
    sizes = joined.groupby(["ievent", "from_collectionID", "from_index"]).size().sort_values(ascending=False)
    selected = list(sizes.head(4).index)
    selected_rows = [joined[(joined["ievent"] == event) &
                            (joined["from_collectionID"] == collection_id) &
                            (joined["from_index"] == index)]
                     for event, collection_id, index in selected]
    charges = np.concatenate([group["eDep"].to_numpy(float) for group in selected_rows])
    if not np.isfinite(charges).all() or (charges <= 0).any():
        raise ValueError("pixel eDep must be finite and positive for a logarithmic colour scale")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eceff1")
    norm = LogNorm(vmin=charges.min(), vmax=charges.max())
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    image = None
    for ax, (event, _collection_id, index), group in zip(axes.flat, selected, selected_rows):
        if (group[["position.x", "position.y"]].to_numpy(float) % 1 != 0).any():
            raise ValueError("pixel positions are not integer sensor coordinates")
        x = group["position.x"].to_numpy(int)
        y = group["position.y"].to_numpy(int)
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        grid = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan)
        for col, row, charge in zip(x, y, group["eDep"].to_numpy(float)):
            previous = grid[row - y0, col - x0]
            grid[row - y0, col - x0] = charge if np.isnan(previous) else previous + charge
        image = ax.imshow(grid, origin="lower", cmap=cmap, norm=norm, interpolation="nearest",
                          extent=[x0 - .5, x1 + .5, y0 - .5, y1 + .5], aspect="equal")
        ax.set_xticks(np.arange(x0, x1 + 1))
        ax.set_yticks(np.arange(y0, y1 + 1))
        ax.set(xlabel="pixel column", ylabel="pixel row",
               title=f"event {event}, cluster {index} · {len(group)} pixels")
        ax.grid(color="white", alpha=0.3)
    fig.colorbar(image, ax=list(axes.flat), label="Pixel eDep (stored units)", shrink=0.8)
    fig.suptitle("Four clusters with the most linked pixels · separate signal-only sample", fontsize=12)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"pixel_rows": len(pixels), "pixel_links": len(links), "example_clusters": len(selected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linked-stub", type=Path, default=DEFAULT_LINKED_STUB)
    parser.add_argument("--pixel-stub", type=Path, default=DEFAULT_PIXEL_STUB)
    parser.add_argument("--pixel-links", type=Path, default=DEFAULT_PIXEL_LINKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clusters = read_clusters(args.linked_stub)
    pixel_counts = count_pixels(clusters, args.pixel_links)
    signal = clusters["n_signal_sim_hits"].to_numpy() > 0
    bib = (clusters["n_signal_sim_hits"].to_numpy() == 0) & (clusters["n_overlay_sim_hits"].to_numpy() > 0)
    if not np.all(signal | bib):
        raise ValueError("some clusters have neither a signal nor an overlay sim hit")

    save_occupancy(clusters, signal, args.output_dir / "occupancy.png")
    save_features(clusters, signal, pixel_counts, args.output_dir / "cluster_features.png")
    link_stats = save_link_quality(Path(f"{args.linked_stub}.vtxClusterSimHitLinksDf.parquet"),
                                   args.output_dir / "link_quality.png")
    pixel_stats = save_pixel_examples(args.pixel_stub, args.output_dir / "signal_only_pixel_examples.png")
    np.save(args.output_dir / "pixel_counts.npy", pixel_counts)
    manifest = json.loads(Path(f"{args.linked_stub}.linkManifest.json").read_text())
    summary = {
        "bib_sample": str(args.linked_stub), "events": int(clusters["ievent"].nunique()),
        "clusters": len(clusters), "signal_linked_clusters": int(signal.sum()),
        "bib_only_clusters": int(bib.sum()), "pixel_links": int(pixel_counts.sum()),
        "signal_cluster_pixel_counts": pixel_counts[signal].tolist(),
        "pixel_table_for_bib_available": bool(manifest["roles"].get("pixels")),
        "link_quality": link_stats, "signal_only_pixel_sample": pixel_stats,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Plots saved in {args.output_dir}")


if __name__ == "__main__":
    main()
