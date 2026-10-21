"""
Plot total model parameters vs weighted background rejection for the input bits sweep.

Loads roc_metrics_summary.csv (already-computed metrics) and H5 model files
(for parameter counts) from each of the 4 result directories, then produces
a single scatter plot with one colour per input bit-width.

Usage
-----
    python plot_input_bits_sweep_params_vs_bkgrej.py
    python plot_input_bits_sweep_params_vs_bkgrej.py --output my_plot.png

Author: Eric
Date: 2026
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')

import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import load_model

try:
    from qkeras.utils import _add_supported_quantized_objects
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("Warning: QKeras not available — model loading may fail")

RESULT_DIRS = {
    4:  "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/"
        "model2.5.fixedinputbits_quantized_6w0i_ib4b_inputsweep_hyperparameter_results_20260421_145824",
    6:  "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/"
        "model2.5.fixedinputbits_quantized_6w0i_ib6b_inputsweep_hyperparameter_results_20260421_160132",
    8:  "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/"
        "model2.5.fixedinputbits_quantized_6w0i_ib8b_inputsweep_hyperparameter_results_20260421_170819",
    10: "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/"
        "model2.5.fixedinputbits_quantized_6w0i_ib10b_inputsweep_hyperparameter_results_20260421_181459",
}

COLORS  = {4: "#e41a1c", 6: "#ff7f00", 8: "#377eb8", 10: "#4daf4a"}
MARKERS = {4: "o",       6: "s",       8: "^",       10: "D"}


def get_custom_objects():
    co = {}
    if QKERAS_AVAILABLE:
        _add_supported_quantized_objects(co)
    return co


def count_parameters(h5_path):
    custom_objects = get_custom_objects()
    model = load_model(h5_path, custom_objects=custom_objects, compile=False)
    n = int(model.count_params())
    del model
    tf.keras.backend.clear_session()
    return n


def load_all_results(result_dirs):
    rows = []
    custom_objects = get_custom_objects()

    for input_bits, d in result_dirs.items():
        csv_path = os.path.join(d, "roc_metrics_summary.csv")
        if not os.path.exists(csv_path):
            print(f"[warn] missing {csv_path}")
            continue

        metrics = pd.read_csv(csv_path)

        for _, row in metrics.iterrows():
            tid = str(row["trial_id"]).zfill(2)
            h5_path = os.path.join(d, f"model_trial_{tid}.h5")
            if not os.path.exists(h5_path):
                print(f"[warn] missing model file {h5_path}")
                continue

            print(f"  {input_bits}-bit  trial {tid}  counting parameters...", end=" ")
            n_params = count_parameters(h5_path)
            print(f"{n_params:,}")

            rows.append({
                "input_bits":       input_bits,
                "trial_id":         tid,
                "parameters":       n_params,
                "weighted_bkg_rej": row["weighted_bkg_rej"],
                "auc":              row["auc"],
            })

    return pd.DataFrame(rows)


def make_plot(data, output_path):
    fig, ax = plt.subplots(figsize=(11, 7))

    for bits, grp in data.groupby("input_bits"):
        ax.scatter(
            grp["parameters"], grp["weighted_bkg_rej"],
            color=COLORS[bits], marker=MARKERS[bits],
            s=80, alpha=0.85, edgecolors="black", linewidths=0.5,
            label=f"{bits}-bit input", zorder=3,
        )

    ax.set_xlabel("Total Model Parameters", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weighted Background Rejection", fontsize=13, fontweight="bold")
    ax.set_title(
        "Model2.5 Input Quantization Sweep\nTotal Parameters vs Weighted Background Rejection\n"
        "(6-bit weights/biases, 8-bit activations)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Input Quantization", fontsize=11, title_fontsize=11,
              loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--", zorder=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


def print_summary(data):
    print("\n" + "=" * 55)
    print("Best model per input bit-width (weighted_bkg_rej):")
    print("=" * 55)
    for bits, grp in data.groupby("input_bits"):
        best = grp.loc[grp["weighted_bkg_rej"].idxmax()]
        print(f"  {bits:2d}-bit:  trial {best['trial_id']}  "
              f"params={int(best['parameters']):,}  "
              f"weighted_bkg_rej={best['weighted_bkg_rej']:.4f}  "
              f"AUC={best['auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Parameters vs weighted bkg rej for input bits sweep")
    parser.add_argument("--output", default="input_bits_sweep_params_vs_bkgrej.png")
    args = parser.parse_args()

    print("Loading results and counting parameters...")
    data = load_all_results(RESULT_DIRS)

    if data.empty:
        print("No data found — check RESULT_DIRS.")
        return

    print_summary(data)
    make_plot(data, args.output)


if __name__ == "__main__":
    main()
