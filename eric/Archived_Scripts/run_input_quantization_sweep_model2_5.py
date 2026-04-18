"""
Input-Quantization Sweep for Model2.5
======================================

This script has two main entry points:

  1. extract_model2_5_architecture(h5_path)
        Load a saved Model2.5 h5 file and return:
          - arch_params  : dict of constructor kwargs  (dense_units, etc.)
          - layer_table  : list of dicts describing every layer
          - weight_bits  : inferred weight-quantisation width (or None if unquantised)

  2. run_input_quantization_sweep(h5_path, ...)
        Rebuild the Model2.5 architecture from the h5, then train one fully-
        quantised model per input-bit width in [2, 3, 4, 6, 8, 12, 16] for
        80 epochs each (overrideable).  Results (ROC, background rejection,
        training history) are saved to a timestamped output directory.

Usage
-----
    python run_input_quantization_sweep_model2_5.py \
        --h5 /path/to/model_trial_003.h5

Or import the two functions directly in a notebook / script.

Author: Eric
Date:   2026
"""

import sys
import os
import argparse
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import pandas as pd

# ── project path setup ────────────────────────────────────────────────────────
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/')
sys.path.append('../MuC_Smartpix_ML/')

import tensorflow as tf

# QKeras (needed for loading quantised h5 files and for the sweep models)
try:
    import qkeras
    from qkeras.utils import load_qmodel
    QKERAS_AVAILABLE = True
except ImportError:
    print("⚠  QKeras not found – quantised models cannot be loaded / built.")
    QKERAS_AVAILABLE = False

from sklearn.metrics import roc_curve, auc as sklearn_auc


# ══════════════════════════════════════════════════════════════════════════════
# ── 1.  Architecture extraction ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Layer names that appear in every Model2.5 variant (quantised or not).
# Maps layer name  →  Model2_5_QuantizedInputs constructor kwarg
_MODEL2_5_LAYER_MAP = {
    "other_dense":            "dense_units",
    "nmodule_xlocal_dense":   "nmodule_xlocal_units",
    "dense2":                 "dense2_units",
    "dense3":                 "dense3_units",
}


def extract_model2_5_architecture(h5_path: str, verbose: bool = True):
    """
    Load a saved Model2.5 h5 file and extract its architecture.

    Parameters
    ----------
    h5_path : str
        Path to the .h5 model file.
    verbose : bool
        If True, print a formatted layer table.

    Returns
    -------
    arch_params : dict
        Constructor kwargs for Model2_5_QuantizedInputs:
            dense_units, nmodule_xlocal_units, dense2_units, dense3_units,
            nmodule_xlocal_weight_bits  (inferred; None when unquantised),
            weight_bits                 (same, for the main branches)
    layer_table : list[dict]
        One dict per Keras layer with keys:
            name, type, units (if applicable), input_shape, output_shape,
            trainable_params, quantizer (if QDense/QActivation)
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"h5 file not found: {h5_path}")

    print(f"\nLoading model from: {h5_path}")

    # ── load ──────────────────────────────────────────────────────────────────
    if QKERAS_AVAILABLE:
        try:
            model = load_qmodel(h5_path)
        except Exception:
            # Fallback: plain Keras loader with QKeras custom objects
            model = tf.keras.models.load_model(
                h5_path, custom_objects=qkeras.get_custom_objects()
            )
    else:
        model = tf.keras.models.load_model(h5_path)

    # ── iterate over layers ───────────────────────────────────────────────────
    layer_table = []
    arch_params = {
        "dense_units":            None,
        "nmodule_xlocal_units":   None,
        "dense2_units":           None,
        "dense3_units":           None,
        "weight_bits":            None,   # inferred from main dense branches
        "nmodule_xlocal_weight_bits": None,
    }

    for layer in model.layers:
        layer_type = type(layer).__name__
        info = {
            "name":             layer.name,
            "type":             layer_type,
            "units":            None,
            "input_shape":      None,
            "output_shape":     None,
            "trainable_params": layer.count_params(),
            "quantizer":        None,
        }

        # ── output shape ──────────────────────────────────────────────────────
        try:
            info["output_shape"] = str(layer.output_shape)
        except Exception:
            pass

        # ── input shape ───────────────────────────────────────────────────────
        try:
            info["input_shape"] = str(layer.input_shape)
        except Exception:
            pass

        # ── units (Dense / QDense) ────────────────────────────────────────────
        if hasattr(layer, "units"):
            info["units"] = int(layer.units)

            # map to constructor kwarg
            if layer.name in _MODEL2_5_LAYER_MAP:
                arch_params[_MODEL2_5_LAYER_MAP[layer.name]] = int(layer.units)

            # infer weight-bit width from kernel_quantizer string
            if hasattr(layer, "kernel_quantizer") and layer.kernel_quantizer is not None:
                q_str = str(layer.kernel_quantizer)
                info["quantizer"] = q_str
                # parse "quantized_bits(N, ...)" → N
                try:
                    bits_str = q_str.split("(")[1].split(",")[0].strip()
                    bits = int(bits_str)
                    if layer.name == "nmodule_xlocal_dense":
                        arch_params["nmodule_xlocal_weight_bits"] = bits
                    elif layer.name in ("other_dense", "dense2", "dense3", "output"):
                        arch_params["weight_bits"] = bits
                except Exception:
                    pass

        # ── QActivation quantizer string ──────────────────────────────────────
        if layer_type == "QActivation" and hasattr(layer, "activation"):
            info["quantizer"] = str(layer.activation)

        layer_table.append(info)

    # ── pretty-print ──────────────────────────────────────────────────────────
    if verbose:
        _print_layer_table(layer_table, arch_params)

    return arch_params, layer_table


def _print_layer_table(layer_table: list, arch_params: dict):
    """Print a formatted summary of the extracted layer table."""
    col_w = [32, 18, 8, 10, 12, 36]
    header = ["Layer name", "Type", "Units", "Params", "Output shape", "Quantizer"]
    sep    = "  ".join("-" * w for w in col_w)

    print("\n" + "=" * sum(col_w + [2 * (len(col_w) - 1)]))
    print("  Model2.5 Architecture")
    print("=" * sum(col_w + [2 * (len(col_w) - 1)]))
    print("  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print(sep)

    for info in layer_table:
        units  = str(info["units"]) if info["units"] is not None else "—"
        params = str(info["trainable_params"])
        q      = (info["quantizer"] or "—")[:col_w[5]]
        out_sh = (info["output_shape"] or "—")[:col_w[4]]
        row    = [info["name"], info["type"], units, params, out_sh, q]
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    print(sep)
    total = sum(d["trainable_params"] for d in layer_table)
    print(f"  Total trainable parameters: {total:,}")
    print()
    print("  Extracted constructor kwargs:")
    for k, v in arch_params.items():
        print(f"    {k:35s} = {v}")
    print("=" * sum(col_w + [2 * (len(col_w) - 1)]))


# ══════════════════════════════════════════════════════════════════════════════
# ── 2.  Input-quantisation sweep ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT_BIT_WIDTHS = [2, 3, 4, 6, 8, 12, 16]
DEFAULT_NUM_EPOCHS       = 80
DEFAULT_DATA_FOLDER      = (
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/"
    "Data_Files/Data_Set_2026V2_Apr/TF_Records/filtering_records16384_data_shuffled_single_bigData"
)


def _bkg_rej_at_eff(y_true, y_score, target_eff: float) -> float:
    """Background rejection (1 – FPR) at fixed signal efficiency."""
    y_true  = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    sig_scores = y_score[y_true == 1]
    bkg_scores = y_score[y_true == 0]
    if len(sig_scores) == 0 or len(bkg_scores) == 0:
        return float("nan")
    threshold = np.quantile(sig_scores, 1.0 - target_eff)
    fpr = float(np.mean(bkg_scores >= threshold))
    return 1.0 - fpr


def run_input_quantization_sweep(
    h5_path: str,
    input_bit_widths: list = None,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    data_folder: str = DEFAULT_DATA_FOLDER,
    weight_bits_override: int = None,
    int_bits: int = 0,
    input_int_bits: int = 0,
    output_dir: str = None,
    early_stopping_patience: int = 15,
    initial_lr: float = 1e-3,
    end_lr: float = 1e-4,
):
    """
    Rebuild the Model2.5 architecture from *h5_path*, then train one fully-
    quantised model per value in *input_bit_widths* and compare performance.

    Parameters
    ----------
    h5_path : str
        Path to a reference Model2.5 h5 file.  The architecture (layer unit
        counts) is extracted from this file and used for every sweep run.
    input_bit_widths : list[int]
        Input quantisation widths to test. Default: [2, 3, 4, 6, 8, 12, 16].
    num_epochs : int
        Training epochs per trial (default 80).
    data_folder : str
        Path to the TFRecords directory.
    weight_bits_override : int or None
        Override the weight-bit width inferred from the h5 file.
    int_bits : int
        Integer bits for weight quantization (default 0).
    input_int_bits : int
        Integer bits for input quantization (default 0).
    output_dir : str or None
        Directory to store results.  Auto-named with timestamp if None.
    early_stopping_patience : int
        Patience for early stopping on val_loss (0 to disable).
    initial_lr, end_lr : float
        Learning-rate schedule endpoints.

    Returns
    -------
    results_df : pd.DataFrame
        One row per input-bit width with columns:
        input_bits, auc, bkg_rej_95, bkg_rej_98, bkg_rej_99,
        best_val_accuracy, num_epochs_trained.
    """
    # ── lazy import (avoids TF startup on bare import of this module) ─────────
    from model2_5_quantized_inputs import Model2_5_QuantizedInputs

    if input_bit_widths is None:
        input_bit_widths = DEFAULT_INPUT_BIT_WIDTHS

    # ── step 1: extract architecture from h5 ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  Step 1 – Extracting Model2.5 architecture from h5")
    print("=" * 70)
    arch_params, layer_table = extract_model2_5_architecture(h5_path, verbose=True)

    # fill in any missing keys with safe defaults
    dense_units          = arch_params.get("dense_units")          or 64
    nmodule_xlocal_units = arch_params.get("nmodule_xlocal_units") or 8
    dense2_units         = arch_params.get("dense2_units")         or 32
    dense3_units         = arch_params.get("dense3_units")         or 16
    nm_bits              = arch_params.get("nmodule_xlocal_weight_bits") or 6
    inferred_weight_bits = arch_params.get("weight_bits")          or 6
    weight_bits = weight_bits_override if weight_bits_override is not None else inferred_weight_bits

    print(f"\nUsing weight_bits = {weight_bits}  (input_int_bits = {input_int_bits})")
    print(f"Sweep:  input_bit_widths = {input_bit_widths}")
    print(f"Epochs per trial: {num_epochs}")

    # ── step 2: create output directory ──────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f"model2_5_input_quant_sweep_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    plots_dir  = os.path.join(output_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}/")

    # ── step 3: sweep ─────────────────────────────────────────────────────────
    sweep_results = []
    roc_rows      = []   # for combined ROC plot

    for ib in input_bit_widths:
        print("\n" + "=" * 70)
        print(f"  input_bits = {ib}  ({weight_bits}-bit weights, {int_bits} int-bits)")
        print("=" * 70)

        config_name = f"quantized_{weight_bits}w{int_bits}i"

        # ── build model ───────────────────────────────────────────────────────
        model_obj = Model2_5_QuantizedInputs(
            tfRecordFolder=data_folder,
            dense_units=dense_units,
            nmodule_xlocal_units=nmodule_xlocal_units,
            dense2_units=dense2_units,
            dense3_units=dense3_units,
            dropout_rate=0.05,
            initial_lr=initial_lr,
            end_lr=end_lr,
            power=2,
            bit_configs=[(weight_bits, int_bits)],
            nmodule_xlocal_weight_bits=nm_bits,
            nmodule_xlocal_int_bits=int_bits,
            input_bits=ib,
            input_int_bits=input_int_bits,
        )

        # ── load data ─────────────────────────────────────────────────────────
        model_obj.loadTfRecords()

        # ── build quantised variant ───────────────────────────────────────────
        model_obj.makeQuantizedModel()

        # ── train ─────────────────────────────────────────────────────────────
        model_obj.trainModel(
            epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            save_best=False,
            run_eagerly=True,
            config_name=config_name,
            clipnorm=1.0,
        )

        history = model_obj.histories[config_name]

        # ── evaluate: ROC + background rejection ─────────────────────────────
        print(f"\nEvaluating input_bits={ib} ...")
        keras_model = model_obj.models[config_name]

        y_true_list, y_pred_list = [], []
        for x_batch, y_batch in model_obj.validation_generator:
            preds = keras_model.predict(x_batch, verbose=0).ravel()
            y_true_list.append(np.asarray(y_batch).ravel())
            y_pred_list.append(preds)

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        n      = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]

        fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
        roc_auc     = sklearn_auc(fpr, tpr)
        br95        = _bkg_rej_at_eff(y_true, y_pred, 0.95)
        br98        = _bkg_rej_at_eff(y_true, y_pred, 0.98)
        br99        = _bkg_rej_at_eff(y_true, y_pred, 0.99)

        best_val_acc = max(history.history.get("val_binary_accuracy", [float("nan")]))
        epochs_trained = len(history.history.get("val_binary_accuracy", []))

        row = {
            "input_bits":         ib,
            "weight_bits":        weight_bits,
            "auc":                float(roc_auc),
            "bkg_rej_95":         float(br95),
            "bkg_rej_98":         float(br98),
            "bkg_rej_99":         float(br99),
            "best_val_accuracy":  float(best_val_acc),
            "epochs_trained":     epochs_trained,
        }
        sweep_results.append(row)
        roc_rows.append({"input_bits": ib, "fpr": fpr, "tpr": tpr, "auc": roc_auc})

        print(f"  AUC={roc_auc:.4f}  BR95={br95:.4f}  BR98={br98:.4f}  BR99={br99:.4f}")

        # ── save this model ───────────────────────────────────────────────────
        model_filename = os.path.join(models_dir, f"model_input_{ib}bit.h5")
        keras_model.save(model_filename)
        print(f"  ✓ Model saved → {model_filename}")

        # ── per-trial ROC plot ────────────────────────────────────────────────
        _save_roc_plot(
            fpr, tpr, roc_auc, ib, weight_bits,
            path=os.path.join(plots_dir, f"roc_input_{ib}bit.png")
        )

        # ── training-history plot ─────────────────────────────────────────────
        _save_history_plot(
            history, ib,
            path=os.path.join(plots_dir, f"history_input_{ib}bit.png")
        )

        # ── save per-trial ROC csv ────────────────────────────────────────────
        roc_csv = os.path.join(plots_dir, f"roc_data_input_{ib}bit.csv")
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(roc_csv, index=False)

        # free RAM between trials
        del model_obj
        tf.keras.backend.clear_session()

    # ── step 4: combined outputs ──────────────────────────────────────────────
    results_df = pd.DataFrame(sweep_results)
    results_csv = os.path.join(output_dir, "sweep_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Sweep results saved → {results_csv}")

    _save_combined_roc_plot(
        roc_rows, path=os.path.join(output_dir, "roc_combined.png")
    )
    _save_metric_vs_bits_plot(
        results_df, output_dir=output_dir
    )

    # ── save architecture snapshot ────────────────────────────────────────────
    arch_snapshot = {
        "source_h5":             h5_path,
        "dense_units":           dense_units,
        "nmodule_xlocal_units":  nmodule_xlocal_units,
        "dense2_units":          dense2_units,
        "dense3_units":          dense3_units,
        "weight_bits":           weight_bits,
        "int_bits":              int_bits,
        "input_int_bits":        input_int_bits,
        "input_bit_widths_swept": input_bit_widths,
        "num_epochs":            num_epochs,
        "layer_table":           layer_table,
    }
    with open(os.path.join(output_dir, "architecture_snapshot.json"), "w") as f:
        json.dump(arch_snapshot, f, indent=4)

    print("\n" + "=" * 70)
    print("  Sweep complete – summary")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nAll outputs in: {output_dir}/")
    print("=" * 70)

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# ── Plotting helpers ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _save_roc_plot(fpr, tpr, roc_auc, input_bits, weight_bits, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate",  fontsize=12)
    ax.set_ylabel("True Positive Rate",   fontsize=12)
    ax.set_title(f"ROC – input {input_bits}-bit  |  weights {weight_bits}-bit",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ ROC plot saved → {path}")


def _save_combined_roc_plot(roc_rows, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(roc_rows)))
    for color, row in zip(colors, sorted(roc_rows, key=lambda r: r["input_bits"])):
        ax.plot(row["fpr"], row["tpr"], lw=2, color=color, alpha=0.85,
                label=f"{row['input_bits']:2d}-bit input  (AUC={row['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3, label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate",  fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate",   fontsize=13, fontweight="bold")
    ax.set_title("ROC Curves – Input Quantisation Sweep (Model2.5)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Combined ROC plot saved → {path}")


def _save_history_plot(history, input_bits, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in zip(
        axes,
        [("binary_accuracy", "val_binary_accuracy"), ("loss", "val_loss")],
        [("Train acc", "Val acc"), ("Train loss", "Val loss")]
    ):
        train_key, val_key = metric
        if train_key in history.history:
            ax.plot(history.history[train_key], label=label[0])
        if val_key in history.history:
            ax.plot(history.history[val_key],   label=label[1])
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)
    axes[0].set_title(f"Accuracy – input {input_bits}-bit",  fontweight="bold")
    axes[1].set_title(f"Loss – input {input_bits}-bit",      fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_metric_vs_bits_plot(df: pd.DataFrame, output_dir: str):
    """Bar charts of AUC and background-rejection metrics vs input bit width."""
    metrics = [
        ("auc",        "AUC",                         "steelblue"),
        ("bkg_rej_95", "Bkg Rejection @ 95% sig eff", "tomato"),
        ("bkg_rej_98", "Bkg Rejection @ 98% sig eff", "seagreen"),
        ("bkg_rej_99", "Bkg Rejection @ 99% sig eff", "goldenrod"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, (col, title, color) in zip(axes, metrics):
        if col not in df.columns:
            continue
        vals = df[col].astype(float)
        bars = ax.bar([str(b) for b in df["input_bits"]], vals, color=color, alpha=0.85)
        ax.set_xlabel("Input bit width", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        y_min = max(0, vals.min() - 0.05)
        y_max = min(1, vals.max() + 0.05)
        ax.set_ylim([y_min, y_max])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model2.5 – Performance vs Input Quantisation Width",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_vs_input_bits.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Metric comparison plot saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI entry point ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Model2.5 input-quantisation sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--h5", required=True,
        help="Path to a reference Model2.5 .h5 file (architecture is extracted from it)"
    )
    parser.add_argument(
        "--bits", nargs="+", type=int,
        default=DEFAULT_INPUT_BIT_WIDTHS,
        metavar="B",
        help="Input bit widths to sweep, e.g. --bits 2 3 4 6 8 12 16"
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_NUM_EPOCHS,
        help="Training epochs per trial"
    )
    parser.add_argument(
        "--data_folder", type=str, default=DEFAULT_DATA_FOLDER,
        help="Path to TFRecords directory"
    )
    parser.add_argument(
        "--weight_bits", type=int, default=None,
        help="Override inferred weight-bit width (e.g. 6)"
    )
    parser.add_argument(
        "--int_bits", type=int, default=0,
        help="Integer bits for weight quantisation"
    )
    parser.add_argument(
        "--input_int_bits", type=int, default=0,
        help="Integer bits for input quantisation"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (auto-named if omitted)"
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early-stopping patience (0 = disabled)"
    )
    parser.add_argument(
        "--extract_only", action="store_true",
        help="Only extract and print the architecture; do not run training sweep"
    )
    args = parser.parse_args()

    if args.extract_only:
        extract_model2_5_architecture(args.h5, verbose=True)
        return

    run_input_quantization_sweep(
        h5_path              = args.h5,
        input_bit_widths     = args.bits,
        num_epochs           = args.epochs,
        data_folder          = args.data_folder,
        weight_bits_override = args.weight_bits,
        int_bits             = args.int_bits,
        input_int_bits       = args.input_int_bits,
        output_dir           = args.output_dir,
        early_stopping_patience = args.patience,
    )


if __name__ == "__main__":
    main()
