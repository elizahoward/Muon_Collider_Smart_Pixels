"""
Bit-width sweep across Model1, Model2, and Model3.

Trains each model at fixed (reasonable mid-range) hyperparameters:
  - Unquantized (Float32)
  - Quantized at 2, 4, 6, 8, 10 bits

For 30 epochs each, then produces 6 plots (2 per model):
  1. Validation accuracy curves over epochs  (line plot)
  2. Best validation accuracy bar chart

Run from the eric/ directory:
    conda activate mlgpu_qkeras
    python run_bit_sweep_all_models.py [--epochs N] [--output-dir DIR]
"""

import argparse
import os
import sys
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

sys.path.insert(0, os.path.join(_ROOT, "MuC_Smartpix_ML"))
sys.path.insert(0, os.path.join(_ROOT, "MuC_Smartpix_Data_Production", "tfRecords"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "ryan"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate

from qkeras import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

from model2_5 import Model2_5
from model3 import Model3
from model1 import Model1

# ── constants ─────────────────────────────────────────────────────────────────
BIT_CONFIGS = [(2, 0), (4, 0), (6, 0), (8, 0), (10, 0)]
DEFAULT_EPOCHS = 30

DATA_V2_APR = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData"
)
DATA_V2_APR_NORM = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData_normalized"
)

# Colors matching the reference image style
_BIT_COLORS = {
    2:  "#66c2a5",  # teal-green
    4:  "#fc8d62",  # orange
    6:  "#8da0cb",  # medium blue
    8:  "#e78ac3",  # pink
    10: "#a6d854",  # yellow-green
}
_UNQ_COLOR = "#000000"    # black  (non-quantized)
_QBAR_COLOR = "#8b0000"   # dark crimson (quantized bars)
_FP32_COLOR = "#7f8c8d"   # slate-gray (Float32 bar)


# ── Model1 builders ───────────────────────────────────────────────────────────
# Model1's makeUnquantizedModel uses only 2 units per layer, which is
# inconsistent with its quantized architecture (17-20-9-16-8).  We build both
# versions here with the same layer sizes so the comparison is apples-to-apples.

def _build_m1_unquantized(m1: Model1) -> None:
    i1 = Input(shape=(1,), name="z_global")
    i2 = Input(shape=(1,), name="x_size")
    i3 = Input(shape=(1,), name="y_size")
    i4 = Input(shape=(1,), name="y_local")
    x = Concatenate()([i1, i2, i3, i4])
    for k, units in enumerate([17, 20, 9, 16, 8]):
        x = Dense(units, activation="relu", name=f"dense{k+1}")(x)
    out = Dense(1, activation="sigmoid", name="output")(x)
    m1.models["Unquantized"] = tf.keras.Model(
        inputs=[i1, i2, i3, i4], outputs=out, name="model1_unquantized"
    )


def _build_m1_quantized(m1: Model1, weight_bits: int, int_bits: int) -> None:
    config = f"quantized_{weight_bits}w{int_bits}i"
    wq = quantized_bits(weight_bits, int_bits, alpha=1)
    aq = quantized_relu(8, 0)
    i1 = Input(shape=(1,), name="z_global")
    i2 = Input(shape=(1,), name="x_size")
    i3 = Input(shape=(1,), name="y_size")
    i4 = Input(shape=(1,), name="y_local")
    x = Concatenate()([i1, i2, i3, i4])
    for k, units in enumerate([17, 20, 9, 16, 8]):
        x = QDense(units, kernel_quantizer=wq, bias_quantizer=wq, name=f"qdense{k+1}")(x)
        x = QActivation(activation=aq, name=f"q_relu{k+1}")(x)
    x = QDense(1, kernel_quantizer=wq, bias_quantizer=wq, name="output_dense")(x)
    out = QActivation("quantized_sigmoid(8,0)", name="output_activation")(x)
    m1.models[config] = tf.keras.Model(
        inputs=[i1, i2, i3, i4], outputs=out, name=f"model1_{config}"
    )


# ── Checkpoint helpers --------------------------------------------------------

def _save_config_checkpoint(model_instance, config_name, val_acc_list, output_dir):
    """Save model weights + per-config history JSON immediately after training."""
    import json
    model_name = model_instance.modelName
    stem = f"{model_name}_{config_name}"

    # weights
    weights_path = os.path.join(output_dir, f"{stem}.weights.h5")
    model_instance.models[config_name].save_weights(weights_path)

    # per-config history
    hist_path = os.path.join(output_dir, f"{stem}_history.json")
    with open(hist_path, "w") as f:
        json.dump({"val_binary_accuracy": val_acc_list}, f, indent=2)

    print(f"  [checkpoint] saved weights → {weights_path}")
    print(f"  [checkpoint] saved history → {hist_path}")


def _update_master_json(all_histories, output_dir):
    """Overwrite histories.json with everything collected so far."""
    import json
    out_path = os.path.join(output_dir, "histories.json")
    with open(out_path, "w") as f:
        json.dump(all_histories, f, indent=2)


# ── Training ------------------------------------------------------------------

def train_all_configs(model_instance, bit_configs, num_epochs, output_dir, all_histories):
    """Load data once, train each config, and checkpoint after every one.

    Skips any config whose history is already present in all_histories so that
    a crashed run can be resumed by re-running with the same --output-dir.
    """
    model_name = model_instance.modelName
    if model_name not in all_histories:
        all_histories[model_name] = {}
    histories = all_histories[model_name]

    # Build the ordered list of (config_name, is_quantized) pairs
    all_configs = [("Unquantized", False)] + [
        (f"quantized_{wb}w{ib}i", True) for wb, ib in bit_configs
    ]

    pending = [(cfg, qk) for cfg, qk in all_configs if cfg not in histories]
    already_done = [cfg for cfg, _ in all_configs if cfg in histories]

    if already_done:
        print(f"  [resume] {model_name}: skipping already-completed configs: {already_done}")
    if not pending:
        print(f"  [resume] {model_name}: all configs done — skipping training entirely.")
        return histories

    # Only load data if there is something left to train
    if model_instance.training_generator is None:
        model_instance.loadTfRecords()

    for config, is_quantized in pending:
        label = "Unquantized (Float32)" if config == "Unquantized" else f"{config.split('_')[1].replace('w0i','')}-bit quantized"
        print(f"\n{'='*60}")
        print(f"  {model_name} — {label}")
        print(f"{'='*60}")
        h = model_instance.trainModel(
            epochs=num_epochs,
            config_name=config,
            early_stopping_patience=0,
            save_best=False,
            run_eagerly=is_quantized,
        )
        histories[config] = h.history["val_binary_accuracy"]
        _save_config_checkpoint(model_instance, config, histories[config], output_dir)
        _update_master_json(all_histories, output_dir)

    return histories


# ── Plotting ------------------------------------------------------------------

def _setup_ax(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#cccccc", alpha=0.8, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")


def plot_val_acc_curves(histories, bit_configs, model_name, output_path):
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("white")
    _setup_ax(ax)

    ax.plot(
        histories["Unquantized"],
        color=_UNQ_COLOR, lw=2.8, zorder=10,
        label="Non-quantized",
    )
    for wb, ib in bit_configs:
        config = f"quantized_{wb}w{ib}i"
        ax.plot(
            histories[config],
            color=_BIT_COLORS[wb], lw=1.8, zorder=5,
            label=f"{wb}-bit",
        )

    ax.set_title(
        f"Validation Accuracy Comparison\n{model_name}",
        fontsize=16, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Accuracy", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="lower right", framealpha=0.9, edgecolor="#aaaaaa")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_best_val_acc_bar(histories, bit_configs, model_name, output_path):
    labels = ["Float32"] + [f"{wb}-bit" for wb, ib in bit_configs]
    best_vals = [max(histories["Unquantized"])]
    for wb, ib in bit_configs:
        best_vals.append(max(histories[f"quantized_{wb}w{ib}i"]))

    colors = [_FP32_COLOR] + [_QBAR_COLOR] * len(bit_configs)
    val_range = max(best_vals) - min(best_vals) if max(best_vals) != min(best_vals) else 0.01

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    _setup_ax(ax)

    bars = ax.bar(labels, best_vals, color=colors, width=0.6, zorder=3)

    for bar, val, lbl in zip(bars, best_vals, labels):
        txt_color = "#555555" if lbl == "Float32" else _QBAR_COLOR
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + val_range * 0.04,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=txt_color,
        )

    y_floor = max(0.0, min(best_vals) - val_range * 0.8)
    y_ceil = max(best_vals) + val_range * 0.25
    ax.set_ylim(y_floor, y_ceil)

    ax.set_title(
        f"{model_name}: Best Validation Accuracy",
        fontsize=16, fontweight="bold", pad=12,
    )
    ax.set_ylabel("Best Validation Accuracy", fontsize=14)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_histories_json(all_histories, output_dir):
    _update_master_json(all_histories, output_dir)
    print(f"  Saved raw histories: {os.path.join(output_dir, 'histories.json')}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bit-width sweep for Model1/2/3")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs per config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: auto-timestamped)")
    parser.add_argument(
        "--models", type=str, default="1,2,3",
        help="Comma-separated list of models to run, e.g. '1,3' (default: '1,2,3')"
    )
    args = parser.parse_args()

    run_models = {int(m.strip()) for m in args.models.split(",")}
    num_epochs = args.epochs

    import json

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"bit_sweep_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutputs: {output_dir}/")
    print(f"Epochs:  {num_epochs}")
    print(f"Bits:    {[wb for wb, _ in BIT_CONFIGS]}")

    # Resume: load any histories already saved in this output directory
    histories_path = os.path.join(output_dir, "histories.json")
    if os.path.exists(histories_path):
        with open(histories_path) as f:
            all_histories = json.load(f)
        done = {m: list(h.keys()) for m, h in all_histories.items()}
        print(f"[resume] Found existing histories.json — already completed: {done}")
    else:
        all_histories = {}

    # ── Model 1 ───────────────────────────────────────────────────────────────
    if 1 in run_models:
        print("\n" + "="*70)
        print("MODEL 1 — 5-layer dense | z_global, x_size, y_size, y_local")
        print("  Architecture: Dense(17)-Dense(20)-Dense(9)-Dense(16)-Dense(8)")
        print("="*70)

        m1 = Model1(tfRecordFolder=DATA_V2_APR, bit_configs=BIT_CONFIGS)
        _build_m1_unquantized(m1)
        for wb, ib in BIT_CONFIGS:
            _build_m1_quantized(m1, wb, ib)

        h1 = train_all_configs(m1, BIT_CONFIGS, num_epochs, output_dir, all_histories)

        plot_val_acc_curves(h1, BIT_CONFIGS, "Model1",
                            os.path.join(output_dir, "model1_val_acc_curves.png"))
        plot_best_val_acc_bar(h1, BIT_CONFIGS, "Model1",
                              os.path.join(output_dir, "model1_best_val_acc_bar.png"))

    # ── Model 2.5 ─────────────────────────────────────────────────────────────
    if 2 in run_models:
        print("\n" + "="*70)
        print("MODEL 2.5 — 2-branch dense | x_profile, nModule, x_local, y_profile, y_local")
        print("  Architecture: Spatial-branch(128) + nModule_xlocal-branch(32) → Dense(128,64)")
        print("="*70)

        m2 = Model2_5(
            tfRecordFolder=DATA_V2_APR_NORM,
            bit_configs=BIT_CONFIGS,
            dense_units=128,
            nmodule_xlocal_units=32,
            dense2_units=128,
            dense3_units=64,
            dropout_rate=0.1,
            initial_lr=1e-3,
            end_lr=1e-4,
            nmodule_xlocal_weight_bits=8,
            nmodule_xlocal_int_bits=0,
        )
        m2.makeUnquantizedModel()
        m2.makeQuantizedModel()

        h2 = train_all_configs(m2, BIT_CONFIGS, num_epochs, output_dir, all_histories)

        plot_val_acc_curves(h2, BIT_CONFIGS, "Model2.5",
                            os.path.join(output_dir, "model2_5_val_acc_curves.png"))
        plot_best_val_acc_bar(h2, BIT_CONFIGS, "Model2.5",
                              os.path.join(output_dir, "model2_5_best_val_acc_bar.png"))

    # ── Model 3 ───────────────────────────────────────────────────────────────
    if 3 in run_models:
        print("\n" + "="*70)
        print("MODEL 3 — Conv2D + scalar | cluster, nModule, x_local, y_local")
        print("  Architecture: Conv2D(32,3x5) + Scalar-Dense(32) → Dense(200,100)")
        print("="*70)

        m3 = Model3(
            tfRecordFolder=DATA_V2_APR_NORM,
            bit_configs=BIT_CONFIGS,
            conv_filters=32,
            kernel_rows=3,
            kernel_cols=5,
            scalar_dense_units=32,
            merged_dense_1=200,
            merged_dense_2=100,
            dropout_rate=0.0,
            initial_lr=0.000871145,
            end_lr=5.3e-5,
        )
        m3.makeUnquantizedModel()
        m3.makeQuantizedModel()

        h3 = train_all_configs(m3, BIT_CONFIGS, num_epochs, output_dir, all_histories)

        plot_val_acc_curves(h3, BIT_CONFIGS, "Model3",
                            os.path.join(output_dir, "model3_val_acc_curves.png"))
        plot_best_val_acc_bar(h3, BIT_CONFIGS, "Model3",
                              os.path.join(output_dir, "model3_best_val_acc_bar.png"))

    # ── Save raw histories + print summary ────────────────────────────────────
    save_histories_json(all_histories, output_dir)

    print("\n" + "="*70)
    print("BIT SWEEP COMPLETE — Summary")
    print("="*70)
    for mname, hist in all_histories.items():
        best_unq = max(hist["Unquantized"])
        print(f"\n{mname}:")
        print(f"  Float32 : {best_unq:.4f}")
        for wb, ib in BIT_CONFIGS:
            config = f"quantized_{wb}w{ib}i"
            best_q = max(hist[config])
            pct = (best_unq - best_q) / best_unq * 100
            print(f"  {wb:2d}-bit  : {best_q:.4f}  ({pct:+.1f}% vs Float32)")

    print(f"\nAll plots saved to: {output_dir}/\n")


if __name__ == "__main__":
    main()
