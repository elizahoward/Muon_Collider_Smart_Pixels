"""
run_int_bits_sweep.py

Sweep `input_int_bits` ∈ {0, 1, 2, 3, 4, 5} for both Model 2.5 (profiles) and
Model 3 (full cluster), train each as a *single* model (no HP search) for a
fixed number of epochs, and emit one summary table.

Why this exists
---------------
The diagnostic in eric/diagnostics/inspect_input_saturation.py shows that with
`quantized_bits(input_bits, 0)` (the current setup) every nonzero cluster pixel
AND every nonzero x_profile/y_profile entry saturates at the quantizer's
+0.998 ceiling. The model effectively learns from a binary mask. Increasing
`input_int_bits` extends the representable range to ±2**int_bits, which the
diagnostic predicts will recover real charge magnitudes (e.g. with int_bits=4,
cluster goes from 3 distinct quantized levels to 378). This script tests
whether that prediction translates into real per-model BR / AUC gains.

What this is NOT
----------------
- Not a hyperparameter search. Each (model, int_bits) cell is one model with
  fixed architecture HPs (we use the model class defaults, which match the
  best HP each model has produced in its own searches).
- Not a final fix; it's a controlled sweep to decide whether to invest in a
  data-pipeline / quantizer config change before retuning everything.

Usage
-----
    cd .../eric
    /home/youeric/miniconda3/envs/mlgpu_qkeras/bin/python \
        quantized_inputs_int_bits_test/run_int_bits_sweep.py \
        --epochs 50 --weight-bits 10

Outputs
-------
- One subfolder per cell at quantized_inputs_int_bits_test/results/<model>_int<N>/
  containing model history JSON, the trained .h5, ROC CSV, and metrics JSON.
- quantized_inputs_int_bits_test/sweep_summary.csv, the comparison table.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

# XLA needs to find CUDA libdevice. Use the running Python's prefix (this is
# the conda env we are actually executing in, not whichever one the parent
# shell has activated as CONDA_PREFIX). Must be set BEFORE importing tensorflow.
_LIBDEVICE_PARENT = sys.prefix
# Sanity check: bail loudly rather than silently fall back if the file isn't
# where we expect.
_expected_libdevice = os.path.join(
    _LIBDEVICE_PARENT, "nvvm", "libdevice", "libdevice.10.bc"
)
if not os.path.isfile(_expected_libdevice):
    print(f"[warn] libdevice.10.bc not found under {_LIBDEVICE_PARENT}/nvvm/libdevice/. "
          f"XLA may fail when training begins.", flush=True)
os.environ["XLA_FLAGS"] = (
    f"--xla_gpu_cuda_data_dir={_LIBDEVICE_PARENT}"
)
# GPU memory growth so this doesn't OOM the card before TF starts.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
ERIC_DIR = os.path.dirname(HERE)
sys.path.insert(0, ERIC_DIR)
sys.path.insert(0, os.path.join(ERIC_DIR, "..", "MuC_Smartpix_ML"))

DEFAULT_DATA_FOLDER = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData_normalized"
)

# Reasonable architecture defaults per model. These mirror what the HP searches
# have repeatedly converged to (Model 2.5 6w0i best: spatial=128; Model 3 10w0i
# best: conv=8, merged_dense_1=72). They are NOT tuned per int_bits cell — the
# point of this sweep is to isolate the input-quantization variable.
M25_DEFAULTS = dict(
    dense_units=128,
    nmodule_xlocal_units=8,
    dense2_units=64,
    dense3_units=32,
    dropout_rate=0.1,
    initial_lr=1e-3,
    end_lr=1e-4,
    power=2,
    nmodule_xlocal_weight_bits=10,
    nmodule_xlocal_int_bits=0,
)
M3_DEFAULTS = dict(
    conv_filters=8,
    kernel_rows=3,
    kernel_cols=3,
    scalar_dense_units=32,
    merged_dense_1=72,
    merged_dense_2=44,
    dropout_rate=0.1,
    initial_lr=8.71145e-4,
    end_lr=5.3e-5,
    power=2,
)


def report_tf_runtime(require_gpu: bool) -> None:
    """Print TF/CUDA runtime info and optionally enforce GPU presence."""
    print("=" * 78)
    print("TensorFlow runtime")
    print("=" * 78)
    print(f"  tf version       : {tf.__version__}")
    print(f"  built with CUDA  : {tf.test.is_built_with_cuda()}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"  physical GPUs    : {gpus}")
    if gpus:
        # Explicitly enabling memory growth avoids pre-allocating all VRAM.
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"  [warn] set_memory_growth failed for {gpu}: {e}", flush=True)
    else:
        print("  [warn] No GPUs detected; training will run on CPU.", flush=True)

    if require_gpu and not gpus:
        raise RuntimeError(
            "No TensorFlow GPU device detected. Check NVIDIA driver/CUDA runtime "
            "and rerun without --require-gpu only if CPU training is intended."
        )


def bkg_rej_at_eff(y_true: np.ndarray, y_score: np.ndarray, target_eff: float) -> float:
    """Background rejection (1 - FPR) at fixed signal efficiency."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    sig = y_score[y_true == 1]
    bkg = y_score[y_true == 0]
    if sig.size == 0 or bkg.size == 0:
        return float("nan")
    threshold = np.quantile(sig, 1.0 - target_eff)
    fpr = float(np.mean(bkg >= threshold))
    return 1.0 - fpr


def collect_validation_labels(generator) -> np.ndarray:
    """Concatenate all validation labels into one array (matches the existing
    ROC-evaluation idiom in model3.py)."""
    return np.concatenate(
        [np.asarray(y).ravel() for _, y in generator],
        axis=0,
    )


def build_model25(data_folder: str, weight_bits: int, input_bits: int,
                  input_int_bits: int):
    from model2_5_quantized_inputs import Model2_5_QuantizedInputs

    m = Model2_5_QuantizedInputs(
        tfRecordFolder=data_folder,
        bit_configs=[(weight_bits, 0)],
        input_bits=input_bits,
        input_int_bits=input_int_bits,
        **M25_DEFAULTS,
    )
    m.loadTfRecords()
    m.makeQuantizedModel()
    config_name = f"quantized_{weight_bits}w0i"
    return m, m.models[config_name], config_name


def build_model3(data_folder: str, weight_bits: int, input_bits: int,
                 input_int_bits: int):
    from model3_quantized_inputs import Model3_QuantizedInputs

    m = Model3_QuantizedInputs(
        tfRecordFolder=data_folder,
        input_bits=input_bits,
        input_int_bits=input_int_bits,
        **M3_DEFAULTS,
    )
    # Restrict Model 3 to a single weight-bit config (it iterates over
    # self.bit_configs in makeQuantizedModel and we only want one config per
    # cell to keep the sweep tractable).
    m.bit_configs = [(weight_bits, 0)]
    m.loadTfRecords()
    m.makeQuantizedModel()
    config_name = f"quantized_{weight_bits}w0i"
    return m, m.models[config_name], config_name


def train_and_evaluate(model_obj, model, run_dir: str, epochs: int,
                       weights_for_obj: dict[float, float]) -> dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]

    history = model.fit(
        model_obj.training_generator,
        validation_data=model_obj.validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Save history.
    hist_path = os.path.join(run_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    # Save model.
    model.save(os.path.join(run_dir, "model.h5"))

    # ROC + BR metrics on validation.
    y_true = collect_validation_labels(model_obj.validation_generator)
    y_pred = model.predict(model_obj.validation_generator, verbose=0).ravel()
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    fpr, tpr, thr = roc_curve(y_true, y_pred, drop_intermediate=False)
    roc_auc = float(auc(fpr, tpr))
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thr}).to_csv(
        os.path.join(run_dir, "roc.csv"), index=False
    )

    br95 = bkg_rej_at_eff(y_true, y_pred, 0.95)
    br98 = bkg_rej_at_eff(y_true, y_pred, 0.98)
    br99 = bkg_rej_at_eff(y_true, y_pred, 0.99)
    weighted_br = (
        weights_for_obj.get(0.95, 0.0) * br95
        + weights_for_obj.get(0.98, 0.0) * br98
        + weights_for_obj.get(0.99, 0.0) * br99
    )

    final_val_loss = float(history.history.get("val_loss", [float("nan")])[-1])
    final_val_acc = float(history.history.get("val_binary_accuracy", [float("nan")])[-1])

    metrics = dict(
        auc=roc_auc,
        bkg_rej_95=float(br95),
        bkg_rej_98=float(br98),
        bkg_rej_99=float(br99),
        weighted_bkg_rej=float(weighted_br),
        final_val_loss=final_val_loss,
        final_val_binary_accuracy=final_val_acc,
        epochs_trained=len(history.history.get("val_loss", [])),
    )
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weight-bits", dest="weight_bits", type=int, default=10,
                        help="Weight quantization bits (int_bits = 0).")
    parser.add_argument("--int-bits", dest="int_bits_list", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5],
                        help="List of input_int_bits values to sweep.")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["model25", "model3"],
                        choices=["model25", "model3"],
                        help="Which model classes to sweep.")
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(HERE, "results"))
    parser.add_argument("--br95", type=float, default=0.1)
    parser.add_argument("--br98", type=float, default=0.7)
    parser.add_argument("--br99", type=float, default=0.2)
    parser.add_argument("--input-bits", type=int, default=None,
                        help="input_bits for the QActivation. Defaults to "
                             "weight_bits + 2 (matches the runner scripts).")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail fast if TensorFlow cannot see at least one GPU.",
    )
    args = parser.parse_args()
    report_tf_runtime(require_gpu=args.require_gpu)

    weights_for_obj = {0.95: args.br95, 0.98: args.br98, 0.99: args.br99}
    input_bits = args.input_bits if args.input_bits is not None else args.weight_bits + 2

    rows: list[dict[str, Any]] = []
    summary_path = os.path.join(HERE, "sweep_summary.csv")
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 78)
    print(f"int_bits sweep starting at {started}")
    print(f"  data_folder : {args.data_folder}")
    print(f"  epochs      : {args.epochs}")
    print(f"  weight bits : {args.weight_bits}w0i")
    print(f"  input bits  : {input_bits} (int_bits sweeps over {args.int_bits_list})")
    print(f"  models      : {args.models}")
    print(f"  weights     : BR95={args.br95}, BR98={args.br98}, BR99={args.br99}")
    print("=" * 78)

    for model_key in args.models:
        for int_bits in args.int_bits_list:
            cell_name = f"{model_key}_int{int_bits}"
            run_dir = os.path.join(args.results_dir, cell_name)
            os.makedirs(run_dir, exist_ok=True)
            print(f"\n{'-' * 78}\n[{cell_name}] training one model "
                  f"(input_bits={input_bits}, input_int_bits={int_bits}, "
                  f"weight_bits={args.weight_bits}w0i)\n{'-' * 78}")
            t0 = time.time()
            try:
                tf.keras.backend.clear_session()
                gc.collect()

                if model_key == "model25":
                    obj, model, _ = build_model25(
                        args.data_folder, args.weight_bits, input_bits, int_bits
                    )
                else:
                    obj, model, _ = build_model3(
                        args.data_folder, args.weight_bits, input_bits, int_bits
                    )

                metrics = train_and_evaluate(
                    obj, model, run_dir, args.epochs, weights_for_obj
                )
                metrics["elapsed_seconds"] = round(time.time() - t0, 1)
                metrics["status"] = "ok"
            except Exception as e:
                metrics = {"status": "error", "error": f"{type(e).__name__}: {e}"}
                metrics["elapsed_seconds"] = round(time.time() - t0, 1)
                print(f"[{cell_name}] FAILED: {metrics['error']}")

            row = dict(
                model=model_key,
                input_int_bits=int_bits,
                input_bits=input_bits,
                weight_bits=args.weight_bits,
                **metrics,
            )
            rows.append(row)

            # Re-emit the summary CSV after every cell so a crash doesn't lose
            # progress.
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            print(f"[{cell_name}] done in {metrics['elapsed_seconds']:.1f}s; "
                  f"summary at {summary_path}")

    # Final pretty print.
    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("SUMMARY (sorted by model, int_bits)")
    print("=" * 78)
    cols = ["model", "input_int_bits", "weighted_bkg_rej", "bkg_rej_95",
            "bkg_rej_98", "bkg_rej_99", "auc", "final_val_loss",
            "epochs_trained", "elapsed_seconds", "status"]
    display = df[[c for c in cols if c in df.columns]].copy()
    print(display.to_string(index=False))
    print(f"\nWritten: {summary_path}")


if __name__ == "__main__":
    main()
