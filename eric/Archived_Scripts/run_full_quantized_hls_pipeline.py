#!/usr/bin/env python3
"""
End-to-end pipeline runner for quantized models (Model2.5 and Model3).

This script automates the following steps for a **single quantization configuration**:

1. Run quantized hyperparameter tuning (Model2.5 or Model3)
2. Run ROC-based Pareto selection on the hyperparameter tuning results
3. Run parallel HLS synthesis on the selected Pareto models
4. Analyze HLS results and generate a resource-utilization plot

**Conda Environment Management:**
The script automatically switches conda environments between steps:
- Steps 1-2 (hyperparameter tuning, Pareto selection): Uses `mlgpu_qkeras`
- Steps 3-4 (HLS synthesis, HLS analysis): Uses `newHLSEnviro`

The script is intentionally conservative and reuses your existing scripts:
- `run_quantized_hyperparam_tuning_model2_5.py`
- `run_quantized_hyperparam_tuning_model3.py`
- `ericHLS/analyze_and_select_pareto_roc.py`
- `ericHLS/parallel_hls_synthesis.py`
- `ericHLS/analyze_hls_results.py`

It **infers** the newest hyperparameter tuning directory based on standard
naming conventions:
- Model2.5: `model2.5_quantized_*_hyperparameter_results_*`
- Model3:   `model3_quantized_*_hyperparameter_results_*`

Usage examples (from this `eric` directory):

    # Run full pipeline for Model2.5 with defaults
    python run_full_quantized_hls_pipeline.py --model 2.5

    # Run full pipeline for Model3 with defaults
    python run_full_quantized_hls_pipeline.py --model 3

    # Override number of HLS workers and FPGA part
    python run_full_quantized_hls_pipeline.py --model 2.5 \\
        --num_workers 4 --fpga_part xc7z020clg400-1

Author: Eric (pipeline wrapper authored by assistant)
Date: 2026
"""

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ERERIC_DIR = Path(__file__).resolve().parent
ERICHLS_DIR = ERERIC_DIR / "ericHLS"


def run_subprocess(cmd, cwd=None, conda_env=None):
    """
    Run a subprocess, streaming output, and fail fast on errors.
    
    Args:
        cmd: List of command arguments (e.g., [sys.executable, "script.py", "--arg", "value"])
        cwd: Working directory (optional)
        conda_env: Conda environment name (optional). If provided, wraps command with `conda run -n <env>`
    """
    print("\n" + "=" * 80)
    if conda_env:
        print(f"RUNNING (conda env: {conda_env}):", " ".join(cmd))
    else:
        print("RUNNING:", " ".join(cmd))
    print("=" * 80)
    sys.stdout.flush()

    # If conda_env is specified, wrap the command
    if conda_env:
        full_cmd = ["conda", "run", "-n", conda_env] + cmd
    else:
        full_cmd = cmd

    result = subprocess.run(full_cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(full_cmd)}")


def find_latest_hyperparam_dir(model: str) -> Path:
    """
    Find the most recent hyperparameter results directory for the given model.

    Model-specific patterns (relative to `eric/`):
      - '2.5' -> model2.5_quantized_*_hyperparameter_results_*
      - '3'   -> model3_quantized_*_hyperparameter_results_*
    """
    if model == "2.5":
        pattern = "model2.5_quantized_*_hyperparameter_results_*"
    elif model == "3":
        pattern = "model3_quantized_*_hyperparameter_results_*"
    else:
        raise ValueError(f"Unsupported model type: {model}")

    candidates = [
        Path(p)
        for p in glob.glob(str(ERERIC_DIR / pattern))
        if Path(p).is_dir()
    ]

    if not candidates:
        raise FileNotFoundError(f"No hyperparameter result directories found for pattern: {pattern}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"\nDetected latest hyperparameter results for model {model}: {latest}")
    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Run full quantized → Pareto ROC → HLS pipeline for Model2.5 or Model3."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["2.5", "3"],
        required=True,
        help="Which model to run the pipeline for: '2.5' or '3'.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026V2_Apr/TF_Records/filtering_records16384_data_shuffled_single_bigData",
        help="TFRecords base data directory used by hyperparameter tuning and ROC analysis.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for HLS synthesis.",
    )

    parser.add_argument(
        "--fpga_part",
        type=str,
        default="xc7z020clg400-1",
        help="FPGA part for HLS synthesis (passed to parallel_hls_synthesis.py).",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="model_trial_*.h5",
        help="Pattern for H5 files in the Pareto-selected directory (for HLS synthesis).",
    )

    parser.add_argument(
        "--skip_hparam",
        action="store_true",
        help="Skip running hyperparameter tuning and reuse the latest existing results.",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # 1. Run quantized hyperparameter tuning (unless skipped)
    # ------------------------------------------------------------------
    if not args.skip_hparam:
        if args.model == "2.5":
            hparam_script = ERERIC_DIR / "run_quantized_hyperparam_tuning_model2_5.py"
        else:
            hparam_script = ERERIC_DIR / "run_quantized_hyperparam_tuning_model3.py"

        if not hparam_script.exists():
            raise FileNotFoundError(f"Hyperparameter tuning script not found: {hparam_script}")

        cmd = [sys.executable, str(hparam_script)]
        # Use mlgpu_qkeras environment for hyperparameter tuning
        run_subprocess(cmd, cwd=str(ERERIC_DIR), conda_env="mlgpu_qkeras")
    else:
        print("\nSkipping hyperparameter tuning (`--skip_hparam` enabled).")

    # ------------------------------------------------------------------
    # 2. Locate latest hyperparameter results directory
    # ------------------------------------------------------------------
    hyperparam_dir = find_latest_hyperparam_dir(args.model)

    # ------------------------------------------------------------------
    # 3. Run ROC-based Pareto selection
    # ------------------------------------------------------------------
    pareto_output_dir = hyperparam_dir.parent / f"{hyperparam_dir.name}_pareto_roc_selected_{timestamp}"
    pareto_output_dir_str = str(pareto_output_dir)
    os.makedirs(pareto_output_dir_str, exist_ok=True)

    analyze_pareto_script = ERICHLS_DIR / "analyze_and_select_pareto_roc.py"
    if not analyze_pareto_script.exists():
        raise FileNotFoundError(f"Pareto ROC analysis script not found: {analyze_pareto_script}")

    cmd_pareto = [
        sys.executable,
        str(analyze_pareto_script),
        "--input_dir",
        str(hyperparam_dir),
        "--data_dir",
        args.data_dir,
        "--output_dir",
        pareto_output_dir_str,
        "--use_weighted",
    ]
    # Use mlgpu_qkeras environment for Pareto ROC selection
    run_subprocess(cmd_pareto, cwd=str(ERICHLS_DIR), conda_env="mlgpu_qkeras")

    # ------------------------------------------------------------------
    # 4. Run parallel HLS synthesis on Pareto-selected models
    # ------------------------------------------------------------------
    parallel_hls_script = ERICHLS_DIR / "parallel_hls_synthesis.py"
    if not parallel_hls_script.exists():
        raise FileNotFoundError(f"parallel_hls_synthesis.py not found: {parallel_hls_script}")

    cmd_hls = [
        sys.executable,
        str(parallel_hls_script),
        "--input_dir",
        pareto_output_dir_str,
        "--num_workers",
        str(args.num_workers),
        "--pattern",
        args.pattern,
        "--fpga_part",
        args.fpga_part,
    ]
    # Use newHLSEnviro environment for HLS synthesis
    run_subprocess(cmd_hls, cwd=str(ERICHLS_DIR), conda_env="newHLSEnviro")

    # parallel_hls_synthesis.py by default writes results to:
    #   <input_dir>/hls_outputs
    hls_results_dir = os.path.join(pareto_output_dir_str, "hls_outputs")

    # ------------------------------------------------------------------
    # 5. Analyze HLS results and generate plot
    # ------------------------------------------------------------------
    analyze_hls_script = ERICHLS_DIR / "analyze_hls_results.py"
    if not analyze_hls_script.exists():
        raise FileNotFoundError(f"analyze_hls_results.py not found: {analyze_hls_script}")

    # Default CSV and plot will be placed one level above `hls_results_dir`
    plot_output = os.path.join(
        os.path.dirname(hls_results_dir),
        f"resource_utilization_{timestamp}.png",
    )

    cmd_analyze_hls = [
        sys.executable,
        str(analyze_hls_script),
        "--results_dir",
        hls_results_dir,
        "--plot",
        "--plot_output",
        plot_output,
    ]
    # Use newHLSEnviro environment for HLS analysis
    run_subprocess(cmd_analyze_hls, cwd=str(ERICHLS_DIR), conda_env="newHLSEnviro")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FULL QUANTIZED → PARETO ROC → HLS PIPELINE COMPLETED")
    print("=" * 80)
    print(f"\nModel:          Model {args.model}")
    print(f"Hyperparam dir: {hyperparam_dir}")
    print(f"Pareto dir:     {pareto_output_dir_str}")
    print(f"HLS results:    {hls_results_dir}")
    print(f"HLS plot:       {plot_output}")
    print("=" * 80)


if __name__ == "__main__":
    main()

