#!/usr/bin/env python3
"""
HLS Pipeline Runner

Orchestrates the three-step HLS workflow:

  Step 1 — Pareto Selection (analyze_and_select_pareto_roc.py)
           Input:  H5 model files + TFRecords validation data
           Output: Pareto-selected H5 models, CSVs, Pareto-front plot

  Step 2 — HLS Synthesis  (parallel_hls_synthesis.py)
           Input:  Pareto-selected H5 models
           Output: Vivado synthesis reports, optional tarballs

  Step 3 — Resource Analysis (analyze_hls_results.py)
           Input:  HLS synthesis outputs
           Output: Resource-utilization CSV + scatter plot

Directory layout (auto-derived from --output_base_dir):
  <output_base_dir>/
    pareto_selected/            ← Step 1 output (overridable with --pareto_dir)
      hls_outputs/              ← Step 2 output (overridable with --hls_output_dir)
        hls_model_trial_XXX/
        synthesis_results.json
      resource_utilization.csv  ← Step 3 output
      resource_utilization.png

Usage examples:

  # Full pipeline
  python run_hls_pipeline.py \\
      --models_dir  ../hyperparameter_results \\
      --data_dir    /path/to/TF_Records/validation_data \\
      --output_base_dir ../my_pipeline_run

  # Only Pareto selection (Step 1)
  python run_hls_pipeline.py \\
      --models_dir  ../hyperparameter_results \\
      --data_dir    /path/to/TF_Records/validation_data \\
      --output_base_dir ../my_pipeline_run \\
      --stop_step 1

  # Start from HLS synthesis (skip Step 1, use existing pareto dir)
  python run_hls_pipeline.py \\
      --pareto_dir ../my_pipeline_run/pareto_selected \\
      --output_base_dir ../my_pipeline_run \\
      --start_step 2

  # Only run resource analysis on existing HLS outputs (Step 3 only)
  python run_hls_pipeline.py \\
      --hls_output_dir ../my_pipeline_run/pareto_selected/hls_outputs \\
      --start_step 3 --stop_step 3 --step3_plot

Author: Eric
Date: April 2026
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Path to this script's directory — used to locate sibling scripts
SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================================
# HELPERS
# ============================================================================

def _banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _run(cmd: list[str], step_name: str) -> int:
    """Run a subprocess command, stream its output, and return the exit code."""
    print(f"\n[Pipeline] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"\n[Pipeline] ERROR: {step_name} failed with exit code {result.returncode}")
    return result.returncode


def _require_dir(path: str, name: str):
    """Exit if a required directory does not exist."""
    if not os.path.isdir(path):
        print(f"[Pipeline] ERROR: {name} directory does not exist: {path}")
        sys.exit(1)


# ============================================================================
# STEP FUNCTIONS
# ============================================================================

def run_step1(args, pareto_dir: str) -> int:
    """Step 1: Pareto selection via analyze_and_select_pareto_roc.py"""
    _banner("STEP 1 — Pareto Selection (analyze_and_select_pareto_roc.py)")

    _require_dir(args.models_dir, "--models_dir")
    _require_dir(args.data_dir, "--data_dir")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "analyze_and_select_pareto_roc.py"),
        "--input_dir",  args.models_dir,
        "--data_dir",   args.data_dir,
        "--output_dir", pareto_dir,
    ]

    if args.step1_use_weighted:
        cmd.append("--use_weighted")

    if args.step1_signal_efficiency is not None:
        cmd += ["--signal_efficiency", str(args.step1_signal_efficiency)]

    if args.step1_bkg_rej_weights is not None:
        cmd += ["--bkg_rej_weights", args.step1_bkg_rej_weights]

    if args.step1_features is not None:
        cmd += ["--features", args.step1_features]

    return _run(cmd, "Step 1 (Pareto selection)")


def run_step2(args, pareto_dir: str, hls_output_dir: str) -> int:
    """Step 2: HLS synthesis via parallel_hls_synthesis.py"""
    _banner("STEP 2 — HLS Synthesis (parallel_hls_synthesis.py)")

    _require_dir(pareto_dir, "pareto (--pareto_dir / Step 1 output)")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "parallel_hls_synthesis.py"),
        "--input_dir",  pareto_dir,
        "--output_dir", hls_output_dir,
        "--num_workers", str(args.step2_num_workers),
        "--fpga_part",   args.step2_fpga_part,
        "--pattern",     args.step2_pattern,
    ]

    if args.step2_no_tarball:
        cmd.append("--no_tarball")

    if args.step2_limit is not None:
        cmd += ["--limit", str(args.step2_limit)]

    return _run(cmd, "Step 2 (HLS synthesis)")


def run_step3(args, hls_output_dir: str) -> int:
    """Step 3: Resource analysis via analyze_hls_results.py"""
    _banner("STEP 3 — Resource Analysis (analyze_hls_results.py)")

    _require_dir(hls_output_dir, "HLS outputs (--hls_output_dir / Step 2 output)")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "analyze_hls_results.py"),
        "--results_dir", hls_output_dir,
    ]

    if args.step3_output_csv is not None:
        cmd += ["--output_csv", args.step3_output_csv]

    if args.step3_plot:
        cmd.append("--plot")

    if args.step3_plot_output is not None:
        cmd += ["--plot_output", args.step3_plot_output]

    cmd += ["--pink-luts",   str(args.step3_pink_luts)]
    cmd += ["--pink-ffs",    str(args.step3_pink_ffs)]
    cmd += ["--xilinx-luts", str(args.step3_xilinx_luts)]
    cmd += ["--xilinx-ffs",  str(args.step3_xilinx_ffs)]

    return _run(cmd, "Step 3 (resource analysis)")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HLS Pipeline Runner — Pareto Selection → HLS Synthesis → Resource Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------
    pipeline = parser.add_argument_group("Pipeline control")
    pipeline.add_argument(
        "--start_step",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="First pipeline step to execute (default: 1)",
    )
    pipeline.add_argument(
        "--stop_step",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Last pipeline step to execute (default: 3)",
    )

    # ------------------------------------------------------------------
    # Shared / directory arguments
    # ------------------------------------------------------------------
    dirs = parser.add_argument_group(
        "Shared directories",
        "Paths shared across pipeline steps. All intermediate paths are "
        "auto-derived from --output_base_dir but can be overridden individually.",
    )
    dirs.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="[Step 1] Directory containing the initial H5 model files (model_trial_*.h5). "
             "Required when --start_step is 1.",
    )
    dirs.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="[Step 1] Directory containing TFRecords (must have tfrecords_validation/ subdir). "
             "Required when --start_step is 1.",
    )
    dirs.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base output directory. Intermediate dirs are created inside here "
             "unless overridden with --pareto_dir / --hls_output_dir. "
             "Required unless both --pareto_dir and --hls_output_dir are set explicitly.",
    )
    dirs.add_argument(
        "--pareto_dir",
        type=str,
        default=None,
        help="[Step 1 output / Step 2 input] Directory for Pareto-selected models. "
             "Default: <output_base_dir>/pareto_selected",
    )
    dirs.add_argument(
        "--hls_output_dir",
        type=str,
        default=None,
        help="[Step 2 output / Step 3 input] Directory for HLS synthesis outputs. "
             "Default: <pareto_dir>/hls_outputs",
    )

    # ------------------------------------------------------------------
    # Step 1 options
    # ------------------------------------------------------------------
    s1 = parser.add_argument_group("Step 1 — Pareto selection options")
    s1.add_argument(
        "--step1_use_weighted",
        action="store_true",
        default=True,
        help="Use weighted background rejection metric (default: True)",
    )
    s1.add_argument(
        "--step1_no_weighted",
        dest="step1_use_weighted",
        action="store_false",
        help="Disable weighted metric — use a single signal efficiency point instead",
    )
    s1.add_argument(
        "--step1_signal_efficiency",
        type=float,
        default=None,
        help="Signal efficiency for background rejection when not using weighted "
             "metric (default: 0.95 inside the script)",
    )
    s1.add_argument(
        "--step1_bkg_rej_weights",
        type=str,
        default=None,
        help='Weighted metric weights, format "sig_eff:weight,..." '
             '(default: "0.95:0.1,0.98:0.7,0.99:0.2")',
    )
    s1.add_argument(
        "--step1_features",
        type=str,
        default=None,
        help="Comma-separated feature names to parse from TFRecords. "
             "If omitted, features are auto-detected from the first model.",
    )

    # ------------------------------------------------------------------
    # Step 2 options
    # ------------------------------------------------------------------
    s2 = parser.add_argument_group("Step 2 — HLS synthesis options")
    s2.add_argument(
        "--step2_num_workers",
        type=int,
        default=4,
        help="Number of parallel synthesis workers (default: 4)",
    )
    s2.add_argument(
        "--step2_fpga_part",
        type=str,
        default="xc7z020clg400-1",
        help="FPGA part number for HLS synthesis (default: xc7z020clg400-1)",
    )
    s2.add_argument(
        "--step2_pattern",
        type=str,
        default="model_trial_*.h5",
        help="Glob pattern to select H5 files for synthesis (default: model_trial_*.h5)",
    )
    s2.add_argument(
        "--step2_no_tarball",
        action="store_true",
        help="Skip creating a tarball of synthesis outputs",
    )
    s2.add_argument(
        "--step2_limit",
        type=int,
        default=None,
        help="Limit the number of models synthesized (useful for testing)",
    )

    # ------------------------------------------------------------------
    # Step 3 options
    # ------------------------------------------------------------------
    s3 = parser.add_argument_group("Step 3 — Resource analysis options")
    s3.add_argument(
        "--step3_output_csv",
        type=str,
        default=None,
        help="Output CSV file path (default: <pareto_dir>/resource_utilization.csv)",
    )
    s3.add_argument(
        "--step3_plot",
        action="store_true",
        help="Generate a resource-utilization scatter plot",
    )
    s3.add_argument(
        "--step3_plot_output",
        type=str,
        default=None,
        help="Output plot file path (default: <pareto_dir>/resource_utilization.png)",
    )
    s3.add_argument(
        "--step3_pink_luts",
        type=int,
        default=53200,
        help="Max LUTs for PYNQ-Z1/Z2 board constraint line (default: 53200)",
    )
    s3.add_argument(
        "--step3_pink_ffs",
        type=int,
        default=106400,
        help="Max FFs for PYNQ-Z1/Z2 board constraint line (default: 106400)",
    )
    s3.add_argument(
        "--step3_xilinx_luts",
        type=int,
        default=215360,
        help="Max LUTs for Xilinx XC7A200T constraint line (default: 215360)",
    )
    s3.add_argument(
        "--step3_xilinx_ffs",
        type=int,
        default=430720,
        help="Max FFs for Xilinx XC7A200T constraint line (default: 430720)",
    )

    return parser


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate step range
    if args.start_step > args.stop_step:
        parser.error(
            f"--start_step ({args.start_step}) must be <= --stop_step ({args.stop_step})"
        )

    # ------------------------------------------------------------------
    # Resolve directory paths
    # ------------------------------------------------------------------
    # Determine pareto_dir
    if args.pareto_dir:
        pareto_dir = os.path.abspath(args.pareto_dir)
    elif args.output_base_dir:
        pareto_dir = os.path.join(os.path.abspath(args.output_base_dir), "pareto_selected")
    else:
        if args.start_step <= 1 or args.stop_step >= 2:
            parser.error(
                "Provide --output_base_dir (or --pareto_dir) so the pipeline "
                "knows where to write/read Pareto-selected models."
            )
        pareto_dir = None  # not needed if running step 3 only with explicit hls_output_dir

    # Determine hls_output_dir
    if args.hls_output_dir:
        hls_output_dir = os.path.abspath(args.hls_output_dir)
    elif pareto_dir:
        hls_output_dir = os.path.join(pareto_dir, "hls_outputs")
    else:
        parser.error(
            "Cannot determine HLS output directory. "
            "Provide --output_base_dir, --pareto_dir, or --hls_output_dir."
        )

    # Validate required args for step 1
    if args.start_step == 1:
        if not args.models_dir:
            parser.error("--models_dir is required when --start_step is 1")
        if not args.data_dir:
            parser.error("--data_dir is required when --start_step is 1")

    # ------------------------------------------------------------------
    # Print pipeline summary
    # ------------------------------------------------------------------
    _banner("HLS PIPELINE RUNNER")

    steps_to_run = list(range(args.start_step, args.stop_step + 1))
    print(f"\nSteps to run     : {steps_to_run}")
    print(f"Start time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.models_dir:
        print(f"Models dir       : {args.models_dir}")
    if args.data_dir:
        print(f"Data dir         : {args.data_dir}")
    if pareto_dir:
        print(f"Pareto dir       : {pareto_dir}")
    print(f"HLS output dir   : {hls_output_dir}")
    if args.step1_bkg_rej_weights:
        print(f"Bkg rej weights  : {args.step1_bkg_rej_weights}")
    print(f"FPGA part        : {args.step2_fpga_part}")
    print(f"Parallel workers : {args.step2_num_workers}")

    # ------------------------------------------------------------------
    # Execute steps
    # ------------------------------------------------------------------
    step_results: dict[int, str] = {}
    start_time = datetime.now()

    for step in steps_to_run:
        step_start = datetime.now()

        if step == 1:
            rc = run_step1(args, pareto_dir)
        elif step == 2:
            rc = run_step2(args, pareto_dir, hls_output_dir)
        elif step == 3:
            rc = run_step3(args, hls_output_dir)
        else:
            continue

        elapsed = (datetime.now() - step_start).total_seconds()
        status = "SUCCESS" if rc == 0 else f"FAILED (exit {rc})"
        step_results[step] = f"{status}  [{elapsed:.1f}s]"

        if rc != 0:
            print(f"\n[Pipeline] Step {step} failed. Stopping pipeline.")
            break

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = (datetime.now() - start_time).total_seconds()
    _banner("PIPELINE SUMMARY")

    print(f"\n{'Step':<8} {'Status'}")
    print("-" * 50)
    for step, status in step_results.items():
        step_names = {
            1: "Pareto selection",
            2: "HLS synthesis",
            3: "Resource analysis",
        }
        print(f"  {step} ({step_names[step]:<20}) : {status}")

    skipped = [s for s in steps_to_run if s not in step_results]
    for step in skipped:
        print(f"  {step} : SKIPPED (previous step failed)")

    print(f"\nTotal elapsed : {total_elapsed:.1f}s")
    print(f"End time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_ok = all("SUCCESS" in v for v in step_results.values())
    if all_ok and not skipped:
        print("\nAll steps completed successfully.")
        print(f"\nOutputs:")
        if 1 in step_results:
            print(f"  Pareto models & plots : {pareto_dir}/")
        if 2 in step_results:
            print(f"  HLS synthesis results : {hls_output_dir}/")
        if 3 in step_results:
            csv_path = args.step3_output_csv or os.path.join(pareto_dir, "resource_utilization.csv")
            print(f"  Resource CSV          : {csv_path}")
            if args.step3_plot:
                plot_path = args.step3_plot_output or os.path.join(pareto_dir, "resource_utilization.png")
                print(f"  Resource plot         : {plot_path}")

    print("\n" + "=" * 80)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
