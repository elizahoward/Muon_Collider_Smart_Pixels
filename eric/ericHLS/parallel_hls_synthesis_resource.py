#!/usr/bin/env python3
"""
Parallel HLS Synthesis Script (Resource / FF+LUT focus)

Combines the parallel/pruning flow of parallel_hls_synthesis_quick.py with the
memory-aware single-model flow of synthesize_single_model.py:

  - Parallel processing of many H5 files (ProcessPoolExecutor)
  - Opt-in build steps (synth-only by default; cosim/export/vsynth require flags)
  - Host + HLS thread caps (OMP/MKL/Vitis)
  - Build TCL patched to honor HLS_MAX_THREADS
  - TF model freed + gc.collect() before launching Vitis HLS
  - Output pruned to only essentials (myproject_csynth.rpt for FF/LUT estimates,
    vitis_hls.log, project.tcl)
  - Optional tarballs of the pruned output
  - High default ReuseFactor (conv=128, dense=16) to suppress IR blow-up

FF/LUT numbers come from the Vitis HLS csynth report (estimates).
Vivado synthesis (--vsynth) is NOT run by default — it takes far longer
and has been observed to OOM-kill on large models. The HLS estimates
are higher than post-implementation truth but don't require Vivado.

Usage:
    python parallel_hls_synthesis_resource.py --input_dir <path> --num_workers 4

Default command example (single worker, HLS estimates only):
    python /home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/ericHLS/parallel_hls_synthesis_resource.py \
        --input_dir /home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/model3_6w0i_pareto_roc_selected/pareto_primary \
        --num_workers 1
"""

import gc
import os
import sys
import argparse
import glob
import shutil
import tarfile
import json
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import tensorflow as tf
import qkeras
import hls4ml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Files to keep after synthesis.
# myproject_csynth.rpt is the top-level Vitis HLS C-synth report with
# FF/LUT/DSP/BRAM *estimates* (no Vivado synthesis needed).
KEEP_FILES = [
    'vivado_synth.rpt'
    'project.tcl',
    'vitis_hls.log',
    'myproject_csynth.rpt',
]


def configure_host_thread_limits(max_threads):
    """Apply conservative thread limits to reduce peak RAM (applied per-worker)."""
    if max_threads <= 0:
        return

    thread_limit = str(max_threads)
    os.environ['OMP_NUM_THREADS'] = thread_limit
    os.environ['OPENBLAS_NUM_THREADS'] = thread_limit
    os.environ['MKL_NUM_THREADS'] = thread_limit
    os.environ['NUMEXPR_NUM_THREADS'] = thread_limit
    os.environ['VECLIB_MAXIMUM_THREADS'] = thread_limit
    os.environ['HLS_MAX_THREADS'] = thread_limit


def setup_tf_config(no_gpu=True, tf_threads=1):
    if no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    if tf_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)


def apply_resource_profile(config, conv_reuse_factor, dense_reuse_factor):
    """Resource strategy + per-layer ReuseFactor to keep HLS IR bounded."""
    for layer_name in config.get('LayerName', {}):
        config['LayerName'][layer_name]['Strategy'] = 'Resource'
        if 'conv' in layer_name.lower():
            config['LayerName'][layer_name]['ReuseFactor'] = conv_reuse_factor
        else:
            config['LayerName'][layer_name]['ReuseFactor'] = dense_reuse_factor


def patch_build_tcl_for_thread_limit(output_dir):
    """Inject a thread cap hook into the generated build_prj.tcl."""
    build_tcl = Path(output_dir) / 'build_prj.tcl'
    if not build_tcl.exists():
        return

    existing = build_tcl.read_text()
    marker = "source [file join $tcldir project.tcl]"
    hook = """
if {[info exists ::env(HLS_MAX_THREADS)]} {
    set _max_threads $::env(HLS_MAX_THREADS)
    if {[string is integer -strict $_max_threads] && $_max_threads > 0} {
        catch {set_param general.maxThreads $_max_threads}
        puts "INFO: hls4ml requested max HLS threads: $_max_threads"
    }
}
""".strip("\n")

    if marker not in existing or "Requested max HLS threads" in existing:
        return

    patched = existing.replace(marker, f"{marker}\n{hook}", 1)
    build_tcl.write_text(patched)


def synthesize_model(
    h5_file,
    output_base_dir,
    fpga_part,
    io_type,
    build_options,
    conv_reuse_factor,
    dense_reuse_factor,
    max_threads,
    tf_threads,
    create_tarball,
    noDSP = True,
):
    """Synthesize a single H5 model, then prune output to KEEP_FILES."""
    result = {
        'h5_file': h5_file,
        'status': 'pending',
        'output_dir': None,
        'kept_files': [],
        'tarball': None,
        'error': None,
        'start_time': datetime.now().isoformat(),
    }

    try:
        configure_host_thread_limits(max_threads)
        setup_tf_config(no_gpu=True, tf_threads=tf_threads)

        model_name = Path(h5_file).stem
        print(f"[{model_name}] Starting HLS synthesis (resource/FF+LUT mode)...")

        output_dir = os.path.join(output_base_dir, f"hls_{model_name}")
        result['output_dir'] = output_dir

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[{model_name}] Loading model from {h5_file}...")
        custom_objects = {}
        qkeras.utils._add_supported_quantized_objects(custom_objects)
        quantized_model = tf.keras.models.load_model(
            h5_file, custom_objects=custom_objects, compile=False
        )

        print(f"[{model_name}] Creating HLS configuration...")
        config = hls4ml.utils.config_from_keras_model(quantized_model, granularity='name')
        apply_resource_profile(config, conv_reuse_factor, dense_reuse_factor)
        print(
            f"[{model_name}] Resource profile: conv RF={conv_reuse_factor}, "
            f"dense RF={dense_reuse_factor}"
        )
        if noDSP:
            print("Setting the DSPs to be unused")
            config['Model']['use_dsp'] = False
            # Force the Vitis HLS backend compiler to strictly enforce 0 DSPs globally
            #Seems to do nothing
            if 'HLSConfig' not in config:
                config['HLSConfig'] = {}
            config['HLSConfig']['CompilerOptions'] = '-max_dsp 0'

            # VERIFIED HLS4ML API: Force a global optimization strategy 
            # This forces activation layers to utilize standard LUT logic mappings
            # config['Model']['Strategy'] = 'Resource'
            config['Model']['BramFactor'] = 0 

        print(f"[{model_name}] Converting to HLS model (io_type={io_type})...")
        hls_model = hls4ml.converters.convert_from_keras_model(
            quantized_model,
            hls_config=config,
            part=fpga_part,
            output_dir=output_dir,
            backend="Vitis",
            io_type=io_type,
            clock_period = 10,
        )

        print(f"[{model_name}] Writing HLS files...")
        hls_model.write()
        patch_build_tcl_for_thread_limit(output_dir)


        if noDSP:
            # Force the Vitis HLS 2024.1 backend to map all operations to standard fabric logic
            tcl_path = os.path.join(output_dir, "build_prj.tcl")
            if os.path.exists(tcl_path):
                with open(tcl_path, "r") as f:
                    tcl_content = f.read()
                
                # 1. Clean up the deprecated array partition warning/error line
                tcl_content = tcl_content.replace(
                    "config_array_partition -maximum_size 4096", 
                    "# config_array_partition -maximum_size 4096"
                )
                
                # 2. Inject valid Vitis HLS 2024.1 global operator fabric overrides
                if "csynth_design" in tcl_content:
                    patch_directives = (
                        "config_op mul -impl fabric\n"  # Maps all inferred multiplications to LUTs
                        "config_op add -impl fabric\n"  # Maps all inferred additions to LUTs
                        "csynth_design"
                    )
                    tcl_content = tcl_content.replace("csynth_design", patch_directives)
                    
                    with open(tcl_path, "w") as f:
                        f.write(tcl_content)
                    print(f"[{model_name}] Patched build_prj.tcl to force Vitis 2024.1 operator fabric mapping.")



        del quantized_model
        gc.collect()

        print(f"[{model_name}] Building HLS project... options={build_options}")
        hls_model.build(**build_options)

        print(f"[{model_name}] Synthesis completed. Pruning output...")

        kept_files = []
        for root, _dirs, files in os.walk(output_dir):
            for file in files:
                if file in KEEP_FILES:
                    kept_files.append(os.path.join(root, file))

        temp_dir = output_dir + "_temp"
        os.makedirs(temp_dir, exist_ok=True)
        for file_path in kept_files:
            rel_path = os.path.relpath(file_path, output_dir)
            dest_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(file_path, dest_path)

        result['kept_files'] = kept_files

        shutil.rmtree(output_dir)
        os.rename(temp_dir, output_dir)
        print(f"[{model_name}] Kept {len(kept_files)} essential files")

        if create_tarball:
            tarball_path = output_dir + ".tar.gz"
            print(f"[{model_name}] Creating tarball: {tarball_path}")
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(output_dir, arcname=os.path.basename(output_dir))
            result['tarball'] = tarball_path

        result['status'] = 'success'
        result['end_time'] = datetime.now().isoformat()
        print(f"[{model_name}] ✓ Complete!")

    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = str(exc)
        result['traceback'] = traceback.format_exc()
        result['end_time'] = datetime.now().isoformat()
        print(f"[{Path(h5_file).stem}] ✗ Failed: {str(exc)}")
        print(traceback.format_exc())

    return result


def find_h5_files(input_dir, pattern="model_trial_*.h5"):
    h5_files = glob.glob(os.path.join(input_dir, pattern))
    h5_files.sort()
    return h5_files


def save_results_summary(results, output_file):
    summary = {
        'total': len(results),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'results': results,
    }
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel HLS synthesis focused on FF/LUT resource metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Typical run: parallel synth + Vivado synth to get FFs/LUTs
  python parallel_hls_synthesis_resource.py --input_dir ../model3_results --num_workers 4 --vsynth

  # Full flow (cosim + export + vsynth)
  python parallel_hls_synthesis_resource.py --input_dir ../model3_results --num_workers 4 --full_flow

  # Override reuse factors for even more aggressive resource sharing
  python parallel_hls_synthesis_resource.py --input_dir ../model3_results \\
      --num_workers 2 --vsynth --conv_reuse_factor 256 --dense_reuse_factor 32
        """,
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing H5 model files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Base directory for HLS outputs (default: input_dir/hls_outputs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--pattern', type=str, default='model_trial_*.h5',
                        help='Glob pattern for H5 files (default: model_trial_*.h5)')
    parser.add_argument('--fpga_part', type=str, default='xc7z020clg400-1',
                        help='FPGA part number (default: xc7z020clg400-1)')
    parser.add_argument('--io_type', type=str, default='io_parallel',
                        choices=['io_parallel', 'io_stream'],
                        help='hls4ml io_type (default: io_parallel)')

    parser.add_argument('--conv_reuse_factor', type=int, default=128,
                        help='ReuseFactor for conv layers (default: 128)')
    parser.add_argument('--dense_reuse_factor', type=int, default=16,
                        help='ReuseFactor for non-conv layers (default: 16)')

    parser.add_argument('--max_threads', type=int, default=4,
                        help='Per-worker thread cap for host + HLS (0 disables; default: 4)')
    parser.add_argument('--tf_threads', type=int, default=1,
                        help='TensorFlow intra/inter-op threads per worker (default: 1)')

    # Opt-in build steps (mirrors synthesize_single_model.py).
    parser.add_argument('--csim', action='store_true', help='Enable C simulation')
    parser.add_argument('--skip_synth', action='store_true',
                        help='Skip csynth_design (requires existing project with --no_reset).')
    parser.add_argument('--cosim', action='store_true',
                        help='Enable C/RTL co-simulation (memory-heavy)')
    parser.add_argument('--validation', action='store_true',
                        help='Enable result validation between csim and cosim outputs')
    parser.add_argument('--export', action='store_true', help='Enable export_design (IP export)')
    parser.add_argument('--vsynth', action='store_true',
                        help='Enable Vivado synthesis (required for FF/LUT numbers)')
    parser.add_argument('--full_flow', action='store_true',
                        help='Shortcut to enable cosim + export + vsynth')
    parser.add_argument('--no_reset', action='store_true',
                        help='Do not reset existing HLS project')

    parser.add_argument('--no_tarball', action='store_true',
                        help='Skip creating tarball of pruned output')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of models to process (for testing)')

    args = parser.parse_args()

    if args.full_flow:
        args.cosim = True
        args.export = True
        args.vsynth = True

    if args.skip_synth and not args.no_reset:
        print("Warning: --skip_synth implies --no_reset to reuse existing synthesis artifacts.")
        args.no_reset = True

    if args.vsynth:
        print("WARNING: --vsynth enabled. Vivado synthesis is memory-heavy and has "
              "been known to OOM-kill on large models; HLS csynth estimates "
              "(myproject_csynth.rpt) are produced without it.")

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'hls_outputs')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Searching for H5 files in: {args.input_dir}")
    print(f"Pattern: {args.pattern}")
    h5_files = find_h5_files(args.input_dir, args.pattern)
    if not h5_files:
        print(f"Error: No H5 files found matching pattern '{args.pattern}'")
        sys.exit(1)

    if args.limit:
        h5_files = h5_files[:args.limit]
        print(f"Limited to first {args.limit} files")

    build_options = {
        'csim': args.csim,
        'synth': not args.skip_synth,
        'cosim': args.cosim,
        'validation': args.validation,
        'export': args.export,
        'vsynth': args.vsynth,
        'reset': not args.no_reset,
    }

    print(f"Found {len(h5_files)} H5 files to process")
    print(f"Workers: {args.num_workers} | FPGA: {args.fpga_part} | io_type: {args.io_type}")
    print(f"Strategy: Resource | conv RF={args.conv_reuse_factor}, dense RF={args.dense_reuse_factor}")
    print(f"Per-worker max_threads cap: {args.max_threads} | tf_threads: {args.tf_threads}")
    print(f"Build options: {build_options}")
    print(f"Output directory: {args.output_dir}")
    print(f"Create tarballs: {not args.no_tarball}")
    print("-" * 80)

    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_h5 = {
            executor.submit(
                synthesize_model,
                h5_file,
                args.output_dir,
                args.fpga_part,
                args.io_type,
                build_options,
                args.conv_reuse_factor,
                args.dense_reuse_factor,
                args.max_threads,
                args.tf_threads,
                not args.no_tarball,
            ): h5_file
            for h5_file in h5_files
        }

        for future in as_completed(future_to_h5):
            h5_file = future_to_h5[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Error processing {h5_file}: {str(exc)}")
                results.append({
                    'h5_file': h5_file,
                    'status': 'failed',
                    'error': str(exc),
                    'traceback': traceback.format_exc(),
                })

    summary_file = os.path.join(args.output_dir, 'synthesis_results.json')
    save_results_summary(results, summary_file)

    print("\n" + "=" * 80)
    print("SYNTHESIS COMPLETE")
    print("=" * 80)
    print(f"Total models processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Output directory: {args.output_dir}")
    print(f"Results summary: {summary_file}")

    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  - {Path(r['h5_file']).name}: {r['error']}")


if __name__ == "__main__":
    main()
