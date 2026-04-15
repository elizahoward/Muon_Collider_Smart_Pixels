#!/usr/bin/env python3
"""
Single Model HLS Synthesis Script

This script synthesizes a single H5 model file to HLS implementation.
Default behavior is now tuned for lower RAM usage:
  - Resource strategy + higher reuse factors
  - synth-only flow (cosim/export/vsynth are opt-in)
  - optional host thread cap for Vitis/TensorFlow
"""

import gc
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# TensorFlow and related imports
import tensorflow as tf
import qkeras
import hls4ml

# Configure GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging


def configure_host_thread_limits(max_threads):
    """Apply conservative thread limits to reduce peak RAM."""
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
    """Configure TensorFlow settings."""
    if no_gpu:
        tf.config.set_visible_devices([], 'GPU')

    if tf_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)


def apply_low_memory_hls_profile(config, use_low_memory_profile, conv_reuse_factor, dense_reuse_factor):
    """Apply per-layer config that reduces unroll/LTO blow-up."""
    if not use_low_memory_profile:
        return "Latency profile (legacy)"

    for layer_name in config.get('LayerName', {}):
        config['LayerName'][layer_name]['Strategy'] = 'Resource'
        if 'conv' in layer_name.lower():
            config['LayerName'][layer_name]['ReuseFactor'] = conv_reuse_factor
        else:
            config['LayerName'][layer_name]['ReuseFactor'] = dense_reuse_factor

    return f"Low-memory profile (Resource, conv RF={conv_reuse_factor}, dense RF={dense_reuse_factor})"


def patch_build_tcl_for_thread_limit(output_dir):
    """Inject a thread cap hook into generated build_prj.tcl."""
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
    output_dir,
    fpga_part,
    io_type,
    build_options,
    use_low_memory_profile,
    conv_reuse_factor,
    dense_reuse_factor
):
    """
    Synthesize a single H5 model to HLS implementation.

    Args:
        h5_file: Path to the H5 model file
        output_dir: Directory for HLS output
        fpga_part: FPGA part number (e.g., 'xc7z020clg400-1')
        io_type: hls4ml io_type
        build_options: build options passed to hls_model.build()
        use_low_memory_profile: if True, apply Resource + ReuseFactor tuning
        conv_reuse_factor: ReuseFactor used for conv layers in low-memory profile
        dense_reuse_factor: ReuseFactor used for non-conv layers in low-memory profile

    Returns:
        dict: Results dictionary with status and information
    """
    result = {
        'h5_file': h5_file,
        'status': 'pending',
        'output_dir': output_dir,
        'error': None,
        'start_time': datetime.now().isoformat()
    }

    try:
        model_name = Path(h5_file).stem
        print(f"[{model_name}] Starting HLS synthesis...")
        print(f"[{model_name}] Output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        print(f"[{model_name}] Loading model from {h5_file}...")
        custom_objects = {}
        qkeras.utils._add_supported_quantized_objects(custom_objects)
        quantized_model = tf.keras.models.load_model(h5_file, custom_objects=custom_objects, compile=False)

        print(f"[{model_name}] Creating HLS configuration...")
        config = hls4ml.utils.config_from_keras_model(quantized_model, granularity='name')
        config_mode = apply_low_memory_hls_profile(
            config,
            use_low_memory_profile=use_low_memory_profile,
            conv_reuse_factor=conv_reuse_factor,
            dense_reuse_factor=dense_reuse_factor
        )
        print(f"[{model_name}] Config mode: {config_mode}")

        print(f"[{model_name}] Converting to HLS model (io_type={io_type})...")
        hls_model = hls4ml.converters.convert_from_keras_model(
            quantized_model,
            hls_config=config,
            part=fpga_part,
            output_dir=output_dir,
            backend="Vitis",
            io_type=io_type
        )

        print(f"[{model_name}] Writing HLS files...")
        hls_model.write()
        patch_build_tcl_for_thread_limit(output_dir)

        # Free TensorFlow-side model memory before launching Vitis HLS.
        del quantized_model
        gc.collect()

        print(f"[{model_name}] Building HLS project...")
        print(f"[{model_name}] Build options: {build_options}")
        hls_model.build(**build_options)

        print(f"[{model_name}] Synthesis completed.")
        print(f"[{model_name}] All files kept in: {output_dir}")

        result['status'] = 'success'
        result['end_time'] = datetime.now().isoformat()
        print(f"[{model_name}] ✓ Complete!")

    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = str(exc)
        result['end_time'] = datetime.now().isoformat()
        print(f"[{model_name}] ✗ Failed: {str(exc)}")
        import traceback
        print(traceback.format_exc())

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Synthesize a single H5 model to HLS implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Low-memory default run (synth only)
  python synthesize_single_model.py --h5_file model_trial_101.h5

  # Enable heavy steps explicitly
  python synthesize_single_model.py --h5_file model_trial_101.h5 --cosim --export --vsynth

  # Legacy high-throughput profile (higher RAM risk)
  python synthesize_single_model.py --h5_file model_trial_101.h5 --high_throughput_profile
        """
    )

    parser.add_argument(
        '--h5_file',
        type=str,
        required=True,
        help='Path to the H5 model file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for HLS output (default: hls_<model_name> next to H5 file)'
    )
    parser.add_argument(
        '--fpga_part',
        type=str,
        default='xc7z020clg400-1',
        help='FPGA part number (default: xc7z020clg400-1)'
    )
    parser.add_argument(
        '--io_type',
        type=str,
        default='io_parallel',
        choices=['io_parallel', 'io_stream'],
        help='hls4ml io_type (default: io_parallel)'
    )
    parser.add_argument(
        '--high_throughput_profile',
        action='store_true',
        help='Use legacy Latency-style behavior (higher RAM usage, faster if it fits).'
    )
    parser.add_argument(
        '--conv_reuse_factor',
        type=int,
        default=64,
        help='ReuseFactor for conv layers in low-memory mode (default: 64)'
    )
    parser.add_argument(
        '--dense_reuse_factor',
        type=int,
        default=8,
        help='ReuseFactor for non-conv layers in low-memory mode (default: 8)'
    )
    parser.add_argument(
        '--max_threads',
        type=int,
        default=4,
        help='Thread cap for host + HLS subprocesses (0 disables cap, default: 4)'
    )
    parser.add_argument(
        '--tf_threads',
        type=int,
        default=1,
        help='TensorFlow intra/inter-op threads (default: 1)'
    )
    parser.add_argument(
        '--csim',
        action='store_true',
        help='Enable C simulation'
    )
    parser.add_argument(
        '--skip_synth',
        action='store_true',
        help='Skip csynth_design (requires existing synthesized project with --no_reset).'
    )
    parser.add_argument(
        '--cosim',
        action='store_true',
        help='Enable C/RTL co-simulation (memory-heavy)'
    )
    parser.add_argument(
        '--validation',
        action='store_true',
        help='Enable result validation between csim and cosim outputs'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Enable export_design (IP export)'
    )
    parser.add_argument(
        '--vsynth',
        action='store_true',
        help='Enable Vivado synthesis (memory-heavy)'
    )
    parser.add_argument(
        '--full_flow',
        action='store_true',
        help='Shortcut to enable cosim + export + vsynth'
    )
    parser.add_argument(
        '--no_reset',
        action='store_true',
        help='Do not reset existing HLS project'
    )

    args = parser.parse_args()

    if args.full_flow:
        args.cosim = True
        args.export = True
        args.vsynth = True

    if args.skip_synth and not args.no_reset:
        print("Warning: --skip_synth implies --no_reset to reuse existing synthesis artifacts.")
        args.no_reset = True

    use_low_memory_profile = not args.high_throughput_profile

    configure_host_thread_limits(args.max_threads)
    setup_tf_config(no_gpu=True, tf_threads=args.tf_threads)

    if not os.path.isfile(args.h5_file):
        print(f"Error: H5 file does not exist: {args.h5_file}")
        sys.exit(1)

    if args.output_dir is None:
        h5_path = Path(args.h5_file)
        args.output_dir = str(h5_path.parent / f"hls_{h5_path.stem}")

    os.makedirs(args.output_dir, exist_ok=True)

    build_options = {
        'csim': args.csim,
        'synth': not args.skip_synth,
        'cosim': args.cosim,
        'validation': args.validation,
        'export': args.export,
        'vsynth': args.vsynth,
        'reset': not args.no_reset
    }

    print("=" * 80)
    print("HLS SYNTHESIS - SINGLE MODEL")
    print("=" * 80)
    print(f"Input H5 file: {args.h5_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"FPGA part: {args.fpga_part}")
    print(f"io_type: {args.io_type}")
    print(f"Low-memory profile: {use_low_memory_profile}")
    if use_low_memory_profile:
        print(f"  conv_reuse_factor={args.conv_reuse_factor}, dense_reuse_factor={args.dense_reuse_factor}")
    print(f"max_threads cap: {args.max_threads}")
    print(f"Build options: {build_options}")
    print("-" * 80)

    result = synthesize_model(
        h5_file=args.h5_file,
        output_dir=args.output_dir,
        fpga_part=args.fpga_part,
        io_type=args.io_type,
        build_options=build_options,
        use_low_memory_profile=use_low_memory_profile,
        conv_reuse_factor=args.conv_reuse_factor,
        dense_reuse_factor=args.dense_reuse_factor
    )

    print("\n" + "=" * 80)
    print("SYNTHESIS COMPLETE")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Output directory: {result['output_dir']}")

    if result['status'] == 'success':
        print("✓ Synthesis completed successfully!")
        print(f"  All files are available in: {result['output_dir']}")
    else:
        print(f"✗ Synthesis failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
