#!/usr/bin/env python3
"""
Single Model HLS Synthesis Script (Quick / Low-IR variant)

Same as synthesize_single_model.py but uses:
  - Strategy='Resource' for all layers      (uses resource-sharing implementation
                                              that respects ReuseFactor; 'Latency'
                                              strategy ignores ReuseFactor)
  - ReuseFactor=64 (Conv) / 8 (Dense)       (limits HLS IR unrolling per layer type)
  - io_type="io_parallel"                   (default; io_stream was tried but
                                              hls4ml streaming Conv2D ignores
                                              ReuseFactor, producing 3M+ IR)

These settings prevent Vitis HLS from hanging on large models (e.g. Model3)
during the Unroll/Inline phase. Model3's 1504->110 dense layer and Conv2D over
a 13x21 input are the main culprits. Tradeoff: higher latency, but synthesis
actually completes.

Usage:
    python synthesize_single_model_quick.py --h5_file <path_to_h5_file> [options]

Author: Eric
Date: 2026
"""

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


def setup_tf_config(no_gpu=True):
    """Configure TensorFlow settings."""
    if no_gpu:
        tf.config.set_visible_devices([], 'GPU')


def synthesize_model(h5_file, output_dir, fpga_part):
    """
    Synthesize a single H5 model to HLS implementation.

    Args:
        h5_file: Path to the H5 model file
        output_dir: Directory for HLS output (all files will be kept)
        fpga_part: FPGA part number (e.g., 'xc7z020clg400-1')

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
        # Extract model name from file
        model_name = Path(h5_file).stem
        print(f"[{model_name}] Starting HLS synthesis (quick/low-IR mode)...")
        print(f"[{model_name}] Output directory: {output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load the model
        print(f"[{model_name}] Loading model from {h5_file}...")
        co = {}
        qkeras.utils._add_supported_quantized_objects(co)
        quantized_model = tf.keras.models.load_model(h5_file, custom_objects=co, compile=False)

        # Create HLS config
        print(f"[{model_name}] Creating HLS configuration...")
        config = hls4ml.utils.config_from_keras_model(quantized_model, granularity='name')

        # Use io_parallel (default) with Resource strategy + ReuseFactor on dense layers.
        # io_stream was tried but hls4ml's streaming Conv2D (shift_line_buffer,
        # pooling2d_cl) ignores ReuseFactor entirely and unrolls spatially over the
        # full 13x21 input map, producing 3M+ instructions at Unroll/Inline.
        # io_parallel unrolls Conv2D less aggressively, and Resource strategy with
        # a high ReuseFactor on the large dense layers (1504->110) is what controls
        # the IR explosion.
        for layer in config['LayerName']:
            config['LayerName'][layer]['Strategy'] = 'Resource'
            # Dense layers get a moderate ReuseFactor to reduce unrolling.
            # Conv layers get a higher ReuseFactor to limit spatial unrolling.
            # Check layer name — hls4ml adds internal layers (e.g. conv2d_linear)
            # not present in the Keras model, so we cannot use get_layer().
            if 'conv' in layer.lower():
                config['LayerName'][layer]['ReuseFactor'] = 64
            else:
                config['LayerName'][layer]['ReuseFactor'] = 8

        # Convert to HLS model using io_parallel (default)
        print(f"[{model_name}] Converting to HLS model (io_parallel, Strategy=Resource, ReuseFactor=64(conv)/8(dense))...")
        hls_model = hls4ml.converters.convert_from_keras_model(
            quantized_model,
            hls_config=config,
            part=fpga_part,
            output_dir=output_dir,
            backend="Vitis",
            io_type="io_parallel"
        )

        # Write HLS files
        print(f"[{model_name}] Writing HLS files...")
        hls_model.write()

        # Build with synthesis
        print(f"[{model_name}] Building HLS project (this may take a while)...")
        print(f"[{model_name}] Running: csim=False, synth=True, cosim=True, validation=False, export=True, vsynth=True")
        hls_model.build(
            csim=False,
            synth=True,
            cosim=True,
            validation=False,
            export=True,
            vsynth=True,
            reset=True
        )

        print(f"[{model_name}] Synthesis completed!")
        print(f"[{model_name}] All files kept in: {output_dir}")

        result['status'] = 'success'
        result['end_time'] = datetime.now().isoformat()
        print(f"[{model_name}] ✓ Complete!")

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['end_time'] = datetime.now().isoformat()
        print(f"[{model_name}] ✗ Failed: {str(e)}")
        import traceback
        print(traceback.format_exc())

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Synthesize a single H5 model to HLS (quick/low-IR mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthesize a single model
  python synthesize_single_model_quick.py --h5_file model_trial_008.h5 --output_dir ./hls_output

  # Use default output directory (hls_<model_name>)
  python synthesize_single_model_quick.py --h5_file model_trial_008.h5

  # Use different FPGA part
  python synthesize_single_model_quick.py --h5_file model_trial_008.h5 --fpga_part xcu250-figd2104-2L-e
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
        help='Directory for HLS output (default: hls_<model_name> in same directory as H5 file)'
    )

    parser.add_argument(
        '--fpga_part',
        type=str,
        default='xc7z020clg400-1',
        help='FPGA part number (default: xc7z020clg400-1)'
    )

    args = parser.parse_args()

    # Setup
    setup_tf_config(no_gpu=True)

    # Validate input file
    if not os.path.isfile(args.h5_file):
        print(f"Error: H5 file does not exist: {args.h5_file}")
        sys.exit(1)

    # Set output directory
    if args.output_dir is None:
        h5_path = Path(args.h5_file)
        model_name = h5_path.stem
        args.output_dir = str(h5_path.parent / f"hls_{model_name}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("HLS SYNTHESIS - SINGLE MODEL (quick/low-IR mode)")
    print("  Strategy    : Resource (all layers)")
    print("  ReuseFactor : 64 (Conv layers) / 8 (Dense layers)")
    print("  io_type     : io_parallel")
    print("=" * 80)
    print(f"Input H5 file: {args.h5_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"FPGA part: {args.fpga_part}")
    print(f"Note: All synthesis files will be kept (no deletion)")
    print("-" * 80)

    # Synthesize model
    result = synthesize_model(args.h5_file, args.output_dir, args.fpga_part)

    # Print final summary
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
