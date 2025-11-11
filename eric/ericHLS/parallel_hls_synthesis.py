#!/usr/bin/env python3
"""
Parallel HLS Synthesis Script for Neural Network Models

This script processes multiple H5 model files in parallel, synthesizing them 
to HLS implementations and keeping only essential output files.

Usage:
    python parallel_hls_synthesis.py --input_dir <path_to_h5_files> --num_workers <workers> [options]

Author: Eric
Date: 2025
"""

import os
import sys
import argparse
import glob
import shutil
import tarfile
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import traceback

# TensorFlow and related imports
import tensorflow as tf
import qkeras
import hls4ml

# Configure GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Files to keep after synthesis
KEEP_FILES = [
    'project.tcl',
    'vitis_hls.log',
    'vivado_synth.rpt',
    'vivado_synth.tcl',
    'vivado.jou',
    'vivado.log'
]


def setup_tf_config(no_gpu=True):
    """Configure TensorFlow settings."""
    if no_gpu:
        tf.config.set_visible_devices([], 'GPU')


def synthesize_model(h5_file, output_base_dir, fpga_part, create_tarball=True):
    """
    Synthesize a single H5 model to HLS implementation.
    
    Args:
        h5_file: Path to the H5 model file
        output_base_dir: Base directory for output
        fpga_part: FPGA part number (e.g., 'xc7z020clg400-1')
        create_tarball: Whether to create a tarball of kept files
        
    Returns:
        dict: Results dictionary with status and information
    """
    result = {
        'h5_file': h5_file,
        'status': 'pending',
        'output_dir': None,
        'kept_files': [],
        'tarball': None,
        'error': None,
        'start_time': datetime.now().isoformat()
    }
    
    try:
        # Extract model name from file
        model_name = Path(h5_file).stem
        print(f"[{model_name}] Starting HLS synthesis...")
        
        # Create output directory
        output_dir = os.path.join(output_base_dir, f"hls_{model_name}")
        result['output_dir'] = output_dir
        
        # Remove existing output directory if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # Load the model
        print(f"[{model_name}] Loading model from {h5_file}...")
        co = {}
        qkeras.utils._add_supported_quantized_objects(co)
        quantized_model = tf.keras.models.load_model(h5_file, custom_objects=co, compile=False)
        
        # Create HLS config
        print(f"[{model_name}] Creating HLS configuration...")
        config = hls4ml.utils.config_from_keras_model(quantized_model, granularity='name')
        
        # Convert to HLS model
        print(f"[{model_name}] Converting to HLS model...")
        hls_model = hls4ml.converters.convert_from_keras_model(
            quantized_model,
            hls_config=config,
            part=fpga_part,
            output_dir=output_dir,
            backend="Vitis"
        )
        
        # Write HLS files
        print(f"[{model_name}] Writing HLS files...")
        hls_model.write()
        
        # Build with synthesis
        print(f"[{model_name}] Building HLS project (this may take a while)...")
        hls_model.build(
            csim=False,
            synth=True,
            cosim=True,
            validation=False,
            export=True,
            vsynth=True,
            reset=True
        )
        
        print(f"[{model_name}] Synthesis completed! Cleaning up...")
        
        # Find and keep only specified files
        kept_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file in KEEP_FILES:
                    file_path = os.path.join(root, file)
                    kept_files.append(file_path)
        
        # Create a temporary directory with only kept files
        temp_dir = output_dir + "_temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        for file_path in kept_files:
            rel_path = os.path.relpath(file_path, output_dir)
            dest_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(file_path, dest_path)
        
        result['kept_files'] = kept_files
        
        # Remove original output directory
        shutil.rmtree(output_dir)
        
        # Rename temp directory to original name
        os.rename(temp_dir, output_dir)
        
        print(f"[{model_name}] Kept {len(kept_files)} essential files")
        
        # Create tarball if requested
        if create_tarball:
            tarball_path = output_dir + ".tar.gz"
            print(f"[{model_name}] Creating tarball: {tarball_path}")
            
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(output_dir, arcname=os.path.basename(output_dir))
            
            result['tarball'] = tarball_path
            print(f"[{model_name}] Tarball created successfully")
        
        result['status'] = 'success'
        result['end_time'] = datetime.now().isoformat()
        print(f"[{model_name}] ✓ Complete!")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        result['end_time'] = datetime.now().isoformat()
        print(f"[{model_name}] ✗ Failed: {str(e)}")
        print(traceback.format_exc())
    
    return result


def find_h5_files(input_dir, pattern="model_trial_*.h5"):
    """Find all H5 files matching the pattern in the input directory."""
    h5_files = glob.glob(os.path.join(input_dir, pattern))
    h5_files.sort()
    return h5_files


def save_results_summary(results, output_file):
    """Save synthesis results to a JSON file."""
    summary = {
        'total': len(results),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel HLS Synthesis for Neural Network Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthesize all models in a directory with 4 workers
  python parallel_hls_synthesis.py --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 --num_workers 4
  
  # Synthesize specific trials only
  python parallel_hls_synthesis.py --input_dir ../model2_results --pattern "model_trial_00*.h5" --num_workers 2
  
  # Use different FPGA part
  python parallel_hls_synthesis.py --input_dir ../model2_results --num_workers 4 --fpga_part xcu250-figd2104-2L-e
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing H5 model files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Base directory for HLS outputs (default: input_dir/hls_outputs)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='model_trial_*.h5',
        help='Pattern to match H5 files (default: model_trial_*.h5)'
    )
    
    parser.add_argument(
        '--fpga_part',
        type=str,
        default='xc7z020clg400-1',
        help='FPGA part number (default: xc7z020clg400-1)'
    )
    
    parser.add_argument(
        '--no_tarball',
        action='store_true',
        help='Skip creating tarball of output files'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of models to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_tf_config(no_gpu=True)
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'hls_outputs')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find H5 files
    print(f"Searching for H5 files in: {args.input_dir}")
    print(f"Pattern: {args.pattern}")
    h5_files = find_h5_files(args.input_dir, args.pattern)
    
    if not h5_files:
        print(f"Error: No H5 files found matching pattern '{args.pattern}'")
        sys.exit(1)
    
    if args.limit:
        h5_files = h5_files[:args.limit]
        print(f"Limited to first {args.limit} files")
    
    print(f"Found {len(h5_files)} H5 files to process")
    print(f"Using {args.num_workers} parallel workers")
    print(f"FPGA part: {args.fpga_part}")
    print(f"Output directory: {args.output_dir}")
    print(f"Create tarballs: {not args.no_tarball}")
    print("-" * 80)
    
    # Process files in parallel
    results = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all jobs
        future_to_h5 = {
            executor.submit(
                synthesize_model,
                h5_file,
                args.output_dir,
                args.fpga_part,
                not args.no_tarball
            ): h5_file
            for h5_file in h5_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_h5):
            h5_file = future_to_h5[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {h5_file}: {str(e)}")
                results.append({
                    'h5_file': h5_file,
                    'status': 'failed',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
    
    # Save results summary
    summary_file = os.path.join(args.output_dir, 'synthesis_results.json')
    save_results_summary(results, summary_file)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("SYNTHESIS COMPLETE")
    print("=" * 80)
    print(f"Total models processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Results summary: {summary_file}")
    
    # List failed models if any
    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  - {Path(r['h5_file']).name}: {r['error']}")


if __name__ == "__main__":
    main()

