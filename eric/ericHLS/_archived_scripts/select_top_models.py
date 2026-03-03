#!/usr/bin/env python3
"""
Select Top Models from Hyperparameter Search

This script filters models from a hyperparameter search directory,
selecting only those in the upper quartile (top 25%) by validation accuracy,
and copies them to a new directory ready for HLS synthesis.

Usage:
    python select_top_models.py --input_dir <hyperparam_search_dir> --output_dir <output_dir> [options]

Author: Eric
Date: 2025
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
from pathlib import Path


def load_trials_from_keras_tuner(search_dir):
    """
    Load trial data from Keras Tuner directory structure (trial_XXX folders).
    
    Args:
        search_dir: Keras Tuner search directory with trial_XXX subdirectories
        
    Returns:
        list: List of trial dictionaries with scores
    """
    trials = []
    
    # Find all trial directories
    trial_dirs = sorted([d for d in os.listdir(search_dir) 
                        if d.startswith('trial_') and 
                        os.path.isdir(os.path.join(search_dir, d))])
    
    for trial_dir in trial_dirs:
        trial_json = os.path.join(search_dir, trial_dir, 'trial.json')
        
        if not os.path.exists(trial_json):
            continue
        
        try:
            with open(trial_json, 'r') as f:
                trial_data = json.load(f)
            
            # Extract key information
            trial_id = int(trial_dir.split('_')[1])
            score = trial_data.get('score')
            hyperparams = trial_data.get('hyperparameters', {}).get('values', {})
            
            trials.append({
                'trial_id': trial_id,
                'score': score,
                'val_accuracy': score,  # Keras Tuner score is typically val_accuracy
                'hyperparameters': hyperparams,
                'model_file': f'model_trial_{trial_id:03d}.h5',
                'hyperparams_file': f'hyperparams_trial_{trial_id:03d}.json'
            })
        
        except Exception as e:
            print(f"Warning: Error loading {trial_json}: {str(e)}")
    
    return trials


def load_trials_summary(input_dir):
    """
    Load trials summary from the hyperparameter search directory.
    
    Args:
        input_dir: Directory containing trials_summary.json
        
    Returns:
        list: List of trial dictionaries
    """
    summary_file = os.path.join(input_dir, 'trials_summary.json')
    
    if not os.path.exists(summary_file):
        print(f"Error: trials_summary.json not found in {input_dir}")
        return None
    
    try:
        with open(summary_file, 'r') as f:
            trials = json.load(f)
        
        print(f"Loaded {len(trials)} trials from {summary_file}")
        return trials
    
    except Exception as e:
        print(f"Error loading trials_summary.json: {str(e)}")
        return None


def extract_validation_accuracy(trial):
    """
    Extract validation accuracy from a trial entry.
    Tries multiple possible keys where accuracy might be stored.
    
    Args:
        trial: Trial dictionary
        
    Returns:
        float: Validation accuracy or None if not found
    """
    # Common keys where validation accuracy might be stored
    possible_keys = [
        'val_accuracy',
        'validation_accuracy',
        'val_acc',
        'accuracy',
        'score',
        'best_val_accuracy',
        'final_val_accuracy'
    ]
    
    # First check if there's a metrics or score field
    if 'metrics' in trial:
        metrics = trial['metrics']
        for key in possible_keys:
            if key in metrics:
                value = metrics[key]
                if value is not None:
                    return float(value)
    
    # Check trial dict directly
    for key in possible_keys:
        if key in trial:
            value = trial[key]
            if value is not None:
                return float(value)
    
    return None


def filter_top_quartile(trials, metric='val_accuracy', percentile=75):
    """
    Filter trials to get only those in the upper quartile.
    
    Args:
        trials: List of trial dictionaries
        metric: Metric name to filter by
        percentile: Percentile threshold (default: 75 for upper quartile)
        
    Returns:
        tuple: (filtered_trials, threshold_value)
    """
    # Extract accuracies
    accuracies = []
    valid_trials = []
    
    for trial in trials:
        acc = extract_validation_accuracy(trial)
        if acc is not None:
            accuracies.append(acc)
            valid_trials.append(trial)
        else:
            print(f"Warning: Could not extract validation accuracy for trial {trial.get('trial_id', 'unknown')}")
    
    if not accuracies:
        print("Error: No validation accuracies found in trials")
        return [], None
    
    # Calculate threshold
    threshold = np.percentile(accuracies, percentile)
    
    # Filter trials
    filtered = [trial for trial, acc in zip(valid_trials, accuracies) if acc >= threshold]
    
    return filtered, threshold


def copy_model_files(trial, input_dir, output_dir, copy_hyperparams=True):
    """
    Copy H5 model file and optionally hyperparameters for a trial.
    
    Args:
        trial: Trial dictionary
        input_dir: Source directory
        output_dir: Destination directory
        copy_hyperparams: Whether to copy hyperparams JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get file paths
        model_file = trial.get('model_file')
        if not model_file:
            print(f"Warning: No model_file field in trial {trial.get('trial_id', 'unknown')}")
            return False
        
        src_model = os.path.join(input_dir, model_file)
        dst_model = os.path.join(output_dir, model_file)
        
        if not os.path.exists(src_model):
            print(f"Warning: Model file not found: {src_model}")
            return False
        
        # Copy model file
        shutil.copy2(src_model, dst_model)
        
        # Copy hyperparameters if requested
        if copy_hyperparams:
            hyperparam_file = trial.get('hyperparams_file')
            if hyperparam_file:
                src_hyperparam = os.path.join(input_dir, hyperparam_file)
                dst_hyperparam = os.path.join(output_dir, hyperparam_file)
                
                if os.path.exists(src_hyperparam):
                    shutil.copy2(src_hyperparam, dst_hyperparam)
        
        return True
    
    except Exception as e:
        print(f"Error copying files for trial {trial.get('trial_id', 'unknown')}: {str(e)}")
        return False


def create_filtered_summary(filtered_trials, output_dir, threshold):
    """
    Create a summary file for the filtered trials.
    
    Args:
        filtered_trials: List of filtered trial dictionaries
        output_dir: Output directory
        threshold: Accuracy threshold used
    """
    summary = {
        'total_trials': len(filtered_trials),
        'selection_criterion': 'upper_quartile',
        'validation_accuracy_threshold': float(threshold),
        'trials': filtered_trials
    }
    
    summary_file = os.path.join(output_dir, 'filtered_trials_summary.json')
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Created filtered summary: {summary_file}")


def print_statistics(trials, filtered_trials, threshold):
    """Print statistics about the filtering."""
    accuracies = [extract_validation_accuracy(t) for t in trials]
    accuracies = [a for a in accuracies if a is not None]
    
    filtered_accuracies = [extract_validation_accuracy(t) for t in filtered_trials]
    
    print("\n" + "=" * 80)
    print("FILTERING STATISTICS")
    print("=" * 80)
    print(f"\nTotal trials: {len(trials)}")
    print(f"Valid trials (with accuracy): {len(accuracies)}")
    print(f"Selected trials (top 25%): {len(filtered_trials)}")
    print(f"\nValidation Accuracy Statistics:")
    print(f"  All trials:")
    print(f"    Min:    {min(accuracies):.4f}")
    print(f"    Max:    {max(accuracies):.4f}")
    print(f"    Mean:   {np.mean(accuracies):.4f}")
    print(f"    Median: {np.median(accuracies):.4f}")
    print(f"    Q1 (25%): {np.percentile(accuracies, 25):.4f}")
    print(f"    Q3 (75%): {np.percentile(accuracies, 75):.4f}")
    print(f"\n  Threshold (75th percentile): {threshold:.4f}")
    print(f"\n  Selected trials:")
    print(f"    Min:    {min(filtered_accuracies):.4f}")
    print(f"    Max:    {max(filtered_accuracies):.4f}")
    print(f"    Mean:   {np.mean(filtered_accuracies):.4f}")
    print(f"    Median: {np.median(filtered_accuracies):.4f}")
    
    # Show top 5 models
    print(f"\n  Top 5 models:")
    sorted_trials = sorted(filtered_trials, 
                          key=lambda t: extract_validation_accuracy(t), 
                          reverse=True)[:5]
    
    for i, trial in enumerate(sorted_trials, 1):
        trial_id = trial.get('trial_id', 'unknown')
        acc = extract_validation_accuracy(trial)
        model_file = trial.get('model_file', 'unknown')
        print(f"    {i}. Trial {trial_id:3d}: {acc:.4f} ({model_file})")


def main():
    parser = argparse.ArgumentParser(
        description='Select top models from hyperparameter search for HLS synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select top 25% models (upper quartile)
  python select_top_models.py \\
      --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --output_dir ../model2_top_models_for_hls
  
  # Select top 10% models
  python select_top_models.py \\
      --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --output_dir ../model2_top_10_for_hls \\
      --percentile 90
  
  # Select specific number of top models
  python select_top_models.py \\
      --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --output_dir ../model2_top_10_for_hls \\
      --top_n 10
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing H5 model files OR Keras Tuner search directory'
    )
    
    parser.add_argument(
        '--search_dir',
        type=str,
        default=None,
        help='Keras Tuner search directory with trial_XXX folders (if different from input_dir)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for selected models'
    )
    
    parser.add_argument(
        '--percentile',
        type=float,
        default=75,
        help='Percentile threshold (default: 75 for upper quartile)'
    )
    
    parser.add_argument(
        '--top_n',
        type=int,
        default=None,
        help='Select top N models instead of using percentile'
    )
    
    parser.add_argument(
        '--no_hyperparams',
        action='store_true',
        help='Do not copy hyperparameter JSON files'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be copied without actually copying'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Determine search directory for validation accuracies
    search_directory = args.search_dir if args.search_dir else args.input_dir
    models_directory = args.input_dir
    
    if args.search_dir and not os.path.isdir(args.search_dir):
        print(f"Error: Search directory does not exist: {args.search_dir}")
        sys.exit(1)
    
    # Load trials - try Keras Tuner format first, then trials_summary.json
    print(f"Loading trials from: {search_directory}")
    
    # Check if this is a Keras Tuner directory (has trial_XXX folders)
    trial_dirs = [d for d in os.listdir(search_directory) 
                  if d.startswith('trial_') and 
                  os.path.isdir(os.path.join(search_directory, d))]
    
    if trial_dirs:
        print(f"Detected Keras Tuner directory format ({len(trial_dirs)} trial folders)")
        trials = load_trials_from_keras_tuner(search_directory)
    else:
        print("Loading from trials_summary.json...")
        trials = load_trials_summary(search_directory)
    
    if trials is None or len(trials) == 0:
        print("Error: No trials found")
        print("\nHint: Provide --search_dir pointing to the Keras Tuner directory")
        print("      (the one with trial_000, trial_001, etc. folders)")
        sys.exit(1)
    
    # Filter trials
    if args.top_n is not None:
        # Select top N models
        print(f"\nSelecting top {args.top_n} models...")
        
        # Extract accuracies and sort
        trials_with_acc = []
        for trial in trials:
            acc = extract_validation_accuracy(trial)
            if acc is not None:
                trials_with_acc.append((trial, acc))
        
        if not trials_with_acc:
            print("Error: No validation accuracies found")
            sys.exit(1)
        
        # Sort by accuracy (descending) and take top N
        trials_with_acc.sort(key=lambda x: x[1], reverse=True)
        filtered_trials = [t for t, _ in trials_with_acc[:args.top_n]]
        threshold = trials_with_acc[args.top_n-1][1] if len(trials_with_acc) >= args.top_n else trials_with_acc[-1][1]
        
    else:
        # Select by percentile
        print(f"\nFiltering for top {100 - args.percentile:.0f}% (>= {args.percentile}th percentile)...")
        filtered_trials, threshold = filter_top_quartile(trials, percentile=args.percentile)
    
    if not filtered_trials:
        print("Error: No trials met the filtering criteria")
        sys.exit(1)
    
    # Print statistics
    print_statistics(trials, filtered_trials, threshold)
    
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - No files will be copied")
        print("=" * 80)
        print(f"\nWould copy {len(filtered_trials)} model files to: {args.output_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Copy files
    print("\n" + "-" * 80)
    print("COPYING FILES")
    print("-" * 80)
    
    success_count = 0
    for trial in filtered_trials:
        trial_id = trial.get('trial_id', 'unknown')
        if copy_model_files(trial, models_directory, args.output_dir, 
                           copy_hyperparams=not args.no_hyperparams):
            success_count += 1
            model_file = trial.get('model_file', 'unknown')
            acc = extract_validation_accuracy(trial)
            print(f"  ✓ Copied trial {trial_id:3d}: {model_file} (acc: {acc:.4f})")
        else:
            print(f"  ✗ Failed to copy trial {trial_id:3d}")
    
    # Create filtered summary
    create_filtered_summary(filtered_trials, args.output_dir, threshold)
    
    # Final summary
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully copied: {success_count}/{len(filtered_trials)} models")
    print(f"Output directory: {args.output_dir}")
    print(f"\nReady for HLS synthesis! You can now run:")
    print(f"  python parallel_hls_synthesis.py \\")
    print(f"      --input_dir {args.output_dir} \\")
    print(f"      --num_workers 4")


if __name__ == "__main__":
    main()

