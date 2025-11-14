#!/usr/bin/env python3
"""
Add Metadata to Existing Trials Summary

This script enriches existing trials_summary.json files by adding:
- Validation accuracy (from Keras Tuner trial data)
- Training metrics
- Total parameters
- Layer structure

Author: Eric
Date: 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path


def load_trial_data_from_search_dir(search_dir):
    """
    Load trial data (including scores) from Keras Tuner search directory.
    
    Returns:
        dict: Mapping from trial_id to trial data
    """
    trial_data_map = {}
    
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
            
            trial_id = int(trial_dir.split('_')[1])
            
            # Extract relevant information
            info = {
                'trial_id': trial_id,
                'score': trial_data.get('score'),  # Validation accuracy
                'metrics': trial_data.get('metrics', {}),
                'hyperparameters': trial_data.get('hyperparameters', {}).get('values', {})
            }
            
            trial_data_map[trial_id] = info
            
        except Exception as e:
            print(f"Warning: Could not load {trial_json}: {str(e)}")
    
    return trial_data_map


def calculate_model_parameters_from_hyperparams(hyperparams):
    """
    Calculate model parameters from hyperparameters without loading the model.
    This is much faster and works for Model2 architecture.
    
    Assumes Model2 architecture:
    - Input: xz (24 features), yl (24 features)  
    - xz_dense1: xz_units
    - yl_dense1: yl_units
    - concatenate: xz_units + yl_units
    - merged_dense1: merged_units1
    - merged_dense2: merged_units2
    - merged_dense3: merged_units3
    - output: 1 unit
    
    Returns:
        dict: Model metadata including estimated parameters
    """
    try:
        # Extract hyperparameters
        xz_units = hyperparams.get('xz_units', 0)
        yl_units = hyperparams.get('yl_units', 0)
        merged_units1 = hyperparams.get('merged_units1', 0)
        merged_units2 = hyperparams.get('merged_units2', 0)
        merged_units3 = hyperparams.get('merged_units3', 0)
        
        # Input dimensions
        xz_input_dim = 24  # z_signal + x_signal (12 each)
        yl_input_dim = 24  # y_signal + l_signal (12 each)
        
        # Calculate parameters for each layer (weights + biases)
        # xz_dense1: (24 x xz_units) + xz_units
        xz_dense1_params = (xz_input_dim * xz_units) + xz_units
        
        # yl_dense1: (24 x yl_units) + yl_units
        yl_dense1_params = (yl_input_dim * yl_units) + yl_units
        
        # merged_dense1: ((xz_units + yl_units) x merged_units1) + merged_units1
        concat_dim = xz_units + yl_units
        merged_dense1_params = (concat_dim * merged_units1) + merged_units1
        
        # merged_dense2: (merged_units1 x merged_units2) + merged_units2
        merged_dense2_params = (merged_units1 * merged_units2) + merged_units2
        
        # merged_dense3: (merged_units2 x merged_units3) + merged_units3
        merged_dense3_params = (merged_units2 * merged_units3) + merged_units3
        
        # output: (merged_units3 x 1) + 1
        output_params = merged_units3 + 1
        
        # Total parameters
        total_params = (xz_dense1_params + yl_dense1_params + 
                       merged_dense1_params + merged_dense2_params + 
                       merged_dense3_params + output_params)
        
        # Layer structure
        layer_structure = [
            {'name': 'xz_input', 'type': 'Input', 'shape': xz_input_dim},
            {'name': 'yl_input', 'type': 'Input', 'shape': yl_input_dim},
            {'name': 'xz_dense1', 'type': 'QDense', 'units': xz_units, 'parameters': xz_dense1_params},
            {'name': 'yl_dense1', 'type': 'QDense', 'units': yl_units, 'parameters': yl_dense1_params},
            {'name': 'concatenate', 'type': 'Concatenate', 'units': concat_dim},
            {'name': 'merged_dense1', 'type': 'QDense', 'units': merged_units1, 'parameters': merged_dense1_params},
            {'name': 'merged_dense2', 'type': 'QDense', 'units': merged_units2, 'parameters': merged_dense2_params},
            {'name': 'merged_dense3', 'type': 'QDense', 'units': merged_units3, 'parameters': merged_dense3_params},
            {'name': 'output', 'type': 'QActivation/QDense', 'units': 1, 'parameters': output_params}
        ]
        
        metadata = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(total_params),  # All are trainable in this architecture
            'non_trainable_parameters': 0,
            'num_layers': len(layer_structure),
            'layer_structure': layer_structure,
            'calculated_from_hyperparams': True
        }
        
        return metadata
        
    except Exception as e:
        print(f"  Warning: Could not calculate parameters: {str(e)}")
        return None


def enrich_trials_summary(results_dir, search_dir=None, dry_run=False):
    """
    Enrich an existing trials_summary.json with additional metadata.
    
    Args:
        results_dir: Directory containing models and trials_summary.json
        search_dir: Keras Tuner search directory with trial data
        dry_run: If True, don't write changes
    """
    trials_summary_path = os.path.join(results_dir, 'trials_summary.json')
    
    if not os.path.exists(trials_summary_path):
        print(f"Error: {trials_summary_path} not found")
        return False
    
    # Load existing trials summary
    print(f"Loading {trials_summary_path}...")
    with open(trials_summary_path, 'r') as f:
        trials_summary = json.load(f)
    
    print(f"Found {len(trials_summary)} trials")
    
    # Load trial data from search directory if provided
    trial_data_map = {}
    if search_dir and os.path.exists(search_dir):
        print(f"\nLoading validation scores from {search_dir}...")
        trial_data_map = load_trial_data_from_search_dir(search_dir)
        print(f"Loaded data for {len(trial_data_map)} trials")
    
    # Enrich each trial
    print(f"\nEnriching trial metadata...")
    enriched_count = 0
    
    for i, trial in enumerate(trials_summary):
        trial_id = trial.get('trial_id', i)
        model_file = trial.get('model_file')
        
        if not model_file:
            continue
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(trials_summary)} trials...")
        
        # Add validation score if available
        if trial_id in trial_data_map:
            trial_info = trial_data_map[trial_id]
            trial['val_accuracy'] = trial_info.get('score')
            trial['metrics'] = trial_info.get('metrics', {})
        
        # Calculate model parameters from hyperparameters (much faster than loading)
        hyperparams = trial.get('hyperparameters', {})
        if hyperparams:
            metadata = calculate_model_parameters_from_hyperparams(hyperparams)
            if metadata:
                trial.update(metadata)
                enriched_count += 1
    
    print(f"\nSuccessfully enriched {enriched_count}/{len(trials_summary)} trials")
    
    # Save enriched summary
    if not dry_run:
        # Backup original
        backup_path = trials_summary_path + '.backup'
        if not os.path.exists(backup_path):
            os.rename(trials_summary_path, backup_path)
            print(f"✓ Created backup: {backup_path}")
        
        # Save enriched version
        with open(trials_summary_path, 'w') as f:
            json.dump(trials_summary, f, indent=4)
        print(f"✓ Saved enriched summary to: {trials_summary_path}")
    else:
        print(f"\n[DRY RUN] Would save enriched summary to: {trials_summary_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Add metadata to existing trials summary files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enrich model2 results
  python add_metadata_to_trials.py \\
      --results_dir model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --search_dir hyperparameter_tuning/model2_quantized_4w0i_hyperparameter_search

  # Enrich model3 results
  python add_metadata_to_trials.py \\
      --results_dir model3_quantized_4w0i_hyperparameter_results_20251110_194507 \\
      --search_dir hyperparameter_tuning/model3_quantized_4w0i_hyperparameter_search

  # Dry run to preview changes
  python add_metadata_to_trials.py \\
      --results_dir model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --search_dir hyperparameter_tuning/model2_quantized_4w0i_hyperparameter_search \\
      --dry_run
        """
    )
    
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing models and trials_summary.json')
    parser.add_argument('--search_dir', type=str, default=None,
                       help='Keras Tuner search directory with trial validation scores')
    parser.add_argument('--dry_run', action='store_true',
                       help='Preview changes without writing files')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    if args.search_dir and not os.path.isdir(args.search_dir):
        print(f"Error: Search directory not found: {args.search_dir}")
        sys.exit(1)
    
    # Enrich the trials summary
    print("="*80)
    print("ADD METADATA TO TRIALS SUMMARY")
    print("="*80)
    print(f"\nResults directory: {args.results_dir}")
    if args.search_dir:
        print(f"Search directory: {args.search_dir}")
    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be modified]")
    print()
    
    success = enrich_trials_summary(args.results_dir, args.search_dir, args.dry_run)
    
    if success:
        print("\n" + "="*80)
        print("COMPLETE")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("FAILED")
        print("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()

