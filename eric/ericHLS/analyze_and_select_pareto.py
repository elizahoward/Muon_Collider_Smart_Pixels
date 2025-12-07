#!/usr/bin/env python3
"""
Combined Analysis and Pareto Selection for HLS Synthesis

This script combines hyperparameter complexity analysis with Pareto optimal model selection.
It takes hyperparameter tuning results, analyzes complexity, selects Pareto models (two-tier),
and prepares them for HLS synthesis by copying H5 files and plots to a single output directory.

Workflow:
1. Analyze hyperparameter results (complexity vs accuracy)
2. Generate plots (parameters and nodes vs accuracy)
3. Select Pareto optimal models (primary + secondary for redundancy)
4. Copy selected H5 model files to output directory
5. Copy all plots to output directory
6. Generate summary files and statistics

Output directory structure:
    output_dir/
    ├── model_trial_XXX.h5 (selected Pareto models)
    ├── complexity_vs_accuracy_parameters.png
    ├── complexity_vs_accuracy_nodes.png
    ├── pareto_front_parameters_combined.png
    ├── pareto_front_nodes_combined.png
    ├── pareto_optimal_models_parameters_primary.csv
    ├── pareto_optimal_models_parameters_secondary.csv
    ├── pareto_optimal_models_parameters_combined.csv
    ├── pareto_optimal_models_parameters.json
    ├── hyperparameter_complexity_summary.csv
    ├── hyperparameter_detailed_results.csv
    └── analysis_summary.json

Usage:
    python analyze_and_select_pareto.py --input_dir <hyperparam_results_dir> --output_dir <output_dir> [options]

Author: Eric
Date: December 2025
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import TensorFlow for H5 loading (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ============================================================================
# COMPLEXITY ANALYSIS FUNCTIONS
# ============================================================================

def calculate_model2_5_complexity(hyperparams):
    """Calculate complexity for Model2.5 architecture."""
    spatial_units = hyperparams.get('spatial_units', 0)
    z_global_units = hyperparams.get('z_global_units', 0)
    dense2_units = hyperparams.get('dense2_units', 0)
    dense3_units = hyperparams.get('dense3_units', 0)
    
    total_nodes = spatial_units + z_global_units + dense2_units + dense3_units + 1
    
    params_spatial = (35 * spatial_units) + spatial_units
    params_z_global = (1 * z_global_units) + z_global_units
    params_dense2 = ((spatial_units + z_global_units) * dense2_units) + dense2_units
    params_dense3 = (dense2_units * dense3_units) + dense3_units
    params_output = (dense3_units * 1) + 1
    
    total_params = params_spatial + params_z_global + params_dense2 + params_dense3 + params_output
    
    return total_nodes, total_params


def detect_model_type(hyperparams):
    """Detect model type from hyperparameters."""
    if 'conv_filters' in hyperparams:
        return 'model3'
    elif 'spatial_units' in hyperparams:
        return 'model2_5'
    elif 'xz_units' in hyperparams:
        return 'model2'
    else:
        raise ValueError(f"Cannot detect model type from hyperparameters: {hyperparams.keys()}")


def parse_flat_format_trial(search_dir, trial_id):
    """Parse trial from flat directory format."""
    trial_id_str = f"{trial_id:03d}"
    
    hyperparam_file = os.path.join(search_dir, f'hyperparams_trial_{trial_id_str}.json')
    if not os.path.exists(hyperparam_file):
        return None
    
    model_file = os.path.join(search_dir, f'model_trial_{trial_id_str}.h5')
    if not os.path.exists(model_file):
        return None
    
    try:
        with open(hyperparam_file, 'r') as f:
            hyperparams = json.load(f)
        
        summary_file = os.path.join(search_dir, 'trials_summary.json')
        val_accuracy = None
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                for trial in summary_data:
                    if trial.get('trial_id') == trial_id or trial.get('trial_id') == trial_id_str:
                        val_accuracy = trial.get('val_accuracy') or trial.get('validation_accuracy') or trial.get('score')
                        break
            except Exception:
                pass
        
        if val_accuracy is not None and val_accuracy > 0:
            return {
                'trial_id': trial_id_str,
                'hyperparams': hyperparams,
                'val_accuracy': val_accuracy,
                'status': 'COMPLETED',
                'model_file': model_file
            }
    
    except Exception as e:
        print(f"  Warning: Error parsing trial {trial_id_str}: {e}")
    
    return None


def analyze_complexity(input_dir, min_accuracy=0.55):
    """
    Analyze model complexity from hyperparameter results.
    
    Returns:
        DataFrame with trial results including nodes, parameters, and accuracy
    """
    print("\n" + "=" * 80)
    print("STEP 1: COMPLEXITY ANALYSIS")
    print("=" * 80)
    
    h5_files = sorted([f for f in os.listdir(input_dir) 
                       if f.startswith('model_trial_') and f.endswith('.h5')])
    
    if not h5_files:
        raise ValueError(f"No H5 model files found in {input_dir}")
    
    print(f"\nFound {len(h5_files)} H5 models")
    print(f"Analyzing complexity...")
    
    # Extract trial IDs
    trial_ids = []
    for h5_file in h5_files:
        try:
            trial_id = int(h5_file.replace('model_trial_', '').replace('.h5', ''))
            trial_ids.append(trial_id)
        except:
            pass
    
    results = []
    detected_model_type = None
    complexity_func = None
    
    for trial_id in sorted(trial_ids):
        trial_result = parse_flat_format_trial(input_dir, trial_id)
        
        if trial_result:
            # Auto-detect model type from first trial
            if complexity_func is None:
                detected_model_type = detect_model_type(trial_result['hyperparams'])
                if detected_model_type == 'model2_5':
                    complexity_func = calculate_model2_5_complexity
                    print(f"  Auto-detected model type: MODEL2.5")
                else:
                    raise ValueError(f"Unsupported model type: {detected_model_type}")
            
            nodes, params = complexity_func(trial_result['hyperparams'])
            
            results.append({
                'model': os.path.basename(input_dir),
                'trial_id': trial_result['trial_id'],
                'nodes': nodes,
                'parameters': params,
                'val_accuracy': trial_result['val_accuracy'],
                'hyperparams': trial_result['hyperparams'],
                'model_file': trial_result['model_file']
            })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        total_trials = len(df)
        df_filtered = df[df['val_accuracy'] >= min_accuracy].copy()
        excluded_count = total_trials - len(df_filtered)
        
        print(f"\n  Completed trials: {total_trials}")
        if excluded_count > 0:
            print(f"  Excluded {excluded_count} failed models (accuracy < {min_accuracy})")
        print(f"  Valid trials after filtering: {len(df_filtered)}")
        print(f"  Nodes range: {df_filtered['nodes'].min():.0f} - {df_filtered['nodes'].max():.0f}")
        print(f"  Parameters range: {df_filtered['parameters'].min():.0f} - {df_filtered['parameters'].max():.0f}")
        print(f"  Accuracy range: {df_filtered['val_accuracy'].min():.4f} - {df_filtered['val_accuracy'].max():.4f}")
        
        return df_filtered
    
    return df


# ============================================================================
# PARETO SELECTION FUNCTIONS
# ============================================================================

def is_dominated(point, other_points, maximize_cols, minimize_cols):
    """Check if a point is dominated by any other point."""
    for _, other in other_points.iterrows():
        better_in_all = True
        strictly_better_in_one = False
        
        for col in maximize_cols:
            if other[col] < point[col]:
                better_in_all = False
                break
            if other[col] > point[col]:
                strictly_better_in_one = True
        
        if not better_in_all:
            continue
        
        for col in minimize_cols:
            if other[col] > point[col]:
                better_in_all = False
                break
            if other[col] < point[col]:
                strictly_better_in_one = True
        
        if better_in_all and strictly_better_in_one:
            return True
    
    return False


def find_pareto_front(df, maximize_cols=['val_accuracy'], minimize_cols=['parameters']):
    """Find Pareto optimal points in a DataFrame."""
    pareto_indices = []
    
    for idx, row in df.iterrows():
        other_points = df.drop(idx)
        if not is_dominated(row, other_points, maximize_cols, minimize_cols):
            pareto_indices.append(idx)
    
    pareto_df = df.loc[pareto_indices].copy()
    pareto_df = pareto_df.sort_values(maximize_cols[0], ascending=False)
    
    return pareto_df


def select_pareto_models(df, complexity_metric='parameters'):
    """
    Select Pareto optimal models (two-tier for redundancy).
    
    Returns:
        Tuple of (primary_pareto_df, secondary_pareto_df)
    """
    print(f"\n--- Tier 1: Primary Pareto Front ({complexity_metric}) ---")
    pareto_df = find_pareto_front(df, 
                                  maximize_cols=['val_accuracy'], 
                                  minimize_cols=[complexity_metric])
    
    print(f"  ✓ Found {len(pareto_df)} primary Pareto optimal models")
    
    print(f"\n--- Tier 2: Secondary Pareto Front (Redundancy) ---")
    primary_trial_ids = set(pareto_df['trial_id'].values)
    df_remaining = df[~df['trial_id'].isin(primary_trial_ids)].copy()
    
    if len(df_remaining) > 1:
        pareto_df_secondary = find_pareto_front(df_remaining,
                                               maximize_cols=['val_accuracy'],
                                               minimize_cols=[complexity_metric])
        
        if not pareto_df_secondary.empty:
            print(f"  ✓ Found {len(pareto_df_secondary)} secondary Pareto optimal models")
        else:
            pareto_df_secondary = None
    else:
        pareto_df_secondary = None
    
    return pareto_df, pareto_df_secondary


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_complexity_vs_accuracy(df, output_dir, model_name, complexity_metric='parameters'):
    """Create scatter plot of complexity vs accuracy."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df[complexity_metric], df['val_accuracy'], 
               alpha=0.6, s=80, c='blue', edgecolors='black', linewidth=0.5)
    
    metric_label = 'Number of Parameters' if complexity_metric == 'parameters' else 'Number of Nodes'
    ax.set_xlabel(metric_label, fontsize=14)
    ax.set_ylabel('Validation Accuracy', fontsize=14)
    ax.set_title(f'{model_name}: {metric_label} vs Validation Accuracy', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'complexity_vs_accuracy_{complexity_metric}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()


def plot_pareto_front(df, pareto_df, pareto_df_secondary, output_dir, model_name, complexity_metric='parameters'):
    """Create Pareto front visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # All models
    ax.scatter(df[complexity_metric], df['val_accuracy'], 
               alpha=0.4, s=60, c='lightgray', edgecolors='gray', 
               linewidth=0.5, label='All models', zorder=1)
    
    # Primary Pareto front
    ax.scatter(pareto_df[complexity_metric], pareto_df['val_accuracy'], 
               alpha=0.9, s=120, c='red', edgecolors='darkred', 
               linewidth=1.5, label='Primary Pareto front', zorder=3, marker='D')
    
    pareto_sorted = pareto_df.sort_values(complexity_metric)
    ax.plot(pareto_sorted[complexity_metric], pareto_sorted['val_accuracy'],
            'r--', alpha=0.6, linewidth=2, zorder=2)
    
    # Secondary Pareto front
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        ax.scatter(pareto_df_secondary[complexity_metric], pareto_df_secondary['val_accuracy'], 
                   alpha=0.8, s=100, c='orange', edgecolors='darkorange', 
                   linewidth=1.5, label='Secondary Pareto front (redundancy)', zorder=3, marker='s')
        
        pareto_sorted_2 = pareto_df_secondary.sort_values(complexity_metric)
        ax.plot(pareto_sorted_2[complexity_metric], pareto_sorted_2['val_accuracy'],
                'orange', linestyle=':', alpha=0.6, linewidth=2, zorder=2)
    
    # Annotate primary Pareto points
    for _, row in pareto_df.iterrows():
        trial_id = row['trial_id']
        label = trial_id if isinstance(trial_id, str) else f"{int(trial_id):03d}"
        ax.annotate(label, 
                   xy=(row[complexity_metric], row['val_accuracy']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, color='darkred', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                            alpha=0.7, edgecolor='darkred', linewidth=1),
                   zorder=4)
    
    # Annotate secondary Pareto points
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        for _, row in pareto_df_secondary.iterrows():
            trial_id = row['trial_id']
            label = trial_id if isinstance(trial_id, str) else f"{int(trial_id):03d}"
            ax.annotate(label, 
                       xy=(row[complexity_metric], row['val_accuracy']),
                       xytext=(8, -12), textcoords='offset points',
                       fontsize=8, color='darkorange', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                alpha=0.7, edgecolor='darkorange', linewidth=1),
                       zorder=4)
    
    # Labels and title
    metric_label = 'Number of Parameters' if complexity_metric == 'parameters' else 'Number of Nodes'
    ax.set_xlabel(metric_label, fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name}: Pareto Front - {metric_label} vs Accuracy (Two-Tier)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Statistics box
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        stats_text = (
            f"Total models: {len(df)}\n"
            f"Primary Pareto: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
            f"Secondary Pareto: {len(pareto_df_secondary)} ({100*len(pareto_df_secondary)/len(df):.1f}%)\n"
            f"Total selected: {len(pareto_df) + len(pareto_df_secondary)}\n"
            f"Acc range: {df['val_accuracy'].min():.4f} - {df['val_accuracy'].max():.4f}"
        )
    else:
        stats_text = (
            f"Total models: {len(df)}\n"
            f"Primary Pareto: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
            f"Acc range: {df['val_accuracy'].min():.4f} - {df['val_accuracy'].max():.4f}"
        )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'pareto_front_{complexity_metric}_combined.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def copy_model_files(pareto_df, pareto_df_secondary, output_dir):
    """Copy H5 model files for selected Pareto models."""
    print("\n" + "=" * 80)
    print("STEP 3: COPYING MODEL FILES")
    print("=" * 80)
    
    success_count = 0
    
    # Copy primary Pareto models
    print(f"\nCopying {len(pareto_df)} primary Pareto models...")
    for _, row in pareto_df.iterrows():
        src_path = row['model_file']
        dst_path = os.path.join(output_dir, os.path.basename(src_path))
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ {os.path.basename(src_path)}")
            success_count += 1
        else:
            print(f"  ✗ File not found: {src_path}")
    
    # Copy secondary Pareto models
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        print(f"\nCopying {len(pareto_df_secondary)} secondary Pareto models (redundancy)...")
        for _, row in pareto_df_secondary.iterrows():
            src_path = row['model_file']
            dst_path = os.path.join(output_dir, os.path.basename(src_path))
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"  ✓ {os.path.basename(src_path)}")
                success_count += 1
            else:
                print(f"  ✗ File not found: {src_path}")
    
    print(f"\nTotal copied: {success_count} model files")
    return success_count


def save_results(df, pareto_df_params, pareto_df_secondary_params, 
                pareto_df_nodes, pareto_df_secondary_nodes, output_dir):
    """Save all analysis results and summaries."""
    print("\n" + "=" * 80)
    print("STEP 4: SAVING RESULTS")
    print("=" * 80)
    
    # Save complexity analysis CSVs
    summary_csv = os.path.join(output_dir, 'hyperparameter_complexity_summary.csv')
    csv_df = df[['model', 'trial_id', 'nodes', 'parameters', 'val_accuracy']].copy()
    csv_df = csv_df.sort_values('val_accuracy', ascending=False)
    csv_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Complexity summary: {summary_csv}")
    
    detailed_csv = os.path.join(output_dir, 'hyperparameter_detailed_results.csv')
    detailed_rows = []
    for _, row in df.iterrows():
        detailed_row = {
            'model': row['model'],
            'trial_id': row['trial_id'],
            'nodes': row['nodes'],
            'parameters': row['parameters'],
            'val_accuracy': row['val_accuracy']
        }
        detailed_row.update(row['hyperparams'])
        detailed_rows.append(detailed_row)
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"✓ Detailed results: {detailed_csv}")
    
    # Save Pareto results (parameters)
    if pareto_df_params is not None:
        primary_csv = os.path.join(output_dir, 'pareto_optimal_models_parameters_primary.csv')
        pareto_df_params.to_csv(primary_csv, index=False)
        print(f"\n✓ Primary Pareto (parameters): {primary_csv}")
        
        if pareto_df_secondary_params is not None and not pareto_df_secondary_params.empty:
            secondary_csv = os.path.join(output_dir, 'pareto_optimal_models_parameters_secondary.csv')
            pareto_df_secondary_params.to_csv(secondary_csv, index=False)
            print(f"✓ Secondary Pareto (parameters): {secondary_csv}")
            
            combined_df = pd.concat([pareto_df_params, pareto_df_secondary_params], ignore_index=True)
            combined_df = combined_df.sort_values('val_accuracy', ascending=False)
            combined_csv = os.path.join(output_dir, 'pareto_optimal_models_parameters_combined.csv')
            combined_df.to_csv(combined_csv, index=False)
            print(f"✓ Combined Pareto (parameters): {combined_csv}")
    
    # Save Pareto results (nodes)
    if pareto_df_nodes is not None:
        primary_csv = os.path.join(output_dir, 'pareto_optimal_models_nodes_primary.csv')
        pareto_df_nodes.to_csv(primary_csv, index=False)
        print(f"\n✓ Primary Pareto (nodes): {primary_csv}")
        
        if pareto_df_secondary_nodes is not None and not pareto_df_secondary_nodes.empty:
            secondary_csv = os.path.join(output_dir, 'pareto_optimal_models_nodes_secondary.csv')
            pareto_df_secondary_nodes.to_csv(secondary_csv, index=False)
            print(f"✓ Secondary Pareto (nodes): {secondary_csv}")
            
            combined_df = pd.concat([pareto_df_nodes, pareto_df_secondary_nodes], ignore_index=True)
            combined_df = combined_df.sort_values('val_accuracy', ascending=False)
            combined_csv = os.path.join(output_dir, 'pareto_optimal_models_nodes_combined.csv')
            combined_df.to_csv(combined_csv, index=False)
            print(f"✓ Combined Pareto (nodes): {combined_csv}")
    
    # Save JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': os.path.basename(output_dir.rstrip('_pareto_hls_ready')),
        'total_models': len(df),
        'primary_pareto_parameters': len(pareto_df_params) if pareto_df_params is not None else 0,
        'secondary_pareto_parameters': len(pareto_df_secondary_params) if pareto_df_secondary_params is not None and not pareto_df_secondary_params.empty else 0,
        'primary_pareto_nodes': len(pareto_df_nodes) if pareto_df_nodes is not None else 0,
        'secondary_pareto_nodes': len(pareto_df_secondary_nodes) if pareto_df_secondary_nodes is not None and not pareto_df_secondary_nodes.empty else 0,
        'accuracy_range': {
            'min': float(df['val_accuracy'].min()),
            'max': float(df['val_accuracy'].max()),
            'mean': float(df['val_accuracy'].mean())
        },
        'parameters_range': {
            'min': int(df['parameters'].min()),
            'max': int(df['parameters'].max()),
            'mean': float(df['parameters'].mean())
        },
        'nodes_range': {
            'min': int(df['nodes'].min()),
            'max': int(df['nodes'].max()),
            'mean': float(df['nodes'].mean())
        }
    }
    
    summary_json = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Analysis summary: {summary_json}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Combined Analysis and Pareto Selection for HLS Synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Model2.5 results for HLS
  python analyze_and_select_pareto.py \\
      --input_dir ../model2.5_quantized_4w0i_hyperparameter_results_20251205_174921 \\
      --output_dir ../model2_5_pareto_hls_ready
  
  # Filter by minimum accuracy
  python analyze_and_select_pareto.py \\
      --input_dir ../model2_5_results \\
      --output_dir ../model2_5_pareto \\
      --min_accuracy 0.85
  
  # Then run HLS synthesis:
  python parallel_hls_synthesis.py --input_dir ../model2_5_pareto_hls_ready --num_workers 4
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing hyperparameter tuning results (with H5 files)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for plots, CSVs, and HLS-ready H5 files'
    )
    
    parser.add_argument(
        '--min_accuracy',
        type=float,
        default=0.55,
        help='Minimum accuracy threshold (default: 0.55)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name = os.path.basename(args.input_dir.rstrip('/'))
    
    print("\n" + "=" * 80)
    print("COMBINED ANALYSIS AND PARETO SELECTION FOR HLS SYNTHESIS")
    print("=" * 80)
    print(f"\nInput directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {model_name}")
    print(f"Minimum accuracy: {args.min_accuracy}")
    
    # Step 1: Analyze complexity
    df = analyze_complexity(args.input_dir, min_accuracy=args.min_accuracy)
    
    if df.empty:
        print("\nError: No valid trials found!")
        sys.exit(1)
    
    # Step 2: Generate complexity plots
    print("\n" + "=" * 80)
    print("STEP 2: GENERATING PLOTS")
    print("=" * 80)
    
    print("\nCreating complexity vs accuracy plots...")
    plot_complexity_vs_accuracy(df, args.output_dir, model_name, 'parameters')
    plot_complexity_vs_accuracy(df, args.output_dir, model_name, 'nodes')
    
    # Select Pareto models for parameters
    print("\n" + "=" * 80)
    print("PARETO SELECTION: PARAMETERS")
    print("=" * 80)
    pareto_df_params, pareto_df_secondary_params = select_pareto_models(df, 'parameters')
    
    # Select Pareto models for nodes
    print("\n" + "=" * 80)
    print("PARETO SELECTION: NODES")
    print("=" * 80)
    pareto_df_nodes, pareto_df_secondary_nodes = select_pareto_models(df, 'nodes')
    
    # Generate Pareto front plots
    print("\nCreating Pareto front plots...")
    plot_pareto_front(df, pareto_df_params, pareto_df_secondary_params, 
                     args.output_dir, model_name, 'parameters')
    plot_pareto_front(df, pareto_df_nodes, pareto_df_secondary_nodes, 
                     args.output_dir, model_name, 'nodes')
    
    # Combine all Pareto models (union of parameters and nodes selections)
    print("\n" + "=" * 80)
    print("COMBINING PARETO SELECTIONS")
    print("=" * 80)
    
    all_pareto_ids = set()
    all_pareto_ids.update(pareto_df_params['trial_id'].values)
    all_pareto_ids.update(pareto_df_nodes['trial_id'].values)
    if pareto_df_secondary_params is not None:
        all_pareto_ids.update(pareto_df_secondary_params['trial_id'].values)
    if pareto_df_secondary_nodes is not None:
        all_pareto_ids.update(pareto_df_secondary_nodes['trial_id'].values)
    
    all_pareto_df = df[df['trial_id'].isin(all_pareto_ids)].copy()
    
    print(f"\nTotal unique Pareto models: {len(all_pareto_df)}")
    print(f"  From parameters (primary): {len(pareto_df_params)}")
    print(f"  From parameters (secondary): {len(pareto_df_secondary_params) if pareto_df_secondary_params is not None else 0}")
    print(f"  From nodes (primary): {len(pareto_df_nodes)}")
    print(f"  From nodes (secondary): {len(pareto_df_secondary_nodes) if pareto_df_secondary_nodes is not None else 0}")
    
    # Copy model files
    copy_model_files(all_pareto_df, None, args.output_dir)
    
    # Save all results
    save_results(df, pareto_df_params, pareto_df_secondary_params,
                pareto_df_nodes, pareto_df_secondary_nodes, args.output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("COMPLETE - HLS SYNTHESIS READY!")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nContents:")
    print(f"  - {len(all_pareto_df)} H5 model files (Pareto optimal)")
    print(f"  - 4 complexity analysis plots")
    print(f"  - 2 Pareto front plots (parameters + nodes)")
    print(f"  - Multiple CSV files with results")
    print(f"  - JSON summary file")
    print(f"\nReady for HLS synthesis! Run:")
    print(f"  python parallel_hls_synthesis.py \\")
    print(f"      --input_dir {args.output_dir} \\")
    print(f"      --num_workers 4")


if __name__ == "__main__":
    main()

