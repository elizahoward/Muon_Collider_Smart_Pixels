#!/usr/bin/env python3
"""
Select Pareto Optimal Models from Hyperparameter Search

This script identifies Pareto optimal models from complexity analysis results,
selecting models that represent the best trade-off between model complexity 
(nodes/parameters) and validation accuracy.

A model is Pareto optimal if no other model exists that is both:
- More accurate (higher validation accuracy)
- Less complex (fewer nodes/parameters)

Usage:
    python select_pareto_models.py --input_dir <complexity_analysis_dir> [options]

Author: Eric
Date: 2025
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


def is_dominated(point, other_points, maximize_cols, minimize_cols):
    """
    Check if a point is dominated by any other point.
    
    A point is dominated if there exists another point that is:
    - Better or equal in all objectives
    - Strictly better in at least one objective
    
    Args:
        point: DataFrame row to check
        other_points: DataFrame of other points to compare against
        maximize_cols: Columns to maximize (e.g., 'val_accuracy')
        minimize_cols: Columns to minimize (e.g., 'parameters', 'nodes')
        
    Returns:
        bool: True if point is dominated, False if on Pareto front
    """
    for _, other in other_points.iterrows():
        # Check if 'other' dominates 'point'
        better_in_all = True
        strictly_better_in_one = False
        
        # Check maximize objectives (higher is better)
        for col in maximize_cols:
            if other[col] < point[col]:
                better_in_all = False
                break
            if other[col] > point[col]:
                strictly_better_in_one = True
        
        if not better_in_all:
            continue
        
        # Check minimize objectives (lower is better)
        for col in minimize_cols:
            if other[col] > point[col]:
                better_in_all = False
                break
            if other[col] < point[col]:
                strictly_better_in_one = True
        
        # If other is better in all and strictly better in at least one
        if better_in_all and strictly_better_in_one:
            return True
    
    return False


def find_pareto_front(df, maximize_cols=['val_accuracy'], minimize_cols=['parameters']):
    """
    Find Pareto optimal points in a DataFrame.
    
    Args:
        df: DataFrame with model data
        maximize_cols: Columns to maximize
        minimize_cols: Columns to minimize
        
    Returns:
        DataFrame: Subset of df containing only Pareto optimal points
    """
    pareto_indices = []
    
    for idx, row in df.iterrows():
        # Get all other points
        other_points = df.drop(idx)
        
        # Check if this point is dominated
        if not is_dominated(row, other_points, maximize_cols, minimize_cols):
            pareto_indices.append(idx)
    
    pareto_df = df.loc[pareto_indices].copy()
    
    # Sort by accuracy (descending) for nice visualization
    pareto_df = pareto_df.sort_values(maximize_cols[0], ascending=False)
    
    return pareto_df


def load_complexity_data(input_dir):
    """
    Load complexity analysis data from CSV files.
    
    Args:
        input_dir: Directory containing complexity CSV files
        
    Returns:
        DataFrame: Model complexity and accuracy data
    """
    # Try to find the summary CSV file
    summary_file = os.path.join(input_dir, 'hyperparameter_complexity_summary.csv')
    detailed_file = os.path.join(input_dir, 'hyperparameter_detailed_results.csv')
    
    if os.path.exists(detailed_file):
        print(f"Loading data from: {detailed_file}")
        df = pd.read_csv(detailed_file)
    elif os.path.exists(summary_file):
        print(f"Loading data from: {summary_file}")
        df = pd.read_csv(summary_file)
    else:
        print(f"Error: No complexity CSV file found in {input_dir}")
        print("Expected: hyperparameter_complexity_summary.csv or hyperparameter_detailed_results.csv")
        return None
    
    # Validate required columns
    required_cols = ['val_accuracy', 'trial_id']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV missing required columns: {required_cols}")
        return None
    
    # Check for complexity metrics
    if 'parameters' not in df.columns and 'nodes' not in df.columns:
        print("Error: CSV missing complexity metrics (parameters or nodes)")
        return None
    
    print(f"Loaded {len(df)} models from CSV")
    return df


def plot_pareto_front(df, pareto_df, output_dir, model_name, complexity_metric='parameters', 
                     pareto_df_secondary=None, tier='primary'):
    """
    Create scatter plots showing all models with Pareto front highlighted.
    
    Args:
        df: DataFrame with all models
        pareto_df: DataFrame with Pareto optimal models
        output_dir: Directory to save plots
        model_name: Name of the model for titles
        complexity_metric: 'parameters' or 'nodes'
        pareto_df_secondary: Optional secondary Pareto front (for two-tier visualization)
        tier: 'primary', 'secondary', or 'combined'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with larger size for better visibility
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all models (gray, smaller, transparent)
    ax.scatter(df[complexity_metric], df['val_accuracy'], 
               alpha=0.4, s=60, c='lightgray', edgecolors='gray', 
               linewidth=0.5, label='All models', zorder=1)
    
    # Plot primary Pareto front models (red, larger, opaque)
    ax.scatter(pareto_df[complexity_metric], pareto_df['val_accuracy'], 
               alpha=0.9, s=120, c='red', edgecolors='darkred', 
               linewidth=1.5, label='Primary Pareto front', zorder=3, marker='D')
    
    # Connect primary Pareto front points with a line
    pareto_sorted = pareto_df.sort_values(complexity_metric)
    ax.plot(pareto_sorted[complexity_metric], pareto_sorted['val_accuracy'],
            'r--', alpha=0.6, linewidth=2, zorder=2)
    
    # Plot secondary Pareto front if provided (orange, slightly smaller)
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        ax.scatter(pareto_df_secondary[complexity_metric], pareto_df_secondary['val_accuracy'], 
                   alpha=0.8, s=100, c='orange', edgecolors='darkorange', 
                   linewidth=1.5, label='Secondary Pareto front (redundancy)', zorder=3, marker='s')
        
        # Connect secondary Pareto front points with a line
        pareto_sorted_2 = pareto_df_secondary.sort_values(complexity_metric)
        ax.plot(pareto_sorted_2[complexity_metric], pareto_sorted_2['val_accuracy'],
                'orange', linestyle=':', alpha=0.6, linewidth=2, zorder=2)
    
    # Annotate primary Pareto points with trial IDs
    for _, row in pareto_df.iterrows():
        trial_id = row['trial_id']
        # Format trial_id - handle both int and string formats
        if isinstance(trial_id, str):
            label = trial_id
        else:
            label = f"{int(trial_id):03d}"
        
        ax.annotate(label, 
                   xy=(row[complexity_metric], row['val_accuracy']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, color='darkred', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                            alpha=0.7, edgecolor='darkred', linewidth=1),
                   zorder=4)
    
    # Annotate secondary Pareto points with trial IDs if provided
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        for _, row in pareto_df_secondary.iterrows():
            trial_id = row['trial_id']
            if isinstance(trial_id, str):
                label = trial_id
            else:
                label = f"{int(trial_id):03d}"
            
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
    
    title_suffix = ''
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        title_suffix = ' (Two-Tier with Redundancy)'
    ax.set_title(f'{model_name}: Pareto Front - {metric_label} vs Accuracy{title_suffix}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Add statistics box
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
            f"Pareto optimal: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
            f"Acc range: {df['val_accuracy'].min():.4f} - {df['val_accuracy'].max():.4f}\n"
            f"{metric_label.split()[0]} range: {df[complexity_metric].min():.0f} - {df[complexity_metric].max():.0f}"
        )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot - use tier suffix if specified
    filename = f'pareto_front_{complexity_metric}'
    if tier != 'primary':
        filename = f'{filename}_{tier}'
    plot_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    plt.close()


def save_pareto_summary(pareto_df, output_dir, complexity_metric, tier='primary', pareto_df_secondary=None):
    """
    Save Pareto optimal models to CSV and JSON.
    
    Args:
        pareto_df: DataFrame with Pareto optimal models
        output_dir: Output directory
        complexity_metric: Complexity metric used ('parameters' or 'nodes')
        tier: 'primary', 'secondary', or 'combined'
        pareto_df_secondary: Optional secondary Pareto front DataFrame
    """
    # Save primary Pareto CSV
    csv_path = os.path.join(output_dir, f'pareto_optimal_models_{complexity_metric}_primary.csv')
    pareto_df.to_csv(csv_path, index=False)
    print(f"✓ Saved primary CSV: {csv_path}")
    
    # Save secondary Pareto CSV if provided
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        csv_path_secondary = os.path.join(output_dir, f'pareto_optimal_models_{complexity_metric}_secondary.csv')
        pareto_df_secondary.to_csv(csv_path_secondary, index=False)
        print(f"✓ Saved secondary CSV: {csv_path_secondary}")
        
        # Save combined CSV
        combined_df = pd.concat([pareto_df, pareto_df_secondary], ignore_index=True)
        combined_df = combined_df.sort_values('val_accuracy', ascending=False)
        csv_path_combined = os.path.join(output_dir, f'pareto_optimal_models_{complexity_metric}_combined.csv')
        combined_df.to_csv(csv_path_combined, index=False)
        print(f"✓ Saved combined CSV: {csv_path_combined}")
    
    # Save JSON with metadata
    summary = {
        'selection_method': 'pareto_optimal_two_tier',
        'complexity_metric': complexity_metric,
        'primary_pareto_models': len(pareto_df),
        'primary_models': pareto_df.to_dict('records')
    }
    
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        summary['secondary_pareto_models'] = len(pareto_df_secondary)
        summary['secondary_models'] = pareto_df_secondary.to_dict('records')
        summary['total_models'] = len(pareto_df) + len(pareto_df_secondary)
    
    json_path = os.path.join(output_dir, f'pareto_optimal_models_{complexity_metric}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved JSON: {json_path}")


def copy_pareto_model_files(pareto_df, models_dir, output_dir):
    """
    Copy H5 model files for Pareto optimal models.
    
    Args:
        pareto_df: DataFrame with Pareto optimal models
        models_dir: Directory containing model H5 files
        output_dir: Destination directory
        
    Returns:
        int: Number of successfully copied files
    """
    if not os.path.exists(models_dir):
        print(f"Warning: Models directory not found: {models_dir}")
        return 0
    
    success_count = 0
    
    for _, row in pareto_df.iterrows():
        trial_id = row['trial_id']
        
        # Handle both int and string trial IDs
        if isinstance(trial_id, str):
            # Already formatted (e.g., "032")
            formatted_id = trial_id
        else:
            # Format as zero-padded 3-digit string
            formatted_id = f"{int(trial_id):03d}"
        
        # Try different possible file naming patterns
        possible_names = [
            f'model_trial_{formatted_id}.h5',
            f'best_model_trial_{formatted_id}.h5',
            f'trial_{formatted_id}_model.h5',
            f'model_{formatted_id}.h5'
        ]
        
        src_path = None
        for name in possible_names:
            candidate = os.path.join(models_dir, name)
            if os.path.exists(candidate):
                src_path = candidate
                break
        
        if src_path:
            dst_path = os.path.join(output_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ Copied: {os.path.basename(src_path)} (trial {formatted_id})")
            success_count += 1
        else:
            print(f"  ✗ Model file not found for trial {formatted_id}")
    
    return success_count


def print_pareto_statistics(df, pareto_df, complexity_metric, pareto_df_secondary=None):
    """Print statistics about Pareto optimal selection."""
    print("\n" + "=" * 80)
    print("PARETO OPTIMAL SELECTION STATISTICS")
    print("=" * 80)
    
    print(f"\nComplexity metric: {complexity_metric}")
    print(f"Total models: {len(df)}")
    print(f"Primary Pareto optimal: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)")
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        print(f"Secondary Pareto optimal: {len(pareto_df_secondary)} ({100*len(pareto_df_secondary)/len(df):.1f}%)")
        print(f"Total selected: {len(pareto_df) + len(pareto_df_secondary)} ({100*(len(pareto_df) + len(pareto_df_secondary))/len(df):.1f}%)")
    
    print(f"\n{'All Models':<30} {'Primary Pareto':<30}")
    print("-" * 80)
    
    # Accuracy stats
    print(f"\n{'Validation Accuracy:':<30}")
    print(f"  Min:  {df['val_accuracy'].min():<20.4f} {pareto_df['val_accuracy'].min():<20.4f}")
    print(f"  Max:  {df['val_accuracy'].max():<20.4f} {pareto_df['val_accuracy'].max():<20.4f}")
    print(f"  Mean: {df['val_accuracy'].mean():<20.4f} {pareto_df['val_accuracy'].mean():<20.4f}")
    
    # Complexity stats
    print(f"\n{complexity_metric.capitalize()}:")
    print(f"  Min:  {df[complexity_metric].min():<20.0f} {pareto_df[complexity_metric].min():<20.0f}")
    print(f"  Max:  {df[complexity_metric].max():<20.0f} {pareto_df[complexity_metric].max():<20.0f}")
    print(f"  Mean: {df[complexity_metric].mean():<20.1f} {pareto_df[complexity_metric].mean():<20.1f}")
    
    # List all Primary Pareto optimal models
    print(f"\n{'Primary Pareto Optimal Models:':<30}")
    print(f"{'Trial ID':<12} {'Accuracy':<12} {complexity_metric.capitalize():<15}")
    print("-" * 80)
    
    pareto_sorted = pareto_df.sort_values('val_accuracy', ascending=False)
    for _, row in pareto_sorted.iterrows():
        trial_id = row['trial_id']
        if isinstance(trial_id, str):
            trial_str = trial_id
        else:
            trial_str = f"{int(trial_id):03d}"
        
        print(f"{trial_str:<12} {row['val_accuracy']:<12.4f} {row[complexity_metric]:<15.0f}")
    
    # List Secondary Pareto models if present
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        print(f"\n{'Secondary Pareto Optimal Models (Redundancy):':<30}")
        print(f"{'Trial ID':<12} {'Accuracy':<12} {complexity_metric.capitalize():<15}")
        print("-" * 80)
        
        pareto_sorted_2 = pareto_df_secondary.sort_values('val_accuracy', ascending=False)
        for _, row in pareto_sorted_2.iterrows():
            trial_id = row['trial_id']
            if isinstance(trial_id, str):
                trial_str = trial_id
            else:
                trial_str = f"{int(trial_id):03d}"
            
            print(f"{trial_str:<12} {row['val_accuracy']:<12.4f} {row[complexity_metric]:<15.0f}")


def main():
    parser = argparse.ArgumentParser(
        description='Select Pareto optimal models from complexity analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select Pareto optimal models based on parameters
  python select_pareto_models.py \\
      --input_dir ../complexity_analysis/model2_quantized_4w0i_hyperparameter_search \\
      --output_dir ../model2_pareto_models
  
  # Select based on nodes instead of parameters
  python select_pareto_models.py \\
      --input_dir ../complexity_analysis/model3_quantized_4w0i_hyperparameter_search \\
      --output_dir ../model3_pareto_models \\
      --complexity_metric nodes
  
  # Generate plots only, don't copy model files
  python select_pareto_models.py \\
      --input_dir ../complexity_analysis/model2_quantized_4w0i_hyperparameter_search \\
      --output_dir ../model2_pareto_analysis \\
      --plot_only
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing complexity analysis CSV files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for plots and selected models'
    )
    
    parser.add_argument(
        '--complexity_metric',
        type=str,
        choices=['parameters', 'nodes', 'both'],
        default='parameters',
        help='Complexity metric to use for Pareto selection (default: parameters)'
    )
    
    parser.add_argument(
        '--models_dir',
        type=str,
        default=None,
        help='Directory containing model H5 files (if different from input_dir parent)'
    )
    
    parser.add_argument(
        '--plot_only',
        action='store_true',
        help='Only generate plots, do not copy model files'
    )
    
    parser.add_argument(
        '--min_accuracy',
        type=float,
        default=None,
        help='Minimum accuracy threshold to consider models (e.g., 0.55)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Load complexity data
    df = load_complexity_data(args.input_dir)
    if df is None:
        sys.exit(1)
    
    # Filter by minimum accuracy if specified
    if args.min_accuracy is not None:
        original_len = len(df)
        df = df[df['val_accuracy'] >= args.min_accuracy].copy()
        print(f"Filtered {original_len - len(df)} models below accuracy threshold {args.min_accuracy}")
        print(f"Remaining models: {len(df)}")
    
    # Extract model name from input directory
    model_name = os.path.basename(args.input_dir.rstrip('/'))
    if not model_name:
        model_name = "Model"
    
    # Create output directory for model files
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory (for model files): {args.output_dir}")
    
    # Plots will be saved in the input directory
    print(f"Plots directory: {args.input_dir}")
    
    # Determine which complexity metrics to use
    metrics_to_process = []
    if args.complexity_metric == 'both':
        if 'parameters' in df.columns:
            metrics_to_process.append('parameters')
        if 'nodes' in df.columns:
            metrics_to_process.append('nodes')
    else:
        if args.complexity_metric not in df.columns:
            print(f"Error: Complexity metric '{args.complexity_metric}' not found in data")
            sys.exit(1)
        metrics_to_process.append(args.complexity_metric)
    
    # Process each complexity metric
    for complexity_metric in metrics_to_process:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING WITH COMPLEXITY METRIC: {complexity_metric.upper()}")
        print(f"{'=' * 80}")
        
        # Find PRIMARY Pareto front
        print(f"\n--- TIER 1: PRIMARY PARETO FRONT ---")
        print(f"Finding primary Pareto optimal models...")
        pareto_df = find_pareto_front(df, 
                                      maximize_cols=['val_accuracy'], 
                                      minimize_cols=[complexity_metric])
        
        if pareto_df.empty:
            print("Warning: No primary Pareto optimal models found")
            continue
        
        print(f"✓ Found {len(pareto_df)} primary Pareto optimal models")
        
        # Find SECONDARY Pareto front (for redundancy)
        print(f"\n--- TIER 2: SECONDARY PARETO FRONT (REDUNDANCY) ---")
        print(f"Removing primary Pareto models and finding secondary front...")
        
        # Get all models except primary Pareto models
        primary_trial_ids = set(pareto_df['trial_id'].values)
        df_remaining = df[~df['trial_id'].isin(primary_trial_ids)].copy()
        
        if len(df_remaining) > 1:
            pareto_df_secondary = find_pareto_front(df_remaining,
                                                   maximize_cols=['val_accuracy'],
                                                   minimize_cols=[complexity_metric])
            
            if not pareto_df_secondary.empty:
                print(f"✓ Found {len(pareto_df_secondary)} secondary Pareto optimal models")
            else:
                print("  No secondary Pareto models found")
                pareto_df_secondary = None
        else:
            print("  Insufficient remaining models for secondary Pareto front")
            pareto_df_secondary = None
        
        # Print statistics
        print_pareto_statistics(df, pareto_df, complexity_metric, pareto_df_secondary)
        
        # Create plots (saved to input_dir)
        print(f"\n{'Creating visualizations...'}")
        plot_pareto_front(df, pareto_df, args.input_dir, model_name, complexity_metric,
                         pareto_df_secondary=pareto_df_secondary, tier='combined')
        
        # Save summary files (saved to output_dir)
        print(f"\n{'Saving results...'}")
        save_pareto_summary(pareto_df, args.output_dir, complexity_metric, 
                          tier='combined', pareto_df_secondary=pareto_df_secondary)
        
        # Copy model files if requested
        if not args.plot_only:
            print(f"\n{'Copying model files...'}")
            
            # Determine models directory
            if args.models_dir:
                models_dir = args.models_dir
            else:
                # Try to find models in parent directory or hyperparameter_tuning
                parent_dir = os.path.dirname(args.input_dir)
                possible_dirs = [
                    parent_dir,
                    os.path.join(parent_dir, 'hyperparameter_tuning'),
                    os.path.join(parent_dir, '..', 'hyperparameter_tuning'),
                ]
                
                models_dir = None
                for pdir in possible_dirs:
                    if os.path.exists(pdir) and any(f.endswith('.h5') for f in os.listdir(pdir) if os.path.isfile(os.path.join(pdir, f))):
                        models_dir = pdir
                        break
            
            if models_dir:
                print(f"  Models directory: {models_dir}")
                
                # Copy primary Pareto models
                print(f"\n  Copying primary Pareto models...")
                success_count_primary = copy_pareto_model_files(pareto_df, models_dir, args.output_dir)
                print(f"  Successfully copied: {success_count_primary}/{len(pareto_df)} primary model files")
                
                # Copy secondary Pareto models if they exist
                if pareto_df_secondary is not None and not pareto_df_secondary.empty:
                    print(f"\n  Copying secondary Pareto models (redundancy)...")
                    success_count_secondary = copy_pareto_model_files(pareto_df_secondary, models_dir, args.output_dir)
                    print(f"  Successfully copied: {success_count_secondary}/{len(pareto_df_secondary)} secondary model files")
                    print(f"\n  Total copied: {success_count_primary + success_count_secondary}/{len(pareto_df) + len(pareto_df_secondary)} model files")
            else:
                print("  Warning: Could not find models directory")
                print("  Specify with --models_dir to copy model files")
    
    # Final summary
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nPlots saved to: {args.input_dir}")
    print(f"Model files and CSVs saved to: {args.output_dir}")
    print(f"\nTwo-tier Pareto selection completed:")
    print(f"  - PRIMARY tier: Best accuracy/complexity trade-offs")
    print(f"  - SECONDARY tier: Redundancy alternatives (backup models)")
    print(f"\nPareto optimal models represent the best trade-off between:")
    print(f"  - Maximizing validation accuracy")
    print(f"  - Minimizing model complexity")
    print(f"\nThe secondary tier provides redundancy for robustness and alternatives.")


if __name__ == "__main__":
    main()

