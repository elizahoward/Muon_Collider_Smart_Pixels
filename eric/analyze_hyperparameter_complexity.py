#!/usr/bin/env python3
"""
Analyze Hyperparameter Tuning Results: Model Complexity vs Accuracy

This script parses hyperparameter tuning results from Model2 and Model3,
calculates model complexity (total number of nodes/parameters), and plots
the relationship between model complexity and validation accuracy.

Usage:
    python analyze_hyperparameter_complexity.py

Author: Eric
Date: 2024
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def calculate_model2_complexity(hyperparams):
    """
    Calculate total number of nodes and parameters for Model2.
    
    Model2 architecture:
    - Input: x_local, z_global, y_local (scalars)
    - Branch 1: x_local, z_global -> xz_units
    - Branch 2: y_local -> yl_units
    - Merged: xz_units + yl_units -> merged_units1 -> merged_units2 -> merged_units3 -> 1
    
    Total nodes = xz_units + yl_units + merged_units1 + merged_units2 + merged_units3 + 1
    """
    xz_units = hyperparams.get('xz_units', 0)
    yl_units = hyperparams.get('yl_units', 0)
    merged_units1 = hyperparams.get('merged_units1', 0)
    merged_units2 = hyperparams.get('merged_units2', 0)
    merged_units3 = hyperparams.get('merged_units3', 0)
    
    # Calculate nodes
    total_nodes = xz_units + yl_units + merged_units1 + merged_units2 + merged_units3 + 1
    
    # Calculate parameters (weights + biases)
    # Branch 1: 2 inputs (x_local, z_global) -> xz_units
    params_xz = (2 * xz_units) + xz_units
    # Branch 2: 1 input (y_local) -> yl_units  
    params_yl = (1 * yl_units) + yl_units
    # Merged layers
    params_merged1 = ((xz_units + yl_units) * merged_units1) + merged_units1
    params_merged2 = (merged_units1 * merged_units2) + merged_units2
    params_merged3 = (merged_units2 * merged_units3) + merged_units3
    params_output = (merged_units3 * 1) + 1
    
    total_params = params_xz + params_yl + params_merged1 + params_merged2 + params_merged3 + params_output
    
    return total_nodes, total_params


def calculate_model3_complexity(hyperparams):
    """
    Calculate total number of nodes and parameters for Model3.
    
    Model3 architecture:
    - Conv2D: cluster (13x21x1) -> conv_filters filters (3x3 kernel) -> pooling (6x10) -> flatten
    - Scalar branch: z_global + y_local (2 inputs) -> scalar_dense_units
    - Merged: conv_output + scalar_dense_units -> merged_dense_1 -> merged_dense_2 -> 1
    
    Total nodes = conv_filters*60 (approx flattened) + scalar_dense_units + merged_dense_1 + merged_dense_2 + 1
    """
    conv_filters = hyperparams.get('conv_filters', 0)
    kernel_rows = hyperparams.get('kernel_rows', 3)
    kernel_cols = hyperparams.get('kernel_cols', 3)
    scalar_dense_units = hyperparams.get('scalar_dense_units', 0)
    merged_dense_1 = hyperparams.get('merged_dense_1', 0)
    merged_dense_2 = hyperparams.get('merged_dense_2', 0)
    
    # Calculate nodes
    # After Conv2D (13x21) -> pooling (6x10) -> flatten
    # Approximate flattened conv output size: 6 * 10 * conv_filters = 60 * conv_filters
    conv_nodes = 60 * conv_filters
    total_nodes = conv_nodes + scalar_dense_units + merged_dense_1 + merged_dense_2 + 1
    
    # Calculate parameters (weights + biases)
    # Conv2D layer: (kernel_rows * kernel_cols * input_channels * conv_filters) + conv_filters
    # Input has 1 channel
    params_conv = (kernel_rows * kernel_cols * 1 * conv_filters) + conv_filters
    
    # Scalar dense layer: 2 inputs (z_global, y_local concatenated) -> scalar_dense_units
    params_scalar = (2 * scalar_dense_units) + scalar_dense_units
    
    # Merged layers
    # Input to merged_dense_1: 60*conv_filters (flattened conv) + scalar_dense_units
    conv_flattened_size = 60 * conv_filters
    params_merged1 = ((conv_flattened_size + scalar_dense_units) * merged_dense_1) + merged_dense_1
    params_merged2 = (merged_dense_1 * merged_dense_2) + merged_dense_2
    params_output = (merged_dense_2 * 1) + 1
    
    total_params = params_conv + params_scalar + params_merged1 + params_merged2 + params_output
    
    return total_nodes, total_params


def parse_trial_folder(trial_path):
    """Parse a single trial folder and extract hyperparameters and metrics."""
    trial_json_path = os.path.join(trial_path, 'trial.json')
    
    if not os.path.exists(trial_json_path):
        return None
    
    try:
        with open(trial_json_path, 'r') as f:
            trial_data = json.load(f)
        
        hyperparams = trial_data['hyperparameters']['values']
        score = trial_data.get('score', None)
        status = trial_data.get('status', 'UNKNOWN')
        
        # Only include completed trials with valid scores
        if status == 'COMPLETED' and score is not None:
            return {
                'trial_id': trial_data['trial_id'],
                'hyperparams': hyperparams,
                'val_accuracy': score,
                'status': status
            }
    except Exception as e:
        print(f"Error parsing {trial_path}: {e}")
    
    return None


def analyze_hyperparameter_search(search_dir, model_name, complexity_func):
    """
    Analyze all trials in a hyperparameter search directory.
    
    Args:
        search_dir: Path to hyperparameter search directory
        model_name: Name of the model (for labeling)
        complexity_func: Function to calculate model complexity
    
    Returns:
        DataFrame with trial results
    """
    results = []
    
    # Find all trial directories
    trial_dirs = sorted([d for d in os.listdir(search_dir) 
                        if d.startswith('trial_') and os.path.isdir(os.path.join(search_dir, d))])
    
    print(f"\nAnalyzing {model_name} ({len(trial_dirs)} trials)...")
    
    for trial_dir in trial_dirs:
        trial_path = os.path.join(search_dir, trial_dir)
        trial_result = parse_trial_folder(trial_path)
        
        if trial_result:
            nodes, params = complexity_func(trial_result['hyperparams'])
            results.append({
                'model': model_name,
                'trial_id': trial_result['trial_id'],
                'nodes': nodes,
                'parameters': params,
                'val_accuracy': trial_result['val_accuracy'],
                'hyperparams': trial_result['hyperparams']
            })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        print(f"  Completed trials: {len(df)}")
        print(f"  Nodes range: {df['nodes'].min():.0f} - {df['nodes'].max():.0f}")
        print(f"  Parameters range: {df['parameters'].min():.0f} - {df['parameters'].max():.0f}")
        print(f"  Accuracy range: {df['val_accuracy'].min():.4f} - {df['val_accuracy'].max():.4f}")
        print(f"  Best accuracy: {df['val_accuracy'].max():.4f} (nodes: {df.loc[df['val_accuracy'].idxmax(), 'nodes']:.0f}, params: {df.loc[df['val_accuracy'].idxmax(), 'parameters']:.0f})")
    
    return df


def plot_complexity_vs_accuracy(df_model2, df_model3, output_dir):
    """Create simple scatter plots for each model showing nodes and parameters vs accuracy."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Model2 - Nodes vs Accuracy
    if not df_model2.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df_model2['nodes'], df_model2['val_accuracy'], 
                   alpha=0.6, s=80, c='blue', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Number of Nodes', fontsize=14)
        ax.set_ylabel('Validation Accuracy', fontsize=14)
        ax.set_title('Model2: Nodes vs Validation Accuracy', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'model2_nodes_vs_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    
    # Model2 - Parameters vs Accuracy
    if not df_model2.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df_model2['parameters'], df_model2['val_accuracy'], 
                   alpha=0.6, s=80, c='blue', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Number of Parameters', fontsize=14)
        ax.set_ylabel('Validation Accuracy', fontsize=14)
        ax.set_title('Model2: Parameters vs Validation Accuracy', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'model2_parameters_vs_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    
    # Model3 - Nodes vs Accuracy
    if not df_model3.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df_model3['nodes'], df_model3['val_accuracy'], 
                   alpha=0.6, s=80, c='red', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Number of Nodes', fontsize=14)
        ax.set_ylabel('Validation Accuracy', fontsize=14)
        ax.set_title('Model3: Nodes vs Validation Accuracy', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'model3_nodes_vs_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    
    # Model3 - Parameters vs Accuracy
    if not df_model3.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df_model3['parameters'], df_model3['val_accuracy'], 
                   alpha=0.6, s=80, c='red', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Number of Parameters', fontsize=14)
        ax.set_ylabel('Validation Accuracy', fontsize=14)
        ax.set_title('Model3: Parameters vs Validation Accuracy', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'model3_parameters_vs_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()


def save_summary_csv(df_model2, df_model3, output_dir):
    """Save combined results to CSV."""
    combined_df = pd.concat([df_model2, df_model3], ignore_index=True)
    
    # Select relevant columns for CSV
    csv_df = combined_df[['model', 'trial_id', 'nodes', 'parameters', 'val_accuracy']].copy()
    csv_df = csv_df.sort_values(['model', 'val_accuracy'], ascending=[True, False])
    
    csv_path = os.path.join(output_dir, 'hyperparameter_complexity_summary.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"✓ Summary CSV saved to: {csv_path}")
    
    # Save detailed results with hyperparameters
    detailed_path = os.path.join(output_dir, 'hyperparameter_detailed_results.csv')
    
    # Expand hyperparameters into separate columns
    detailed_rows = []
    for _, row in combined_df.iterrows():
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
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✓ Detailed results saved to: {detailed_path}")


def main():
    """Main function to analyze hyperparameter tuning results."""
    print("=== Hyperparameter Tuning Complexity Analysis ===")
    
    # Define paths
    base_dir = "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/hyperparameter_tuning"
    model2_dir = os.path.join(base_dir, "model2_hyperparameter_search")
    model3_dir = os.path.join(base_dir, "model3_hyperparameter_search")
    output_dir = "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/complexity_analysis"
    
    # Check if directories exist
    if not os.path.exists(model2_dir):
        print(f"WARNING: Model2 directory not found: {model2_dir}")
        df_model2 = pd.DataFrame()
    else:
        df_model2 = analyze_hyperparameter_search(model2_dir, 'Model2', calculate_model2_complexity)
    
    if not os.path.exists(model3_dir):
        print(f"WARNING: Model3 directory not found: {model3_dir}")
        df_model3 = pd.DataFrame()
    else:
        df_model3 = analyze_hyperparameter_search(model3_dir, 'Model3', calculate_model3_complexity)
    
    if df_model2.empty and df_model3.empty:
        print("ERROR: No valid trial data found for either model!")
        return
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    plot_complexity_vs_accuracy(df_model2, df_model3, output_dir)
    
    # Save results to CSV
    print("\n=== Saving Results ===")
    save_summary_csv(df_model2, df_model3, output_dir)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if not df_model2.empty:
        print("\nModel2:")
        print(f"  Total trials: {len(df_model2)}")
        print(f"  Mean accuracy: {df_model2['val_accuracy'].mean():.4f} ± {df_model2['val_accuracy'].std():.4f}")
        print(f"  Best accuracy: {df_model2['val_accuracy'].max():.4f}")
        print(f"  Mean nodes: {df_model2['nodes'].mean():.1f}")
        print(f"  Mean parameters: {df_model2['parameters'].mean():.1f}")
        corr_nodes = df_model2['nodes'].corr(df_model2['val_accuracy'])
        corr_params = df_model2['parameters'].corr(df_model2['val_accuracy'])
        print(f"  Nodes-Accuracy correlation: {corr_nodes:.4f}")
        print(f"  Parameters-Accuracy correlation: {corr_params:.4f}")
    
    if not df_model3.empty:
        print("\nModel3:")
        print(f"  Total trials: {len(df_model3)}")
        print(f"  Mean accuracy: {df_model3['val_accuracy'].mean():.4f} ± {df_model3['val_accuracy'].std():.4f}")
        print(f"  Best accuracy: {df_model3['val_accuracy'].max():.4f}")
        print(f"  Mean nodes: {df_model3['nodes'].mean():.1f}")
        print(f"  Mean parameters: {df_model3['parameters'].mean():.1f}")
        corr_nodes = df_model3['nodes'].corr(df_model3['val_accuracy'])
        corr_params = df_model3['parameters'].corr(df_model3['val_accuracy'])
        print(f"  Nodes-Accuracy correlation: {corr_nodes:.4f}")
        print(f"  Parameters-Accuracy correlation: {corr_params:.4f}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()

