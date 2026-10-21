#!/usr/bin/env python3
"""
Analyze Hyperparameter Tuning Results: Model Complexity vs Accuracy

This script parses hyperparameter tuning results from a specified directory,
calculates model complexity (total number of nodes/parameters), and plots
the relationship between model complexity and validation accuracy.

Supports Model2, Model2.5, and Model3 architectures with automatic model type detection.

Usage:
    python analyze_hyperparameter_complexity.py [search_directory]

    If no directory is provided, uses the default path specified in the script.

Examples:
    # Analyze Model2 results
    python analyze_hyperparameter_complexity.py hyperparameter_tuning/model2_quantized_4w0i_hyperparameter_search
    
    # Analyze Model2.5 results
    python analyze_hyperparameter_complexity.py hyperparameter_tuning/model2_5_quantized_4w0i_hyperparameter_search
    
    # Analyze Model3 results
    python analyze_hyperparameter_complexity.py hyperparameter_tuning/model3_quantized_4w0i_hyperparameter_search

Author: Eric
Date: 2024
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Try to import TensorFlow for loading H5 models
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Will use manual parameter calculation.")


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


def calculate_model2_5_complexity(hyperparams):
    """
    Calculate total number of nodes and parameters for Model2.5.
    
    Model2.5 architecture:
    - Input: x_profile (21), z_global (1), y_profile (13), y_local (1)
    - Spatial features branch: x_profile + y_profile + y_local (35 inputs) -> spatial_units
    - z_global branch: z_global (1 input) -> z_global_units
    - Merge: Concatenate both branches -> (spatial_units + z_global_units)
    - Dense2: (spatial_units + z_global_units) -> dense2_units
    - Dense3: dense2_units -> dense3_units
    - Output: dense3_units -> 1
    
    Total nodes = spatial_units + z_global_units + dense2_units + dense3_units + 1
    """
    spatial_units = hyperparams.get('spatial_units', 0)
    z_global_units = hyperparams.get('z_global_units', 0)
    dense2_units = hyperparams.get('dense2_units', 0)
    dense3_units = hyperparams.get('dense3_units', 0)
    
    # Calculate nodes
    total_nodes = spatial_units + z_global_units + dense2_units + dense3_units + 1
    
    # Calculate parameters (weights + biases)
    # Spatial features branch: 35 inputs (21 + 13 + 1) -> spatial_units
    params_spatial = (35 * spatial_units) + spatial_units
    
    # Z-global branch: 1 input -> z_global_units
    params_z_global = (1 * z_global_units) + z_global_units
    
    # Dense2: (spatial_units + z_global_units) -> dense2_units
    params_dense2 = ((spatial_units + z_global_units) * dense2_units) + dense2_units
    
    # Dense3: dense2_units -> dense3_units
    params_dense3 = (dense2_units * dense3_units) + dense3_units
    
    # Output layer: dense3_units -> 1
    params_output = (dense3_units * 1) + 1
    
    total_params = params_spatial + params_z_global + params_dense2 + params_dense3 + params_output
    
    return total_nodes, total_params


def calculate_model3_complexity(hyperparams):
    """
    Calculate total number of nodes and parameters for Model3.
    
    Model3 architecture:
    - Input: cluster (13x21), z_global, y_local (scalars)
    - Conv2D branch: cluster -> Reshape to (13x21x1) -> Conv2D -> MaxPooling2D (2x2) -> Flatten
    - Scalar branch: z_global + y_local -> Concatenate -> Dense (scalar_dense_units)
    - Merge: Concatenate conv output with scalar dense output
    - Head: Dense (merged_dense_1) -> Dropout -> Dense (merged_dense_2) -> Output (1)
    
    Total nodes = flattened_conv_units + scalar_dense_units + merged_dense_1 + merged_dense_2 + 1
    """
    conv_filters = hyperparams.get('conv_filters', 32)
    kernel_rows = hyperparams.get('kernel_rows', 3)
    kernel_cols = hyperparams.get('kernel_cols', 3)
    scalar_dense_units = hyperparams.get('scalar_dense_units', 32)
    merged_dense_1 = hyperparams.get('merged_dense_1', 200)
    
    # Calculate merged_dense_2 from multiplier if available
    merged_multiplier_2 = hyperparams.get('merged_multiplier_2', None)
    if merged_multiplier_2 is not None:
        merged_dense_2 = int(round(merged_dense_1 * merged_multiplier_2))
    else:
        merged_dense_2 = hyperparams.get('merged_dense_2', 100)
    
    # Calculate Conv2D output shape
    # Input: (13, 21, 1)
    # After Conv2D with "same" padding: (13, 21, conv_filters)
    # After MaxPooling2D (2x2): (floor(13/2), floor(21/2), conv_filters) = (6, 10, conv_filters)
    # After Flatten: 6 * 10 * conv_filters = 60 * conv_filters
    flattened_conv_units = 6 * 10 * conv_filters
    
    # Calculate nodes
    total_nodes = flattened_conv_units + scalar_dense_units + merged_dense_1 + merged_dense_2 + 1
    
    # Calculate parameters (weights + biases)
    # Conv2D layer: (kernel_rows * kernel_cols * input_channels * conv_filters) + conv_filters
    params_conv = (kernel_rows * kernel_cols * 1 * conv_filters) + conv_filters
    
    # Scalar branch: 2 inputs (z_global, y_local) -> scalar_dense_units
    params_scalar = (2 * scalar_dense_units) + scalar_dense_units
    
    # Merged dense 1: (flattened_conv_units + scalar_dense_units) -> merged_dense_1
    params_merged1 = ((flattened_conv_units + scalar_dense_units) * merged_dense_1) + merged_dense_1
    
    # Merged dense 2: merged_dense_1 -> merged_dense_2
    params_merged2 = (merged_dense_1 * merged_dense_2) + merged_dense_2
    
    # Output layer: merged_dense_2 -> 1
    params_output = (merged_dense_2 * 1) + 1
    
    total_params = params_conv + params_scalar + params_merged1 + params_merged2 + params_output
    
    return total_nodes, total_params


def extract_complexity_from_h5(h5_path):
    """
    Extract parameter count and node count directly from H5 model file.
    
    Args:
        h5_path: Path to H5 model file
        
    Returns:
        tuple: (total_nodes, total_params) or (None, None) if failed
    """
    if not TF_AVAILABLE:
        return None, None
    
    if not os.path.exists(h5_path):
        return None, None
    
    try:
        # Load model with custom objects for QKeras if needed
        custom_objects = {}
        try:
            from qkeras import QDense, QActivation, quantized_bits, quantized_relu
            custom_objects = {
                'QDense': QDense,
                'QActivation': QActivation,
                'quantized_bits': quantized_bits,
                'quantized_relu': quantized_relu
            }
        except ImportError:
            pass
        
        # Load the model
        model = keras.models.load_model(h5_path, custom_objects=custom_objects, compile=False)
        
        # Count total trainable parameters
        total_params = model.count_params()
        
        # Count nodes (neurons) in dense/conv layers
        total_nodes = 0
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            
            # For Dense and QDense layers, count units
            if 'Dense' in layer_type:
                units = layer.units
                total_nodes += units
            
            # For Conv2D layers, count filters
            elif 'Conv2D' in layer_type:
                filters = layer.filters
                # After pooling and flattening, this contributes to node count
                # For simplicity, we count output feature map size
                # This is approximate - actual node count after flatten depends on input shape
                pass  # Will be counted when flattened
            
            # For Flatten layers, get output shape
            elif 'Flatten' in layer_type:
                output_shape = layer.output_shape
                if output_shape and len(output_shape) > 1:
                    # Product of all dimensions except batch
                    nodes_from_flatten = int(np.prod(output_shape[1:]))
                    total_nodes += nodes_from_flatten
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
        
        return total_nodes, total_params
        
    except Exception as e:
        print(f"  Warning: Could not load H5 model {os.path.basename(h5_path)}: {e}")
        return None, None


def detect_model_type(hyperparams):
    """
    Detect whether hyperparameters are for Model2, Model2.5, or Model3.
    
    Args:
        hyperparams: Dictionary of hyperparameters
    
    Returns:
        'model2', 'model2_5', or 'model3'
    """
    # Model3 has conv_filters
    if 'conv_filters' in hyperparams:
        return 'model3'
    # Model2.5 has spatial_units
    elif 'spatial_units' in hyperparams:
        return 'model2_5'
    # Model2 has xz_units
    elif 'xz_units' in hyperparams:
        return 'model2'
    else:
        raise ValueError(f"Cannot detect model type from hyperparameters: {hyperparams.keys()}")


def parse_trial_folder(trial_path, trial_id=None):
    """
    Parse a single trial folder and extract hyperparameters and metrics.
    Supports both Keras Tuner format (trial_XXX/trial.json) and flat format.
    """
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
                'status': status,
                'is_keras_tuner_format': True
            }
    except Exception as e:
        print(f"Error parsing {trial_path}: {e}")
    
    return None


def parse_flat_format_trial(search_dir, trial_id):
    """
    Parse trial from flat directory format (model_trial_XXX.h5 + hyperparams_trial_XXX.json).
    
    Args:
        search_dir: Base directory containing model and hyperparam files
        trial_id: Trial ID number (e.g., 1, 2, 3...)
        
    Returns:
        dict with trial information or None
    """
    # Format trial ID as zero-padded 3-digit string
    trial_id_str = f"{trial_id:03d}"
    
    # Look for hyperparameters file
    hyperparam_file = os.path.join(search_dir, f'hyperparams_trial_{trial_id_str}.json')
    if not os.path.exists(hyperparam_file):
        return None
    
    # Look for model file
    model_file = os.path.join(search_dir, f'model_trial_{trial_id_str}.h5')
    if not os.path.exists(model_file):
        return None
    
    try:
        # Load hyperparameters
        with open(hyperparam_file, 'r') as f:
            hyperparams = json.load(f)
        
        # Try to load validation accuracy from trials_summary.json
        summary_file = os.path.join(search_dir, 'trials_summary.json')
        val_accuracy = None
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                # Find this trial in the summary
                for trial in summary_data:
                    if trial.get('trial_id') == trial_id or trial.get('trial_id') == trial_id_str:
                        val_accuracy = trial.get('val_accuracy') or trial.get('validation_accuracy') or trial.get('score')
                        break
            except Exception as e:
                print(f"  Warning: Could not parse trials_summary.json: {e}")
        
        # Only return if we have validation accuracy
        if val_accuracy is not None and val_accuracy > 0:
            return {
                'trial_id': trial_id_str,
                'hyperparams': hyperparams,
                'val_accuracy': val_accuracy,
                'status': 'COMPLETED',
                'is_keras_tuner_format': False,
                'model_file': model_file
            }
    
    except Exception as e:
        print(f"  Warning: Error parsing trial {trial_id_str}: {e}")
    
    return None


def analyze_hyperparameter_search(search_dir, model_name, complexity_func=None, min_accuracy=0.55):
    """
    Analyze all trials in a hyperparameter search directory.
    Supports both Keras Tuner format and flat H5 format.
    
    Args:
        search_dir: Path to hyperparameter search directory
        model_name: Name of the model (for labeling)
        complexity_func: Function to calculate model complexity (auto-detected if None)
        min_accuracy: Minimum accuracy threshold to include (default: 0.55 to exclude failed models ~0.5)
    
    Returns:
        DataFrame with trial results
    """
    results = []
    
    # Detect directory format
    trial_dirs = sorted([d for d in os.listdir(search_dir) 
                        if d.startswith('trial_') and os.path.isdir(os.path.join(search_dir, d))])
    
    # Check for flat format (model_trial_XXX.h5 files)
    h5_files = sorted([f for f in os.listdir(search_dir) 
                       if f.startswith('model_trial_') and f.endswith('.h5')])
    
    is_flat_format = len(h5_files) > 0 and len(trial_dirs) == 0
    use_h5_extraction = TF_AVAILABLE and (is_flat_format or len(h5_files) > 0)
    
    if is_flat_format:
        print(f"\nAnalyzing {model_name} (flat format, {len(h5_files)} H5 models)...")
        print(f"  Will extract parameters directly from H5 files")
        
        # Extract trial IDs from H5 filenames
        trial_ids = []
        for h5_file in h5_files:
            try:
                trial_id = int(h5_file.replace('model_trial_', '').replace('.h5', ''))
                trial_ids.append(trial_id)
            except:
                pass
        
        # Parse each trial
        for trial_id in sorted(trial_ids):
            trial_result = parse_flat_format_trial(search_dir, trial_id)
            
            if trial_result:
                # Try to extract from H5 first
                nodes, params = extract_complexity_from_h5(trial_result['model_file'])
                
                # Fall back to manual calculation if H5 extraction failed
                if nodes is None or params is None:
                    # Auto-detect model type if needed
                    if complexity_func is None:
                        detected_model_type = detect_model_type(trial_result['hyperparams'])
                        if detected_model_type == 'model2':
                            complexity_func = calculate_model2_complexity
                        elif detected_model_type == 'model2_5':
                            complexity_func = calculate_model2_5_complexity
                        elif detected_model_type == 'model3':
                            complexity_func = calculate_model3_complexity
                    
                    nodes, params = complexity_func(trial_result['hyperparams'])
                
                results.append({
                    'model': model_name,
                    'trial_id': trial_result['trial_id'],
                    'nodes': nodes,
                    'parameters': params,
                    'val_accuracy': trial_result['val_accuracy'],
                    'hyperparams': trial_result['hyperparams']
                })
    
    else:
        # Keras Tuner format
        print(f"\nAnalyzing {model_name} (Keras Tuner format, {len(trial_dirs)} trials)...")
        if use_h5_extraction:
            print(f"  Will try to extract parameters from H5 files when available")
        
        # Auto-detect model type from first trial if complexity_func not provided
        detected_model_type = None
        if complexity_func is None:
            for trial_dir in trial_dirs:
                trial_path = os.path.join(search_dir, trial_dir)
                trial_result = parse_trial_folder(trial_path)
                if trial_result:
                    detected_model_type = detect_model_type(trial_result['hyperparams'])
                    if detected_model_type == 'model2':
                        complexity_func = calculate_model2_complexity
                    elif detected_model_type == 'model2_5':
                        complexity_func = calculate_model2_5_complexity
                    elif detected_model_type == 'model3':
                        complexity_func = calculate_model3_complexity
                    print(f"  Auto-detected model type: {detected_model_type.upper()}")
                    break
            
            if complexity_func is None:
                raise ValueError(f"Could not auto-detect model type from trials in {search_dir}")
        
        for trial_dir in trial_dirs:
            trial_path = os.path.join(search_dir, trial_dir)
            trial_result = parse_trial_folder(trial_path)
            
            if trial_result:
                # Try to find and load H5 file for this trial
                nodes, params = None, None
                
                if use_h5_extraction:
                    # Look for H5 file in trial directory or parent
                    possible_h5_paths = [
                        os.path.join(trial_path, 'best_model.h5'),
                        os.path.join(trial_path, 'model.h5'),
                        os.path.join(search_dir, f"model_trial_{trial_result['trial_id']}.h5")
                    ]
                    
                    for h5_path in possible_h5_paths:
                        if os.path.exists(h5_path):
                            nodes, params = extract_complexity_from_h5(h5_path)
                            if nodes is not None and params is not None:
                                break
                
                # Fall back to manual calculation
                if nodes is None or params is None:
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
        total_trials = len(df)
        # Filter out failed models with accuracy around 0.5
        df_filtered = df[df['val_accuracy'] >= min_accuracy].copy()
        excluded_count = total_trials - len(df_filtered)
        
        print(f"  Completed trials: {total_trials}")
        if excluded_count > 0:
            print(f"  Excluded {excluded_count} failed models (accuracy < {min_accuracy})")
        print(f"  Valid trials after filtering: {len(df_filtered)}")
        if not df_filtered.empty:
            print(f"  Nodes range: {df_filtered['nodes'].min():.0f} - {df_filtered['nodes'].max():.0f}")
            print(f"  Parameters range: {df_filtered['parameters'].min():.0f} - {df_filtered['parameters'].max():.0f}")
            print(f"  Accuracy range: {df_filtered['val_accuracy'].min():.4f} - {df_filtered['val_accuracy'].max():.4f}")
            print(f"  Best accuracy: {df_filtered['val_accuracy'].max():.4f} (nodes: {df_filtered.loc[df_filtered['val_accuracy'].idxmax(), 'nodes']:.0f}, params: {df_filtered.loc[df_filtered['val_accuracy'].idxmax(), 'parameters']:.0f})")
    
    return df_filtered if not df.empty else df


def plot_complexity_vs_accuracy(df, model_name, output_dir):
    """Create scatter plots showing nodes and parameters vs accuracy for a single model."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        print("WARNING: No data to plot!")
        return
    
    # Nodes vs Accuracy
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df['nodes'], df['val_accuracy'], 
               alpha=0.6, s=80, c='blue', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Number of Nodes', fontsize=14)
    ax.set_ylabel('Validation Accuracy', fontsize=14)
    ax.set_title(f'{model_name}: Nodes vs Validation Accuracy', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name.lower()}_nodes_vs_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    # Parameters vs Accuracy
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df['parameters'], df['val_accuracy'], 
               alpha=0.6, s=80, c='blue', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Number of Parameters', fontsize=14)
    ax.set_ylabel('Validation Accuracy', fontsize=14)
    ax.set_title(f'{model_name}: Parameters vs Validation Accuracy', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name.lower()}_parameters_vs_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()


def save_summary_csv(df, output_dir):
    """Save results to CSV."""
    if df.empty:
        print("WARNING: No data to save!")
        return
    
    # Select relevant columns for CSV
    csv_df = df[['model', 'trial_id', 'nodes', 'parameters', 'val_accuracy']].copy()
    csv_df = csv_df.sort_values('val_accuracy', ascending=False)
    
    csv_path = os.path.join(output_dir, 'hyperparameter_complexity_summary.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"✓ Summary CSV saved to: {csv_path}")
    
    # Save detailed results with hyperparameters
    detailed_path = os.path.join(output_dir, 'hyperparameter_detailed_results.csv')
    
    # Expand hyperparameters into separate columns
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
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✓ Detailed results saved to: {detailed_path}")


def main():
    """Main function to analyze hyperparameter tuning results."""
    print("=== Hyperparameter Tuning Complexity Analysis ===")
    
    # Get search directory from command line argument or use default
    if len(sys.argv) > 1:
        search_dir = sys.argv[1]
    else:
        # Default directory - can be modified here
        base_dir = "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/hyperparameter_tuning"
        search_dir = os.path.join(base_dir, "model2_quantized_4w0i_hyperparameter_search")
    
    # Check if directory exists
    if not os.path.exists(search_dir):
        print(f"ERROR: Search directory not found: {search_dir}")
        print("Usage: python analyze_hyperparameter_complexity.py [search_directory]")
        return
    
    # Extract model name from directory path
    model_name = os.path.basename(search_dir.rstrip('/'))
    if not model_name:
        model_name = os.path.basename(os.path.dirname(search_dir))
    
    # Create output directory based on model name
    output_dir = os.path.join("/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/complexity_analysis", model_name)
    
    print(f"Search directory: {search_dir}")
    print(f"Model name: {model_name}")
    print(f"Output directory: {output_dir}")
    
    # Analyze hyperparameter search (auto-detects model type, excludes failed models with accuracy ~0.5)
    df = analyze_hyperparameter_search(search_dir, model_name, complexity_func=None, min_accuracy=0.55)
    
    if df.empty:
        print("ERROR: No valid trial data found!")
        return
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    plot_complexity_vs_accuracy(df, model_name, output_dir)
    
    # Save results to CSV
    print("\n=== Saving Results ===")
    save_summary_csv(df, output_dir)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\n{model_name}:")
    print(f"  Total trials: {len(df)}")
    print(f"  Mean accuracy: {df['val_accuracy'].mean():.4f} ± {df['val_accuracy'].std():.4f}")
    print(f"  Best accuracy: {df['val_accuracy'].max():.4f}")
    print(f"  Mean nodes: {df['nodes'].mean():.1f}")
    print(f"  Mean parameters: {df['parameters'].mean():.1f}")
    corr_nodes = df['nodes'].corr(df['val_accuracy'])
    corr_params = df['parameters'].corr(df['val_accuracy'])
    print(f"  Nodes-Accuracy correlation: {corr_nodes:.4f}")
    print(f"  Parameters-Accuracy correlation: {corr_params:.4f}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()

        

