#!/usr/bin/env python3
"""
Combined Analysis and Pareto Selection using ROC Metrics

This script combines the functionality of analyze_and_select_pareto.py with
ROC-based background rejection metrics from evaluate_and_select_by_roc_simple.py.

It selects Pareto optimal models based on:
- Minimize: Number of parameters (complexity)
- Maximize: Background rejection at a given signal efficiency (or weighted metric)

Workflow:
1. Find H5 models in input directory
2. Auto-detect required input features by inspecting first model (smart!)
3. Load validation data with only the required features (efficient!)
4. Evaluate models on validation data to compute ROC metrics
5. Select Pareto optimal models (two-tier) based on complexity vs background rejection
6. Copy selected H5 model files to output directory
7. Generate Pareto front plots and summary files

Key Feature:
- Automatically detects input features from model architecture (not TFRecords)
- Only parses features the model actually needs (saves RAM and time)
- Works with any model architecture without code changes

Author: Eric
Date: February 2026


Examples:
  # Auto-detect features from model (default - recommended)
  python analyze_and_select_pareto_roc.py
      --input_dir ../model2.5_quantized_4w0i_hyperparameter_results_20260214_211815
      --data_dir /local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData
      --output_dir ../model2_5_pareto_roc_selected
      --use_weighted
  
  # Manually specify features (if needed)
  python analyze_and_select_pareto_roc.py
      --input_dir ../model2.5_quantized_4w0i_hyperparameter_results_20260214_211815
      --data_dir /local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData
      --output_dir ../model2_5_pareto_roc_selected
      --features "x_profile,nModule,x_local,y_profile,y_local"
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
from sklearn.metrics import roc_curve, auc

# Add parent directory to path for imports
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/ryan/')
sys.path.append('/local/d1/smartpixML/filtering_models/shuffling_data/')

# Import TensorFlow
try:
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Error: TensorFlow not available")
    sys.exit(1)

# Import QKeras
try:
    from qkeras.utils import _add_supported_quantized_objects
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("Warning: QKeras not available")


def _detect_model_input_features(model_file):
    """
    Detect required input features by inspecting a model's input layers.
    
    Args:
        model_file: Path to H5 model file
    
    Returns:
        dict: Feature description dictionary with feature names as keys
    """
    try:
        custom_objects = get_custom_objects()
        model = load_model(model_file, custom_objects=custom_objects, compile=False)
        
        # Get input layer names
        feature_names = []
        if hasattr(model, 'input_names'):
            feature_names = model.input_names
        elif hasattr(model, 'input'):
            # Handle single or multiple inputs
            if isinstance(model.input, list):
                feature_names = [inp.name.split(':')[0].split('/')[-1] for inp in model.input]
            else:
                feature_names = [model.input.name.split(':')[0].split('/')[-1]]
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
        
        if not feature_names:
            raise ValueError(f"Could not detect input features from model: {model_file}")
        
        # Build feature description (all features are serialized tensors)
        feature_description = {name: tf.io.FixedLenFeature([], tf.string) for name in feature_names}
        
        # Always include 'y' for labels
        feature_description['y'] = tf.io.FixedLenFeature([], tf.string)
        
        print(f"  Detected {len(feature_names)} input features from model: {sorted(feature_names)}")
        return feature_description
        
    except Exception as e:
        raise ValueError(f"Failed to detect features from model {model_file}: {e}")


def _parse_tfrecord_fn(example, feature_description=None):
    """
    Parse a single TFRecord example.
    
    Args:
        example: Raw TFRecord example
        feature_description: Dict of feature descriptions. If None, uses default.
    
    Returns:
        Tuple of (X dict, y): Input features and labels
    """
    if feature_description is None:
        # Default feature description for backward compatibility
        feature_description = {
            'y': tf.io.FixedLenFeature([], tf.string),
            'x_local': tf.io.FixedLenFeature([], tf.string),
            'y_profile': tf.io.FixedLenFeature([], tf.string),
            'y_local': tf.io.FixedLenFeature([], tf.string),
        }
    
    parsed = tf.io.parse_single_example(example, feature_description)
    
    # Extract label (must be 'y')
    if 'y' not in parsed:
        raise ValueError("TFRecord must contain 'y' feature for labels")
    
    y = tf.io.parse_tensor(parsed['y'], out_type=tf.float32)
    
    # Extract all other features as input
    X = {}
    for feature_name, feature_value in parsed.items():
        if feature_name != 'y':
            X[feature_name] = tf.io.parse_tensor(feature_value, out_type=tf.float32)
    
    return X, y


def build_tfrecord_dataset(tfrecord_dir, feature_description=None):
    """
    Build TensorFlow dataset from TFRecord files.
    
    Args:
        tfrecord_dir: Directory containing TFRecord files
        feature_description: Feature description dict (required)
    
    Returns:
        TensorFlow dataset
    """
    if feature_description is None:
        raise ValueError("feature_description is required. Use _detect_model_input_features() to get it from a model.")
    
    # Create parse function with feature description
    parse_fn = lambda ex: _parse_tfrecord_fn(ex, feature_description)
    
    pattern = os.path.join(tfrecord_dir, "*.tfrecord")
    files = tf.data.Dataset.list_files(pattern, shuffle=False)
    ds = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_custom_objects():
    """Get custom objects dictionary for loading QKeras models."""
    if not QKERAS_AVAILABLE:
        return {}
    co = {}
    _add_supported_quantized_objects(co)
    return co


# ============================================================================
# ROC METRICS COMPUTATION
# ============================================================================

def compute_background_rejection_direct(y_true, y_score, signal_efficiency=0.95):
    """
    Compute background rejection at a fixed signal efficiency.
    
    Args:
        y_true: True labels (0=background, 1=signal)
        y_score: Predicted scores
        signal_efficiency: Target signal efficiency (default: 0.95)
    
    Returns:
        dict with threshold, achieved_signal_efficiency, fpr, and background_rejection
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    sig_scores = y_score[y_true == 1]
    bkg_scores = y_score[y_true == 0]

    if len(sig_scores) == 0 or len(bkg_scores) == 0:
        return {
            'threshold_at_target': np.nan,
            'achieved_signal_efficiency': np.nan,
            'fpr_at_target': np.nan,
            'background_rejection': np.nan
        }

    # Choose threshold so approximately `signal_efficiency` of signal passes
    threshold = np.quantile(sig_scores, 1.0 - signal_efficiency)

    achieved_sig_eff = float(np.mean(sig_scores >= threshold))
    fpr_at_target = float(np.mean(bkg_scores >= threshold))
    background_rejection = 1.0 - fpr_at_target

    return {
        'threshold_at_target': float(threshold),
        'achieved_signal_efficiency': achieved_sig_eff,
        'fpr_at_target': fpr_at_target,
        'background_rejection': float(background_rejection)
    }


def compute_weighted_background_rejection(y_true, y_score, bkg_rej_weights=None):
    """
    Compute weighted background rejection at multiple signal efficiencies.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        bkg_rej_weights: Dict mapping signal efficiency to weight
                        Default: {0.95: 0.3, 0.98: 0.6, 0.99: 0.1}
    
    Returns:
        float: Weighted background rejection score
    """
    if bkg_rej_weights is None:
        bkg_rej_weights = {0.95: 0.3, 0.98: 0.6, 0.99: 0.1}
    
    weighted_score = 0.0
    for sig_eff, weight in bkg_rej_weights.items():
        result = compute_background_rejection_direct(y_true, y_score, sig_eff)
        weighted_score += weight * result['background_rejection']
    
    return weighted_score


def evaluate_model_roc(model_file, validation_dataset, signal_efficiencies, use_weighted=True, bkg_rej_weights=None):
    """
    Evaluate a single model and compute ROC metrics.
    
    Args:
        model_file: Path to H5 model file
        validation_dataset: TensorFlow dataset for validation
        signal_efficiencies: List of signal efficiencies to evaluate
        use_weighted: Whether to use weighted background rejection
        bkg_rej_weights: Weights for weighted metric
        
    Returns:
        dict with metrics: parameters, auc, background_rejection, etc.
    """
    model_name = Path(model_file).stem
    print(f"  Evaluating {model_name}...", end=' ')
    
    # Load model
    try:
        custom_objects = get_custom_objects()
        model = load_model(model_file, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return None
    
    # Get parameter count
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = int(trainable_count + non_trainable_count)
    
    # Predict on validation set
    y_pred = model.predict(validation_dataset, verbose=0).ravel()
    
    # Get true labels
    y_true = np.concatenate(
        [y.numpy().ravel() for _, y in validation_dataset],
        axis=0
    )
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    
    # Compute background rejection at different signal efficiencies
    bg_rej_results = {}
    for sig_eff in signal_efficiencies:
        result = compute_background_rejection_direct(y_true, y_pred, sig_eff)
        bg_rej_results[sig_eff] = result['background_rejection']
    
    # Compute primary metric (weighted or single point)
    if use_weighted:
        primary_metric = compute_weighted_background_rejection(y_true, y_pred, bkg_rej_weights)
        metric_name = 'weighted_bkg_rej'
    else:
        # Use the first signal efficiency as primary
        primary_sig_eff = signal_efficiencies[0]
        primary_metric = bg_rej_results[primary_sig_eff]
        metric_name = f'bkg_rej_@{primary_sig_eff:.0%}'
    
    # Clean up
    del model
    tf.keras.backend.clear_session()
    
    print(f"✓ (params={total_params}, AUC={roc_auc:.4f}, metric={primary_metric:.4f})")
    
    return {
        'model_name': model_name,
        'model_file': model_file,
        'parameters': total_params,
        'auc': roc_auc,
        'primary_metric': primary_metric,
        'metric_name': metric_name,
        'bg_rej_results': bg_rej_results
    }


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


def find_pareto_front(df, maximize_cols=['primary_metric'], minimize_cols=['parameters']):
    """Find Pareto optimal points in a DataFrame."""
    pareto_indices = []
    
    for idx, row in df.iterrows():
        other_points = df.drop(idx)
        if not is_dominated(row, other_points, maximize_cols, minimize_cols):
            pareto_indices.append(idx)
    
    pareto_df = df.loc[pareto_indices].copy()
    pareto_df = pareto_df.sort_values(maximize_cols[0], ascending=False)
    
    return pareto_df


def select_pareto_models(df):
    """
    Select Pareto optimal models (two-tier for redundancy).
    
    Returns:
        Tuple of (primary_pareto_df, secondary_pareto_df)
    """
    print("\n--- Tier 1: Primary Pareto Front ---")
    pareto_df = find_pareto_front(df, 
                                  maximize_cols=['primary_metric'], 
                                  minimize_cols=['parameters'])
    
    print(f"  ✓ Found {len(pareto_df)} primary Pareto optimal models")
    
    print(f"\n--- Tier 2: Secondary Pareto Front (Redundancy) ---")
    primary_trial_ids = set(pareto_df['trial_id'].values)
    df_remaining = df[~df['trial_id'].isin(primary_trial_ids)].copy()
    
    if len(df_remaining) > 1:
        pareto_df_secondary = find_pareto_front(df_remaining,
                                               maximize_cols=['primary_metric'],
                                               minimize_cols=['parameters'])
        
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

def plot_pareto_front(df, pareto_df, pareto_df_secondary, output_dir, model_name, metric_name):
    """Create Pareto front visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # All models
    ax.scatter(df['parameters'], df['primary_metric'], 
               alpha=0.4, s=60, c='lightgray', edgecolors='gray', 
               linewidth=0.5, label='All models', zorder=1)
    
    # Primary Pareto front
    ax.scatter(pareto_df['parameters'], pareto_df['primary_metric'], 
               alpha=0.9, s=120, c='red', edgecolors='darkred', 
               linewidth=1.5, label='Primary Pareto front', zorder=3, marker='D')
    
    pareto_sorted = pareto_df.sort_values('parameters')
    ax.plot(pareto_sorted['parameters'], pareto_sorted['primary_metric'],
            'r--', alpha=0.6, linewidth=2, zorder=2)
    
    # Secondary Pareto front
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        ax.scatter(pareto_df_secondary['parameters'], pareto_df_secondary['primary_metric'], 
                   alpha=0.8, s=100, c='orange', edgecolors='darkorange', 
                   linewidth=1.5, label='Secondary Pareto front (redundancy)', zorder=3, marker='s')
        
        pareto_sorted_2 = pareto_df_secondary.sort_values('parameters')
        ax.plot(pareto_sorted_2['parameters'], pareto_sorted_2['primary_metric'],
                'orange', linestyle=':', alpha=0.6, linewidth=2, zorder=2)
    
    # Annotate primary Pareto points
    for _, row in pareto_df.iterrows():
        trial_id = row['trial_id']
        label = trial_id if isinstance(trial_id, str) else f"{int(trial_id):03d}"
        ax.annotate(label, 
                   xy=(row['parameters'], row['primary_metric']),
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
                       xy=(row['parameters'], row['primary_metric']),
                       xytext=(8, -12), textcoords='offset points',
                       fontsize=8, color='darkorange', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                alpha=0.7, edgecolor='darkorange', linewidth=1),
                       zorder=4)
    
    # Labels and title
    metric_display = {
        'weighted_bkg_rej': 'Weighted Background Rejection',
        'bkg_rej_@95%': 'Background Rejection @ 95% Signal Eff.',
        'bkg_rej_@98%': 'Background Rejection @ 98% Signal Eff.',
        'bkg_rej_@99%': 'Background Rejection @ 99% Signal Eff.'
    }.get(metric_name, metric_name)
    
    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_display, fontsize=14, fontweight='bold')
    ax.set_title('Model 2.5: Pareto Front', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend at bottom right, above statistics box
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9, bbox_to_anchor=(0.98, 0.25))
    
    # Statistics box at bottom right, below legend
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        stats_text = (
            f"Total models: {len(df)}\n"
            f"Primary Pareto: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
            f"Secondary Pareto: {len(pareto_df_secondary)} ({100*len(pareto_df_secondary)/len(df):.1f}%)\n"
            f"Total selected: {len(pareto_df) + len(pareto_df_secondary)}\n"
            f"Metric range: {df['primary_metric'].min():.4f} - {df['primary_metric'].max():.4f}"
        )
    else:
        stats_text = (
            f"Total models: {len(df)}\n"
            f"Primary Pareto: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
            f"Metric range: {df['primary_metric'].min():.4f} - {df['primary_metric'].max():.4f}"
        )
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'pareto_front_roc_based.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def copy_model_files(pareto_df, pareto_df_secondary, output_dir):
    """Copy H5 model files for selected Pareto models."""
    print("\n" + "=" * 80)
    print("COPYING MODEL FILES")
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


def save_results(df, pareto_df, pareto_df_secondary, output_dir, metric_name, bkg_rej_weights):
    """Save all analysis results and summaries."""
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save detailed results CSV
    detailed_csv = os.path.join(output_dir, 'roc_based_analysis_detailed.csv')
    save_df = df[['trial_id', 'parameters', 'auc', 'primary_metric']].copy()
    
    # Add individual background rejection columns
    if 'bg_rej_results' in df.columns:
        for sig_eff in sorted(df.iloc[0]['bg_rej_results'].keys()):
            save_df[f'bkg_rej_@{sig_eff:.0%}'] = df['bg_rej_results'].apply(lambda x: x[sig_eff])
    
    save_df = save_df.sort_values('primary_metric', ascending=False)
    save_df.to_csv(detailed_csv, index=False)
    print(f"\n✓ Detailed results: {detailed_csv}")
    
    # Save Pareto results
    primary_csv = os.path.join(output_dir, 'pareto_optimal_models_roc_primary.csv')
    pareto_save = pareto_df[['trial_id', 'parameters', 'auc', 'primary_metric']].copy()
    pareto_save.to_csv(primary_csv, index=False)
    print(f"✓ Primary Pareto: {primary_csv}")
    
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        secondary_csv = os.path.join(output_dir, 'pareto_optimal_models_roc_secondary.csv')
        secondary_save = pareto_df_secondary[['trial_id', 'parameters', 'auc', 'primary_metric']].copy()
        secondary_save.to_csv(secondary_csv, index=False)
        print(f"✓ Secondary Pareto: {secondary_csv}")
        
        combined_df = pd.concat([pareto_save, secondary_save], ignore_index=True)
        combined_df = combined_df.sort_values('primary_metric', ascending=False)
        combined_csv = os.path.join(output_dir, 'pareto_optimal_models_roc_combined.csv')
        combined_df.to_csv(combined_csv, index=False)
        print(f"✓ Combined Pareto: {combined_csv}")
    
    # Save JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'metric_name': metric_name,
        'bkg_rej_weights': bkg_rej_weights if bkg_rej_weights else 'N/A',
        'total_models': len(df),
        'primary_pareto_models': len(pareto_df),
        'secondary_pareto_models': len(pareto_df_secondary) if pareto_df_secondary is not None and not pareto_df_secondary.empty else 0,
        'primary_metric_range': {
            'min': float(df['primary_metric'].min()),
            'max': float(df['primary_metric'].max()),
            'mean': float(df['primary_metric'].mean())
        },
        'parameters_range': {
            'min': int(df['parameters'].min()),
            'max': int(df['parameters'].max()),
            'mean': float(df['parameters'].mean())
        },
        'auc_range': {
            'min': float(df['auc'].min()),
            'max': float(df['auc'].max()),
            'mean': float(df['auc'].mean())
        }
    }
    
    summary_json = os.path.join(output_dir, 'pareto_roc_analysis_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Analysis summary: {summary_json}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pareto Selection using ROC-based Background Rejection Metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use weighted background rejection (default)
  python analyze_and_select_pareto_roc.py \\
      --input_dir ../model2.5_quantized_4w0i_hyperparameter_results_20260214_211815 \\
      --data_dir /local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData \\
      --output_dir ../model2_5_pareto_roc_selected \\
      --use_weighted
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing hyperparameter tuning results (with H5 files)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing TFRecords (with tfrecords_validation/ subdirectory)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for Pareto-selected models and plots'
    )
    
    parser.add_argument(
        '--use_weighted',
        action='store_true',
        default=True,
        help='Use weighted background rejection metric (default: True)'
    )
    
    parser.add_argument(
        '--signal_efficiency',
        type=float,
        default=0.95,
        help='Signal efficiency for background rejection (if not using weighted, default: 0.95)'
    )
    
    parser.add_argument(
        '--bkg_rej_weights',
        type=str,
        default='0.95:0.3,0.98:0.6,0.99:0.1',
        help='Weights for background rejection (format: "sig_eff:weight,..." default: "0.95:0.3,0.98:0.6,0.99:0.1")'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='Comma-separated list of feature names to parse (default: auto-detect from first model)'
    )
    
    args = parser.parse_args()
    
    # Parse background rejection weights
    bkg_rej_weights = {}
    if args.use_weighted:
        for pair in args.bkg_rej_weights.split(','):
            sig_eff, weight = pair.split(':')
            bkg_rej_weights[float(sig_eff)] = float(weight)
        signal_efficiencies = list(bkg_rej_weights.keys())
    else:
        signal_efficiencies = [args.signal_efficiency]
        bkg_rej_weights = None
    
    # Parse feature description if provided
    feature_description = None
    if args.features:
        feature_list = [f.strip() for f in args.features.split(',')]
        # Always include 'y' for labels
        if 'y' not in feature_list:
            feature_list.append('y')
        feature_description = {name: tf.io.FixedLenFeature([], tf.string) for name in feature_list}
        print(f"Using specified features: {sorted(feature_list)}")
    
    # Validate directories
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name = os.path.basename(args.input_dir.rstrip('/'))
    
    print("\n" + "=" * 80)
    print("PARETO SELECTION USING ROC-BASED METRICS")
    print("=" * 80)
    print(f"\nInput directory: {args.input_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {model_name}")
    if args.use_weighted:
        print(f"Metric: Weighted background rejection")
        print(f"Weights: {bkg_rej_weights}")
    else:
        print(f"Metric: Background rejection @ {args.signal_efficiency:.0%} signal efficiency")
    
    # Find all H5 files first
    print("\n" + "=" * 80)
    print("FINDING MODEL FILES")
    print("=" * 80)
    
    h5_files = sorted([os.path.join(args.input_dir, f) 
                      for f in os.listdir(args.input_dir) 
                      if f.endswith('.h5') and f.startswith('model_trial_')])
    
    if not h5_files:
        print(f"\nError: No H5 model files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} models to evaluate")
    
    # Detect features from first model if not specified
    if feature_description is None:
        print("\n" + "=" * 80)
        print("DETECTING INPUT FEATURES FROM MODEL")
        print("=" * 80)
        print(f"Inspecting first model: {os.path.basename(h5_files[0])}")
        feature_description = _detect_model_input_features(h5_files[0])
    
    # Load validation data
    print("\n" + "=" * 80)
    print("LOADING VALIDATION DATA")
    print("=" * 80)
    val_dir = os.path.join(args.data_dir, "tfrecords_validation/")
    
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found: {val_dir}")
        sys.exit(1)
    
    print(f"Loading validation data from: {val_dir}")
    validation_dataset = build_tfrecord_dataset(val_dir, feature_description=feature_description)
    print("✓ Validation dataset loaded")
    
    # Evaluate all models
    print("\n" + "=" * 80)
    print("EVALUATING MODELS WITH ROC METRICS")
    print("=" * 80)
    
    results = []
    for model_file in h5_files:
        result = evaluate_model_roc(
            model_file,
            validation_dataset,
            signal_efficiencies,
            use_weighted=args.use_weighted,
            bkg_rej_weights=bkg_rej_weights
        )
        if result is not None:
            # Extract trial ID
            trial_id = Path(model_file).stem.replace('model_trial_', '')
            result['trial_id'] = trial_id
            results.append(result)
    
    if not results:
        print("\nError: No models were successfully evaluated")
        sys.exit(1)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n✓ Successfully evaluated {len(df)} models")
    print(f"  Parameters range: {df['parameters'].min():,} - {df['parameters'].max():,}")
    print(f"  Primary metric range: {df['primary_metric'].min():.4f} - {df['primary_metric'].max():.4f}")
    print(f"  AUC range: {df['auc'].min():.4f} - {df['auc'].max():.4f}")
    
    # Pareto selection
    print("\n" + "=" * 80)
    print("PARETO SELECTION")
    print("=" * 80)
    
    pareto_df, pareto_df_secondary = select_pareto_models(df)
    
    # Generate Pareto front plot
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    metric_name = df.iloc[0]['metric_name']
    plot_pareto_front(df, pareto_df, pareto_df_secondary, args.output_dir, model_name, metric_name)
    
    # Copy model files
    copy_model_files(pareto_df, pareto_df_secondary, args.output_dir)
    
    # Save results
    save_results(df, pareto_df, pareto_df_secondary, args.output_dir, metric_name, bkg_rej_weights)
    
    # Final summary
    print("\n" + "=" * 80)
    print("COMPLETE - PARETO SELECTION FINISHED!")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nSelected models:")
    print(f"  Primary Pareto: {len(pareto_df)}")
    if pareto_df_secondary is not None:
        print(f"  Secondary Pareto: {len(pareto_df_secondary)}")
    print(f"  Total: {len(pareto_df) + (len(pareto_df_secondary) if pareto_df_secondary is not None else 0)}")
    print(f"\nFiles created:")
    print(f"  - Pareto front plot")
    print(f"  - CSV files with results")
    print(f"  - JSON summary")
    print(f"  - {len(pareto_df) + (len(pareto_df_secondary) if pareto_df_secondary is not None else 0)} H5 model files")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
