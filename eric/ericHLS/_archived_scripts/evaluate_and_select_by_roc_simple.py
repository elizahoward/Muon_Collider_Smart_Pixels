#!/usr/bin/env python3
"""
Add ROC Analysis to Pareto-Selected Models (using simple TFRecord loading)

This script combines the functionality of evaluate_and_select_by_roc.py with
the simpler TFRecord loading approach from simple_predict_roc.py.

Author: Eric
Date: February 2026
"""

import os
import sys
import argparse
import json
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


def _parse_tfrecord_fn(example):
    """Parse a single TFRecord example."""
    feature_description = {
        'y': tf.io.FixedLenFeature([], tf.string),
        'x_profile': tf.io.FixedLenFeature([], tf.string),
        'z_global': tf.io.FixedLenFeature([], tf.string),
        'y_profile': tf.io.FixedLenFeature([], tf.string),
        'y_local': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, feature_description)
    y = tf.io.parse_tensor(example['y'], out_type=tf.float32)
    X = {
        'x_profile': tf.io.parse_tensor(example['x_profile'], out_type=tf.float32),
        'z_global': tf.io.parse_tensor(example['z_global'], out_type=tf.float32),
        'y_profile': tf.io.parse_tensor(example['y_profile'], out_type=tf.float32),
        'y_local': tf.io.parse_tensor(example['y_local'], out_type=tf.float32),
    }
    return X, y


def build_tfrecord_dataset(tfrecord_dir):
    """Build TensorFlow dataset from TFRecord files."""
    pattern = os.path.join(tfrecord_dir, "*.tfrecord")
    files = tf.data.Dataset.list_files(pattern, shuffle=False)
    ds = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_custom_objects():
    """Get custom objects dictionary for loading QKeras models."""
    if not QKERAS_AVAILABLE:
        return {}
    co = {}
    _add_supported_quantized_objects(co)
    return co


def compute_background_rejection_direct(y_true, y_score, signal_efficiency=0.95):
    """
    Compute background rejection at a fixed signal efficiency directly from scores.

    This does not use ROC interpolation. It selects a threshold from signal scores
    and then evaluates FPR on background scores at that threshold.

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

    # Choose threshold so approximately `signal_efficiency` of signal passes.
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


def evaluate_model_roc(model_file, validation_dataset, roc_dir, signal_efficiency):
    """
    Evaluate a single model and compute ROC metrics.
    
    Args:
        model_file: Path to H5 model file
        validation_dataset: TensorFlow dataset for validation
        roc_dir: Directory to save ROC plots and CSVs
        
    Returns:
        dict with metrics: fpr, tpr, auc, fixed-point background rejection, parameters, etc.
    """
    model_name = Path(model_file).stem
    print(f"\nEvaluating {model_name}...")
    
    # Load model
    try:
        custom_objects = get_custom_objects()
        model = load_model(model_file, custom_objects=custom_objects, compile=False)
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return None
    
    # Get parameter count
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = int(trainable_count + non_trainable_count)
    print(f"  Parameters: {total_params:,}")
    
    # Predict on validation set
    print(f"  Running inference...")
    y_pred = model.predict(validation_dataset, verbose=0).ravel()
    
    # Get true labels
    y_true = np.concatenate(
        [y.numpy().ravel() for _, y in validation_dataset],
        axis=0
    )
    
    print(f"  ✓ Inference complete: {len(y_true)} samples")
    
    # Compute ROC curve using all threshold points for finer ROC traces.
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    
    print(f"  AUC: {roc_auc:.4f}")
    
    # Clean up
    del model
    tf.keras.backend.clear_session()
    
    # Save ROC plot
    plot_path = os.path.join(roc_dir, f'{model_name}_roc.png')
    plot_single_roc(fpr, tpr, roc_auc, model_name, plot_path)
    
    # Save ROC data as CSV
    csv_path = os.path.join(roc_dir, f'{model_name}_roc_data.csv')
    roc_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })
    roc_df.to_csv(csv_path, index=False)
    print(f"  ✓ ROC data saved: {os.path.basename(csv_path)}")
    
    fixed_point_metrics = compute_background_rejection_direct(
        y_true, y_pred, signal_efficiency=signal_efficiency
    )

    return {
        'model_name': model_name,
        'model_file': model_file,
        'parameters': total_params,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'threshold_at_target': fixed_point_metrics['threshold_at_target'],
        'achieved_signal_efficiency': fixed_point_metrics['achieved_signal_efficiency'],
        'fpr_at_target': fixed_point_metrics['fpr_at_target'],
        'background_rejection': fixed_point_metrics['background_rejection']
    }


def plot_single_roc(fpr, tpr, roc_auc, model_name, output_path):
    """Plot ROC curve for a single model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Background Efficiency)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Signal Efficiency)', fontsize=12)
    ax.set_title(f'ROC Curve: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_roc(results_df, output_path):
    """Plot ROC curves for all models on a single plot."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(results_df)))
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        fpr = row['fpr']
        tpr = row['tpr']
        auc_val = row['auc']
        model_name = row['model_name']
        
        # Extract trial number for cleaner labels
        trial_num = model_name.replace('model_trial_', '')
        
        ax.plot(fpr, tpr, lw=2, alpha=0.8, color=colors[idx],
                label=f'Trial {trial_num} (AUC={auc_val:.3f})')
    
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.3, label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Background Efficiency)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Signal Efficiency)', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves: All Pareto Models Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Combined ROC plot saved: {os.path.basename(output_path)}")


def plot_background_rejection_vs_parameters(results_df, output_path, signal_efficiency):
    """Plot background rejection vs number of parameters."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    ax.scatter(results_df['parameters'], results_df['background_rejection'],
               alpha=0.8, s=150, c='steelblue', edgecolors='darkblue',
               linewidth=2, zorder=3)
    
    # Annotate points
    for _, row in results_df.iterrows():
        trial_num = row['model_name'].replace('model_trial_', '')
        ax.annotate(trial_num,
                   xy=(row['parameters'], row['background_rejection']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, color='darkblue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                            alpha=0.8, edgecolor='darkblue', linewidth=1.5),
                   zorder=4)
    
    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Background Rejection @ {signal_efficiency:.0%} Signal Eff.', 
                  fontsize=14, fontweight='bold')
    ax.set_title('ROC Performance: Background Rejection vs Model Complexity',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Statistics box
    stats_text = (
        f"Models evaluated: {len(results_df)}\n"
        f"Signal efficiency: {signal_efficiency:.0%}\n"
        f"Bkg rejection range: {results_df['background_rejection'].min():.4f} - {results_df['background_rejection'].max():.4f}\n"
        f"Parameters range: {results_df['parameters'].min():,} - {results_df['parameters'].max():,}\n"
        f"AUC range: {results_df['auc'].min():.4f} - {results_df['auc'].max():.4f}"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Background rejection plot saved: {os.path.basename(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description='Add ROC analysis to Pareto-selected models (simple version)'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing Pareto-selected H5 models'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing TFRecords (with tfrecords_validation/ subdirectory)'
    )
    
    parser.add_argument(
        '--signal_efficiency',
        type=float,
        default=0.95,
        help='Target signal efficiency for background rejection calculation (default: 0.95)'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    # Create ROC subdirectory
    roc_dir = os.path.join(args.input_dir, 'roc_analysis')
    os.makedirs(roc_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ROC ANALYSIS FOR PARETO-SELECTED MODELS (SIMPLE VERSION)")
    print("=" * 80)
    print(f"\nInput directory: {args.input_dir}")
    print(f"ROC output directory: {roc_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Target signal efficiency: {args.signal_efficiency:.1%}")
    
    # Find all H5 files in the input directory
    h5_files = sorted([os.path.join(args.input_dir, f) 
                      for f in os.listdir(args.input_dir) 
                      if f.endswith('.h5') and f.startswith('model_trial_')])
    
    if not h5_files:
        print(f"\nError: No H5 model files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(h5_files)} Pareto-selected models to evaluate")
    
    # Load validation data
    print("\n" + "=" * 80)
    print("LOADING VALIDATION DATA")
    print("=" * 80)
    val_dir = os.path.join(args.data_dir, "tfrecords_validation/")
    
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found: {val_dir}")
        sys.exit(1)
    
    print(f"Loading validation data from: {val_dir}")
    validation_dataset = build_tfrecord_dataset(val_dir)
    print("✓ Validation dataset loaded")
    
    # Evaluate all models
    print("\n" + "=" * 80)
    print("EVALUATING MODELS WITH ROC")
    print("=" * 80)
    
    results = []
    for model_file in h5_files:
        result = evaluate_model_roc(
            model_file,
            validation_dataset,
            roc_dir,
            signal_efficiency=args.signal_efficiency
        )
        if result is not None:
            print(
                f"  Background rejection @ {args.signal_efficiency:.0%}: "
                f"{result['background_rejection']:.4f} "
                f"(FPR={result['fpr_at_target']:.4f}, "
                f"Achieved SigEff={result['achieved_signal_efficiency']:.4f})"
            )
            results.append(result)
    
    if not results:
        print("\nError: No models were successfully evaluated")
        sys.exit(1)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save ROC metrics summary
    summary_csv = results_df[
        [
            'model_name',
            'parameters',
            'auc',
            'background_rejection',
            'fpr_at_target',
            'achieved_signal_efficiency',
            'threshold_at_target'
        ]
    ].copy()
    summary_csv_path = os.path.join(args.input_dir, 'roc_metrics_summary.csv')
    summary_csv.to_csv(summary_csv_path, index=False)
    print(f"\n✓ ROC metrics summary saved: {os.path.basename(summary_csv_path)}")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING ROC PLOTS")
    print("=" * 80)
    
    # Combined ROC curves
    combined_roc_path = os.path.join(args.input_dir, 'roc_combined_all_models.png')
    plot_combined_roc(results_df, combined_roc_path)
    
    # Background rejection vs parameters
    bg_rej_plot_path = os.path.join(args.input_dir, 'background_rejection_vs_parameters.png')
    plot_background_rejection_vs_parameters(results_df, bg_rej_plot_path, args.signal_efficiency)
    
    # Update analysis summary JSON
    existing_summary_path = os.path.join(args.input_dir, 'analysis_summary.json')
    
    if os.path.exists(existing_summary_path):
        with open(existing_summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Add ROC metrics
    summary['roc_analysis'] = {
        'timestamp': datetime.now().isoformat(),
        'signal_efficiency': args.signal_efficiency,
        'auc_range': {
            'min': float(results_df['auc'].min()),
            'max': float(results_df['auc'].max()),
            'mean': float(results_df['auc'].mean())
        },
        'background_rejection_range': {
            'min': float(results_df['background_rejection'].min()),
            'max': float(results_df['background_rejection'].max()),
            'mean': float(results_df['background_rejection'].mean())
        }
    }
    
    with open(existing_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Updated analysis summary: {os.path.basename(existing_summary_path)}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("ROC ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nROC analysis added to: {args.input_dir}")
    print(f"\nNew files created:")
    print(f"  - {len(results_df)} individual ROC plots ({roc_dir}/)")
    print(f"  - {len(results_df)} ROC data CSVs ({roc_dir}/)")
    print(f"  - 1 combined ROC comparison plot")
    print(f"  - 1 background rejection vs parameters plot")
    print(f"  - 1 ROC metrics summary CSV")
    print(f"  - Updated analysis_summary.json with ROC metrics")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Models evaluated: {len(results_df)}")
    print(f"AUC range: {results_df['auc'].min():.4f} - {results_df['auc'].max():.4f}")
    print(f"Background rejection @ {args.signal_efficiency:.0%}: {results_df['background_rejection'].min():.4f} - {results_df['background_rejection'].max():.4f}")
    print(f"Parameter range: {results_df['parameters'].min():,} - {results_df['parameters'].max():,}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
