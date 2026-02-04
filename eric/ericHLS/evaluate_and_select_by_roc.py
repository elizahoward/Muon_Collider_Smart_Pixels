#!/usr/bin/env python3
"""
Add ROC Analysis to Pareto-Selected Models

This script adds ROC performance analysis to models that have already been 
processed by analyze_and_select_pareto.py. It evaluates each H5 model and adds:
- Individual ROC curve plot for each model (PNG)
- ROC curve data (CSV with FPR, TPR, thresholds)
- Combined ROC comparison plot
- Background rejection vs Parameters plot
- Updated analysis summary with ROC metrics

This is designed to RUN IN THE SAME DIRECTORY that analyze_and_select_pareto.py 
created, adding ROC information without duplicating model files.

Usage:
    python evaluate_and_select_by_roc.py \\
        --input_dir <pareto_output_dir> \\
        --data_dir <tfrecord_dir> \\
        --signal_efficiency 0.95

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
    # Enable eager execution for QKeras compatibility
    tf.config.run_functions_eagerly(True)
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Error: TensorFlow not available")
    sys.exit(1)

# Import QKeras
try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu, quantized_tanh
    from qkeras.utils import _add_supported_quantized_objects
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("Warning: QKeras not available")

# Import data generator
try:
    import OptimizedDataGenerator4_data_shuffled_bigData as ODG2
    DATA_GEN_AVAILABLE = True
except ImportError:
    DATA_GEN_AVAILABLE = False
    print("Error: Data generator not available")
    sys.exit(1)


def get_custom_objects():
    """Get custom objects dictionary for loading QKeras models."""
    if not QKERAS_AVAILABLE:
        return {}
    
    co = {}
    _add_supported_quantized_objects(co)
    return co


def load_validation_data(data_dir, batch_size=16384):
    """
    Load validation data using the data generator.
    
    Args:
        data_dir: Path to TFRecord folder
        batch_size: Batch size for data loading
    
    Returns:
        validation_generator
    """
    val_dir = os.path.join(data_dir, "tfrecords_validation/")
    
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    print(f"Loading validation data from: {val_dir}")
    
    # Model2.5 uses x_profile, z_global, y_profile, y_local features
    x_feature_description = ['x_profile', 'z_global', 'y_profile', 'y_local']
    
    validation_generator = ODG2.OptimizedDataGeneratorDataShuffledBigData(
        load_records=True,
        tf_records_dir=val_dir,
        x_feature_description=x_feature_description,
        batch_size=batch_size
    )
    
    print(f"Validation batches: {len(validation_generator)}")
    
    return validation_generator


def evaluate_model_roc(model_file, validation_generator, roc_dir):
    """
    Evaluate a single model and compute ROC metrics.
    
    Args:
        model_file: Path to H5 model file
        validation_generator: Data generator for validation set
        roc_dir: Directory to save ROC plots and CSVs
        
    Returns:
        dict with metrics: fpr, tpr, auc, background_rejection, parameters, etc.
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
    y_true_all = []
    y_pred_all = []
    
    for batch_idx in range(len(validation_generator)):
        X_batch, y_batch = validation_generator[batch_idx]
        
        # Run prediction
        y_pred_batch = model.predict(X_batch, verbose=0)
        
        y_true_all.append(y_batch)
        y_pred_all.append(y_pred_batch)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"    Processed {batch_idx + 1}/{len(validation_generator)} batches")
    
    y_true = np.concatenate(y_true_all, axis=0).flatten()
    y_pred = np.concatenate(y_pred_all, axis=0).flatten()
    
    print(f"  ✓ Inference complete: {len(y_true)} samples")
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
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
    
    return {
        'model_name': model_name,
        'model_file': model_file,
        'parameters': total_params,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc
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


def compute_background_rejection(fpr, tpr, signal_efficiency=0.95):
    """
    Compute background rejection at specified signal efficiency.
    
    Args:
        fpr: False positive rate array
        tpr: True positive rate array (signal efficiency)
        signal_efficiency: Target signal efficiency (default: 0.95)
        
    Returns:
        background_rejection: 1/fpr at target signal efficiency
    """
    # Find the index where TPR >= signal_efficiency
    idx = np.where(tpr >= signal_efficiency)[0]
    
    if len(idx) == 0:
        return 0.0  # Model doesn't reach target signal efficiency
    
    # Use the first point that reaches or exceeds target
    idx = idx[0]
    fpr_at_target = fpr[idx]
    
    if fpr_at_target == 0:
        return np.inf  # Perfect rejection
    
    background_rejection = 1.0 / fpr_at_target
    
    return background_rejection


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
        f"Bkg rejection range: {results_df['background_rejection'].min():.1f} - {results_df['background_rejection'].max():.1f}\n"
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
        description='Add ROC analysis to Pareto-selected models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add ROC analysis to existing Pareto output directory
  python evaluate_and_select_by_roc.py \\
      --input_dir ../model2_5_pareto_hls_ready_2026 \\
      --data_dir /local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try3_eric/filtering_records16384_data_shuffled_single_bigData \\
      --signal_efficiency 0.95

  # Use different signal efficiency
  python evaluate_and_select_by_roc.py \\
      --input_dir ../model2_5_pareto_hls_ready_2026 \\
      --data_dir <tfrecord_dir> \\
      --signal_efficiency 0.90

NOTE: This script is designed to ADD ROC information to an existing directory
      created by analyze_and_select_pareto.py. It will NOT copy model files,
      but will add ROC plots and metrics to the same directory.
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing Pareto-selected H5 models (output from analyze_and_select_pareto.py)'
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
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16384,
        help='Batch size for data loading (default: 16384)'
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
    print("ROC ANALYSIS FOR PARETO-SELECTED MODELS")
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
        print("Make sure you run analyze_and_select_pareto.py first!")
        sys.exit(1)
    
    print(f"\nFound {len(h5_files)} Pareto-selected models to evaluate")
    
    # Load validation data
    print("\n" + "=" * 80)
    print("LOADING VALIDATION DATA")
    print("=" * 80)
    validation_generator = load_validation_data(args.data_dir, args.batch_size)
    
    # Evaluate all models
    print("\n" + "=" * 80)
    print("EVALUATING MODELS WITH ROC")
    print("=" * 80)
    
    results = []
    for model_file in h5_files:
        result = evaluate_model_roc(model_file, validation_generator, roc_dir)
        if result is not None:
            # Compute background rejection
            bg_rej = compute_background_rejection(result['fpr'], result['tpr'], 
                                                 args.signal_efficiency)
            result['background_rejection'] = bg_rej
            print(f"  Background rejection @ {args.signal_efficiency:.0%}: {bg_rej:.2f}")
            results.append(result)
    
    if not results:
        print("\nError: No models were successfully evaluated")
        sys.exit(1)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save ROC metrics summary
    summary_csv = results_df[['model_name', 'parameters', 'auc', 'background_rejection']].copy()
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
    print(f"Background rejection @ {args.signal_efficiency:.0%}: {results_df['background_rejection'].min():.2f} - {results_df['background_rejection'].max():.2f}")
    print(f"Parameter range: {results_df['parameters'].min():,} - {results_df['parameters'].max():,}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
