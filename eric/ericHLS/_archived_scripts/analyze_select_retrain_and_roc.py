#!/usr/bin/env python3
"""
Pareto selection + retraining + ROC analysis for selected models.

Workflow:
1. Reuse Pareto-selection workflow from analyze_and_select_pareto.py
2. Copy selected H5 models into output directory
3. Retrain each selected model for additional epochs
4. Save retrained model files
5. Compute and save per-model ROC curves/data
6. Record background rejection at fixed signal efficiencies (95/98/99 by default)

Author: Eric
Date: February 2026
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Add parent directories to path for existing utilities
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/ryan/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/ericHLS/')
sys.path.append('/local/d1/smartpixML/filtering_models/shuffling_data/')

# TensorFlow / QKeras
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Error: TensorFlow not available")
    sys.exit(1)

try:
    from qkeras.utils import _add_supported_quantized_objects
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("Warning: QKeras not available")

# Reuse Pareto functions from existing script
import analyze_and_select_pareto as pareto


def get_custom_objects():
    """Get custom objects for loading QKeras models."""
    if not QKERAS_AVAILABLE:
        return {}
    co = {}
    _add_supported_quantized_objects(co)
    return co


def _extract_batch_index(path):
    """Extract numeric batch index for stable sorting of TFRecord files."""
    name = os.path.basename(path)
    match = re.search(r"batch_(\d+)\.tfrecord$", name)
    if match:
        return int(match.group(1))
    return name


def _parse_tfrecord_fn(example):
    """Parse one serialized TFRecord example into (X, y)."""
    feature_description = {
        'y': tf.io.FixedLenFeature([], tf.string),
        'x_profile': tf.io.FixedLenFeature([], tf.string),
        'z_global': tf.io.FixedLenFeature([], tf.string),
        'y_profile': tf.io.FixedLenFeature([], tf.string),
        'y_local': tf.io.FixedLenFeature([], tf.string),
    }

    parsed = tf.io.parse_single_example(example, feature_description)
    y = tf.io.parse_tensor(parsed['y'], out_type=tf.float32)
    x = {
        'x_profile': tf.io.parse_tensor(parsed['x_profile'], out_type=tf.float32),
        'z_global': tf.io.parse_tensor(parsed['z_global'], out_type=tf.float32),
        'y_profile': tf.io.parse_tensor(parsed['y_profile'], out_type=tf.float32),
        'y_local': tf.io.parse_tensor(parsed['y_local'], out_type=tf.float32),
    }
    return x, y


def build_tfrecord_dataset(tfrecord_dir, shuffle_files=False, deterministic=True):
    """Build tf.data dataset from TFRecord files with controlled file ordering."""
    files = [
        os.path.join(tfrecord_dir, fname)
        for fname in os.listdir(tfrecord_dir)
        if fname.endswith('.tfrecord')
    ]
    if not files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    files = sorted(files, key=_extract_batch_index)
    file_ds = tf.data.Dataset.from_tensor_slices(files)

    if shuffle_files:
        file_ds = file_ds.shuffle(len(files), reshuffle_each_iteration=True)

    options = tf.data.Options()
    options.deterministic = deterministic
    file_ds = file_ds.with_options(options)

    ds = file_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=deterministic
    )
    ds = ds.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, len(files)


def collect_labels(dataset):
    """Collect all labels from a dataset into one flat numpy array."""
    return np.concatenate([y.numpy().ravel() for _, y in dataset], axis=0)


def compute_fixed_eff_metrics(y_true, y_score, target_signal_efficiencies):
    """
    Compute fixed-operating-point metrics directly from score distributions.

    For each target signal efficiency:
      - pick threshold from signal-score quantile
      - compute achieved signal efficiency and FPR at that threshold
      - compute background rejection = 1 - FPR
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    sig_scores = y_score[y_true == 1]
    bkg_scores = y_score[y_true == 0]

    metrics = {}
    if len(sig_scores) == 0 or len(bkg_scores) == 0:
        for eff in target_signal_efficiencies:
            key = int(round(eff * 100))
            metrics[f'threshold_{key}'] = np.nan
            metrics[f'achieved_sig_eff_{key}'] = np.nan
            metrics[f'fpr_{key}'] = np.nan
            metrics[f'bkg_rej_{key}'] = np.nan
        return metrics

    for eff in target_signal_efficiencies:
        key = int(round(eff * 100))
        threshold = np.quantile(sig_scores, 1.0 - eff)
        achieved_sig_eff = float(np.mean(sig_scores >= threshold))
        fpr = float(np.mean(bkg_scores >= threshold))
        bkg_rej = 1.0 - fpr

        metrics[f'threshold_{key}'] = float(threshold)
        metrics[f'achieved_sig_eff_{key}'] = achieved_sig_eff
        metrics[f'fpr_{key}'] = fpr
        metrics[f'bkg_rej_{key}'] = float(bkg_rej)

    return metrics


def plot_single_roc(fpr, tpr, roc_auc, model_name, output_path):
    """Plot one ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Background Efficiency)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Signal Efficiency)', fontsize=12)
    ax.set_title(f'Retrained ROC: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_roc(results_df, output_path):
    """Plot all retrained model ROC curves together."""
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.tab20(np.linspace(0, 1, len(results_df)))

    for idx, (_, row) in enumerate(results_df.iterrows()):
        trial_num = row['model_name'].replace('model_trial_', '')
        ax.plot(
            row['fpr'],
            row['tpr'],
            lw=2,
            alpha=0.8,
            color=colors[idx],
            label=f'Trial {trial_num} (AUC={row["auc"]:.3f})'
        )

    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.3, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Background Efficiency)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Signal Efficiency)', fontsize=14, fontweight='bold')
    ax.set_title('Retrained ROC Curves: Pareto Models', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def retrain_and_evaluate_model(model_path, train_ds, val_ds, y_true_val, epochs, learning_rate,
                               signal_efficiencies, roc_dir, retrained_models_dir, history_dir):
    """Retrain one model and compute ROC/fixed-efficiency background rejection metrics."""
    model_name = Path(model_path).stem
    print(f"\nRetraining {model_name}...")

    try:
        model = load_model(model_path, custom_objects=get_custom_objects(), compile=False)
    except Exception as exc:
        print(f"  ✗ Failed to load {model_name}: {exc}")
        return None

    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = int(trainable_count + non_trainable_count)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
        run_eagerly=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )

    # Save retrained model
    retrained_path = os.path.join(retrained_models_dir, f"{model_name}_retrained_{epochs}ep.h5")
    model.save(retrained_path)
    print(f"  ✓ Saved retrained model: {os.path.basename(retrained_path)}")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(history_dir, f"{model_name}_history.csv")
    history_df.to_csv(history_path, index=False)

    # Predict and evaluate ROC on validation set
    y_pred = model.predict(val_ds, verbose=0).ravel()
    if len(y_pred) != len(y_true_val):
        print(f"  ✗ Prediction/label length mismatch for {model_name}: {len(y_pred)} vs {len(y_true_val)}")
        del model
        tf.keras.backend.clear_session()
        return None

    fpr, tpr, thresholds = roc_curve(y_true_val, y_pred, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    # Save per-model ROC data and plot
    roc_csv_path = os.path.join(roc_dir, f"{model_name}_roc_data.csv")
    pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}).to_csv(roc_csv_path, index=False)

    roc_plot_path = os.path.join(roc_dir, f"{model_name}_roc.png")
    plot_single_roc(fpr, tpr, roc_auc, model_name, roc_plot_path)

    # Fixed signal efficiency metrics (directly from scores)
    fixed_metrics = compute_fixed_eff_metrics(y_true_val, y_pred, signal_efficiencies)

    # Cleanup
    del model
    tf.keras.backend.clear_session()

    result = {
        'model_name': model_name,
        'original_model_path': model_path,
        'retrained_model_path': retrained_path,
        'parameters': total_params,
        'auc': float(roc_auc),
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }
    result.update(fixed_metrics)

    return result


def parse_signal_efficiencies(raw_list):
    """Parse target signal efficiencies and validate range."""
    effs = sorted(set(raw_list))
    for eff in effs:
        if eff <= 0.0 or eff >= 1.0:
            raise ValueError(f"Signal efficiency must be in (0,1): {eff}")
    return effs


def main():
    parser = argparse.ArgumentParser(
        description='Pareto select + retrain selected models + ROC analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python analyze_select_retrain_and_roc.py \
      --input_dir ../model2.5_quantized_4w0i_hyperparameter_results_20260203_114608 \
      --output_dir ../model2_5_pareto_hls_ready_retrained \
      --data_dir /local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try3_eric/filtering_records16384_data_shuffled_single_bigData \
      --epochs 50 \
      --signal_efficiencies 0.95 0.98 0.99
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing hyperparameter tuning H5 results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for Pareto-selected + retrained results')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing TFRecords root (must have train/validation subdirs)')
    parser.add_argument('--min_accuracy', type=float, default=0.55,
                        help='Minimum accuracy filter during Pareto analysis')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Retraining epochs for each selected model')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for retraining')
    parser.add_argument('--signal_efficiencies', type=float, nargs='+', default=[0.95, 0.98, 0.99],
                        help='Target signal efficiencies for background rejection metrics')

    args = parser.parse_args()

    if not TF_AVAILABLE:
        print("Error: TensorFlow not available")
        sys.exit(1)

    if not os.path.isdir(args.input_dir):
        print(f"Error: input_dir does not exist: {args.input_dir}")
        sys.exit(1)
    if not os.path.isdir(args.data_dir):
        print(f"Error: data_dir does not exist: {args.data_dir}")
        sys.exit(1)

    signal_efficiencies = parse_signal_efficiencies(args.signal_efficiencies)

    os.makedirs(args.output_dir, exist_ok=True)
    roc_dir = os.path.join(args.output_dir, 'roc_analysis_retrained')
    retrained_models_dir = os.path.join(args.output_dir, 'retrained_models')
    history_dir = os.path.join(args.output_dir, 'retrain_history')
    for path in [roc_dir, retrained_models_dir, history_dir]:
        os.makedirs(path, exist_ok=True)

    print("\n" + "=" * 80)
    print("PARETO SELECTION + RETRAIN + ROC ANALYSIS")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Retraining epochs: {args.epochs}")
    print(f"Signal efficiencies: {', '.join([f'{e:.0%}' for e in signal_efficiencies])}")

    # ---------------------------------------------------------------------
    # 1) Pareto selection and copy (same as analyze_and_select_pareto.py)
    # ---------------------------------------------------------------------
    df = pareto.analyze_complexity(args.input_dir, min_accuracy=args.min_accuracy)
    if df.empty:
        print("\nError: No valid trials after filtering")
        sys.exit(1)

    model_name = os.path.basename(args.input_dir.rstrip('/'))
    pareto.plot_complexity_vs_accuracy(df, args.output_dir, model_name, 'parameters')

    pareto_primary, pareto_secondary = pareto.select_pareto_models(df, 'parameters')
    pareto.plot_pareto_front(df, pareto_primary, pareto_secondary, args.output_dir, model_name, 'parameters')

    all_ids = set(pareto_primary['trial_id'].values)
    if pareto_secondary is not None:
        all_ids.update(pareto_secondary['trial_id'].values)

    selected_df = df[df['trial_id'].isin(all_ids)].copy()
    selected_df = selected_df.sort_values('val_accuracy', ascending=False)

    print(f"\nSelected models for retraining: {len(selected_df)}")
    pareto.copy_model_files(selected_df, None, args.output_dir)
    pareto.save_results(df, pareto_primary, pareto_secondary, args.output_dir)

    # ---------------------------------------------------------------------
    # 2) Build train/validation datasets
    # ---------------------------------------------------------------------
    train_dir = os.path.join(args.data_dir, 'tfrecords_train')
    val_dir = os.path.join(args.data_dir, 'tfrecords_validation')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"Error: Expected train/validation TFRecord dirs under {args.data_dir}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("LOADING TFRECORD DATASETS")
    print("=" * 80)

    train_ds, n_train_files = build_tfrecord_dataset(train_dir, shuffle_files=True, deterministic=False)
    val_ds, n_val_files = build_tfrecord_dataset(val_dir, shuffle_files=False, deterministic=True)

    print(f"Train TFRecord files: {n_train_files}")
    print(f"Validation TFRecord files: {n_val_files}")

    y_true_val = collect_labels(val_ds)
    print(f"Validation labels loaded: {len(y_true_val)} samples")

    # ---------------------------------------------------------------------
    # 3) Retrain and evaluate each selected copied model
    # ---------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RETRAINING SELECTED MODELS")
    print("=" * 80)

    results = []
    for _, row in selected_df.iterrows():
        copied_model_path = os.path.join(args.output_dir, os.path.basename(row['model_file']))
        if not os.path.exists(copied_model_path):
            print(f"\n✗ Missing copied model: {copied_model_path}")
            continue

        result = retrain_and_evaluate_model(
            model_path=copied_model_path,
            train_ds=train_ds,
            val_ds=val_ds,
            y_true_val=y_true_val,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            signal_efficiencies=signal_efficiencies,
            roc_dir=roc_dir,
            retrained_models_dir=retrained_models_dir,
            history_dir=history_dir,
        )

        if result is None:
            continue

        eff_msg = []
        for eff in signal_efficiencies:
            key = int(round(eff * 100))
            eff_msg.append(
                f"{key}%: bkg_rej={result[f'bkg_rej_{key}']:.4f}, "
                f"fpr={result[f'fpr_{key}']:.4f}, "
                f"achieved={result[f'achieved_sig_eff_{key}']:.4f}"
            )
        print("  " + " | ".join(eff_msg))

        # Keep original Pareto metadata for easier comparison
        result['pre_retrain_val_accuracy'] = float(row['val_accuracy'])
        results.append(result)

    if not results:
        print("\nError: No models were successfully retrained/evaluated")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 4) Save summaries and combined plots
    # ---------------------------------------------------------------------
    results_df = pd.DataFrame(results)

    summary_cols = [
        'model_name',
        'parameters',
        'pre_retrain_val_accuracy',
        'auc',
        'retrained_model_path',
    ]
    for eff in signal_efficiencies:
        key = int(round(eff * 100))
        summary_cols.extend([
            f'bkg_rej_{key}',
            f'fpr_{key}',
            f'achieved_sig_eff_{key}',
            f'threshold_{key}'
        ])

    summary_csv_path = os.path.join(args.output_dir, 'retrained_roc_metrics_summary.csv')
    results_df[summary_cols].to_csv(summary_csv_path, index=False)
    print(f"\n✓ Saved retrained ROC summary: {summary_csv_path}")

    combined_roc_path = os.path.join(args.output_dir, 'roc_combined_retrained_models.png')
    plot_combined_roc(results_df, combined_roc_path)
    print(f"✓ Saved combined retrained ROC plot: {combined_roc_path}")

    # Background rejection vs parameters plots for each target efficiency
    for eff in signal_efficiencies:
        key = int(round(eff * 100))
        plot_path = os.path.join(args.output_dir, f'background_rejection_vs_parameters_{key}pct_retrained.png')

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(
            results_df['parameters'],
            results_df[f'bkg_rej_{key}'],
            alpha=0.8,
            s=150,
            c='steelblue',
            edgecolors='darkblue',
            linewidth=2,
            zorder=3
        )
        for _, r in results_df.iterrows():
            trial_num = r['model_name'].replace('model_trial_', '')
            ax.annotate(
                trial_num,
                xy=(r['parameters'], r[f'bkg_rej_{key}']),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=9,
                color='darkblue',
                fontweight='bold'
            )

        ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Background Rejection @ {key}% Signal Eff.', fontsize=14, fontweight='bold')
        ax.set_title('Retrained Models: Background Rejection vs Parameters', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {plot_path}")

    # Update or create analysis_summary.json with retraining info
    analysis_summary_path = os.path.join(args.output_dir, 'analysis_summary.json')
    if os.path.exists(analysis_summary_path):
        with open(analysis_summary_path, 'r') as f:
            analysis_summary = json.load(f)
    else:
        analysis_summary = {}

    retrain_summary = {
        'timestamp': datetime.now().isoformat(),
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'signal_efficiencies': signal_efficiencies,
        'num_retrained_models': int(len(results_df)),
        'auc_range': {
            'min': float(results_df['auc'].min()),
            'max': float(results_df['auc'].max()),
            'mean': float(results_df['auc'].mean())
        },
        'background_rejection_ranges': {
            str(int(round(eff * 100))): {
                'min': float(results_df[f'bkg_rej_{int(round(eff * 100))}'].min()),
                'max': float(results_df[f'bkg_rej_{int(round(eff * 100))}'].max()),
                'mean': float(results_df[f'bkg_rej_{int(round(eff * 100))}'].mean()),
            }
            for eff in signal_efficiencies
        }
    }
    analysis_summary['retraining_roc_analysis'] = retrain_summary

    with open(analysis_summary_path, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    print(f"✓ Updated analysis summary: {analysis_summary_path}")

    print("\n" + "=" * 80)
    print("DONE: PARETO SELECTION + RETRAIN + ROC")
    print("=" * 80)
    print(f"Models selected: {len(selected_df)}")
    print(f"Models retrained: {len(results_df)}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"ROC dir: {roc_dir}")
    print(f"Retrained models dir: {retrained_models_dir}")


if __name__ == '__main__':
    main()
