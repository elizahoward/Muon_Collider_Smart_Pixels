#!/usr/bin/env python3
"""
Multi-Folder Pareto Selection for Model1 Quantization Trials

Extends analyze_and_select_pareto_roc.py to handle the model1_intermediate
directory structure, where results are split across multiple folders by dense-
layer count (hp2q / hp3q / hp4q / hp5q) but share the same weight/input-bit
configuration (e.g. 8w0i_i10_sigmoid).

Folder name convention expected:
  model1_quantized_hp{N}q_{bit_config}_results_{timestamp}
  e.g. model1_quantized_hp3q_8w0i_i10_sigmoid_results_20260423_170932

Workflow:
  1. Scan --input_dir for result folders matching the convention above
  2. Group folders by {bit_config} (i.e. everything between hpXq_ and _results_)
  3. For each group, collect all model_trial_*.h5 files across all hp variants
  4. Auto-detect input features from the first model in the group
  5. Evaluate every model; tag trial_id as  hp{N}q_trial_XXX  to stay unique
  6. Run two-tier Pareto selection (params vs background rejection)
  7. Copy selected files to  output_dir/{bit_config}/  with renamed filenames
  8. Save plots, CSVs, JSON summary, and architecture JSONs per group

Usage:
  python analyze_and_select_pareto_roc_model1.py \\
      --input_dir  /path/to/model1_intermediate \\
      --data_dir   /local/d1/smartpixML/.../TF_Records/... \\
      --output_dir /path/to/model1_pareto_selected

Author: Eric (generated 2026-05-07)
"""

import os
import re
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_curve, auc

sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/ryan/')
sys.path.append('/local/d1/smartpixML/filtering_models/shuffling_data/')

try:
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Error: TensorFlow not available ({e})")
    sys.exit(1)

try:
    from qkeras.utils import _add_supported_quantized_objects
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("Warning: QKeras not available")

# Pattern: model1_quantized_hp{N}q_{bit_config}_results_{timestamp}
_FOLDER_RE = re.compile(
    r'^model1_quantized_hp(\d+)q_(.+?)_results_\d{8}_\d{6}$'
)


# ============================================================================
# HELPERS (shared with original script)
# ============================================================================

def get_custom_objects():
    if not QKERAS_AVAILABLE:
        return {}
    co = {}
    _add_supported_quantized_objects(co)
    return co


def _detect_model_input_features(model_file):
    custom_objects = get_custom_objects()
    model = load_model(model_file, custom_objects=custom_objects, compile=False)

    feature_names = []
    if hasattr(model, 'input_names'):
        feature_names = model.input_names
    elif hasattr(model, 'input'):
        if isinstance(model.input, list):
            feature_names = [inp.name.split(':')[0].split('/')[-1] for inp in model.input]
        else:
            feature_names = [model.input.name.split(':')[0].split('/')[-1]]

    del model
    tf.keras.backend.clear_session()

    if not feature_names:
        raise ValueError(f"Could not detect input features from model: {model_file}")

    feature_description = {name: tf.io.FixedLenFeature([], tf.string) for name in feature_names}
    feature_description['y'] = tf.io.FixedLenFeature([], tf.string)
    print(f"  Detected {len(feature_names)} input features: {sorted(feature_names)}")
    return feature_description


def _parse_tfrecord_fn(example, feature_description):
    parsed = tf.io.parse_single_example(example, feature_description)
    y = tf.io.parse_tensor(parsed['y'], out_type=tf.float32)
    X = {k: tf.io.parse_tensor(v, out_type=tf.float32)
         for k, v in parsed.items() if k != 'y'}
    return X, y


def build_tfrecord_dataset(tfrecord_dir, feature_description):
    parse_fn = lambda ex: _parse_tfrecord_fn(ex, feature_description)
    pattern = os.path.join(tfrecord_dir, "*.tfrecord")
    files = tf.data.Dataset.list_files(pattern, shuffle=False)
    ds = files.interleave(tf.data.TFRecordDataset,
                          num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_background_rejection_direct(y_true, y_score, signal_efficiency=0.95):
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    sig_scores = y_score[y_true == 1]
    bkg_scores = y_score[y_true == 0]
    if len(sig_scores) == 0 or len(bkg_scores) == 0:
        return {'threshold_at_target': np.nan, 'achieved_signal_efficiency': np.nan,
                'fpr_at_target': np.nan, 'background_rejection': np.nan}
    threshold = np.quantile(sig_scores, 1.0 - signal_efficiency)
    fpr_at_target = float(np.mean(bkg_scores >= threshold))
    return {
        'threshold_at_target': float(threshold),
        'achieved_signal_efficiency': float(np.mean(sig_scores >= threshold)),
        'fpr_at_target': fpr_at_target,
        'background_rejection': 1.0 - fpr_at_target,
    }


def compute_weighted_background_rejection(y_true, y_score, bkg_rej_weights=None):
    if bkg_rej_weights is None:
        bkg_rej_weights = {0.95: 0.1, 0.98: 0.7, 0.99: 0.2}
    return sum(w * compute_background_rejection_direct(y_true, y_score, se)['background_rejection']
               for se, w in bkg_rej_weights.items())


def evaluate_model_roc(model_file, validation_dataset, signal_efficiencies,
                       use_weighted=True, bkg_rej_weights=None, display_name=None):
    label = display_name or Path(model_file).stem
    print(f"  Evaluating {label}...", end=' ', flush=True)

    try:
        model = load_model(model_file, custom_objects=get_custom_objects(), compile=False)
    except Exception as e:
        print(f"FAILED to load: {e}")
        return None

    total_params = int(
        np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]) +
        np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    )

    y_pred = model.predict(validation_dataset, verbose=0).ravel()
    y_true = np.concatenate([y.numpy().ravel() for _, y in validation_dataset], axis=0)

    fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    bg_rej_results = {se: compute_background_rejection_direct(y_true, y_pred, se)['background_rejection']
                      for se in signal_efficiencies}

    if use_weighted:
        primary_metric = compute_weighted_background_rejection(y_true, y_pred, bkg_rej_weights)
        metric_name = 'weighted_bkg_rej'
    else:
        se0 = signal_efficiencies[0]
        primary_metric = bg_rej_results[se0]
        metric_name = f'bkg_rej_@{se0:.0%}'

    del model
    tf.keras.backend.clear_session()

    print(f"params={total_params}, AUC={roc_auc:.4f}, metric={primary_metric:.4f}")

    return {
        'model_file': model_file,
        'parameters': total_params,
        'auc': roc_auc,
        'primary_metric': primary_metric,
        'metric_name': metric_name,
        'bg_rej_results': bg_rej_results,
    }


# ============================================================================
# PARETO SELECTION (unchanged logic from original)
# ============================================================================

def is_dominated(point, other_points, maximize_cols, minimize_cols):
    for _, other in other_points.iterrows():
        better_in_all = True
        strictly_better = False
        for col in maximize_cols:
            if other[col] < point[col]:
                better_in_all = False; break
            if other[col] > point[col]:
                strictly_better = True
        if not better_in_all:
            continue
        for col in minimize_cols:
            if other[col] > point[col]:
                better_in_all = False; break
            if other[col] < point[col]:
                strictly_better = True
        if better_in_all and strictly_better:
            return True
    return False


def find_pareto_front(df, maximize_cols=('primary_metric',), minimize_cols=('parameters',)):
    pareto_indices = [idx for idx, row in df.iterrows()
                      if not is_dominated(row, df.drop(idx), list(maximize_cols), list(minimize_cols))]
    return df.loc[pareto_indices].sort_values(maximize_cols[0], ascending=False).copy()


def select_pareto_models(df):
    print("\n--- Tier 1: Primary Pareto Front ---")
    pareto_df = find_pareto_front(df)
    print(f"  Found {len(pareto_df)} primary Pareto optimal models")

    print("\n--- Tier 2: Secondary Pareto Front ---")
    remaining = df[~df['trial_id'].isin(pareto_df['trial_id'])].copy()
    pareto_df_secondary = None
    if len(remaining) > 1:
        pareto_df_secondary = find_pareto_front(remaining)
        if pareto_df_secondary.empty:
            pareto_df_secondary = None
        else:
            print(f"  Found {len(pareto_df_secondary)} secondary Pareto optimal models")

    return pareto_df, pareto_df_secondary


# ============================================================================
# PLOTTING
# ============================================================================

def plot_pareto_front(df, pareto_df, pareto_df_secondary, output_dir, bit_config, metric_name):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(df['parameters'], df['primary_metric'],
               alpha=0.4, s=60, c='lightgray', edgecolors='gray',
               linewidth=0.5, label='All models', zorder=1)

    ax.scatter(pareto_df['parameters'], pareto_df['primary_metric'],
               alpha=0.9, s=120, c='red', edgecolors='darkred',
               linewidth=1.5, label='Primary Pareto', zorder=3, marker='D')
    ps = pareto_df.sort_values('parameters')
    ax.plot(ps['parameters'], ps['primary_metric'], 'r--', alpha=0.6, linewidth=2, zorder=2)

    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        ax.scatter(pareto_df_secondary['parameters'], pareto_df_secondary['primary_metric'],
                   alpha=0.8, s=100, c='orange', edgecolors='darkorange',
                   linewidth=1.5, label='Secondary Pareto', zorder=3, marker='s')
        ps2 = pareto_df_secondary.sort_values('parameters')
        ax.plot(ps2['parameters'], ps2['primary_metric'],
                'orange', linestyle=':', alpha=0.6, linewidth=2, zorder=2)

    for _, row in pareto_df.iterrows():
        ax.annotate(row['trial_id'],
                    xy=(row['parameters'], row['primary_metric']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=8, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                              alpha=0.7, edgecolor='darkred', linewidth=1), zorder=4)

    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        for _, row in pareto_df_secondary.iterrows():
            ax.annotate(row['trial_id'],
                        xy=(row['parameters'], row['primary_metric']),
                        xytext=(8, -12), textcoords='offset points',
                        fontsize=8, color='darkorange', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  alpha=0.7, edgecolor='darkorange', linewidth=1), zorder=4)

    metric_display = {
        'weighted_bkg_rej': 'Weighted Background Rejection',
        'bkg_rej_@95%': 'Background Rejection @ 95% Signal Eff.',
        'bkg_rej_@98%': 'Background Rejection @ 98% Signal Eff.',
        'bkg_rej_@99%': 'Background Rejection @ 99% Signal Eff.',
    }.get(metric_name, metric_name)

    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_display, fontsize=14, fontweight='bold')
    ax.set_title(f'Model1 Pareto Front — {bit_config}', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9, bbox_to_anchor=(0.98, 0.25))

    n_sec = len(pareto_df_secondary) if pareto_df_secondary is not None else 0
    stats_text = (
        f"Total models: {len(df)}\n"
        f"Primary Pareto: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
        f"Secondary Pareto: {n_sec} ({100*n_sec/len(df):.1f}%)\n"
        f"Metric range: {df['primary_metric'].min():.4f} – {df['primary_metric'].max():.4f}"
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'pareto_front_roc_based.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {plot_path}")
    plt.close()


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def copy_model_files(pareto_df, pareto_df_secondary, output_dir, separate_folders=True):
    """Copy selected models, renaming to avoid filename collisions.

    The 'output_filename' column in the DataFrames carries the pre-computed
    destination basename (e.g. hp3q_model_trial_005.h5).
    """
    print(f"\nCopying model files to {output_dir}...")
    success = 0

    primary_dir = os.path.join(output_dir, 'pareto_primary') if separate_folders else output_dir
    os.makedirs(primary_dir, exist_ok=True)

    for _, row in pareto_df.iterrows():
        dst = os.path.join(primary_dir, row['output_filename'])
        if os.path.exists(row['model_file']):
            shutil.copy2(row['model_file'], dst)
            print(f"  [primary]    {row['output_filename']}")
            success += 1
        else:
            print(f"  MISSING: {row['model_file']}")

    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        sec_dir = os.path.join(output_dir, 'pareto_secondary') if separate_folders else output_dir
        os.makedirs(sec_dir, exist_ok=True)
        for _, row in pareto_df_secondary.iterrows():
            dst = os.path.join(sec_dir, row['output_filename'])
            if os.path.exists(row['model_file']):
                shutil.copy2(row['model_file'], dst)
                print(f"  [secondary]  {row['output_filename']}")
                success += 1
            else:
                print(f"  MISSING: {row['model_file']}")

    print(f"Total copied: {success}")
    return success


def save_results(df, pareto_df, pareto_df_secondary, output_dir, metric_name, bkg_rej_weights):
    cols = ['trial_id', 'hp_tag', 'parameters', 'auc', 'primary_metric']

    save_df = df[cols].copy()
    if 'bg_rej_results' in df.columns:
        for se in sorted(df.iloc[0]['bg_rej_results'].keys()):
            save_df[f'bkg_rej_@{se:.0%}'] = df['bg_rej_results'].apply(lambda x: x[se])
    save_df.sort_values('primary_metric', ascending=False).to_csv(
        os.path.join(output_dir, 'roc_based_analysis_detailed.csv'), index=False)

    pareto_df[cols].to_csv(os.path.join(output_dir, 'pareto_optimal_models_roc_primary.csv'), index=False)

    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        pareto_df_secondary[cols].to_csv(
            os.path.join(output_dir, 'pareto_optimal_models_roc_secondary.csv'), index=False)
        combined = pd.concat([pareto_df[cols], pareto_df_secondary[cols]], ignore_index=True)
        combined.sort_values('primary_metric', ascending=False).to_csv(
            os.path.join(output_dir, 'pareto_optimal_models_roc_combined.csv'), index=False)

    n_sec = len(pareto_df_secondary) if pareto_df_secondary is not None and not pareto_df_secondary.empty else 0
    summary = {
        'timestamp': datetime.now().isoformat(),
        'metric_name': metric_name,
        'bkg_rej_weights': bkg_rej_weights or 'N/A',
        'total_models': len(df),
        'primary_pareto_models': len(pareto_df),
        'secondary_pareto_models': n_sec,
        'primary_metric_range': {'min': float(df['primary_metric'].min()),
                                  'max': float(df['primary_metric'].max()),
                                  'mean': float(df['primary_metric'].mean())},
        'parameters_range': {'min': int(df['parameters'].min()),
                              'max': int(df['parameters'].max()),
                              'mean': float(df['parameters'].mean())},
        'auc_range': {'min': float(df['auc'].min()),
                      'max': float(df['auc'].max()),
                      'mean': float(df['auc'].mean())},
    }
    with open(os.path.join(output_dir, 'pareto_roc_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved results to {output_dir}")


def save_model_architectures(pareto_df, pareto_df_secondary, output_dir):
    arch_dir = os.path.join(output_dir, 'architectures')
    os.makedirs(arch_dir, exist_ok=True)
    custom_objects = get_custom_objects()
    all_summary = {}

    all_rows = []
    if pareto_df is not None:
        tmp = pareto_df.copy(); tmp['tier'] = 'primary'; all_rows.append(tmp)
    if pareto_df_secondary is not None and not pareto_df_secondary.empty:
        tmp = pareto_df_secondary.copy(); tmp['tier'] = 'secondary'; all_rows.append(tmp)
    if not all_rows:
        return

    for _, row in pd.concat(all_rows, ignore_index=True).iterrows():
        name = Path(row['output_filename']).stem
        try:
            model = load_model(row['model_file'], custom_objects=custom_objects, compile=False)
        except Exception as e:
            print(f"  Could not load {name}: {e}")
            continue

        layers_info = []
        for layer in model.layers:
            lc = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'trainable_params': int(np.sum([tf.keras.backend.count_params(w) for w in layer.trainable_weights])),
                'non_trainable_params': int(np.sum([tf.keras.backend.count_params(w) for w in layer.non_trainable_weights])),
            }
            try: lc['output_shape'] = str(layer.output_shape)
            except Exception: lc['output_shape'] = 'unknown'
            try:
                qcfg = layer.get_config()
                for key in ('kernel_quantizer', 'bias_quantizer', 'activation'):
                    if key in qcfg:
                        lc[key] = str(qcfg[key])
            except Exception: pass
            layers_info.append(lc)

        arch = {'model_name': name, 'tier': row['tier'],
                'total_params': int(model.count_params()),
                'num_layers': len(layers_info), 'layers': layers_info}
        with open(os.path.join(arch_dir, f"{name}_architecture.json"), 'w') as f:
            json.dump(arch, f, indent=2)
        all_summary[name] = arch
        print(f"  {name} ({row['tier']}): {len(layers_info)} layers, {arch['total_params']:,} params")

        del model
        tf.keras.backend.clear_session()

    with open(os.path.join(arch_dir, 'all_architectures_summary.json'), 'w') as f:
        json.dump(all_summary, f, indent=2)


# ============================================================================
# FOLDER SCANNING & GROUPING
# ============================================================================

def scan_input_dir(input_dir):
    """Return a dict mapping bit_config -> list of (hp_n, folder_path)."""
    groups = {}
    for name in sorted(os.listdir(input_dir)):
        m = _FOLDER_RE.match(name)
        if not m:
            continue
        hp_n = int(m.group(1))
        bit_config = m.group(2)
        folder = os.path.join(input_dir, name)
        groups.setdefault(bit_config, []).append((hp_n, folder))
    # Sort each group by hp_n
    for bc in groups:
        groups[bc].sort(key=lambda x: x[0])
    return groups


def collect_h5_files(group_entries):
    """
    Given [(hp_n, folder), ...], return list of dicts:
      {'model_file': str, 'hp_tag': str, 'trial_num': str, 'output_filename': str}
    """
    entries = []
    for hp_n, folder in group_entries:
        hp_tag = f"hp{hp_n}q"
        for fname in sorted(os.listdir(folder)):
            if not (fname.startswith('model_trial_') and fname.endswith('.h5')):
                continue
            trial_num = fname[len('model_trial_'):-len('.h5')]  # e.g. "005"
            entries.append({
                'model_file': os.path.join(folder, fname),
                'hp_tag': hp_tag,
                'trial_num': trial_num,
                'output_filename': f"{hp_tag}_model_trial_{trial_num}.h5",
                'trial_id': f"{hp_tag}_trial_{trial_num}",
            })
    return entries


# ============================================================================
# PER-GROUP PROCESSING
# ============================================================================

def process_bit_config_group(bit_config, group_entries, validation_dataset,
                              signal_efficiencies, bkg_rej_weights, use_weighted,
                              output_base_dir, no_secondary, separate_folders):
    print("\n" + "=" * 80)
    print(f"PROCESSING GROUP: {bit_config}")
    print(f"  HP variants: {[f'hp{n}q' for n, _ in group_entries]}")
    print("=" * 80)

    h5_entries = collect_h5_files(group_entries)
    if not h5_entries:
        print(f"  No h5 files found — skipping.")
        return

    print(f"  Total models to evaluate: {len(h5_entries)}")

    # Detect features from first model in group
    print(f"\n  Detecting features from: {h5_entries[0]['output_filename']}")
    feature_description = _detect_model_input_features(h5_entries[0]['model_file'])

    # Evaluate all models
    results = []
    for entry in h5_entries:
        result = evaluate_model_roc(
            entry['model_file'], validation_dataset,
            signal_efficiencies, use_weighted=use_weighted,
            bkg_rej_weights=bkg_rej_weights,
            display_name=entry['output_filename'],
        )
        if result is None:
            continue
        result['trial_id'] = entry['trial_id']
        result['hp_tag'] = entry['hp_tag']
        result['output_filename'] = entry['output_filename']
        results.append(result)

    if not results:
        print("  No models successfully evaluated — skipping.")
        return

    df = pd.DataFrame(results)
    print(f"\n  Evaluated {len(df)} models")
    print(f"  Params range : {df['parameters'].min():,} – {df['parameters'].max():,}")
    print(f"  Metric range : {df['primary_metric'].min():.4f} – {df['primary_metric'].max():.4f}")
    print(f"  AUC range    : {df['auc'].min():.4f} – {df['auc'].max():.4f}")

    # Pareto selection
    pareto_df, pareto_df_secondary = select_pareto_models(df)
    if no_secondary:
        pareto_df_secondary = None

    # Output directory for this group
    out_dir = os.path.join(output_base_dir, bit_config)
    os.makedirs(out_dir, exist_ok=True)

    metric_name = df.iloc[0]['metric_name']
    plot_pareto_front(df, pareto_df, pareto_df_secondary, out_dir, bit_config, metric_name)
    copy_model_files(pareto_df, pareto_df_secondary, out_dir, separate_folders=separate_folders)
    save_results(df, pareto_df, pareto_df_secondary, out_dir, metric_name, bkg_rej_weights)
    save_model_architectures(pareto_df, pareto_df_secondary, out_dir)

    n_sec = len(pareto_df_secondary) if pareto_df_secondary is not None else 0
    print(f"\n  Group done — selected {len(pareto_df)} primary + {n_sec} secondary models")
    return len(pareto_df), n_sec


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-folder Pareto selection for model1 quantization trials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing model1_quantized_hp*q_*_results_* folders')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing TFRecords (with tfrecords_validation/ sub-dir)')
    parser.add_argument('--output_dir', required=True,
                        help='Root output dir; one sub-folder is created per bit_config group')
    parser.add_argument('--use_weighted', action='store_true', default=True,
                        help='Use weighted background rejection (default: True)')
    parser.add_argument('--signal_efficiency', type=float, default=0.95,
                        help='Signal efficiency when not using weighted metric')
    parser.add_argument('--bkg_rej_weights', type=str,
                        default='0.95:0.1,0.98:0.7,0.99:0.2',
                        help='Weights for background rejection (format: "se:w,...)"')
    parser.add_argument('--no_secondary', action='store_true', default=False,
                        help='Disable secondary Pareto tier')
    parser.add_argument('--no_separate_folders', action='store_true', default=False,
                        help='Flat output instead of pareto_primary / pareto_secondary sub-folders')
    parser.add_argument('--bit_configs', type=str, default=None,
                        help='Comma-separated list of bit configs to process (default: all found)')

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: input_dir does not exist: {args.input_dir}"); sys.exit(1)
    if not os.path.isdir(args.data_dir):
        print(f"Error: data_dir does not exist: {args.data_dir}"); sys.exit(1)

    # Parse weights
    if args.use_weighted:
        bkg_rej_weights = {}
        for pair in args.bkg_rej_weights.split(','):
            se, w = pair.split(':')
            bkg_rej_weights[float(se)] = float(w)
        signal_efficiencies = list(bkg_rej_weights.keys())
    else:
        bkg_rej_weights = None
        signal_efficiencies = [args.signal_efficiency]

    # Scan folders
    groups = scan_input_dir(args.input_dir)
    if not groups:
        print(f"No matching folders found in {args.input_dir}"); sys.exit(1)

    if args.bit_configs:
        wanted = {bc.strip() for bc in args.bit_configs.split(',')}
        groups = {k: v for k, v in groups.items() if k in wanted}
        if not groups:
            print(f"None of the requested bit_configs found."); sys.exit(1)

    print("\n" + "=" * 80)
    print("MODEL1 MULTI-FOLDER PARETO SELECTION")
    print("=" * 80)
    print(f"Input dir  : {args.input_dir}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"\nFound {len(groups)} bit-config groups:")
    for bc, entries in sorted(groups.items()):
        print(f"  {bc:30s}  ({len(entries)} hp variants: {[f'hp{n}q' for n,_ in entries]})")

    # Load validation data once (features detected per group, dataset rebuilt if features differ)
    val_dir = os.path.join(args.data_dir, "tfrecords_validation/")
    if not os.path.exists(val_dir):
        print(f"Error: validation dir not found: {val_dir}"); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    separate_folders = not args.no_separate_folders

    # Process each group
    group_summary = {}
    cached_features = None
    cached_dataset = None

    for bit_config, entries in sorted(groups.items()):
        h5_entries = collect_h5_files(entries)
        if not h5_entries:
            continue

        # Detect features; rebuild dataset only if features changed
        feat_desc = _detect_model_input_features(h5_entries[0]['model_file'])
        feat_key = tuple(sorted(feat_desc.keys()))
        if feat_key != cached_features:
            print(f"\nBuilding validation dataset for features: {sorted(feat_desc.keys())}")
            cached_dataset = build_tfrecord_dataset(val_dir, feat_desc)
            cached_features = feat_key

        result = process_bit_config_group(
            bit_config, entries, cached_dataset,
            signal_efficiencies, bkg_rej_weights, args.use_weighted,
            args.output_dir, args.no_secondary, separate_folders,
        )
        if result:
            group_summary[bit_config] = {'primary': result[0], 'secondary': result[1]}

    # Final summary
    print("\n" + "=" * 80)
    print("ALL GROUPS COMPLETE")
    print("=" * 80)
    total_primary = sum(v['primary'] for v in group_summary.values())
    total_secondary = sum(v['secondary'] for v in group_summary.values())
    for bc, counts in sorted(group_summary.items()):
        print(f"  {bc:30s}  primary={counts['primary']}, secondary={counts['secondary']}")
    print(f"\nTotal selected: {total_primary} primary + {total_secondary} secondary")
    print(f"Output root   : {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
