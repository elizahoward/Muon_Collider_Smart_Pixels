#!/usr/bin/env python3
"""
Plot Models and Select by Criteria

Generate a visualization of models (accuracy vs parameters) and select models
based on criteria (threshold, top N, parameter range, etc.)

Author: Eric
Date: 2025
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def load_trials_summary(input_dir):
    """Load trials from trials_summary.json."""
    summary_file = os.path.join(input_dir, 'trials_summary.json')
    
    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found")
        return None
    
    with open(summary_file, 'r') as f:
        return json.load(f)


def extract_validation_accuracy(trial):
    """Extract validation accuracy from trial."""
    possible_keys = ['val_accuracy', 'validation_accuracy', 'val_acc', 
                     'accuracy', 'score']
    
    for key in possible_keys:
        if key in trial and trial[key] is not None:
            return float(trial[key])
    
    return None


def plot_models(trials, output_plot, title="Model Selection"):
    """Create a scatter plot of models."""
    accuracies = []
    param_counts = []
    valid_trials = []
    
    for trial in trials:
        acc = extract_validation_accuracy(trial)
        params = trial.get('total_parameters')
        
        if acc is not None and params is not None:
            accuracies.append(acc)
            param_counts.append(params)
            valid_trials.append(trial)
    
    if len(valid_trials) == 0:
        print("Error: No valid trials with accuracy and parameters")
        return valid_trials
    
    accuracies = np.array(accuracies)
    param_counts = np.array(param_counts)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(param_counts, accuracies, c='blue', alpha=0.6, s=50)
    
    ax.set_xlabel('Total Parameters', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Models: {len(valid_trials)}\n'
    stats_text += f'Accuracy: {accuracies.min():.4f} - {accuracies.max():.4f}\n'
    stats_text += f'Parameters: {param_counts.min():,} - {param_counts.max():,}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Compute and plot Pareto frontier
    pareto_indices = compute_pareto_frontier(accuracies, param_counts)
    if len(pareto_indices) > 0:
        pareto_params = param_counts[pareto_indices]
        pareto_accs = accuracies[pareto_indices]
        sort_idx = np.argsort(pareto_params)
        ax.plot(pareto_params[sort_idx], pareto_accs[sort_idx], 
               'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')
        ax.scatter(pareto_params, pareto_accs, c='red', s=100, 
                  marker='*', edgecolors='darkred', linewidths=2, 
                  label=f'Pareto Optimal ({len(pareto_indices)} models)', zorder=10)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_plot}")
    
    return valid_trials


def compute_pareto_frontier(accuracies, param_counts):
    """Compute Pareto frontier indices."""
    pareto_indices = []
    
    for i in range(len(accuracies)):
        is_pareto = True
        for j in range(len(accuracies)):
            if i != j:
                if (accuracies[j] >= accuracies[i] and 
                    param_counts[j] <= param_counts[i] and
                    (accuracies[j] > accuracies[i] or param_counts[j] < param_counts[i])):
                    is_pareto = False
                    break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return pareto_indices


def select_models_by_criteria(trials, criteria):
    """Select models based on various criteria."""
    accuracies = []
    param_counts = []
    valid_trials = []
    
    for trial in trials:
        acc = extract_validation_accuracy(trial)
        params = trial.get('total_parameters')
        
        if acc is not None and params is not None:
            accuracies.append(acc)
            param_counts.append(params)
            valid_trials.append(trial)
    
    accuracies = np.array(accuracies)
    param_counts = np.array(param_counts)
    
    selected_indices = set()
    
    # Apply criteria
    if criteria.get('top_n'):
        # Select top N by accuracy
        top_indices = np.argsort(accuracies)[::-1][:criteria['top_n']]
        selected_indices.update(top_indices)
        print(f"  Selected top {criteria['top_n']} by accuracy")
    
    if criteria.get('pareto'):
        # Select Pareto frontier
        pareto_indices = compute_pareto_frontier(accuracies, param_counts)
        selected_indices.update(pareto_indices)
        print(f"  Selected {len(pareto_indices)} Pareto optimal models")
    
    if criteria.get('acc_threshold'):
        # Select models above accuracy threshold
        threshold = criteria['acc_threshold']
        above_threshold = np.where(accuracies >= threshold)[0]
        selected_indices.update(above_threshold)
        print(f"  Selected {len(above_threshold)} models with accuracy >= {threshold:.4f}")
    
    if criteria.get('param_range'):
        # Select models within parameter range
        min_p, max_p = criteria['param_range']
        in_range = np.where((param_counts >= min_p) & (param_counts <= max_p))[0]
        selected_indices.update(in_range)
        print(f"  Selected {len(in_range)} models with {min_p:,} <= params <= {max_p:,}")
    
    if criteria.get('top_n_smallest'):
        # Select N smallest models (by parameters)
        smallest_indices = np.argsort(param_counts)[:criteria['top_n_smallest']]
        selected_indices.update(smallest_indices)
        print(f"  Selected {criteria['top_n_smallest']} smallest models")
    
    selected_trials = [valid_trials[i] for i in sorted(selected_indices)]
    
    return selected_trials


def copy_model_files(trial, input_dir, output_dir):
    """Copy model and hyperparameter files."""
    try:
        model_file = trial.get('model_file')
        if not model_file:
            return False
        
        src_model = os.path.join(input_dir, model_file)
        dst_model = os.path.join(output_dir, model_file)
        
        if not os.path.exists(src_model):
            return False
        
        shutil.copy2(src_model, dst_model)
        
        # Copy hyperparameters
        hyperparam_file = trial.get('hyperparams_file')
        if hyperparam_file:
            src_hyperparam = os.path.join(input_dir, hyperparam_file)
            dst_hyperparam = os.path.join(output_dir, hyperparam_file)
            
            if os.path.exists(src_hyperparam):
                shutil.copy2(src_hyperparam, dst_hyperparam)
        
        return True
    
    except Exception as e:
        print(f"Error copying files: {str(e)}")
        return False


def create_selection_summary(selected_trials, output_dir, all_trials, criteria):
    """Create a summary file for the selected trials."""
    all_accs = [extract_validation_accuracy(t) for t in all_trials]
    all_accs = [a for a in all_accs if a is not None]
    
    selected_accs = [extract_validation_accuracy(t) for t in selected_trials]
    
    summary = {
        'total_available_trials': len(all_trials),
        'total_selected': len(selected_trials),
        'selection_method': 'criteria_based',
        'selection_criteria': criteria,
        'validation_accuracy_range': {
            'all_trials': {
                'min': float(min(all_accs)) if all_accs else None,
                'max': float(max(all_accs)) if all_accs else None,
                'mean': float(np.mean(all_accs)) if all_accs else None,
            },
            'selected_trials': {
                'min': float(min(selected_accs)) if selected_accs else None,
                'max': float(max(selected_accs)) if selected_accs else None,
                'mean': float(np.mean(selected_accs)) if selected_accs else None,
            }
        },
        'trials': selected_trials
    }
    
    summary_file = os.path.join(output_dir, 'filtered_trials_summary.json')
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Created selection summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot and select models based on criteria',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plot and select top 25 models by accuracy
  python plot_and_select_models.py \\
      --input_dir model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --output_dir model2_top25_for_hls \\
      --top_n 25

  # Select Pareto optimal models
  python plot_and_select_models.py \\
      --input_dir model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --output_dir model2_pareto_for_hls \\
      --pareto

  # Select models with accuracy >= 0.905 and parameters <= 30000
  python plot_and_select_models.py \\
      --input_dir model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --output_dir model2_filtered_for_hls \\
      --acc_threshold 0.905 \\
      --param_range 0 30000

  # Just generate plot (no selection)
  python plot_and_select_models.py \\
      --input_dir model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --plot_only
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing models and trials_summary.json')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for selected models')
    parser.add_argument('--plot_file', type=str, default='model_selection_plot.png',
                       help='Output plot filename')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only generate plot, do not copy files')
    
    # Selection criteria
    parser.add_argument('--top_n', type=int,
                       help='Select top N models by accuracy')
    parser.add_argument('--pareto', action='store_true',
                       help='Select Pareto optimal models')
    parser.add_argument('--acc_threshold', type=float,
                       help='Minimum validation accuracy threshold')
    parser.add_argument('--param_range', type=int, nargs=2, metavar=('MIN', 'MAX'),
                       help='Parameter range (min max)')
    parser.add_argument('--top_n_smallest', type=int,
                       help='Select N smallest models by parameter count')
    
    args = parser.parse_args()
    
    # Validate
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Load trials
    print(f"Loading trials from: {args.input_dir}")
    trials = load_trials_summary(args.input_dir)
    
    if not trials:
        sys.exit(1)
    
    print(f"Loaded {len(trials)} trials")
    
    # Generate plot
    plot_path = os.path.join(args.input_dir, args.plot_file) if not args.output_dir else os.path.join(args.output_dir, args.plot_file)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    valid_trials = plot_models(trials, plot_path, 
                               title=f"Model Selection - {os.path.basename(args.input_dir)}")
    
    if args.plot_only:
        print(f"\n✓ Plot saved. View it and decide on selection criteria.")
        return
    
    # Build criteria
    criteria = {}
    if args.top_n:
        criteria['top_n'] = args.top_n
    if args.pareto:
        criteria['pareto'] = True
    if args.acc_threshold:
        criteria['acc_threshold'] = args.acc_threshold
    if args.param_range:
        criteria['param_range'] = tuple(args.param_range)
    if args.top_n_smallest:
        criteria['top_n_smallest'] = args.top_n_smallest
    
    if not criteria:
        print("\nNo selection criteria specified. Use --top_n, --pareto, --acc_threshold, etc.")
        print(f"Plot saved to: {plot_path}")
        return
    
    if not args.output_dir:
        print("\nError: --output_dir required when using selection criteria")
        sys.exit(1)
    
    # Select models
    print(f"\nSelecting models based on criteria:")
    selected_trials = select_models_by_criteria(valid_trials, criteria)
    
    print(f"\n✓ Selected {len(selected_trials)} models")
    
    # Create output directory and copy files
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nCopying files to: {args.output_dir}")
    success_count = 0
    for trial in selected_trials:
        trial_id = trial.get('trial_id', 'unknown')
        if copy_model_files(trial, args.input_dir, args.output_dir):
            success_count += 1
            model_file = trial.get('model_file', 'unknown')
            acc = extract_validation_accuracy(trial)
            params = trial.get('total_parameters', 0)
            print(f"  ✓ Trial {trial_id:3d}: {model_file} (acc: {acc:.4f}, params: {params:,})")
        else:
            print(f"  ✗ Failed to copy trial {trial_id:3d}")
    
    # Create summary
    create_selection_summary(selected_trials, args.output_dir, trials, criteria)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Successfully copied: {success_count}/{len(selected_trials)} models")
    print(f"Output directory: {args.output_dir}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()


