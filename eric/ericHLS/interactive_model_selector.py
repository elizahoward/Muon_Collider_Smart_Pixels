#!/usr/bin/env python3
"""
Interactive Model Selector

This script provides an interactive graphical interface to select models
based on validation accuracy vs total parameters. Users can visualize the
Pareto frontier and manually select which models to include for HLS synthesis.

Usage:
    python interactive_model_selector.py --input_dir <hyperparam_search_dir> --output_dir <output_dir>

Controls:
    - Click on points to select/deselect individual models
    - Use Lasso tool to select multiple models at once
    - Press 'a' to select all points
    - Press 'c' to clear selection
    - Press 'p' to show Pareto frontier
    - Close window to finalize selection and copy files

Author: Eric
Date: 2025
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button
from matplotlib.path import Path as MplPath
from pathlib import Path


def load_trials_from_keras_tuner(search_dir):
    """
    Load trial data from Keras Tuner directory structure (trial_XXX folders).
    
    Args:
        search_dir: Keras Tuner search directory with trial_XXX subdirectories
        
    Returns:
        list: List of trial dictionaries with scores
    """
    trials = []
    
    # Find all trial directories
    trial_dirs = sorted([d for d in os.listdir(search_dir) 
                        if d.startswith('trial_') and 
                        os.path.isdir(os.path.join(search_dir, d))])
    
    for trial_dir in trial_dirs:
        trial_json = os.path.join(search_dir, trial_dir, 'trial.json')
        
        if not os.path.exists(trial_json):
            continue
        
        try:
            with open(trial_json, 'r') as f:
                trial_data = json.load(f)
            
            # Extract key information
            trial_id = int(trial_dir.split('_')[1])
            score = trial_data.get('score')
            hyperparams = trial_data.get('hyperparameters', {}).get('values', {})
            
            trials.append({
                'trial_id': trial_id,
                'score': score,
                'val_accuracy': score,
                'hyperparameters': hyperparams,
                'model_file': f'model_trial_{trial_id:03d}.h5',
                'hyperparams_file': f'hyperparams_trial_{trial_id:03d}.json'
            })
        
        except Exception as e:
            print(f"Warning: Error loading {trial_json}: {str(e)}")
    
    return trials


def load_trials_summary(input_dir):
    """Load trials summary from JSON file."""
    summary_file = os.path.join(input_dir, 'trials_summary.json')
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    # Try filtered_trials_summary.json
    filtered_summary = os.path.join(input_dir, 'filtered_trials_summary.json')
    if os.path.exists(filtered_summary):
        with open(filtered_summary, 'r') as f:
            data = json.load(f)
            return data.get('trials', [])
    
    return None


def extract_validation_accuracy(trial):
    """Extract validation accuracy from trial."""
    possible_keys = ['val_accuracy', 'validation_accuracy', 'val_acc', 
                     'accuracy', 'score', 'best_val_accuracy', 'final_val_accuracy']
    
    if 'metrics' in trial:
        metrics = trial['metrics']
        for key in possible_keys:
            if key in metrics and metrics[key] is not None:
                return float(metrics[key])
    
    for key in possible_keys:
        if key in trial and trial[key] is not None:
            return float(trial[key])
    
    return None


class InteractiveModelSelector:
    """Interactive scatter plot for model selection."""
    
    def __init__(self, trials, models_dir):
        self.trials = trials
        self.models_dir = models_dir
        self.selected_indices = set()
        self.show_pareto = False
        
        # Extract data
        self.prepare_data()
        
        # Create figure
        self.setup_plot()
        
    def prepare_data(self):
        """Prepare data for plotting from trials_summary.json metadata."""
        print(f"\nLoading trial data from {len(self.trials)} trials...")
        
        self.valid_trials = []
        self.accuracies = []
        self.param_counts = []
        
        skipped_count = 0
        for i, trial in enumerate(self.trials):
            # Extract validation accuracy
            acc = extract_validation_accuracy(trial)
            if acc is None:
                skipped_count += 1
                continue
            
            # Extract parameter count from metadata (much faster than loading model)
            param_count = trial.get('total_parameters')
            if param_count is None:
                skipped_count += 1
                continue
            
            self.valid_trials.append(trial)
            self.accuracies.append(acc)
            self.param_counts.append(param_count)
        
        self.accuracies = np.array(self.accuracies)
        self.param_counts = np.array(self.param_counts)
        
        print(f"Successfully loaded {len(self.valid_trials)} trials ({skipped_count} skipped due to missing data)")
        
        if len(self.valid_trials) == 0:
            print("\nERROR: No valid trials found!")
            print("Possible reasons:")
            print("  1. trials_summary.json is missing validation accuracy data")
            print("  2. trials_summary.json is missing total_parameters data")
            print("\nTo fix this, run:")
            print("  python add_metadata_to_trials.py --results_dir <results_dir> --search_dir <search_dir>")
            sys.exit(1)
    
    def compute_pareto_frontier(self):
        """
        Compute Pareto frontier (models that are not dominated by any other).
        A model dominates another if it has both higher accuracy and fewer parameters.
        """
        pareto_indices = []
        
        for i in range(len(self.accuracies)):
            is_pareto = True
            for j in range(len(self.accuracies)):
                if i != j:
                    # j dominates i if it has higher accuracy and fewer parameters
                    if (self.accuracies[j] >= self.accuracies[i] and 
                        self.param_counts[j] <= self.param_counts[i] and
                        (self.accuracies[j] > self.accuracies[i] or 
                         self.param_counts[j] < self.param_counts[i])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def setup_plot(self):
        """Setup the interactive plot."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Plot all points
        self.scatter = self.ax.scatter(
            self.param_counts, 
            self.accuracies,
            c='blue', 
            alpha=0.6, 
            s=50,
            picker=True,
            label='Available models'
        )
        
        # Initialize selected points scatter (empty)
        self.selected_scatter = self.ax.scatter(
            [], [], 
            c='red', 
            alpha=0.8, 
            s=80,
            marker='o',
            edgecolors='darkred',
            linewidths=2,
            label='Selected models',
            zorder=10
        )
        
        # Pareto frontier line (initially hidden)
        self.pareto_line, = self.ax.plot([], [], 'g--', linewidth=2, 
                                         alpha=0.7, label='Pareto frontier')
        
        self.ax.set_xlabel('Total Parameters', fontsize=12)
        self.ax.set_ylabel('Validation Accuracy', fontsize=12)
        self.ax.set_title('Interactive Model Selection\n' + 
                         'Click to select/deselect | Use lasso to select multiple | ' +
                         'Press "a" for all, "c" to clear, "p" for Pareto',
                         fontsize=14, pad=20)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Add info text
        self.info_text = self.fig.text(0.5, 0.05, 
                                       f'Selected: 0 / {len(self.valid_trials)} models',
                                       ha='center', fontsize=12,
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add buttons
        self.add_buttons()
        
        # Connect events
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Add lasso selector
        self.lasso = LassoSelector(self.ax, self.on_lasso)
    
    def add_buttons(self):
        """Add control buttons."""
        # Select All button
        ax_select_all = plt.axes([0.2, 0.01, 0.1, 0.04])
        self.btn_select_all = Button(ax_select_all, 'Select All')
        self.btn_select_all.on_clicked(self.select_all)
        
        # Clear button
        ax_clear = plt.axes([0.35, 0.01, 0.1, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self.clear_selection)
        
        # Pareto button
        ax_pareto = plt.axes([0.5, 0.01, 0.15, 0.04])
        self.btn_pareto = Button(ax_pareto, 'Show Pareto')
        self.btn_pareto.on_clicked(self.toggle_pareto)
        
        # Done button
        ax_done = plt.axes([0.7, 0.01, 0.1, 0.04])
        self.btn_done = Button(ax_done, 'Done')
        self.btn_done.on_clicked(lambda event: plt.close(self.fig))
    
    def on_pick(self, event):
        """Handle point click."""
        if event.mouseevent.button == 1:  # Left click
            ind = event.ind[0]
            
            if ind in self.selected_indices:
                self.selected_indices.remove(ind)
            else:
                self.selected_indices.add(ind)
            
            self.update_plot()
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'a':
            self.select_all(None)
        elif event.key == 'c':
            self.clear_selection(None)
        elif event.key == 'p':
            self.toggle_pareto(None)
    
    def on_lasso(self, verts):
        """Handle lasso selection."""
        path = MplPath(verts)
        
        for i in range(len(self.param_counts)):
            point = (self.param_counts[i], self.accuracies[i])
            if path.contains_point(point):
                self.selected_indices.add(i)
        
        self.update_plot()
    
    def select_all(self, event):
        """Select all points."""
        self.selected_indices = set(range(len(self.valid_trials)))
        self.update_plot()
    
    def clear_selection(self, event):
        """Clear all selections."""
        self.selected_indices.clear()
        self.update_plot()
    
    def toggle_pareto(self, event):
        """Toggle Pareto frontier display."""
        self.show_pareto = not self.show_pareto
        
        if self.show_pareto:
            pareto_indices = self.compute_pareto_frontier()
            pareto_params = self.param_counts[pareto_indices]
            pareto_accs = self.accuracies[pareto_indices]
            
            # Sort by parameters for line plot
            sort_idx = np.argsort(pareto_params)
            self.pareto_line.set_data(pareto_params[sort_idx], pareto_accs[sort_idx])
            self.btn_pareto.label.set_text('Hide Pareto')
            
            # Optionally select Pareto models
            # self.selected_indices.update(pareto_indices)
        else:
            self.pareto_line.set_data([], [])
            self.btn_pareto.label.set_text('Show Pareto')
        
        self.fig.canvas.draw()
    
    def update_plot(self):
        """Update the plot with current selection."""
        # Update selected points
        if self.selected_indices:
            selected_idx = list(self.selected_indices)
            self.selected_scatter.set_offsets(
                np.c_[self.param_counts[selected_idx], self.accuracies[selected_idx]]
            )
        else:
            self.selected_scatter.set_offsets(np.empty((0, 2)))
        
        # Update info text
        self.info_text.set_text(
            f'Selected: {len(self.selected_indices)} / {len(self.valid_trials)} models'
        )
        
        self.fig.canvas.draw()
    
    def get_selected_trials(self):
        """Return list of selected trial dictionaries."""
        return [self.valid_trials[i] for i in sorted(self.selected_indices)]
    
    def show(self):
        """Display the interactive plot."""
        plt.show()


def copy_model_files(trial, input_dir, output_dir):
    """Copy model and hyperparameter files."""
    try:
        model_file = trial.get('model_file')
        if not model_file:
            return False
        
        src_model = os.path.join(input_dir, model_file)
        dst_model = os.path.join(output_dir, model_file)
        
        if not os.path.exists(src_model):
            print(f"Warning: Model file not found: {src_model}")
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


def create_selection_summary(selected_trials, output_dir, all_trials):
    """Create a summary file for the selected trials."""
    # Calculate statistics
    all_accs = [extract_validation_accuracy(t) for t in all_trials]
    all_accs = [a for a in all_accs if a is not None]
    
    selected_accs = [extract_validation_accuracy(t) for t in selected_trials]
    
    summary = {
        'total_available_trials': len(all_trials),
        'total_selected': len(selected_trials),
        'selection_method': 'interactive_manual',
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
    
    print(f"Created selection summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive model selector with visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_model_selector.py \\
      --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\
      --output_dir ../model2_selected_for_hls

  # With separate search directory
  python interactive_model_selector.py \\
      --input_dir ../models \\
      --search_dir ../keras_tuner_search \\
      --output_dir ../model2_selected_for_hls
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing H5 model files')
    parser.add_argument('--search_dir', type=str, default=None,
                       help='Keras Tuner search directory (if different from input_dir)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for selected models')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show selection without copying files')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    search_directory = args.search_dir if args.search_dir else args.input_dir
    
    # Load trials
    print(f"Loading trials from: {search_directory}")
    
    trial_dirs = [d for d in os.listdir(search_directory) 
                  if d.startswith('trial_') and 
                  os.path.isdir(os.path.join(search_directory, d))]
    
    if trial_dirs:
        print(f"Detected Keras Tuner directory ({len(trial_dirs)} trial folders)")
        trials = load_trials_from_keras_tuner(search_directory)
    else:
        trials = load_trials_summary(search_directory)
    
    if not trials:
        print("Error: No trials found")
        sys.exit(1)
    
    print(f"Loaded {len(trials)} trials")
    
    # Create interactive selector
    print("\nOpening interactive selection window...")
    print("Controls:")
    print("  - Click on points to select/deselect")
    print("  - Use lasso tool to select multiple points")
    print("  - Press 'a' to select all")
    print("  - Press 'c' to clear selection")
    print("  - Press 'p' to show/hide Pareto frontier")
    print("  - Click 'Done' or close window when finished")
    
    selector = InteractiveModelSelector(trials, args.input_dir)
    selector.show()
    
    # Get selected trials
    selected_trials = selector.get_selected_trials()
    
    if not selected_trials:
        print("\nNo models selected. Exiting.")
        sys.exit(0)
    
    # Print selection summary
    print("\n" + "=" * 80)
    print("SELECTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal trials available: {len(trials)}")
    print(f"Trials selected: {len(selected_trials)}")
    
    selected_accs = [extract_validation_accuracy(t) for t in selected_trials]
    print(f"\nSelected models accuracy range:")
    print(f"  Min:  {min(selected_accs):.4f}")
    print(f"  Max:  {max(selected_accs):.4f}")
    print(f"  Mean: {np.mean(selected_accs):.4f}")
    
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - No files will be copied")
        print("=" * 80)
        return
    
    # Create output directory and copy files
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "-" * 80)
    print("COPYING FILES")
    print("-" * 80)
    
    success_count = 0
    for trial in selected_trials:
        trial_id = trial.get('trial_id', 'unknown')
        if copy_model_files(trial, args.input_dir, args.output_dir):
            success_count += 1
            model_file = trial.get('model_file', 'unknown')
            acc = extract_validation_accuracy(trial)
            print(f"  ✓ Copied trial {trial_id:3d}: {model_file} (acc: {acc:.4f})")
        else:
            print(f"  ✗ Failed to copy trial {trial_id:3d}")
    
    # Create summary
    create_selection_summary(selected_trials, args.output_dir, trials)
    
    # Final summary
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully copied: {success_count}/{len(selected_trials)} models")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

