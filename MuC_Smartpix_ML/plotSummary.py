import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
import argparse
from collections import defaultdict
import pandas as pd

class Plotter:
    def __init__(self, results_dir):
        """
        Initialize with results directory path.
        
        Args:
            results_dir (str or Path): Path to the results directory
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")
            
        print(f"Loading results from: {self.results_dir}")
        
        # Use Set2 color scheme for line plots (matching Model2)
        self.set2_colors = plt.cm.Set2(np.linspace(0, 1, 10))
        # Use Reds color scheme for bar plots (matching Model2)
        self.reds_colors = plt.cm.Reds(np.linspace(0.3, 0.9, 10))  # Avoid very light and very dark
        # Custom red color (139, 0, 33) for histogram bars
        self.custom_red = (139/255, 0/255, 33/255)
        self.custom_grey = (70/255, 70/255, 70/255)
    def load_model1_results_data(self):
        """
        Load data from Model1 results format (model subdirectories with trial subdirectories).
        
        Returns:
            dict: Dictionary containing averaged model data
        """
        print("Loading Model1 results data...")
        
        model_data = {}
        
        # Look for model subdirectories
        for model_dir in self.results_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
                
            # if not any(substring in model_dir.name for substring in ['quantized_', 'non_quantized']):
            if not any(substring in model_dir.name for substring in ['Model']):
                continue
                
            print(f"  Processing: {model_dir.name}")
            
            # Load overall results if available
            overall_results_file = model_dir / "quantization_results.csv"
            if overall_results_file.exists():
                try:
                    with open(overall_results_file, 'r') as f:
                        overall_results = pd.read_csv(f)
                        overall_results["category"] = overall_results["model_type"]
                        
                        for index, row in overall_results.iterrows():
                            if row["model_type"] == "non_quantized":
                                overall_results.loc[index,"category"] = "Float32"
                                overall_results.loc[index,"color"] = "grey"
                            else:
                                overall_results.loc[index,"category"] = str(int(row["weight_bits"]))+"-bit"
                                overall_results.loc[index,"color"] = "teal"
                except Exception as e:
                    print(f"    Error loading overall results: {e}")
                    overall_results = {}
            else:
                overall_results = {}

            
            
            # # Load averaged training history
            # history_files = list(model_dir.glob("*_history.npz"))
            # if not history_files:
            #     print(f"    Warning: No history file found in {model_dir.name}")
            #     continue
            
            # history_file = history_files[0]
            
            try:
            #     # Load training history from .npz file
            #     history_data = np.load(history_file)
            #     history = {
            #         'accuracy': history_data['accuracy'].tolist(),
            #         'val_accuracy': history_data['val_accuracy'].tolist(),
            #         'loss': history_data['loss'].tolist(),
            #         'val_loss': history_data['val_loss'].tolist()
            #     }
                
                model_data[model_dir.name] = {
                    # 'history': history,
                    'eval_results': {
                        'roc_auc': overall_results['roc_auc'],
                        'test_accuracy': overall_results['test_accuracy'],
                        'test_loss': overall_results['test_loss'],
                    },
                    'dir_name': model_dir.name,
                    # 'n_trials': overall_results['n_trials', 1],
                    'weight_bits': overall_results['weight_bits'],
                    'model_type': overall_results['model_type'],
                    'overal_results': overall_results
                }
                print("loaded data is")
                print(model_data[model_dir.name])
                
            #     print(f"    ✓ Loaded {len(history['val_accuracy'])} epochs of averaged data from {overall_results.get('n_trials', 1)} trials")
                
            except Exception as e:
                print(f"    Error loading data from {model_dir.name}: {e}")
                continue
        
        return model_data

    def newPlotter(self, model_data, save_dir):
        for name, data in model_data.items():
            # print(data['overal_results'])
            plt.bar(data['overal_results']['category'],data['overal_results']['test_accuracy'],
                    color=data['overal_results']['color'], edgecolor='black', linewidth=1, width=0.6)
            # print("SAVING\n\n\n")
            plt.show()
            plt.savefig(data['dir_name'] + '/newPlot.png', dpi=300, bbox_inches='tight')
            plt.close()


    def create_best_metrics_plots(self, model_data, save_dir):
        """
        Create a square bar plot for best validation accuracy with custom styling.
        
        Args:
            model_data (dict): Dictionary containing model data
            save_dir (Path): Directory to save plots
        """
        print("Creating best validation accuracy plot...")
        
        # # Sort models: non-quantized first, then by weight bits
        # sorted_models = []
        # model_names = []
        
        # for name, data in model_data.items():
            # if data['model_type'] == 'non_quantized':
            #     sorted_models.append((name, data, 0))
            #     model_names.append('Float32')
            # else:
            #     bits = data.get('weight_bits', 999)
            #     sorted_models.append((name, data, bits))
            #     model_names.append(f'{bits}-bit')
        
        # Sort by weight bits
        # sorted_indices = sorted(range(len(sorted_models)), key=lambda i: sorted_models[i][2])
        # sorted_models = [sorted_models[i] for i in sorted_indices]
        # model_names = [model_names[i] for i in sorted_indices]
        
        # # Extract best validation accuracy
        # best_val_acc = []
        
        # for name, data, _ in sorted_models:
        #     best_val_acc.append(data['eval_results']['test_accuracy'])
        
        
        
        # # Create colors list - grey for Float32, custom red for others
        # colors = []
        # for name in model_names:
        #     if name == 'Float32':
        #         colors.append('grey')
        #     else:
        #         colors.append(self.custom_red)
        for name, data in model_data.items():
            print("neew plotting!!")
            # Create taller figure
            fig, ax = plt.subplots(figsize=(9, 11))  # Made taller as requested
            
            # x_pos = np.arange(len(model_names))
            # print(data['overal_results'])
            bars = plt.bar(data['overal_results']['category'],data['overal_results']['test_accuracy'],
                    color=data['overal_results']['color'], edgecolor='black', linewidth=1, width=0.6)
        
            # Set labels and title with appropriate font sizes
            ax.set_ylabel('Best Validation Accuracy', fontsize=36)  # Increased from 24
            modelName = data['overal_results']['model_path']
            modelName = modelName.loc[0][5]
            ax.set_title(f'Model {modelName}: Best Validation Accuracy', fontsize=28, fontweight='bold')  # Made smaller
            
            # Set tick marks and labels
            # ax.set_xticks(x_pos)
            ax.set_xticklabels(data['overal_results']['category'], rotation=45, ha='right', fontsize=30)  # Increased from 20
            ax.tick_params(axis='x', labelsize=30)  # X-axis tick mark size
            ax.tick_params(axis='y', labelsize=20)  # Smaller Y-axis tick mark size
            
            # Set y-axis limits for consistency
            ax.set_ylim(0.5, 1)
            
            # Add horizontal gridlines
            ax.grid(True, alpha=0.3, axis='y')
            
            # Hide the 1.1 tick label while keeping the tick
            yticks = ax.get_yticks()
            ax.set_yticks(yticks)
            yticklabels = [f'{tick:.2f}' if abs(tick - 1.1) > 0.01 else '' for tick in yticks]
            ax.set_yticklabels(yticklabels) 
            
            # Add value labels on top of bars with color matching bars (non-bold)
            for bar, val, color in zip(bars, data['overal_results']['test_accuracy'], data['overal_results']['color']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=24, color=color)
            
            # Adjust layout to fit the larger fonts and give more space for title
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.20, top=0.80, left=0.15, right=0.95)  # Even more space at top for title separation
            
            # Save the plot
            plt.savefig(data['dir_name'] + '/best_validation_accuracy2.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ Saved: best_validation_accuracy.png")

    def generate_all_plots(self):
        save_dir = self.results_dir
        model_data = self.load_model1_results_data()
        print(model_data)
        self.create_best_metrics_plots(model_data, save_dir)
        print("new plotter")
        self.newPlotter(model_data, save_dir)


def main():
    """Main function to run the plotting script"""
    parser = argparse.ArgumentParser(description='Generate plots from Model1 training results')
    parser.add_argument('--results_dir', help='Path to results directory')
    parser.add_argument('--output', help='Output directory (default: same as results_dir)')
    
    args = parser.parse_args()
    
    try:
        plotter = Plotter(args.results_dir)
        plotter.generate_all_plots()
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

    