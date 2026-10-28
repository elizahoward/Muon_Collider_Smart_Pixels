"""
Script to read parquet files and plot histogram of z-global values for signal.

This script:
1. Loads parquet files from Simulation_Output_Signal directory
2. Extracts z-global values from all signal files
3. Plots histogram of signal z-global distribution

Author: Modified to read parquet files directly
Date: 2025
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_z_global_from_parquet(data_dir):
    """
    Load z-global values from signal parquet files.
    
    Args:
        data_dir: Path to directory containing parquet files
    
    Returns:
        z_global_sig: z-global values for signal events
    """
    # Find all signal label files
    label_files = sorted(glob.glob(os.path.join(data_dir, "labelssig*.parquet")))
    
    if not label_files:
        raise ValueError(f"No signal label files found in {data_dir}")
    
    print(f"Found {len(label_files)} signal files")
    
    # Collect z-global values
    z_global_sig = []
    
    # Process each file
    print("\nLoading signal files...")
    for label_file in tqdm(label_files):
        try:
            # Read the labels file which contains z-global
            df = pd.read_parquet(label_file)
            
            # Check if z-global column exists
            if 'z-global' not in df.columns:
                print(f"Warning: 'z-global' column not found in {label_file}")
                print(f"Available columns: {df.columns.tolist()}")
                continue
            
            # Extract z-global values
            z_global_values = df['z-global'].values
            z_global_sig.extend(z_global_values.tolist())
            
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            continue
    
    # Convert to numpy array
    z_global_sig = np.array(z_global_sig)
    
    print(f"\nLoaded {len(z_global_sig)} signal events")
    
    return z_global_sig


def plot_z_global_histogram(z_global_sig, bins=50, save_path=None):
    """
    Plot histogram of z-global values for signal.
    
    Args:
        z_global_sig: z-global values for signal
        bins: Number of bins for histogram
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(z_global_sig, bins=bins, alpha=0.7, label=f'Signal (n={len(z_global_sig)})', 
            color='blue', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('z-global', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of z-global values: Signal', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (
        f'Mean: μ={np.mean(z_global_sig):.4f}\n'
        f'Std: σ={np.std(z_global_sig):.4f}\n'
        f'Min: {np.min(z_global_sig):.4f}\n'
        f'Max: {np.max(z_global_sig):.4f}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to run the script."""
    # Default data directory
    data_dir = "/local/d1/smartpixML/bigData/Simulation_Output_2"
    
    # Allow command line argument for different directory
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    print("="*60)
    print("z-global Histogram Plotter (Signal Only)")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print("="*60)
    
    # Load data
    try:
        z_global_sig = load_z_global_from_parquet(data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if len(z_global_sig) == 0:
        print("No data loaded. Exiting.")
        return
    
    # Plot histogram
    print("\n" + "="*60)
    print("Plotting histogram...")
    print("="*60)
    
    save_path = "z_global_signal_histogram.png"
    plot_z_global_histogram(z_global_sig, bins=50, save_path=save_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

