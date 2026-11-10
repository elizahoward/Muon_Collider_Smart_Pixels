#!/usr/bin/env python3
"""
Plot Model Parameters vs Hardware Resources (FF + LUT)

This script combines parameter count data with HLS resource utilization data,
plots the relationship, and fits a linear regression (OLS) model.

Usage:
    python plot_parameters_vs_resources.py --resource_csv <resource_utilization.csv> --pareto_csv <pareto_optimal_models_roc_combined.csv> [options]
    
    # Or if both files are in the same directory:
    python plot_parameters_vs_resources.py --input_dir <directory>

Author: Eric
Date: February 2026
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from scipy import stats
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Error: matplotlib and scipy are required")
    print("Install with: pip install matplotlib scipy")
    sys.exit(1)


def load_and_merge_data(resource_csv, pareto_csv):
    """
    Load resource utilization and parameter data, then merge them.
    
    Args:
        resource_csv: Path to resource_utilization.csv
        pareto_csv: Path to pareto_optimal_models_roc_combined.csv
        
    Returns:
        DataFrame with merged data
    """
    # Load resource utilization data
    resource_df = pd.read_csv(resource_csv)
    
    # Extract trial_id from model_name (e.g., "model_trial_101" -> "101")
    resource_df['trial_id'] = resource_df['model_name'].str.extract(r'model_trial_(\d+)')[0]
    
    # Load parameter data
    param_df = pd.read_csv(pareto_csv)
    
    # Ensure trial_id is string in both dataframes for consistent merging
    resource_df['trial_id'] = resource_df['trial_id'].astype(str)
    param_df['trial_id'] = param_df['trial_id'].astype(str).str.zfill(3)  # Pad with zeros
    
    # Merge on trial_id
    merged_df = pd.merge(
        resource_df,
        param_df[['trial_id', 'parameters']],
        on='trial_id',
        how='inner'
    )
    
    # Calculate total resources (FF + LUT)
    merged_df['total_resources'] = merged_df['luts_used'] + merged_df['registers_used']
    
    return merged_df


def fit_linear_regression(x, y):
    """
    Fit OLS linear regression and return statistics.
    
    Args:
        x: Independent variable (parameters)
        y: Dependent variable (total resources)
        
    Returns:
        dict with regression statistics
    """
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate predictions
    y_pred = slope * x + intercept
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate R-squared
    r_squared = r_value ** 2
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
        'y_pred': y_pred,
        'residuals': residuals
    }


def plot_parameters_vs_resources(df, output_path, regression_stats):
    """
    Create scatter plot with linear regression line.
    
    Args:
        df: DataFrame with merged data
        output_path: Path to save the plot
        regression_stats: Dictionary with regression statistics
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    ax.scatter(df['parameters'], df['total_resources'], 
               alpha=0.7, s=100, c='steelblue', edgecolors='navy', 
               linewidth=1.5, label='Models', zorder=3)
    
    # Regression line
    x_line = np.array([df['parameters'].min(), df['parameters'].max()])
    y_line = regression_stats['slope'] * x_line + regression_stats['intercept']
    ax.plot(x_line, y_line, 'r--', linewidth=2.5, 
            label=f'Linear Fit (OLS)', zorder=2, alpha=0.8)
    
    # Annotate some points (optional - can be removed if too cluttered)
    for idx, row in df.iterrows():
        if idx % 5 == 0:  # Annotate every 5th point to avoid clutter
            ax.annotate(f"T{row['trial_id']}", 
                       xy=(row['parameters'], row['total_resources']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.6)
    
    # Labels and title
    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Hardware Resources (FF + LUT)', fontsize=14, fontweight='bold')
    ax.set_title('Model Parameters vs Hardware Resources\nwith Linear Regression (OLS)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    
    # Regression statistics text box
    stats_text = (
        f"Linear Regression (OLS)\n"
        f"─────────────────────\n"
        f"Equation: y = {regression_stats['slope']:.2f}x + {regression_stats['intercept']:.2f}\n"
        f"R² = {regression_stats['r_squared']:.4f}\n"
        f"R = {regression_stats['r_value']:.4f}\n"
        f"p-value = {regression_stats['p_value']:.2e}\n"
        f"Std Error = {regression_stats['std_err']:.2f}\n"
        f"─────────────────────\n"
        f"N = {len(df)} models"
    )
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           family='monospace')
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()


def print_regression_summary(regression_stats, df):
    """Print detailed regression summary."""
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION SUMMARY (OLS)")
    print("=" * 80)
    print(f"\nEquation: Total_Resources = {regression_stats['slope']:.4f} × Parameters + {regression_stats['intercept']:.4f}")
    print(f"\nModel Fit:")
    print(f"  R-squared (R²):     {regression_stats['r_squared']:.6f}")
    print(f"  Correlation (R):    {regression_stats['r_value']:.6f}")
    print(f"  P-value:            {regression_stats['p_value']:.2e}")
    print(f"  Standard Error:     {regression_stats['std_err']:.4f}")
    print(f"\nInterpretation:")
    print(f"  - Each additional parameter adds ~{regression_stats['slope']:.2f} hardware resources (FF+LUT)")
    print(f"  - The model explains {regression_stats['r_squared']*100:.2f}% of the variance")
    print(f"  - Sample size: {len(df)} models")
    
    # Residual statistics
    print(f"\nResidual Statistics:")
    print(f"  Mean:               {regression_stats['residuals'].mean():.2f}")
    print(f"  Std Dev:            {regression_stats['residuals'].std():.2f}")
    print(f"  Min:                {regression_stats['residuals'].min():.2f}")
    print(f"  Max:                {regression_stats['residuals'].max():.2f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Plot model parameters vs hardware resources with linear regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specify both CSV files
  python plot_parameters_vs_resources.py \\
      --resource_csv ../model_results/resource_utilization.csv \\
      --pareto_csv ../model_results/pareto_optimal_models_roc_combined.csv
  
  # Use directory (will look for default filenames)
  python plot_parameters_vs_resources.py --input_dir ../model_results
        """
    )
    
    parser.add_argument(
        '--resource_csv',
        type=str,
        help='Path to resource_utilization.csv'
    )
    
    parser.add_argument(
        '--pareto_csv',
        type=str,
        help='Path to pareto_optimal_models_roc_combined.csv'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Directory containing both CSV files (alternative to specifying each file)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output plot path (default: parameters_vs_resources.png in input directory)'
    )
    
    args = parser.parse_args()
    
    # Determine input files
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Directory does not exist: {args.input_dir}")
            sys.exit(1)
        
        resource_csv = os.path.join(args.input_dir, 'resource_utilization.csv')
        pareto_csv = os.path.join(args.input_dir, 'pareto_optimal_models_roc_combined.csv')
        output_dir = args.input_dir
    else:
        if not args.resource_csv or not args.pareto_csv:
            print("Error: Either --input_dir or both --resource_csv and --pareto_csv must be specified")
            sys.exit(1)
        
        resource_csv = args.resource_csv
        pareto_csv = args.pareto_csv
        output_dir = os.path.dirname(resource_csv)
    
    # Validate input files
    if not os.path.isfile(resource_csv):
        print(f"Error: Resource CSV not found: {resource_csv}")
        sys.exit(1)
    
    if not os.path.isfile(pareto_csv):
        print(f"Error: Pareto CSV not found: {pareto_csv}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(output_dir, 'parameters_vs_resources.png')
    
    print("=" * 80)
    print("PARAMETERS VS HARDWARE RESOURCES ANALYSIS")
    print("=" * 80)
    print(f"\nInput files:")
    print(f"  Resource CSV: {resource_csv}")
    print(f"  Pareto CSV:   {pareto_csv}")
    print(f"\nOutput plot: {output_path}")
    print("-" * 80)
    
    # Load and merge data
    print("\nLoading and merging data...")
    df = load_and_merge_data(resource_csv, pareto_csv)
    print(f"✓ Merged {len(df)} models")
    
    # Fit linear regression
    print("\nFitting linear regression (OLS)...")
    regression_stats = fit_linear_regression(df['parameters'].values, df['total_resources'].values)
    print("✓ Regression complete")
    
    # Print summary
    print_regression_summary(regression_stats, df)
    
    # Create plot
    print("\nGenerating plot...")
    plot_parameters_vs_resources(df, output_path, regression_stats)
    
    # Save detailed results to CSV
    results_csv = output_path.replace('.png', '_regression_data.csv')
    df_output = df[['trial_id', 'model_name', 'parameters', 'luts_used', 'registers_used', 'total_resources']].copy()
    df_output['predicted_resources'] = regression_stats['y_pred']
    df_output['residual'] = regression_stats['residuals']
    df_output.to_csv(results_csv, index=False)
    print(f"✓ Detailed data saved to: {results_csv}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - Plot:         {output_path}")
    print(f"  - Data CSV:     {results_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
