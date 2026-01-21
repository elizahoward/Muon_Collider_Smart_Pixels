#!/usr/bin/env python3
"""
Comprehensive HLS Synthesis Results Analysis

This script combines the functionality of analyze_synthesis_results.py and 
extract_hls_resources.py into a single unified tool that:
- Extracts resource utilization from Vivado reports (LUTs, FFs, BRAM, DSP, Fmax)
- Links to validation accuracy data
- Generates comprehensive statistics
- Creates visualizations with FPGA constraints
- Ranks models by multiple criteria

Usage:
    python analyze_hls_results.py --results_dir <hls_outputs_dir> [options]

Author: Eric
Date: January 2026
"""

import os
import sys
import argparse
import json
import re
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting will be disabled.")
    print("Install with: pip install matplotlib")


def parse_vivado_report(report_file):
    """
    Parse Vivado synthesis report to extract resource utilization.
    
    Returns:
        dict: Resource utilization metrics
    """
    if not os.path.exists(report_file):
        return None
    
    resources = {
        'luts': None,
        'luts_used': None,
        'luts_available': None,
        'luts_percent': None,
        'registers': None,
        'registers_used': None,
        'registers_available': None,
        'registers_percent': None,
        'bram': None,
        'bram_used': None,
        'bram_available': None,
        'bram_percent': None,
        'dsp': None,
        'dsp_used': None,
        'dsp_available': None,
        'dsp_percent': None,
    }
    
    try:
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Parse LUTs - Fixed pattern to match actual Vivado report format
        lut_match = re.search(r'\|\s*Slice LUTs\*?\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|', content)
        if lut_match:
            resources['luts_used'] = int(lut_match.group(1))
            resources['luts_available'] = int(lut_match.group(2))
            resources['luts_percent'] = float(lut_match.group(3))
            resources['luts'] = resources['luts_used']
        
        # Parse Registers (Flip-Flops) - Fixed pattern
        reg_match = re.search(r'\|\s*Slice Registers\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|', content)
        if reg_match:
            resources['registers_used'] = int(reg_match.group(1))
            resources['registers_available'] = int(reg_match.group(2))
            resources['registers_percent'] = float(reg_match.group(3))
            resources['registers'] = resources['registers_used']
        
        # Parse BRAM - Fixed pattern
        bram_match = re.search(r'\|\s*Block RAM Tile\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|', content)
        if bram_match:
            resources['bram_used'] = int(bram_match.group(1))
            resources['bram_available'] = int(bram_match.group(2))
            resources['bram_percent'] = float(bram_match.group(3))
            resources['bram'] = resources['bram_used']
        
        # Parse DSP - Fixed pattern
        dsp_match = re.search(r'\|\s*DSPs\s+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|', content)
        if dsp_match:
            resources['dsp_used'] = int(dsp_match.group(1))
            resources['dsp_available'] = int(dsp_match.group(2))
            resources['dsp_percent'] = float(dsp_match.group(3))
            resources['dsp'] = resources['dsp_used']
        
    except Exception as e:
        print(f"Error parsing {report_file}: {str(e)}")
        return None
    
    return resources


def parse_hls_log(log_file):
    """
    Parse HLS log to extract synthesis information.
    
    Returns:
        dict: HLS synthesis metrics (latency, II, etc.)
    """
    if not os.path.exists(log_file):
        return None
    
    metrics = {
        'latency_min': None,
        'latency_max': None,
        'interval_min': None,
        'interval_max': None,
        'estimated_fmax': None,
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Parse estimated Fmax
        fmax_match = re.search(r'Estimated Fmax:\s+([\d.]+)\s+MHz', content)
        if fmax_match:
            metrics['estimated_fmax'] = float(fmax_match.group(1))
        
    except Exception as e:
        print(f"Error parsing {log_file}: {str(e)}")
        return None
    
    return metrics


def analyze_results(results_dir):
    """
    Analyze all synthesis results in the directory.
    
    Args:
        results_dir: Directory containing HLS outputs
    
    Returns:
        DataFrame with analysis results
    """
    # Load synthesis results JSON
    results_json = os.path.join(results_dir, 'synthesis_results.json')
    
    if not os.path.exists(results_json):
        print(f"Error: Results file not found: {results_json}")
        return None
    
    with open(results_json, 'r') as f:
        results_data = json.load(f)
    
    # Load validation accuracy data if available
    val_accuracy_map = {}
    
    # Try multiple possible locations for trials summary
    possible_summary_locations = [
        os.path.join(os.path.dirname(results_dir), 'trials_summary.json'),
        os.path.join(os.path.dirname(results_dir), 'filtered_trials_summary.json'),
    ]
    
    for trials_summary in possible_summary_locations:
        if os.path.exists(trials_summary):
            try:
                with open(trials_summary, 'r') as f:
                    trials_data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(trials_data, list):
                    trials_list = trials_data
                elif isinstance(trials_data, dict) and 'trials' in trials_data:
                    trials_list = trials_data['trials']
                else:
                    trials_list = []
                
                # Create mapping from model name to validation accuracy
                for trial in trials_list:
                    trial_id = trial.get('trial_id', '')
                    val_acc = trial.get('val_accuracy') or trial.get('validation_accuracy') or trial.get('score')
                    if trial_id and val_acc:
                        # Handle both string and int trial IDs
                        model_name = f"model_trial_{trial_id}"
                        val_accuracy_map[model_name] = val_acc
                
                if val_accuracy_map:
                    print(f"Loaded validation accuracy data from: {trials_summary}")
                    break
            except Exception as e:
                print(f"Warning: Could not load {trials_summary}: {e}")
    
    # Analyze each successful synthesis
    analysis = []
    
    for result in results_data['results']:
        if result['status'] != 'success':
            continue
        
        model_name = Path(result['h5_file']).stem
        
        # Construct output directory path
        output_dir_name = f"hls_{model_name}"
        output_dir = os.path.join(results_dir, output_dir_name)
        
        # Look for report files
        vivado_report = os.path.join(output_dir, 'vivado_synth.rpt')
        vitis_log = os.path.join(output_dir, 'vitis_hls.log')
        
        # Check if files exist
        if not os.path.exists(vivado_report):
            vivado_report = None
        if not os.path.exists(vitis_log):
            vitis_log = None
        
        # Parse reports
        resources = parse_vivado_report(vivado_report) if vivado_report else {}
        hls_metrics = parse_hls_log(vitis_log) if vitis_log else {}
        
        # Combine results
        row = {
            'model_name': model_name,
            'val_accuracy': val_accuracy_map.get(model_name, None),
            'h5_file': result['h5_file'],
            'output_dir': result['output_dir'],
            'tarball': result.get('tarball', None),
        }
        
        if resources:
            row.update(resources)
        
        if hls_metrics:
            row.update(hls_metrics)
        
        analysis.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(analysis)
    
    return df, results_data


def print_summary(df, results_data):
    """Print a summary of the analysis results."""
    print("\n" + "=" * 80)
    print("HLS SYNTHESIS ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal models: {results_data['total']}")
    print(f"Successful: {results_data['successful']}")
    print(f"Failed: {results_data['failed']}")
    
    if len(df) > 0:
        print("\n" + "-" * 80)
        print("MODEL ACCURACY STATISTICS")
        print("-" * 80)
        
        if 'val_accuracy' in df.columns:
            valid_acc = df[df['val_accuracy'].notna()]
            if len(valid_acc) > 0:
                print(f"\nValidation Accuracy:")
                print(f"  Min: {valid_acc['val_accuracy'].min():.4f}")
                print(f"  Max: {valid_acc['val_accuracy'].max():.4f}")
                print(f"  Mean: {valid_acc['val_accuracy'].mean():.4f}")
                print(f"  Median: {valid_acc['val_accuracy'].median():.4f}")
        
        print("\n" + "-" * 80)
        print("RESOURCE UTILIZATION STATISTICS")
        print("-" * 80)
        
        if 'luts_used' in df.columns:
            print(f"\nLUTs:")
            print(f"  Min: {df['luts_used'].min():.0f}")
            print(f"  Max: {df['luts_used'].max():.0f}")
            print(f"  Mean: {df['luts_used'].mean():.0f}")
            print(f"  Median: {df['luts_used'].median():.0f}")
            print(f"  Utilization: {df['luts_percent'].min():.2f}% - {df['luts_percent'].max():.2f}%")
        
        if 'registers_used' in df.columns:
            print(f"\nRegisters (Flip-Flops):")
            print(f"  Min: {df['registers_used'].min():.0f}")
            print(f"  Max: {df['registers_used'].max():.0f}")
            print(f"  Mean: {df['registers_used'].mean():.0f}")
            print(f"  Median: {df['registers_used'].median():.0f}")
            print(f"  Utilization: {df['registers_percent'].min():.2f}% - {df['registers_percent'].max():.2f}%")
        
        if 'bram_used' in df.columns and df['bram_used'].max() > 0:
            print(f"\nBRAM:")
            print(f"  Min: {df['bram_used'].min():.0f}")
            print(f"  Max: {df['bram_used'].max():.0f}")
            print(f"  Mean: {df['bram_used'].mean():.0f}")
        
        if 'dsp_used' in df.columns and df['dsp_used'].max() > 0:
            print(f"\nDSP:")
            print(f"  Min: {df['dsp_used'].min():.0f}")
            print(f"  Max: {df['dsp_used'].max():.0f}")
            print(f"  Mean: {df['dsp_used'].mean():.0f}")
        
        if 'estimated_fmax' in df.columns:
            valid_fmax = df[df['estimated_fmax'].notna()]
            if len(valid_fmax) > 0:
                print(f"\nEstimated Fmax (MHz):")
                print(f"  Min: {valid_fmax['estimated_fmax'].min():.2f}")
                print(f"  Max: {valid_fmax['estimated_fmax'].max():.2f}")
                print(f"  Mean: {valid_fmax['estimated_fmax'].mean():.2f}")
        
        # Find best models
        print("\n" + "-" * 80)
        print("TOP MODELS")
        print("-" * 80)
        
        if 'val_accuracy' in df.columns:
            valid_acc = df[df['val_accuracy'].notna()]
            if len(valid_acc) > 0:
                print("\nBest Accuracy:")
                if 'luts_used' in df.columns:
                    best_acc = valid_acc.nlargest(5, 'val_accuracy')[['model_name', 'val_accuracy', 'luts_used', 'registers_used']]
                else:
                    best_acc = valid_acc.nlargest(5, 'val_accuracy')[['model_name', 'val_accuracy']]
                print(best_acc.to_string(index=False))
        
        if 'luts_used' in df.columns:
            print("\nSmallest (by LUT count):")
            if 'val_accuracy' in df.columns:
                smallest = df.nsmallest(5, 'luts_used')[['model_name', 'val_accuracy', 'luts_used', 'registers_used']]
            else:
                smallest = df.nsmallest(5, 'luts_used')[['model_name', 'luts_used', 'registers_used']]
            print(smallest.to_string(index=False))
        
        if 'estimated_fmax' in df.columns:
            valid_fmax = df[df['estimated_fmax'].notna()]
            if len(valid_fmax) > 0:
                print("\nFastest (by Estimated Fmax):")
                fastest = valid_fmax.nlargest(5, 'estimated_fmax')[['model_name', 'estimated_fmax', 'luts_used']]
                print(fastest.to_string(index=False))


def plot_resource_utilization(df, output_path, fpga_constraints):
    """
    Create a scatter plot of LUT vs FF utilization with FPGA constraint regions.
    
    Args:
        df: DataFrame with resource utilization data
        output_path: Path where the plot image should be saved
        fpga_constraints: List of tuples (name, max_luts, max_ffs, color) for each FPGA
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\nWarning: matplotlib not available. Skipping plot generation.")
        print("Install matplotlib with: pip install matplotlib")
        return
    
    if df.empty or 'luts_used' not in df.columns or 'registers_used' not in df.columns:
        print("\nWarning: Insufficient data for plotting")
        return
    
    # Extract data for plotting
    luts = df['luts_used'].tolist()
    ffs = df['registers_used'].tolist()
    models = df['model_name'].tolist()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw FPGA constraint boxes
    for name, max_luts, max_ffs, color in fpga_constraints:
        rect = patches.Rectangle(
            (0, 0), max_luts, max_ffs,
            linewidth=2, edgecolor='black', facecolor=color, alpha=0.2,
            label=f'{name}\n({max_luts:,} LUTs × {max_ffs:,} FFs)'
        )
        ax.add_patch(rect)
    
    # Color points by validation accuracy if available
    if 'val_accuracy' in df.columns and df['val_accuracy'].notna().any():
        valid_data = df[df['val_accuracy'].notna()]
        invalid_data = df[df['val_accuracy'].isna()]
        
        # Plot models with accuracy data (colored)
        if not valid_data.empty:
            scatter = ax.scatter(
                valid_data['luts_used'], 
                valid_data['registers_used'],
                c=valid_data['val_accuracy'],
                s=150, 
                cmap='RdYlGn',
                vmin=valid_data['val_accuracy'].min() * 0.95,
                vmax=valid_data['val_accuracy'].max() * 1.01,
                zorder=5, 
                alpha=0.8,
                edgecolors='darkred', 
                linewidth=2
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Validation Accuracy', fontsize=12, fontweight='bold')
        
        # Plot models without accuracy data (gray)
        if not invalid_data.empty:
            ax.scatter(
                invalid_data['luts_used'],
                invalid_data['registers_used'],
                c='gray',
                s=150,
                zorder=5,
                alpha=0.5,
                edgecolors='black',
                linewidth=2,
                label='No accuracy data'
            )
    else:
        # No accuracy data - plot all in red
        ax.scatter(luts, ffs, c='red', s=150, zorder=5, alpha=0.7, 
                  edgecolors='darkred', linewidth=2)
    
    # Add labels for each point
    for i, (lut, ff, model) in enumerate(zip(luts, ffs, models)):
        # Extract trial number for cleaner labels
        trial_match = re.search(r'trial_(\d+)', model)
        label = trial_match.group(1) if trial_match else model
        
        ax.annotate(
            f'T{label}' if trial_match else label, 
            (lut, ff),
            xytext=(10, 10), 
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5)
        )
    
    # Set axis labels and title
    ax.set_xlabel('LUTs (Look-Up Tables)', fontsize=14, fontweight='bold')
    ax.set_ylabel('FFs (Flip Flops / Registers)', fontsize=14, fontweight='bold')
    ax.set_title('HLS Model Resource Utilization vs FPGA Constraints', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits with some padding
    max_lut = max(max(luts), max(c[1] for c in fpga_constraints))
    max_ff = max(max(ffs), max(c[2] for c in fpga_constraints))
    ax.set_xlim(-max_lut*0.05, max_lut*1.1)
    ax.set_ylim(-max_ff*0.05, max_ff*1.1)
    
    # Format axis ticks with thousands separator
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add statistics text box
    stats_text = f"Models analyzed: {len(df)}\n"
    stats_text += f"LUT range: {min(luts):,} - {max(luts):,}\n"
    stats_text += f"FF range: {min(ffs):,} - {max(ffs):,}"
    
    if 'val_accuracy' in df.columns:
        valid_acc = df[df['val_accuracy'].notna()]
        if not valid_acc.empty:
            stats_text += f"\nAcc range: {valid_acc['val_accuracy'].min():.4f} - {valid_acc['val_accuracy'].max():.4f}"
    
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive HLS synthesis results analysis with visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full analysis with CSV and plot
    python analyze_hls_results.py --results_dir ../model2_5_pareto_hls_ready/hls_outputs --plot
    
    # Analysis with custom output paths
    python analyze_hls_results.py \\
        --results_dir ./hls_outputs \\
        --output_csv analysis.csv \\
        --plot \\
        --plot_output utilization.png
    
    # Custom FPGA constraints
    python analyze_hls_results.py \\
        --results_dir ./hls_outputs \\
        --plot \\
        --pink-luts 10000 --pink-ffs 20000 \\
        --xilinx-luts 53200 --xilinx-ffs 106400
        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing HLS synthesis outputs (hls_outputs/)'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Output CSV file path (default: <parent_dir>/resource_utilization.csv)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plot of resource utilization'
    )
    
    parser.add_argument(
        '--plot_output',
        type=str,
        default=None,
        help='Output plot file path (default: <parent_dir>/resource_utilization.png)'
    )
    
    parser.add_argument(
        '--pink-luts',
        type=int,
        default=10000,
        help='Maximum LUTs for Pink Board (default: 10000)'
    )
    
    parser.add_argument(
        '--pink-ffs',
        type=int,
        default=20000,
        help='Maximum FFs for Pink Board (default: 20000)'
    )
    
    parser.add_argument(
        '--xilinx-luts',
        type=int,
        default=53200,
        help='Maximum LUTs for Xilinx FPGA (default: 53200)'
    )
    
    parser.add_argument(
        '--xilinx-ffs',
        type=int,
        default=106400,
        help='Maximum FFs for Xilinx FPGA (default: 106400)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.results_dir):
        print(f"Error: Directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    # Determine output paths
    parent_dir = os.path.dirname(args.results_dir)
    
    if args.output_csv:
        csv_output_path = os.path.abspath(args.output_csv)
    else:
        csv_output_path = os.path.join(parent_dir, 'resource_utilization.csv')
    
    if args.plot_output:
        plot_output_path = os.path.abspath(args.plot_output)
    else:
        plot_output_path = os.path.join(parent_dir, 'resource_utilization.png')
    
    print(f"Analyzing synthesis results in: {args.results_dir}")
    
    # Analyze results
    result = analyze_results(args.results_dir)
    if result is None:
        sys.exit(1)
    
    df, results_data = result
    
    if df is None or len(df) == 0:
        print("No successful synthesis results to analyze")
        sys.exit(1)
    
    # Save to CSV
    df.to_csv(csv_output_path, index=False)
    print(f"\n✓ CSV saved to: {csv_output_path}")
    
    # Print summary
    print_summary(df, results_data)
    
    # Generate plot if requested
    if args.plot:
        fpga_constraints = [
            ("Pink Board", args.pink_luts, args.pink_ffs, "pink"),
            ("Xilinx Zynq xc7z020", args.xilinx_luts, args.xilinx_ffs, "lightblue")
        ]
        plot_resource_utilization(df, plot_output_path, fpga_constraints)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - CSV: {csv_output_path}")
    if args.plot:
        print(f"  - Plot: {plot_output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
