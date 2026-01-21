#!/usr/bin/env python3
"""
Extract resource utilization (LUTs and FFs) from HLS synthesis reports.

This script reads Vivado synthesis reports from HLS output folders and extracts
the LUT and FF (Flip Flop) utilization data, saving it to a CSV file and 
optionally creating a visualization plot.

Usage:
    python extract_hls_resources.py <hls_outputs_folder>
    
Example:
    python extract_hls_resources.py /path/to/model_results/hls_outputs --plot
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def parse_vivado_synth_report(report_path: str) -> Optional[Dict[str, int]]:
    """
    Parse a Vivado synthesis report to extract LUTs and FFs.
    
    Args:
        report_path: Path to the vivado_synth.rpt file
        
    Returns:
        Dictionary with 'LUTs' and 'FFs' keys, or None if parsing fails
    """
    if not os.path.exists(report_path):
        print(f"Warning: Report file not found: {report_path}")
        return None
    
    try:
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Look for the Slice Logic table
        # Pattern to match "Slice LUTs*" line with the Used column
        lut_pattern = r'\|\s*Slice LUTs\*?\s*\|\s*(\d+)\s*\|'
        # Pattern to match "Slice Registers" or "Register as Flip Flop" line
        ff_pattern = r'\|\s*Slice Registers\s*\|\s*(\d+)\s*\|'
        
        lut_match = re.search(lut_pattern, content)
        ff_match = re.search(ff_pattern, content)
        
        if lut_match and ff_match:
            luts = int(lut_match.group(1))
            ffs = int(ff_match.group(1))
            return {'LUTs': luts, 'FFs': ffs}
        else:
            print(f"Warning: Could not parse resource data from {report_path}")
            return None
            
    except Exception as e:
        print(f"Error parsing {report_path}: {e}")
        return None


def extract_trial_number(folder_name: str) -> Optional[int]:
    """
    Extract the trial number from a folder name like 'hls_model_trial_0'.
    
    Args:
        folder_name: Name of the folder
        
    Returns:
        Trial number as integer, or None if not found
    """
    match = re.search(r'trial_(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def process_hls_outputs(hls_outputs_dir: str) -> List[Dict]:
    """
    Process all HLS model folders in the given directory.
    
    Args:
        hls_outputs_dir: Path to the hls_outputs directory
        
    Returns:
        List of dictionaries containing model data
    """
    hls_outputs_path = Path(hls_outputs_dir)
    
    if not hls_outputs_path.exists():
        raise FileNotFoundError(f"Directory not found: {hls_outputs_dir}")
    
    results = []
    
    # Find all hls_model_trial_* directories
    model_dirs = sorted([d for d in hls_outputs_path.iterdir() 
                        if d.is_dir() and d.name.startswith('hls_model_trial_')])
    
    if not model_dirs:
        print(f"Warning: No hls_model_trial_* directories found in {hls_outputs_dir}")
        return results
    
    print(f"Found {len(model_dirs)} model directories")
    
    for model_dir in model_dirs:
        trial_num = extract_trial_number(model_dir.name)
        if trial_num is None:
            print(f"Warning: Could not extract trial number from {model_dir.name}")
            continue
        
        report_path = model_dir / 'vivado_synth.rpt'
        resource_data = parse_vivado_synth_report(str(report_path))
        
        if resource_data:
            results.append({
                'Model': f'trial_{trial_num}',
                'Trial_Number': trial_num,
                'LUTs': resource_data['LUTs'],
                'FFs': resource_data['FFs']
            })
            print(f"  Processed {model_dir.name}: LUTs={resource_data['LUTs']}, FFs={resource_data['FFs']}")
        else:
            print(f"  Skipped {model_dir.name}: Could not extract resource data")
    
    # Sort by trial number
    results.sort(key=lambda x: x['Trial_Number'])
    
    return results


def save_to_csv(data: List[Dict], output_path: str):
    """
    Save the extracted data to a CSV file.
    
    Args:
        data: List of dictionaries containing model data
        output_path: Path where the CSV file should be saved
    """
    if not data:
        print("No data to save")
        return
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Trial_Number', 'LUTs', 'FFs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\nCSV file saved to: {output_path}")
    print(f"Total models processed: {len(data)}")


def load_csv_data(csv_path: str) -> List[Dict]:
    """
    Load resource utilization data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of dictionaries containing model data
    """
    data = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'Model': row['Model'],
                'Trial_Number': int(row['Trial_Number']),
                'LUTs': int(row['LUTs']),
                'FFs': int(row['FFs'])
            })
    return data


def plot_resource_utilization(
    data: List[Dict],
    output_path: str,
    fpga_constraints: Optional[List[Tuple[str, int, int, str]]] = None
):
    """
    Create a scatter plot of LUT vs FF utilization with FPGA constraint regions.
    
    Args:
        data: List of dictionaries containing model data
        output_path: Path where the plot image should be saved
        fpga_constraints: List of tuples (name, max_luts, max_ffs, color) for each FPGA
                         If None, uses default constraints
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping plot generation.")
        print("Install matplotlib with: pip install matplotlib")
        return
    
    if not data:
        print("No data to plot")
        return
    
    # Default FPGA constraints if none provided
    if fpga_constraints is None:
        fpga_constraints = [
            ("Pink Board", 10000, 20000, "pink"),
            ("Xilinx Zynq xc7z020", 53200, 106400, "lightblue")
        ]
    
    # Extract data for plotting
    luts = [d['LUTs'] for d in data]
    ffs = [d['FFs'] for d in data]
    models = [d['Model'] for d in data]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw FPGA constraint boxes
    for name, max_luts, max_ffs, color in fpga_constraints:
        rect = patches.Rectangle(
            (0, 0), max_luts, max_ffs,
            linewidth=2, edgecolor='black', facecolor=color, alpha=0.2,
            label=f'{name}\n({max_luts:,} LUTs × {max_ffs:,} FFs)'
        )
        ax.add_patch(rect)
    
    # Plot the models
    scatter = ax.scatter(luts, ffs, c='red', s=100, zorder=5, alpha=0.7, edgecolors='darkred', linewidth=2)
    
    # Add labels for each point
    for i, (lut, ff, model) in enumerate(zip(luts, ffs, models)):
        ax.annotate(
            model, 
            (lut, ff),
            xytext=(10, 10), 
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1)
        )
    
    # Set axis labels and title
    ax.set_xlabel('LUTs (Look-Up Tables)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FFs (Flip Flops)', fontsize=12, fontweight='bold')
    ax.set_title('HLS Model Resource Utilization vs FPGA Constraints', fontsize=14, fontweight='bold')
    
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
    stats_text = f"Models analyzed: {len(data)}\n"
    stats_text += f"LUT range: {min(luts):,} - {max(luts):,}\n"
    stats_text += f"FF range: {min(ffs):,} - {max(ffs):,}"
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Extract LUT and FF resource utilization from HLS synthesis reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract data to CSV only
    python extract_hls_resources.py /path/to/results/hls_outputs
    
    # Extract data and generate plot
    python extract_hls_resources.py /path/to/results/hls_outputs --plot
    
    # Use custom FPGA constraints
    python extract_hls_resources.py /path/to/results/hls_outputs --plot \\
        --pink-luts 10000 --pink-ffs 20000 \\
        --xilinx-luts 53200 --xilinx-ffs 106400
        """
    )
    
    parser.add_argument(
        'hls_outputs_dir',
        type=str,
        help='Path to the hls_outputs directory containing hls_model_trial_* folders'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: <parent_dir>/resource_utilization.csv)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate a visualization plot of the resource utilization'
    )
    
    parser.add_argument(
        '--plot-output',
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
    
    # Resolve the hls_outputs directory path
    hls_outputs_dir = os.path.abspath(args.hls_outputs_dir)
    
    # Determine output paths
    parent_dir = os.path.dirname(hls_outputs_dir)
    
    if args.output:
        csv_output_path = os.path.abspath(args.output)
    else:
        csv_output_path = os.path.join(parent_dir, 'resource_utilization.csv')
    
    if args.plot_output:
        plot_output_path = os.path.abspath(args.plot_output)
    else:
        plot_output_path = os.path.join(parent_dir, 'resource_utilization.png')
    
    print(f"Processing HLS outputs from: {hls_outputs_dir}")
    print(f"Output CSV will be saved to: {csv_output_path}")
    if args.plot:
        print(f"Output plot will be saved to: {plot_output_path}")
    print("-" * 70)
    
    # Process the HLS outputs
    data = process_hls_outputs(hls_outputs_dir)
    
    # Save to CSV
    if data:
        save_to_csv(data, csv_output_path)
        
        # Generate plot if requested
        if args.plot:
            fpga_constraints = [
                ("Pink Board", args.pink_luts, args.pink_ffs, "pink"),
                ("Xilinx Zynq xc7z020", args.xilinx_luts, args.xilinx_ffs, "lightblue")
            ]
            plot_resource_utilization(data, plot_output_path, fpga_constraints)
        
        print("-" * 70)
        print("Done!")
    else:
        print("-" * 70)
        print("No data was extracted. Please check the input directory and report files.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
