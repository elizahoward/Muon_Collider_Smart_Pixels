#!/usr/bin/env python3
"""
Analyze HLS Synthesis Results

This script analyzes the synthesis results from parallel_hls_synthesis.py,
extracting resource utilization information and creating summary reports.

Usage:
    python analyze_synthesis_results.py --results_dir <hls_outputs_dir>

Author: Eric
Date: 2025
"""

import os
import sys
import argparse
import json
import re
import pandas as pd
from pathlib import Path


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
        
        # Parse latency and interval (if present in summary)
        # This is more complex and might need adjustment based on actual log format
        
    except Exception as e:
        print(f"Error parsing {log_file}: {str(e)}")
        return None
    
    return metrics


def analyze_results(results_dir, output_csv=None):
    """
    Analyze all synthesis results in the directory.
    
    Args:
        results_dir: Directory containing HLS outputs
        output_csv: Optional path to save CSV summary
    
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
    trials_summary = os.path.join(os.path.dirname(results_dir), 'filtered_trials_summary.json')
    if os.path.exists(trials_summary):
        with open(trials_summary, 'r') as f:
            trials_data = json.load(f)
        # Create mapping from model name to validation accuracy
        for trial in trials_data.get('trials', []):
            model_file = trial.get('model_file', '')
            if model_file:
                model_name = Path(model_file).stem
                val_accuracy_map[model_name] = trial.get('val_accuracy', None)
    
    # Analyze each successful synthesis
    analysis = []
    
    for result in results_data['results']:
        if result['status'] != 'success':
            continue
        
        model_name = Path(result['h5_file']).stem
        
        # Construct output directory path - assume it's hls_{model_name} in results_dir
        # This is more reliable than trying to resolve the relative paths from JSON
        output_dir_name = f"hls_{model_name}"
        output_dir = os.path.join(results_dir, output_dir_name)
        
        # Look for report files - they're typically in the root of output_dir
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
            'output_dir': result['output_dir'],  # Keep original path in CSV
            'tarball': result.get('tarball', None),
        }
        
        if resources:
            row.update(resources)
        
        if hls_metrics:
            row.update(hls_metrics)
        
        analysis.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(analysis)
    
    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved analysis to: {output_csv}")
    
    return df


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
            print(f"\nRegisters:")
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


def main():
    parser = argparse.ArgumentParser(
        description='Analyze HLS synthesis results and extract resource utilization'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing HLS synthesis outputs'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Output CSV file for detailed results (default: results_dir/analysis.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.results_dir):
        print(f"Error: Directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    # Set default output CSV
    if args.output_csv is None:
        args.output_csv = os.path.join(args.results_dir, 'resource_analysis.csv')
    
    # Load results data
    results_json = os.path.join(args.results_dir, 'synthesis_results.json')
    with open(results_json, 'r') as f:
        results_data = json.load(f)
    
    # Analyze results
    print(f"Analyzing synthesis results in: {args.results_dir}")
    df = analyze_results(args.results_dir, args.output_csv)
    
    if df is None or len(df) == 0:
        print("No successful synthesis results to analyze")
        sys.exit(1)
    
    # Print summary
    print_summary(df, results_data)
    
    print("\n" + "=" * 80)
    print(f"Detailed results saved to: {args.output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()

