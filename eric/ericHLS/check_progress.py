#!/usr/bin/env python3
"""
Check Progress of Parallel HLS Synthesis

This script monitors the progress of ongoing parallel HLS synthesis runs.

Usage:
    python check_progress.py --results_dir <hls_outputs_dir>

Author: Eric
Date: 2025
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime


def check_progress(results_dir, watch=False, interval=30):
    """
    Check the progress of synthesis runs.
    
    Args:
        results_dir: Directory containing HLS outputs
        watch: If True, continuously monitor progress
        interval: Update interval in seconds when watching
    """
    
    def print_status():
        # Check if results file exists
        results_json = os.path.join(results_dir, 'synthesis_results.json')
        
        if not os.path.exists(results_json):
            print(f"Results file not found yet: {results_json}")
            print("Synthesis may still be starting up...")
            return False
        
        # Load results
        with open(results_json, 'r') as f:
            results = json.load(f)
        
        # Count status
        total = results['total']
        successful = results['successful']
        failed = results['failed']
        pending = total - successful - failed
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"SYNTHESIS PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"\nTotal Models: {total}")
        print(f"Completed: {successful + failed} / {total} ({(successful + failed) / total * 100:.1f}%)")
        print(f"  ✓ Successful: {successful}")
        print(f"  ✗ Failed: {failed}")
        print(f"  ⏳ Pending: {pending}")
        
        # Show progress bar
        completed = successful + failed
        bar_length = 50
        filled = int(bar_length * completed / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\n[{bar}] {completed}/{total}")
        
        # List recently completed
        if results['results']:
            print("\n" + "-" * 80)
            print("RECENT COMPLETIONS (last 5):")
            print("-" * 80)
            
            completed_results = [r for r in results['results'] if r['status'] != 'pending']
            recent = sorted(completed_results, key=lambda x: x.get('end_time', ''), reverse=True)[:5]
            
            for r in recent:
                model_name = Path(r['h5_file']).stem
                status_symbol = "✓" if r['status'] == 'success' else "✗"
                end_time = r.get('end_time', 'N/A')
                if end_time != 'N/A':
                    try:
                        dt = datetime.fromisoformat(end_time)
                        end_time = dt.strftime('%H:%M:%S')
                    except:
                        pass
                print(f"  {status_symbol} {model_name:40s} [{end_time}]")
        
        # List failed models
        failed_results = [r for r in results['results'] if r['status'] == 'failed']
        if failed_results:
            print("\n" + "-" * 80)
            print(f"FAILED MODELS ({len(failed_results)}):")
            print("-" * 80)
            for r in failed_results[:10]:  # Show first 10
                model_name = Path(r['h5_file']).stem
                error = r.get('error', 'Unknown error')
                # Truncate error message
                error = error[:60] + '...' if len(error) > 60 else error
                print(f"  ✗ {model_name:40s} - {error}")
            
            if len(failed_results) > 10:
                print(f"  ... and {len(failed_results) - 10} more")
        
        # Check if complete
        all_complete = (completed == total)
        
        if all_complete:
            print("\n" + "=" * 80)
            print("✓ ALL SYNTHESIS RUNS COMPLETE!")
            print("=" * 80)
            print(f"\nFinal Results:")
            print(f"  Successful: {successful}/{total} ({successful/total*100:.1f}%)")
            print(f"  Failed: {failed}/{total} ({failed/total*100:.1f}%)")
            print(f"\nResults saved to: {results_json}")
        
        return all_complete
    
    # Main monitoring loop
    try:
        if watch:
            print(f"Monitoring progress in: {results_dir}")
            print(f"Update interval: {interval} seconds")
            print("Press Ctrl+C to stop monitoring")
            print("=" * 80)
            
            while True:
                try:
                    complete = print_status()
                    if complete:
                        break
                    time.sleep(interval)
                except KeyboardInterrupt:
                    print("\n\nMonitoring stopped by user")
                    break
        else:
            print_status()
    
    except Exception as e:
        print(f"Error checking progress: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Check progress of parallel HLS synthesis'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing HLS synthesis outputs'
    )
    
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Continuously monitor progress (updates every 30 seconds)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Update interval in seconds when watching (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.results_dir):
        print(f"Error: Directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    # Check progress
    check_progress(args.results_dir, args.watch, args.interval)


if __name__ == "__main__":
    main()

