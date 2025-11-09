#!/usr/bin/env python3
"""
Analyze parameter breakdown by layer for Model3 to identify which layer contributes most.
"""

import pandas as pd

def calculate_model3_layer_breakdown(hyperparams):
    """Calculate parameters for each layer in Model3."""
    conv_filters = hyperparams.get('conv_filters', 32)
    kernel_rows = hyperparams.get('kernel_rows', 3)
    kernel_cols = hyperparams.get('kernel_cols', 3)
    scalar_dense_units = hyperparams.get('scalar_dense_units', 32)
    merged_dense_1 = hyperparams.get('merged_dense_1', 200)
    
    # Calculate merged_dense_2 from multiplier if available
    merged_multiplier_2 = hyperparams.get('merged_multiplier_2', None)
    if merged_multiplier_2 is not None:
        merged_dense_2 = int(round(merged_dense_1 * merged_multiplier_2))
    else:
        merged_dense_2 = hyperparams.get('merged_dense_2', 100)
    
    # Calculate Conv2D output shape
    flattened_conv_units = 6 * 10 * conv_filters
    
    # Calculate parameters for each layer
    breakdown = {}
    
    # 1. Conv2D layer
    breakdown['Conv2D'] = {
        'weights': kernel_rows * kernel_cols * 1 * conv_filters,
        'biases': conv_filters,
        'total': (kernel_rows * kernel_cols * 1 * conv_filters) + conv_filters
    }
    
    # 2. Scalar Dense layer
    breakdown['Scalar_Dense'] = {
        'weights': 2 * scalar_dense_units,
        'biases': scalar_dense_units,
        'total': (2 * scalar_dense_units) + scalar_dense_units
    }
    
    # 3. Merged Dense 1 (the big one!)
    input_size = flattened_conv_units + scalar_dense_units
    breakdown['Merged_Dense_1'] = {
        'weights': input_size * merged_dense_1,
        'biases': merged_dense_1,
        'total': (input_size * merged_dense_1) + merged_dense_1,
        'input_size': input_size
    }
    
    # 4. Merged Dense 2
    breakdown['Merged_Dense_2'] = {
        'weights': merged_dense_1 * merged_dense_2,
        'biases': merged_dense_2,
        'total': (merged_dense_1 * merged_dense_2) + merged_dense_2
    }
    
    # 5. Output layer
    breakdown['Output'] = {
        'weights': merged_dense_2 * 1,
        'biases': 1,
        'total': (merged_dense_2 * 1) + 1
    }
    
    total = sum(layer['total'] for layer in breakdown.values())
    breakdown['TOTAL'] = {'total': total}
    
    return breakdown


def main():
    # Read the detailed results
    csv_path = "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/complexity_analysis/model3_quantized_4w0i_hyperparameter_search/hyperparameter_detailed_results.csv"
    df = pd.read_csv(csv_path)
    
    print("="*80)
    print("PARAMETER BREAKDOWN BY LAYER - Model3")
    print("="*80)
    
    # Analyze a few representative trials
    trials_to_analyze = [0, 1, 5, 15]  # Small, medium, large parameter counts
    
    for idx in trials_to_analyze:
        if idx >= len(df):
            continue
            
        row = df.iloc[idx]
        hyperparams = {
            'conv_filters': int(row['conv_filters']),
            'kernel_rows': int(row['kernel_rows']),
            'kernel_cols': int(row['kernel_cols']),
            'scalar_dense_units': int(row['scalar_dense_units']),
            'merged_dense_1': int(row['merged_dense_1']),
            'merged_multiplier_2': row['merged_multiplier_2']
        }
        
        breakdown = calculate_model3_layer_breakdown(hyperparams)
        total_params = breakdown['TOTAL']['total']
        
        print(f"\n{'='*80}")
        print(f"Trial {row['trial_id']:02d} - Total Parameters: {total_params:,}")
        print(f"  Accuracy: {row['val_accuracy']:.4f}")
        print(f"  Hyperparams: conv_filters={hyperparams['conv_filters']}, "
              f"scalar_dense={hyperparams['scalar_dense_units']}, "
              f"merged_dense_1={hyperparams['merged_dense_1']}, "
              f"merged_multiplier_2={hyperparams['merged_multiplier_2']:.2f}")
        print(f"{'='*80}")
        
        for layer_name in ['Conv2D', 'Scalar_Dense', 'Merged_Dense_1', 'Merged_Dense_2', 'Output']:
            layer = breakdown[layer_name]
            percentage = (layer['total'] / total_params) * 100
            
            if layer_name == 'Merged_Dense_1':
                print(f"{layer_name:20s}: {layer['total']:>10,} params ({percentage:>5.1f}%) "
                      f"[Input size: {layer['input_size']:,} × {hyperparams['merged_dense_1']} = {layer['weights']:,} weights + {layer['biases']} biases]")
            else:
                print(f"{layer_name:20s}: {layer['total']:>10,} params ({percentage:>5.1f}%)")
    
    # Calculate average percentages across all trials
    print(f"\n{'='*80}")
    print("AVERAGE PARAMETER DISTRIBUTION ACROSS ALL TRIALS")
    print(f"{'='*80}")
    
    layer_totals = {
        'Conv2D': [],
        'Scalar_Dense': [],
        'Merged_Dense_1': [],
        'Merged_Dense_2': [],
        'Output': []
    }
    
    for _, row in df.iterrows():
        hyperparams = {
            'conv_filters': int(row['conv_filters']),
            'kernel_rows': int(row['kernel_rows']),
            'kernel_cols': int(row['kernel_cols']),
            'scalar_dense_units': int(row['scalar_dense_units']),
            'merged_dense_1': int(row['merged_dense_1']),
            'merged_multiplier_2': row['merged_multiplier_2']
        }
        
        breakdown = calculate_model3_layer_breakdown(hyperparams)
        total_params = breakdown['TOTAL']['total']
        
        for layer_name in layer_totals.keys():
            percentage = (breakdown[layer_name]['total'] / total_params) * 100
            layer_totals[layer_name].append(percentage)
    
    print(f"\n{'Layer':<20s} {'Avg %':>10s} {'Min %':>10s} {'Max %':>10s}")
    print("-" * 50)
    for layer_name in ['Conv2D', 'Scalar_Dense', 'Merged_Dense_1', 'Merged_Dense_2', 'Output']:
        percentages = layer_totals[layer_name]
        avg_pct = sum(percentages) / len(percentages)
        min_pct = min(percentages)
        max_pct = max(percentages)
        print(f"{layer_name:<20s} {avg_pct:>10.1f} {min_pct:>10.1f} {max_pct:>10.1f}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print("="*80)
    print("The 'Merged_Dense_1' layer dominates parameter count because it connects")
    print("the flattened Conv2D output (typically 960-3,840 units) plus scalar")
    print("dense output (16-64 units) to merged_dense_1 units (50-200 units).")
    print("This creates a massive weight matrix: input_size × merged_dense_1")
    print("="*80)


if __name__ == "__main__":
    main()

