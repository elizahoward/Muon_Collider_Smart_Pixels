"""
Quick script to run quantized hyperparameter tuning on Model2.5

Parameters:
- 4-bit quantization (0 integer bits)
- 15 epochs per trial
- 1 execution per trial
- 5 max trials
- Objective: weighted background rejection
  weighted = 0.3*BR95 + 0.6*BR98 + 0.1*BR99
- Progressive layer sizes: dense2_units <= (spatial_units + z_global_units)
                          dense3_units <= dense2_units

Author: Eric
Date: 2024
"""

import sys
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from model2_5 import Model2_5

def main():
    """Run quantized hyperparameter tuning for Model2.5"""
    
    print("="*70)
    print("Model2.5 - Quantized Hyperparameter Tuning")
    print("="*70)
    print("\nConfiguration:")
    print("  - Quantization: 4-bit fractional, 0 integer bits")
    print("  - z_global: 4-bit (matches spatial features)")
    print("  - Epochs per trial: 15")
    print("  - Executions per trial: 1")
    print("  - Max trials: 5")
    print("  - Objective: 0.3*BR95 + 0.6*BR98 + 0.1*BR99")
    print("  - Progressive layer constraint: Each layer <= previous layer")
    print("="*70)
    print()
    
    # Initialize Model2.5
    model25 = Model2_5(
        tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData",
        dense_units=128,        # Will be overridden by hyperparameter search
        z_global_units=32,      # Will be overridden by hyperparameter search
        dense2_units=128,       # Will be overridden by hyperparameter search
        dense3_units=64,        # Will be overridden by hyperparameter search
        dropout_rate=0.1,
        initial_lr=1e-3,
        end_lr=1e-4,
        power=2,
        bit_configs=[(4, 0)],   # 4-bit quantization
        # z_global_weight_bits=8,  # Original: 8-bit z_global
        z_global_weight_bits=4,    # Changed to 4-bit z_global
        z_global_int_bits=0
    )
    
    # Run hyperparameter tuning on 4-bit quantization
    bit_configs = [(4, 0)]  # 4-bit fractional, 0 integer bits
    
    results = model25.runQuantizedHyperparameterTuning(
        bit_configs=bit_configs,
        max_trials=8,
        executions_per_trial=1,
        numEpochs=15,
        use_weighted_bkg_rej=True,
        bkg_rej_weights={0.95: 0.3, 0.98: 0.6, 0.99: 0.1}
    )
    
    # Print results summary
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("="*70)
    
    for config_name, result in results.items():
        print(f"\n{config_name}:")
        print(f"  Results directory: {result['config_dir']}/")
        print(f"  Number of trials completed: {result['num_trials']}")
        print(f"  Best model saved to: {result['model_files'][0]}")
        print(f"  Summary file: {result['summary_file']}")
        print(f"\n  Best hyperparameters found:")
        for param, value in result['best_hyperparameters'].items():
            print(f"    {param}: {value}")
    
    print("\n" + "="*70)
    print("All models and hyperparameters have been saved!")
    print("="*70)

if __name__ == "__main__":
    main()