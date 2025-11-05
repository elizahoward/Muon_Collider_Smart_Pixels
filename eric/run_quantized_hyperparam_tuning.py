"""
Quick script to run 6-bit quantized hyperparameter tuning on Model2

Parameters:
- 6-bit quantization (0 integer bits)
- 30 epochs per trial
- 1 execution per trial
- 120 max trials

Author: Eric
Date: 2024
"""

import sys
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from model2 import Model2

def main():
    """Run 6-bit quantized hyperparameter tuning for Model2"""
    
    print("="*70)
    print("Model2 - 6-bit Quantized Hyperparameter Tuning")
    print("="*70)
    print("\nConfiguration:")
    print("  - Quantization: 6-bit fractional, 0 integer bits")
    print("  - Epochs per trial: 30")
    print("  - Executions per trial: 1")
    print("  - Max trials: 120")
    print("="*70)
    print()
    
    # Initialize Model2
    model2 = Model2(
        tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
        xz_units=32,  # Will be overridden by hyperparameter search
        yl_units=32,
        merged_units_1=128,
        merged_units_2=64,
        merged_units_3=32,
        dropout_rate=0.1,
        initial_lr=1e-3,
        end_lr=1e-4,
        power=2
    )
    
    # Run hyperparameter tuning on 6-bit quantization only
    bit_configs = [(4, 0)]  # 6-bit fractional, 0 integer bits
    
    results = model2.runQuantizedHyperparameterTuning(
        bit_configs=bit_configs,
        max_trials=5,
        executions_per_trial=1,
        numEpochs=2
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




