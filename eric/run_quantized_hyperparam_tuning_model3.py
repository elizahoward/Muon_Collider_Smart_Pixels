"""
Quick script to run quantized hyperparameter tuning on Model3

Parameters:
- 4-bit quantization (0 integer bits)
- 20 epochs per trial
- 1 execution per trial
- 160 max trials

Author: Eric
Date: 2024
"""

import sys
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from model3 import Model3

def main():
    """Run quantized hyperparameter tuning for Model3"""
    
    print("="*70)
    print("Model3 - Quantized Hyperparameter Tuning")
    print("="*70)
    print("\nConfiguration:")
    print("  - Quantization: 4-bit fractional, 0 integer bits")
    print("  - Epochs per trial: 20")
    print("  - Executions per trial: 1")
    print("  - Max trials: 160")
    print("="*70)
    print()
    
    # Initialize Model3
    model3 = Model3(
        tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
        conv_filters=32,  # Will be overridden by hyperparameter search
        kernel_rows=3,
        kernel_cols=3,
        scalar_dense_units=32,
        merged_dense_1=200,
        merged_dense_2=100,
        dropout_rate=0.1,
        initial_lr=0.000871145,
        end_lr=5.3e-05,
        power=2
    )
    
    # Run hyperparameter tuning on 4-bit quantization
    bit_configs = [(4, 0)]  # 4-bit fractional, 0 integer bits
    
    results = model3.runQuantizedHyperparameterTuning(
        bit_configs=bit_configs,
        max_trials=150,
        executions_per_trial=1,
        numEpochs=23
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

