"""
Example script for running quantized hyperparameter tuning on Model2

This script demonstrates how to use the new runQuantizedHyperparameterTuning method
to perform hyperparameter optimization on quantized versions of Model2.

Author: Eric
Date: 2024
"""

import sys
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from model2 import Model2

def main():
    """Run quantized hyperparameter tuning for Model2"""
    
    print("=== Model2 Quantized Hyperparameter Tuning ===\n")
    
    # Initialize Model2
    model2 = Model2(
        tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
        xz_units=32,  # These will be overridden by hyperparameter search
        yl_units=32,
        merged_units_1=128,
        merged_units_2=64,
        merged_units_3=32,
        dropout_rate=0.1,
        initial_lr=1e-3,
        end_lr=1e-4,
        power=2
    )
    
    # Option 1: Run hyperparameter tuning on all default bit configurations
    # results = model2.runQuantizedHyperparameterTuning(
    #     max_trials=50,
    #     executions_per_trial=2,
    #     numEpochs=30
    # )
    
    # Option 2: Run hyperparameter tuning on specific bit configurations
    # For example, only 8-bit, 6-bit, and 4-bit quantization
    bit_configs_to_test = [
        (8, 0),   # 8-bit weights, 0 integer bits
        (6, 0),   # 6-bit weights, 0 integer bits
        (4, 0),   # 4-bit weights, 0 integer bits
    ]
    
    results = model2.runQuantizedHyperparameterTuning(
        bit_configs=bit_configs_to_test,
        max_trials=50,          # Number of different hyperparameter combinations to try
        executions_per_trial=2,  # Number of times to train each combination
        numEpochs=30            # Epochs per training run
    )
    
    # Access the results
    print("\n=== Results Summary ===")
    for config_name, result in results.items():
        print(f"\n{config_name}:")
        print(f"  Results directory: {result['config_dir']}/")
        print(f"  Number of trials: {result['num_trials']}")
        print(f"  All model files: {len(result['model_files'])} models saved")
        print(f"  Best model: {result['model_files'][0]}")
        print(f"  Summary file: {result['summary_file']}")
        print(f"  Best hyperparameters: {result['best_hyperparameters']}")
    
    # You can also access all models (not just the best)
    # For example, to get all 8-bit models:
    if 'quantized_8w0i' in results:
        all_8bit_models = results['quantized_8w0i']['all_models']
        best_8bit_model = results['quantized_8w0i']['best_model']
        
        print(f"\n8-bit quantization results:")
        print(f"  Total models trained: {len(all_8bit_models)}")
        print(f"  Best model summary:")
        best_8bit_model.summary()
        
        # Access specific trial models
        print(f"\n  All 8-bit model files:")
        for idx, model_file in enumerate(results['quantized_8w0i']['model_files']):
            marker = " (BEST)" if idx == 0 else ""
            print(f"    Trial {idx}: {model_file}{marker}")

if __name__ == "__main__":
    main()

