#!/usr/bin/env python3
"""
Simple Hyperparameter Tuning Script for Model3

This script uses the built-in hyperparameter tuning functionality from Model3.
It runs hyperparameter tuning using the big dataset and saves results.

Usage:
    python hyperparameter_tuning_model3.py

Author: Eric
Date: 2024
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to path to import the model classes
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/ryan/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/')

from model3 import Model3


def main():
    """
    Main function to run hyperparameter tuning for Model3 using built-in methods.
    """
    print("=== Model3 Hyperparameter Tuning Script ===")
    
    # Configuration
    tfRecordFolder = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData"
    
    # Check if dataset exists
    if not os.path.exists(tfRecordFolder):
        print(f"ERROR: Dataset path does not exist: {tfRecordFolder}")
        print("Please check the path and try again.")
        return
    
    print(f"Using dataset: {tfRecordFolder}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"model3_hyperparameter_tuning_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}/")
    
    # Initialize Model3
    print("Initializing Model3...")
    model3 = Model3(
        tfRecordFolder=tfRecordFolder,
        conv_filters=32,
        kernel_rows=3,
        kernel_cols=5,
        scalar_dense_units=32,
        merged_dense_1=200,
        merged_dense_2=100,
        dropout_rate=0.0,
        initial_lr=0.000871145,
        end_lr=5.3e-05,
        power=2
    )
    
    # Run hyperparameter tuning using built-in method
    print("Starting hyperparameter tuning...")
    try:
        best_model, results = model3.runHyperparameterTuning(
            max_trials=75,  # Adjust based on computational resources
            executions_per_trial=2
        )
        
        print("✓ Hyperparameter tuning completed!")
        
        # Save the best model
        best_model_path = os.path.join(output_dir, "best_model.h5")
        best_model.save(best_model_path)
        print(f"Best model saved to: {best_model_path}")
        
        # Save tuning results
        results_file = os.path.join(output_dir, "tuning_results.txt")
        with open(results_file, 'w') as f:
            f.write("=== Model3 Hyperparameter Tuning Results ===\n\n")
            f.write(f"Dataset: {tfRecordFolder}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Tuning Results Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(str(results))
        
        print(f"Tuning results saved to: {results_file}")
        
        # Evaluate the best model
        print("Evaluating best model...")
        eval_results = model3.evaluate()
        
        # Save evaluation results
        eval_file = os.path.join(output_dir, "evaluation_results.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Evaluation results saved to: {eval_file}")
        
        # Create comprehensive summary
        summary_file = os.path.join(output_dir, "comprehensive_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== Model3 Hyperparameter Tuning Complete Results ===\n\n")
            f.write(f"Dataset: {tfRecordFolder}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Best Model Performance:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Test Loss: {eval_results['test_loss']:.4f}\n")
            f.write(f"Test Accuracy: {eval_results['test_accuracy']:.4f}\n")
            f.write(f"ROC AUC: {eval_results['roc_auc']:.4f}\n\n")
            
            f.write("Tuning Results:\n")
            f.write("-" * 40 + "\n")
            f.write(str(results))
        
        print(f"Comprehensive summary saved to: {summary_file}")
        
        print("\n=== TUNING COMPLETED SUCCESSFULLY ===")
        print(f"Best model performance:")
        print(f"  Accuracy: {eval_results['test_accuracy']:.4f}")
        print(f"  ROC AUC: {eval_results['roc_auc']:.4f}")
        print(f"  Loss: {eval_results['test_loss']:.4f}")
        
    except Exception as e:
        print(f"ERROR during hyperparameter tuning: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nAll results saved to: {output_dir}/")
    print("Hyperparameter tuning completed!")


if __name__ == "__main__":
    main()

