#!/usr/bin/env python3
"""
Training Script for Model2 using the SmartPixModel Abstract Base Class

This script demonstrates how to use the Model2 implementation that properly
inherits from the SmartPixModel abstract base class.

Usage:
    python train_model2.py [options]

Author: Eric
Date: 2024
"""

import argparse
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from model2 import Model2


def main():
    """Main training script for Model2"""
    parser = argparse.ArgumentParser(
        description='Train Model2 using the SmartPixModel abstract base class',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default="/local/d1/smartpixML/filtering_models/shuffling_data/filtering_records1024_data_shuffled_single/",
        help='Path to TFRecords directory'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-type', 
        choices=['unquantized', 'quantized', 'both'], 
        default='unquantized',
        help='Type of model to train'
    )
    parser.add_argument(
        '--dropout-rate', 
        type=float, 
        default=0.1,
        help='Dropout rate for regularization'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=1e-3,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--early-stopping-patience', 
        type=int, 
        default=20,
        help='Patience for early stopping'
    )
    
    # Quantization arguments
    parser.add_argument(
        '--bit-configs', 
        nargs='+', 
        default=['8_0', '6_0', '4_0'],
        help='Bit configurations for quantized models (format: weight_bits_int_bits)'
    )
    
    # Hyperparameter tuning
    parser.add_argument(
        '--hyperparameter-tuning', 
        action='store_true',
        help='Run hyperparameter tuning before training'
    )
    parser.add_argument(
        '--max-trials', 
        type=int, 
        default=50,
        help='Maximum trials for hyperparameter tuning'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help='Output directory for results (default: timestamp-based)'
    )
    parser.add_argument(
        '--save-plots', 
        action='store_true',
        help='Save training plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"model2_results_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Parse bit configurations
    bit_configs = []
    for config in args.bit_configs:
        try:
            weight_bits, int_bits = map(int, config.split('_'))
            bit_configs.append((weight_bits, int_bits))
        except ValueError:
            print(f"Warning: Invalid bit configuration '{config}'. Expected format: 'weight_bits_int_bits'")
    
    # Initialize Model2
    print("=== Initializing Model2 ===")
    model2 = Model2(
        tfRecordFolder=args.data_dir,
        dropout_rate=args.dropout_rate
    )
    
    # Load data
    print("=== Loading TFRecords ===")
    train_gen, val_gen = model2.loadTfRecords()
    
    # Hyperparameter tuning (optional)
    if args.hyperparameter_tuning:
        print("=== Running Hyperparameter Tuning ===")
        best_model, tuning_results = model2.runHyperparameterTuning(
            max_trials=args.max_trials
        )
        
        # Save tuning results
        tuning_results_file = output_dir / "hyperparameter_tuning_results.json"
        with open(tuning_results_file, 'w') as f:
            json.dump(tuning_results, f, indent=2)
        print(f"Hyperparameter tuning results saved to: {tuning_results_file}")
    
    # Train models
    results = {}
    
    if args.model_type in ['unquantized', 'both']:
        print("\n=== Training Unquantized Model2 ===")
        
        # Build unquantized model
        model2.buildModel("unquantized")
        
        # Train model
        history = model2.trainModel(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience
        )
        
        # Evaluate model
        eval_results = model2.evaluate()
        
        # Save results
        results['unquantized'] = {
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'binary_accuracy': history.history['binary_accuracy'],
                'val_binary_accuracy': history.history['val_binary_accuracy']
            },
            'evaluation': eval_results
        }
        
        # Save model
        model2.saveModel()
        
        # Plot results
        if args.save_plots:
            plot_dir = output_dir / "unquantized_plots"
            plot_dir.mkdir(exist_ok=True)
            model2.plotModel(save_plots=True, output_dir=str(plot_dir))
        
        print("✓ Unquantized Model2 training completed!")
    
    if args.model_type in ['quantized', 'both']:
        print("\n=== Training Quantized Model2 ===")
        
        if not bit_configs:
            print("Warning: No valid bit configurations provided. Using default.")
            bit_configs = [(8, 0), (6, 0), (4, 0)]
        
        # Build quantized models
        quantized_models = model2.buildModel("quantized", bit_configs)
        
        results['quantized'] = {}
        
        # Train each quantized model configuration
        for config_name, q_model in quantized_models.items():
            print(f"\n--- Training {config_name} ---")
            
            # Set the quantized model as the current model for training
            model2.model = q_model
            
            # Train model
            history = model2.trainModel(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                early_stopping_patience=args.early_stopping_patience
            )
            
            # Evaluate model
            eval_results = model2.evaluate()
            
            # Save results
            results['quantized'][config_name] = {
                'training_history': {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'binary_accuracy': history.history['binary_accuracy'],
                    'val_binary_accuracy': history.history['val_binary_accuracy']
                },
                'evaluation': eval_results
            }
            
            # Save model
            model_path = output_dir / f"model2_{config_name}.keras"
            q_model.save(model_path)
            print(f"Model saved to: {model_path}")
            
            # Plot results
            if args.save_plots:
                plot_dir = output_dir / f"quantized_plots_{config_name}"
                plot_dir.mkdir(exist_ok=True)
                model2.plotModel(save_plots=True, output_dir=str(plot_dir))
        
        print("✓ Quantized Model2 training completed!")
    
    # Save overall results
    results_file = output_dir / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Training Completed! ===")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {results_file}")
    
    # Print summary
    print("\n=== Results Summary ===")
    for model_type, type_results in results.items():
        print(f"\n{model_type.upper()} MODELS:")
        if isinstance(type_results, dict) and 'evaluation' in type_results:
            # Unquantized model
            eval_results = type_results['evaluation']
            print(f"  Test Accuracy: {eval_results['test_accuracy']:.4f}")
            print(f"  Test Loss: {eval_results['test_loss']:.4f}")
            print(f"  ROC AUC: {eval_results['roc_auc']:.4f}")
        else:
            # Quantized models
            for config_name, config_results in type_results.items():
                eval_results = config_results['evaluation']
                print(f"  {config_name}:")
                print(f"    Test Accuracy: {eval_results['test_accuracy']:.4f}")
                print(f"    Test Loss: {eval_results['test_loss']:.4f}")
                print(f"    ROC AUC: {eval_results['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

