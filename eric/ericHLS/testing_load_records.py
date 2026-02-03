#!/usr/bin/env python3
"""
Simple test script to load a single H5 model and generate ROC curve.
This helps debug performance issues with model evaluation.

Usage:
    python testing_load_records.py --model_file <path_to_h5> --data_dir <tfrecord_dir>
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Add parent directory to path for imports
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/ryan/')
sys.path.append('/local/d1/smartpixML/filtering_models/shuffling_data/')

# Import TensorFlow
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import load_model

# Import QKeras
try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu, quantized_tanh
    from qkeras.utils import _add_supported_quantized_objects
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("Warning: QKeras not available")

# Import data generator
import OptimizedDataGenerator4_data_shuffled_bigData as ODG2


def get_custom_objects():
    """Get custom objects dictionary for loading QKeras models."""
    if not QKERAS_AVAILABLE:
        return {}
    co = {}
    _add_supported_quantized_objects(co)
    return co


def main():
    parser = argparse.ArgumentParser(description='Test model loading and ROC generation')
    parser.add_argument('--model_file', type=str, required=True, help='Path to H5 model file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to TFRecord directory')
    parser.add_argument('--num_batches', type=int, default=5, help='Number of batches to use (default: 5 for testing)')
    parser.add_argument('--batch_size', type=int, default=16384, help='Batch size (default: 16384)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("TESTING MODEL LOADING AND ROC GENERATION")
    print("=" * 80)
    print(f"\nModel file: {args.model_file}")
    print(f"Data directory: {args.data_dir}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Batch size: {args.batch_size}")
    
    # Step 1: Load model
    print("\n" + "-" * 80)
    print("STEP 1: Loading model...")
    print("-" * 80)
    start_time = time.time()
    
    custom_objects = get_custom_objects()
    model = load_model(args.model_file, custom_objects=custom_objects, compile=False)
    
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f} seconds")
    
    # Get parameter count
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = int(trainable_count + non_trainable_count)
    print(f"  Total parameters: {total_params:,}")
    
    # Step 2: Load validation data
    print("\n" + "-" * 80)
    print("STEP 2: Loading validation data...")
    print("-" * 80)
    start_time = time.time()
    
    val_dir = os.path.join(args.data_dir, "tfrecords_validation/")
    x_feature_description = ['x_profile', 'z_global', 'y_profile', 'y_local']
    
    validation_generator = ODG2.OptimizedDataGeneratorDataShuffledBigData(
        load_records=True,
        tf_records_dir=val_dir,
        x_feature_description=x_feature_description,
        batch_size=args.batch_size
    )
    
    data_load_time = time.time() - start_time
    print(f"✓ Data generator created in {data_load_time:.2f} seconds")
    print(f"  Total batches available: {len(validation_generator)}")
    print(f"  Using first {args.num_batches} batches for testing")
    
    # Step 3: Run inference
    print("\n" + "-" * 80)
    print("STEP 3: Running inference...")
    print("-" * 80)
    
    y_true_all = []
    y_pred_all = []
    
    total_inference_time = 0
    for batch_idx in range(min(args.num_batches, len(validation_generator))):
        print(f"\nBatch {batch_idx + 1}/{args.num_batches}:")
        
        # Load batch
        batch_start = time.time()
        X_batch, y_batch = validation_generator[batch_idx]
        batch_load_time = time.time() - batch_start
        print(f"  Data loading: {batch_load_time:.3f} seconds")
        
        # Print batch info
        if isinstance(X_batch, dict):
            print(f"  Batch type: dict")
            print(f"  Keys: {list(X_batch.keys())}")
            for key, val in X_batch.items():
                print(f"    {key}: {val.shape}")
        elif isinstance(X_batch, (list, tuple)):
            print(f"  Batch shape: {[x.shape for x in X_batch]}")
        else:
            print(f"  Batch shape: {X_batch.shape}")
        print(f"  Labels shape: {y_batch.shape}")
        
        # Run prediction
        pred_start = time.time()
        if isinstance(X_batch, dict):
            # For dictionary input
            y_pred_batch = model.predict(X_batch, verbose=0)
        elif isinstance(X_batch, (list, tuple)):
            y_pred_batch = model.predict(X_batch, verbose=0, batch_size=len(X_batch[0]))
        else:
            y_pred_batch = model.predict(X_batch, verbose=0)
        pred_time = time.time() - pred_start
        total_inference_time += pred_time
        print(f"  Prediction: {pred_time:.3f} seconds")
        print(f"  Throughput: {len(y_batch) / pred_time:.0f} samples/sec")
        
        y_true_all.append(y_batch)
        y_pred_all.append(y_pred_batch)
    
    print(f"\n✓ Total inference time: {total_inference_time:.2f} seconds")
    print(f"  Average per batch: {total_inference_time / args.num_batches:.2f} seconds")
    
    # Step 4: Compute ROC
    print("\n" + "-" * 80)
    print("STEP 4: Computing ROC curve...")
    print("-" * 80)
    start_time = time.time()
    
    y_true = np.concatenate(y_true_all, axis=0).flatten()
    y_pred = np.concatenate(y_pred_all, axis=0).flatten()
    
    print(f"  Total samples: {len(y_true):,}")
    print(f"  True labels - unique values: {np.unique(y_true)}")
    print(f"  Predictions - min: {y_pred.min():.4f}, max: {y_pred.max():.4f}")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    roc_time = time.time() - start_time
    print(f"✓ ROC computed in {roc_time:.3f} seconds")
    print(f"  AUC: {roc_auc:.4f}")
    
    # Step 5: Plot ROC curve
    print("\n" + "-" * 80)
    print("STEP 5: Plotting ROC curve...")
    print("-" * 80)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Background Efficiency)', fontsize=14)
    ax.set_ylabel('True Positive Rate (Signal Efficiency)', fontsize=14)
    ax.set_title(f'ROC Curve: {os.path.basename(args.model_file)}', 
                fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'test_roc_curve.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ ROC curve saved to: {output_path}")
    plt.close()
    
    # Step 6: Compute background rejection
    print("\n" + "-" * 80)
    print("STEP 6: Computing background rejection...")
    print("-" * 80)
    
    for sig_eff in [0.90, 0.95, 0.99]:
        idx = np.where(tpr >= sig_eff)[0]
        if len(idx) > 0:
            idx = idx[0]
            fpr_at_target = fpr[idx]
            if fpr_at_target > 0:
                bg_rej = 1.0 / fpr_at_target
                print(f"  @ {sig_eff:.0%} signal efficiency: Background rejection = {bg_rej:.2f}")
            else:
                print(f"  @ {sig_eff:.0%} signal efficiency: Perfect rejection (FPR = 0)")
        else:
            print(f"  @ {sig_eff:.0%} signal efficiency: Not reached")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model loading:      {load_time:.2f} seconds")
    print(f"Data loading:       {data_load_time:.2f} seconds")
    print(f"Inference (total):  {total_inference_time:.2f} seconds")
    print(f"  Per batch avg:    {total_inference_time / args.num_batches:.2f} seconds")
    print(f"  Throughput:       {len(y_true) / total_inference_time:.0f} samples/sec")
    print(f"ROC computation:    {roc_time:.3f} seconds")
    print(f"\nAUC: {roc_auc:.4f}")
    print(f"Total time: {load_time + data_load_time + total_inference_time + roc_time:.2f} seconds")
    
    # Estimate full dataset
    total_batches = len(validation_generator)
    estimated_full_time = (total_inference_time / args.num_batches) * total_batches
    print(f"\nEstimated time for full dataset ({total_batches} batches): {estimated_full_time / 60:.1f} minutes")
    print("=" * 80)
    
    # Cleanup
    del model
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
