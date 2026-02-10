#!/usr/bin/env python3
"""
Simple script to load a model, run predictions, and compute FPR/TPR metrics.
Edit the MODEL_FILE and DATA_DIR variables below.
"""

import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================
MODEL_FILE = "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/model2.5_quantized_4w0i_hyperparameter_results_20260203_114608/model_trial_0.h5"
DATA_DIR = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try3_eric/filtering_records16384_data_shuffled_single_bigData"
NUM_BATCHES = None  # None = use all batches
# ============================================================================

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
    from qkeras.utils import _add_supported_quantized_objects
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False

def _parse_tfrecord_fn(example):
    feature_description = {
        'y': tf.io.FixedLenFeature([], tf.string),
        'x_profile': tf.io.FixedLenFeature([], tf.string),
        'z_global': tf.io.FixedLenFeature([], tf.string),
        'y_profile': tf.io.FixedLenFeature([], tf.string),
        'y_local': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, feature_description)
    y = tf.io.parse_tensor(example['y'], out_type=tf.float32)
    X = {
        'x_profile': tf.io.parse_tensor(example['x_profile'], out_type=tf.float32),
        'z_global': tf.io.parse_tensor(example['z_global'], out_type=tf.float32),
        'y_profile': tf.io.parse_tensor(example['y_profile'], out_type=tf.float32),
        'y_local': tf.io.parse_tensor(example['y_local'], out_type=tf.float32),
    }
    return X, y


def build_tfrecord_dataset(tfrecord_dir):
    pattern = os.path.join(tfrecord_dir, "*.tfrecord")
    files = tf.data.Dataset.list_files(pattern, shuffle=False)
    ds = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_custom_objects():
    if not QKERAS_AVAILABLE:
        return {}
    co = {}
    _add_supported_quantized_objects(co)
    return co


def sig_eff_and_bkg_rej(y_true, y_score, thr):
    """
    Compute signal efficiency and background rejection at a threshold.
    
    Signal efficiency (ε_sig) = TPR = TP / (TP + FN)
    Background rejection (R_bkg) = 1 - FPR = 1 - FP/(FP + TN)
    
    Args:
        y_true: True labels (0=background, 1=signal)
        y_score: Predicted scores
        thr: Threshold
    
    Returns:
        eff_sig: Signal efficiency
        bkg_rej: Background rejection
    """
    y_hat = (y_score >= thr).astype(int)
    sig = (y_true == 1)
    bkg = (y_true == 0)
    
    tp = np.sum(y_hat[sig] == 1)
    fp = np.sum(y_hat[bkg] == 1)
    tn = np.sum(y_hat[bkg] == 0)
    
    eff_sig = tp / max(np.sum(sig), 1)  # Signal efficiency = TPR
    fpr = fp / max(np.sum(bkg), 1)       # False positive rate
    bkg_rej = 1.0 - fpr                  # Background rejection = 1 - FPR
    return eff_sig, bkg_rej


def main():
    # Load model
    print(f"Loading: {os.path.basename(MODEL_FILE)}")
    model = load_model(MODEL_FILE, custom_objects=get_custom_objects(), compile=False)
    
    # Load validation data
    val_dir = os.path.join(DATA_DIR, "tfrecords_validation/")
    dataset = build_tfrecord_dataset(val_dir)
    if NUM_BATCHES:
        dataset = dataset.take(NUM_BATCHES)
    
    # Predict directly from tf.data pipeline
    y_score = model.predict(dataset, verbose=0).ravel()
    y_true = np.concatenate(
        [y.numpy().ravel() for _, y in dataset],
        axis=0
    )
    
    # Compute ROC using all threshold points for finer ROC traces.
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    
    print(f"AUC: {roc_auc:.4f}")
    print(f"\nThreshold | Signal Eff | Bkg Rej")
    print("-" * 40)
    for thr in [0.3, 0.5, 0.7, 0.9]:
        eff, rej = sig_eff_and_bkg_rej(y_true, y_score, thr)
        print(f"  {thr:.1f}     |   {eff:.4f}   |  {rej:.4f}")
    
    print(f"\nSig Eff | Bkg Rej (1-FPR)")
    print("-" * 30)
    for sig_eff in [0.95, 0.98, 0.99]:
        idx = np.where(tpr >= sig_eff)[0]
        if len(idx) > 0:
            fpr_at_target = fpr[idx[0]]
            bg_rej = 1.0 - fpr_at_target  # Background rejection = 1 - FPR
            print(f" {sig_eff:.0%}    |  {bg_rej:.4f}")
        else:
            print(f" {sig_eff:.0%}    |  N/A")
    
    # Cleanup
    del model
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
