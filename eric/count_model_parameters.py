#!/usr/bin/env python3
"""
Script to count the total number of parameters in a Keras model saved as H5 file.

Usage:
    python count_model_parameters.py <path_to_model.h5>

Example:
    python count_model_parameters.py model2.5_quantized_4w0i_hyperparameter_results_20260119_181421/model_trial_3.h5
"""

import sys
import os
import tensorflow as tf
from tensorflow import keras


def count_parameters(model_path):
    """
    Load a Keras model from H5 file and count its parameters.
    
    Args:
        model_path: Path to the H5 model file
        
    Returns:
        tuple: (total_params, trainable_params, non_trainable_params)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    print(f"Loading model from: {model_path}")
    try:
        # Try loading with custom objects (in case QKeras layers are used)
        try:
            from qkeras import QDense, QActivation
            custom_objects = {'QDense': QDense, 'QActivation': QActivation}
            model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        except ImportError:
            # Fall back to regular loading if QKeras not available
            model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load without compilation...")
        model = keras.models.load_model(model_path, compile=False)
    
    # Count parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    return total_params, trainable_params, non_trainable_params


def main():
    if len(sys.argv) != 2:
        print("Usage: python count_model_parameters.py <path_to_model.h5>")
        print("\nExample:")
        print("    python count_model_parameters.py model2.5_quantized_4w0i_hyperparameter_results_20260119_181421/model_trial_3.h5")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    try:
        total, trainable, non_trainable = count_parameters(model_path)
        
        print("\n" + "="*60)
        print("MODEL PARAMETER COUNT")
        print("="*60)
        print(f"Total parameters:          {total:,}")
        print(f"Trainable parameters:      {trainable:,}")
        print(f"Non-trainable parameters:  {non_trainable:,}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
