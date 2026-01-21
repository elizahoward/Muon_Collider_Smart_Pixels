"""
Load and inspect quantized model h5 files from hyperparameter tuning results.

This script loads a quantized model saved from hyperparameter tuning and displays
its architecture, hyperparameters, and optionally evaluates it on test data.

Usage:
    python load_quantized_model.py <path_to_model.h5>
    
Example:
    python load_quantized_model.py model2.5_quantized_4w0i_hyperparameter_results_20260119_181421/model_trial_1.h5

Author: Eric
Date: 2026
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# Add necessary paths
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')

# Import TensorFlow and related libraries
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import QKeras for quantized layers
try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu, quantized_tanh
    from qkeras.utils import _add_supported_quantized_objects
    QKERAS_AVAILABLE = True
except ImportError:
    print("Warning: QKeras not available. Install with: pip install qkeras")
    QKERAS_AVAILABLE = False


def get_custom_objects():
    """Get custom objects dictionary for loading QKeras models."""
    if not QKERAS_AVAILABLE:
        return {}
    
    co = {}
    _add_supported_quantized_objects(co)
    return co


def load_hyperparameters(model_path):
    """
    Load hyperparameters JSON file corresponding to the model.
    
    Args:
        model_path: Path to the model h5 file
        
    Returns:
        Dictionary of hyperparameters or None if file not found
    """
    model_path = Path(model_path)
    
    # Extract trial number from model filename (e.g., model_trial_1.h5 -> 1)
    model_name = model_path.stem  # e.g., "model_trial_1"
    parts = model_name.split('_')
    
    if len(parts) >= 3 and parts[-2] == 'trial':
        trial_num = parts[-1]
        hyperparam_file = model_path.parent / f"hyperparams_trial_{trial_num}.json"
        
        if hyperparam_file.exists():
            with open(hyperparam_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Hyperparameters file not found: {hyperparam_file}")
            return None
    else:
        print(f"Warning: Could not parse trial number from model name: {model_name}")
        return None


def print_model_info(model, hyperparams=None):
    """
    Print detailed information about the model.
    
    Args:
        model: Loaded Keras model
        hyperparams: Dictionary of hyperparameters (optional)
    """
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    
    # Model name
    print(f"\nModel Name: {model.name}")
    
    # Hyperparameters
    if hyperparams:
        print("\nHyperparameters:")
        for key, value in hyperparams.items():
            if isinstance(value, float) and value < 0.001:
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")
    
    # Model architecture summary
    print("\nModel Architecture:")
    print("-" * 70)
    model.summary()
    
    # Input shapes
    print("\nInput Shapes:")
    for i, input_layer in enumerate(model.inputs):
        print(f"  Input {i} ({input_layer.name}): {input_layer.shape}")
    
    # Output shapes
    print("\nOutput Shapes:")
    for i, output_layer in enumerate(model.outputs):
        print(f"  Output {i} ({output_layer.name}): {output_layer.shape}")
    
    # Layer details with quantization info
    print("\nLayer Details:")
    print("-" * 70)
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        print(f"\n[Layer {i}] {layer.name} ({layer_type})")
        
        # Print layer configuration
        if hasattr(layer, 'units'):
            print(f"  Units: {layer.units}")
        
        # Print quantization info for QKeras layers
        if layer_type == 'QDense':
            config = layer.get_config()
            print(f"  Kernel quantizer: {config.get('kernel_quantizer', 'N/A')}")
            print(f"  Bias quantizer: {config.get('bias_quantizer', 'N/A')}")
        elif layer_type == 'QActivation':
            config = layer.get_config()
            print(f"  Quantizer: {config.get('quantizer', 'N/A')}")
        
        # Print output shape
        if hasattr(layer, 'output_shape'):
            print(f"  Output shape: {layer.output_shape}")
    
    # Parameter counts
    print("\n" + "="*70)
    print("PARAMETER COUNTS")
    print("="*70)
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_count = trainable_count + non_trainable_count
    
    print(f"Total parameters: {total_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")


def create_dummy_input(model):
    """
    Create dummy input data for testing the model.
    
    Args:
        model: Loaded Keras model
        
    Returns:
        List of dummy input arrays
    """
    inputs = []
    for input_layer in model.inputs:
        # Get shape without batch dimension
        shape = input_layer.shape[1:]
        # Create random input with batch size 1
        dummy = np.random.randn(1, *shape).astype(np.float32)
        inputs.append(dummy)
    return inputs


def test_model_inference(model):
    """
    Test the model with dummy input to verify it works.
    
    Args:
        model: Loaded Keras model
        
    Returns:
        Model output
    """
    print("\n" + "="*70)
    print("TESTING MODEL INFERENCE")
    print("="*70)
    
    # Create dummy inputs
    dummy_inputs = create_dummy_input(model)
    
    print("\nCreated dummy inputs:")
    for i, (input_layer, data) in enumerate(zip(model.inputs, dummy_inputs)):
        print(f"  {input_layer.name}: shape={data.shape}, dtype={data.dtype}")
        print(f"    Sample values: {data.flatten()[:5]}")
    
    # Run inference
    print("\nRunning inference...")
    try:
        output = model.predict(dummy_inputs, verbose=0)
        print(f"✓ Inference successful!")
        print(f"\nOutput shape: {output.shape}")
        print(f"Output values: {output.flatten()[:5]}")
        print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
        return output
    except Exception as e:
        print(f"✗ Inference failed with error: {e}")
        return None


def main():
    """Main function to load and inspect a quantized model."""
    parser = argparse.ArgumentParser(
        description='Load and inspect quantized model h5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and inspect a model
  python load_quantized_model.py model2.5_quantized_4w0i_hyperparameter_results_20260119_181421/model_trial_1.h5
  
  # Load model without testing inference
  python load_quantized_model.py model_trial_1.h5 --no-test
  
  # Get only the parameter count (for scripting)
  python load_quantized_model.py model_trial_1.h5 --params-only
        """
    )
    
    parser.add_argument('model_path', type=str,
                        help='Path to the model h5 file')
    parser.add_argument('--no-test', action='store_true',
                        help='Skip inference testing')
    parser.add_argument('--compile', action='store_true',
                        help='Compile the model after loading (default: False)')
    parser.add_argument('--params-only', action='store_true',
                        help='Only output the total parameter count (no other information)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Params-only mode: just output parameter count
    if args.params_only:
        # Load model silently
        try:
            if QKERAS_AVAILABLE:
                custom_objects = get_custom_objects()
                model = load_model(str(model_path), custom_objects=custom_objects, compile=False)
            else:
                model = load_model(str(model_path), compile=False)
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Calculate and print parameter count
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        total_count = trainable_count + non_trainable_count
        
        print(int(total_count))
        return model
    
    # Normal mode with full output
    print("="*70)
    print("LOADING QUANTIZED MODEL")
    print("="*70)
    print(f"\nModel file: {model_path}")
    
    # Load hyperparameters
    print("\nLoading hyperparameters...")
    hyperparams = load_hyperparameters(model_path)
    if hyperparams:
        print("✓ Hyperparameters loaded successfully")
    else:
        print("✗ Hyperparameters not found")
    
    # Load model
    print("\nLoading model...")
    try:
        if QKERAS_AVAILABLE:
            custom_objects = get_custom_objects()
            model = load_model(str(model_path), custom_objects=custom_objects, compile=args.compile)
        else:
            model = load_model(str(model_path), compile=args.compile)
        
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nTrying to load without compilation...")
        try:
            if QKERAS_AVAILABLE:
                custom_objects = get_custom_objects()
                model = load_model(str(model_path), custom_objects=custom_objects, compile=False)
            else:
                model = load_model(str(model_path), compile=False)
            print("✓ Model loaded successfully (without compilation)")
        except Exception as e2:
            print(f"✗ Failed to load model: {e2}")
            sys.exit(1)
    
    # Print model information
    print_model_info(model, hyperparams)
    
    # Test inference if requested
    if not args.no_test:
        test_model_inference(model)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return model


if __name__ == "__main__":
    model = main()
