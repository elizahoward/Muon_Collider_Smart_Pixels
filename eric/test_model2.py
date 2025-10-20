#!/usr/bin/env python3
"""
Test Script for Model2 Implementation

This script tests the Model2 implementation to ensure it properly inherits
from the SmartPixModel abstract base class and all methods work correctly.

Author: Eric
Date: 2024
"""

import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from model2 import Model2


def test_model2_initialization():
    """Test Model2 initialization"""
    print("=== Testing Model2 Initialization ===")
    
    try:
        model2 = Model2()
        print("✓ Model2 initialized successfully")
        print(f"  Model name: {model2.modelName}")
        print(f"  Feature description: {model2.x_feature_description}")
        print(f"  Dropout rate: {model2.dropout_rate}")
        return model2
    except Exception as e:
        print(f"✗ Model2 initialization failed: {e}")
        return None


def test_model2_inheritance():
    """Test that Model2 properly inherits from SmartPixModel"""
    print("\n=== Testing Model2 Inheritance ===")
    
    try:
        model2 = Model2()
        
        # Check inheritance
        from Model_Classes import SmartPixModel
        assert isinstance(model2, SmartPixModel), "Model2 should inherit from SmartPixModel"
        print("✓ Model2 properly inherits from SmartPixModel")
        
        # Check that all abstract methods are implemented
        required_methods = [
            'makeUnquantizedModel',
            'makeUnquatizedModelHyperParameterTuning', 
            'makeQuantizedModel',
            'buildModel',
            'runHyperparameterTuning',
            'trainModel',
            'plotModel',
            'evaluate'
        ]
        
        for method_name in required_methods:
            assert hasattr(model2, method_name), f"Model2 should have {method_name} method"
            method = getattr(model2, method_name)
            assert callable(method), f"{method_name} should be callable"
        
        print("✓ All required abstract methods are implemented")
        return True
        
    except Exception as e:
        print(f"✗ Inheritance test failed: {e}")
        return False


def test_model2_building():
    """Test Model2 model building"""
    print("\n=== Testing Model2 Model Building ===")
    
    try:
        model2 = Model2()
        
        # Test unquantized model building
        print("Testing unquantized model building...")
        unquantized_model = model2.buildModel("unquantized")
        assert unquantized_model is not None, "Unquantized model should be built"
        assert model2.model is not None, "Model should be stored in self.model"
        print("✓ Unquantized model built successfully")
        
        # Check model architecture
        print(f"  Model name: {unquantized_model.name}")
        print(f"  Input layers: {len(unquantized_model.inputs)}")
        print(f"  Output shape: {unquantized_model.output_shape}")
        
        # Test quantized model building (if QKeras available)
        try:
            print("Testing quantized model building...")
            quantized_models = model2.buildModel("quantized", [(8, 0), (6, 0)])
            assert quantized_models is not None, "Quantized models should be built"
            print(f"✓ Quantized models built successfully ({len(quantized_models)} configurations)")
            
            for config_name, q_model in quantized_models.items():
                print(f"  {config_name}: {q_model.name}")
                
        except ImportError:
            print("  QKeras not available, skipping quantized model test")
        
        return True
        
    except Exception as e:
        print(f"✗ Model building test failed: {e}")
        return False


def test_model2_with_dummy_data():
    """Test Model2 with dummy data"""
    print("\n=== Testing Model2 with Dummy Data ===")
    
    try:
        model2 = Model2()
        
        # Build model
        model = model2.buildModel("unquantized")
        
        # Create dummy data matching Model2 input format
        batch_size = 4
        x_profile = np.random.random((batch_size, 21))
        z_global = np.random.random((batch_size, 1))
        y_profile = np.random.random((batch_size, 13))
        y_local = np.random.random((batch_size, 1))
        
        # Test forward pass
        print("Testing forward pass with dummy data...")
        predictions = model.predict([x_profile, z_global, y_profile, y_local])
        assert predictions.shape == (batch_size, 1), f"Expected output shape {(batch_size, 1)}, got {predictions.shape}"
        print("✓ Forward pass successful")
        print(f"  Input shapes: x_profile{x_profile.shape}, z_global{z_global.shape}, y_profile{y_profile.shape}, y_local{y_local.shape}")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Dummy data test failed: {e}")
        return False


def test_model2_hyperparameter_tuning():
    """Test Model2 hyperparameter tuning function"""
    print("\n=== Testing Model2 Hyperparameter Tuning Function ===")
    
    try:
        model2 = Model2()
        
        # Test hyperparameter tuning function (without actually running it)
        print("Testing hyperparameter tuning function creation...")
        import keras_tuner as kt
        
        # Create a mock hyperparameter object
        class MockHP:
            def Int(self, name, min_value, max_value, step=1):
                return np.random.randint(min_value, max_value + 1, step)
            
            def Float(self, name, min_value, max_value, sampling='uniform'):
                return np.random.uniform(min_value, max_value)
        
        mock_hp = MockHP()
        
        # Test the hyperparameter tuning function
        tuning_model = model2.makeUnquatizedModelHyperParameterTuning(mock_hp)
        assert tuning_model is not None, "Hyperparameter tuning model should be built"
        print("✓ Hyperparameter tuning function works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Hyperparameter tuning test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=== Model2 Implementation Tests ===")
    print("Testing Model2 implementation against SmartPixModel abstract base class...\n")
    
    tests = [
        test_model2_initialization,
        test_model2_inheritance,
        test_model2_building,
        test_model2_with_dummy_data,
        test_model2_hyperparameter_tuning
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Model2 implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run the training script: python train_model2.py")
        print("2. Try with real data by specifying --data-dir")
        print("3. Experiment with different model configurations")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

