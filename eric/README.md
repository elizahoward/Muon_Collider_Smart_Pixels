# Model2 Implementation using SmartPixModel Abstract Base Class

This folder contains a complete implementation of Model2 that properly inherits from the `SmartPixModel` abstract base class, demonstrating how to create concrete model implementations following the established framework.

## Files

- **`model2.py`** - Main Model2 class implementation inheriting from SmartPixModel
- **`train_model2.py`** - Training script demonstrating usage of the Model2 class
- **`test_model2.py`** - Test script to verify the implementation works correctly
- **`README.md`** - This documentation file

## Model2 Architecture

Model2 is a CNN-based architecture designed for smart pixel detector classification:

### Input Features
- **x_profile**: 21-dimensional profile data
- **z_global**: 1-dimensional global z coordinate  
- **y_profile**: 13-dimensional profile data
- **y_local**: 1-dimensional local y coordinate

### Architecture
```
x_profile (21) + z_global (1) ──┐
                                ├── Concatenate ──► Dense(32) ──┐
y_profile (13) + y_local (1) ──┘                               │
                                                                ├── Merge ──► Dense(128) ──► Dense(64) ──► Dense(32) ──► Dense(1)
                                                                │
                                                               ──┘
```

## Usage

### 1. Test the Implementation

First, run the test script to ensure everything works:

```bash
cd /home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric
python test_model2.py
```

### 2. Train Model2

#### Basic Training (Unquantized Model)
```bash
python train_model2.py --model-type unquantized --epochs 50
```

#### Train with Quantized Models
```bash
python train_model2.py --model-type quantized --bit-configs 8_0 6_0 4_0 --epochs 50
```

#### Train Both Unquantized and Quantized
```bash
python train_model2.py --model-type both --epochs 50 --save-plots
```

#### Hyperparameter Tuning
```bash
python train_model2.py --hyperparameter-tuning --max-trials 20 --model-type unquantized
```

### 3. Command Line Options

```bash
python train_model2.py --help
```

Key options:
- `--data-dir`: Path to TFRecords directory
- `--model-type`: Choose from 'unquantized', 'quantized', or 'both'
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate for optimizer
- `--bit-configs`: Bit configurations for quantized models (format: weight_bits_int_bits)
- `--hyperparameter-tuning`: Run hyperparameter optimization
- `--save-plots`: Save training plots
- `--output-dir`: Custom output directory

## Key Features

### 1. Proper Inheritance from SmartPixModel
- Implements all required abstract methods
- Follows the established framework pattern
- Compatible with the existing codebase structure

### 2. Complete Implementation
- ✅ `makeUnquantizedModel()` - Build standard neural network
- ✅ `makeQuantizedModel()` - Build quantized versions using QKeras
- ✅ `makeUnquatizedModelHyperParameterTuning()` - Hyperparameter optimization
- ✅ `buildModel()` - Model construction
- ✅ `runHyperparameterTuning()` - Automated hyperparameter search
- ✅ `trainModel()` - Training logic with callbacks
- ✅ `evaluate()` - Model evaluation with ROC metrics
- ✅ `plotModel()` - Visualization of results
- ✅ `loadTfRecords()` - Data loading with OptimizedDataGenerator4

### 3. Advanced Features
- **Multiple Quantization Configurations**: Test different bit precisions
- **Hyperparameter Tuning**: Automated optimization using Keras Tuner
- **Comprehensive Evaluation**: ROC curves, accuracy, loss metrics
- **Flexible Training**: Early stopping, model checkpointing
- **Organized Output**: Timestamped results with detailed logging

## Example Usage in Code

```python
from model2 import Model2

# Initialize Model2
model2 = Model2(
    tfRecordFolder="/path/to/tfrecords/",
    dropout_rate=0.1
)

# Load data
model2.loadTfRecords()

# Build and train unquantized model
model2.buildModel("unquantized")
history = model2.trainModel(epochs=50)

# Evaluate
results = model2.evaluate()

# Plot results
model2.plotModel(save_plots=True)

# Save model
model2.saveModel()
```

## Integration with Existing Codebase

This implementation demonstrates how Eric's training models should be refactored to properly use the abstract base class:

1. **Inherit from SmartPixModel** instead of creating standalone classes
2. **Implement all abstract methods** to ensure consistency
3. **Use the shared data pipeline** with OptimizedDataGenerator4
4. **Follow the established interface** for model building, training, and evaluation

## Dependencies

- TensorFlow 2.x
- QKeras (for quantized models)
- Keras Tuner (for hyperparameter optimization)
- NumPy, Matplotlib, Scikit-learn
- OptimizedDataGenerator4 (from ryan folder)

## Notes

- The implementation includes proper error handling and validation
- All methods follow the abstract base class interface
- Results are automatically saved with timestamps
- The code is well-documented and follows Python best practices
- Both unquantized and quantized models are supported

This serves as a template for implementing other models (Model1, Model3, etc.) using the same pattern.

