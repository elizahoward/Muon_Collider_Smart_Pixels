# Quantized Model Loader

## Overview

`load_quantized_model.py` is a utility script for loading and inspecting quantized model h5 files from hyperparameter tuning results.

## Usage

### Basic Usage - Full Model Information

Load and display complete model information including architecture, hyperparameters, and inference test:

```bash
python load_quantized_model.py model2.5_quantized_4w0i_hyperparameter_results_20260119_181421/model_trial_1.h5
```

### Output Parameter Count Only

Get just the total parameter count (useful for scripting):

```bash
python load_quantized_model.py model_trial_1.h5 --params-only
```

Example output:
```
9961
```

### Load Without Inference Testing

Skip the inference test to load faster:

```bash
python load_quantized_model.py model_trial_1.h5 --no-test
```

## Features

- **Automatic Hyperparameter Loading**: Automatically finds and loads the corresponding `hyperparams_trial_X.json` file
- **QKeras Support**: Handles quantized layers (QDense, QActivation) with proper custom objects
- **Detailed Architecture Display**: Shows all layers with their quantization configurations
- **Parameter Counting**: Displays total, trainable, and non-trainable parameters
- **Inference Testing**: Tests the model with dummy inputs to verify it works
- **Clean Output Mode**: `--params-only` flag for scripting and automation

## Options

- `--no-test`: Skip inference testing (faster loading)
- `--compile`: Compile the model after loading (default: False)
- `--params-only`: Only output the total parameter count with no other information

## Requirements

- TensorFlow/Keras
- QKeras (for quantized models)
- NumPy

Activate the appropriate conda environment before running:
```bash
conda activate mlgpu_qkeras  # or mlproj_qkeras, qk-tf214-gpu, etc.
```

## Examples

### Check parameters for all models in a directory

```bash
for model in model2.5_quantized_4w0i_hyperparameter_results_20260119_181421/model_trial_*.h5; do
    params=$(python load_quantized_model.py $model --params-only 2>/dev/null)
    echo "$(basename $model): $params parameters"
done
```

Output:
```
model_trial_0.h5: 14857 parameters
model_trial_1.h5: 9961 parameters
model_trial_2.h5: 9961 parameters
model_trial_3.h5: 8329 parameters
```

### Use in Python scripts

```python
import subprocess

def get_model_params(model_path):
    result = subprocess.run(
        ['python', 'load_quantized_model.py', model_path, '--params-only'],
        capture_output=True,
        text=True
    )
    return int(result.stdout.strip())

params = get_model_params('model_trial_1.h5')
print(f"Model has {params} parameters")
```

## Model Information Displayed (Full Mode)

1. **Model Name**: The name of the loaded model
2. **Hyperparameters**: All hyperparameters from the JSON file
3. **Architecture Summary**: Keras model.summary() output
4. **Input/Output Shapes**: Shape information for all inputs and outputs
5. **Layer Details**: Detailed information for each layer including:
   - Layer type and name
   - Number of units
   - Quantization configurations (for QKeras layers)
   - Output shapes
6. **Parameter Counts**: Total, trainable, and non-trainable parameters
7. **Inference Test** (if not skipped): Test run with dummy data showing output ranges

## Notes

- The script automatically looks for `hyperparams_trial_X.json` in the same directory as the model file
- Models are loaded without compilation by default for faster loading and compatibility
- The `--params-only` mode suppresses all TensorFlow logging except errors
- Compatible with Model2.5 quantized models (4-bit, 8-bit, etc.)
