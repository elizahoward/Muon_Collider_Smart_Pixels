# Archived Scripts

These scripts are no longer actively used. The current active workflow lives in the parent `eric/` directory and centers on:
- `run_quantized_inputs_hyperparam_tuning_model3.py` (+ `model3.py`, `model3_quantized_inputs.py`)
- `run_quantized_inputs_hyperparam_tuning_model2_5.py` (+ `model2.py`, `model2_5.py`, `model2_5_quantized_inputs.py`)

---

## Model Definitions

### `model2_5_standalone.py`
Standalone re-implementation of Model2.5 that inherits directly from `SmartPixModel` instead of `Model2`. Uses the older `OptimizedDataGenerator4_data_shuffled_bigData` data loader. Predates the refactored `model2_5.py`.

### `make_small_model3.py`
One-off script to build and train (30 epochs) a small, HLS-friendly Model3 variant with hardcoded architecture: Conv2D(4 filters, 3×3) → MaxPool → Flatten → 240 nodes, scalar branch 3→8, head 248→32→16→1. Outputs `model3_small_6w0i.h5`. Uses `OptimizedDataGenerator4_data_shuffled_bigData_NewFormat`.

---

## Hyperparameter Tuning Runners

### `hyperparameter_tuning_model2.py`
Simple wrapper that calls Model2's built-in HP tuning method. Hardcoded data path, no CLI args. Early-stage script before the parameterized `run_quantized_*` runners existed.

### `hyperparameter_tuning_model3.py`
Same as above but for Model3. Hardcoded data path, no CLI args. Superseded by `run_quantized_hyperparam_tuning_model3.py`.

### `run_quantized_hyperparam_tuning_model2_5.py`
Runs Keras Tuner RandomSearch for Model2.5 with quantized weights. Searches layer sizes, dropout, and learning rate. No input quantization — superseded by `run_quantized_inputs_hyperparam_tuning_model2_5.py`.

### `run_quantized_hyperparam_tuning_model3.py`
Same workflow as above but for Model3. Objective: weighted background rejection (0.3×BR95 + 0.6×BR98 + 0.1×BR99), 160 max trials, 20 epochs/trial. No input quantization — superseded by `run_quantized_inputs_hyperparam_tuning_model3.py`.

---

## Pipeline & Analysis Tools

### `run_full_quantized_hls_pipeline.py`
End-to-end automation: HP tuning → ROC Pareto selection → parallel HLS synthesis → resource utilization analysis. Switches conda envs between steps (`mlgpu_qkeras` for training, `newHLSEnviro` for HLS). Invokes the existing individual scripts rather than reimplementing them.

### `run_input_quantization_sweep_model2_5.py`
Loads a saved Model2.5 H5, reconstructs its architecture, then retrains one fully-quantized model per input-bit width in [2, 3, 4, 6, 8, 12, 16] for 80 epochs each. Saves ROC curves and background rejection metrics per width.

### `analyze_hyperparameter_complexity.py`
Parses a Keras Tuner results directory and plots model complexity (total node count) vs. validation accuracy. Supports Model2, Model2.5, and Model3 with auto-detection.

### `add_metadata_to_trials.py`
Post-processing utility: enriches an existing `trials_summary.json` with validation accuracy, training metrics, parameter counts, and layer structure pulled from Keras Tuner trial directories.

### `count_model_parameters.py`
Simple CLI tool: load a `.h5` model and print total / trainable / non-trainable parameter counts. Handles QKeras custom objects.

### `plot_z_global_histogram.py`
Reads signal parquet files from a data directory and plots a histogram of the `z_global` distribution. One-off diagnostic/exploration script.

---

## HLS Testing

### `hls_trail.py`
Early HLS exploration script. Loads a Model2.5 H5 via `hls4ml`, runs synthesis, and checks GPU availability. Uses the legacy `OptimizedDataGenerator4` data loader (only script that still imports it). Superseded by the `ericHLS/` subdirectory pipeline.

### `train_hls_test.py`
Trains a minimal 2-epoch Model2.5 on CPU and saves an H5 for quick HLS synthesis smoke-testing. Hardcoded 4-bit config, outputs `model2_5_4w0i_hls_test.h5`.

---

## Data Generator

### `OptimizedDataGenerator4.py`
Legacy TFRecord/parquet data generator. Implements `OptimizedDataGenerator` (a `tf.keras.utils.Sequence`) with QKeras-based quantization preprocessing. Only ever imported by `hls_trail.py` in this directory. The production data loading is now handled by `OptimizedDataGenerator4_data_shuffled_bigData_NewFormat` in `MuC_Smartpix_Data_Production/tfRecords/`.
