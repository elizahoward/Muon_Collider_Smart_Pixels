# Streamlined HLS Workflow (Updated Jan 2026)

## Quick Start - Two Simple Steps

### Step 1: Select Pareto Optimal Models

```bash
python analyze_and_select_pareto.py \
    --input_dir ../model2.5_quantized_4w0i_hyperparameter_results_20260119_181421 \
    --output_dir ../model2_5_pareto_hls_ready \
    --min_accuracy 0.80
```

**What it does:**
- Loads each H5 model file directly (model-agnostic!)
- Extracts actual parameter counts from the models
- Performs two-tier Pareto selection (primary + secondary for redundancy)
- Copies selected models to output directory
- Generates plots and CSV summaries

**Output:**
```
model2_5_pareto_hls_ready/
├── model_trial_*.h5                               (Pareto optimal models)
├── complexity_vs_accuracy_parameters.png          (scatter plot)
├── pareto_front_parameters_combined.png           (Pareto visualization)
├── pareto_optimal_models_parameters_primary.csv   (primary Pareto)
├── pareto_optimal_models_parameters_secondary.csv (secondary Pareto)
├── pareto_optimal_models_parameters_combined.csv  (all selected)
├── hyperparameter_complexity_summary.csv          (all trials)
├── hyperparameter_detailed_results.csv            (with hyperparameters)
└── analysis_summary.json                          (metadata)
```

### Step 2: Run HLS Synthesis

```bash
python parallel_hls_synthesis.py \
    --input_dir ../model2_5_pareto_hls_ready \
    --num_workers 4
```

**What it does:**
- Converts selected models to HLS implementations using hls4ml
- Runs Vitis HLS synthesis in parallel
- Automatically cleans up large intermediate files
- Creates tarballs of synthesis results
- Generates synthesis summary

**Output:**
```
model2_5_pareto_hls_ready/hls_outputs/
├── hls_model_trial_*/               (HLS projects)
├── hls_model_trial_*.tar.gz         (compressed results)
└── synthesis_results.json           (synthesis summary)
```

---

## Optional Post-Processing

### Analyze HLS Results (Comprehensive)

After HLS synthesis, analyze results with comprehensive statistics and visualization:

```bash
python analyze_hls_results.py \
    --results_dir ../model2_5_pareto_hls_ready/hls_outputs \
    --plot
```

**What it does:**
- Extracts all resource utilization (LUTs, FFs, BRAM, DSP, Fmax)
- Links to validation accuracy data
- Generates comprehensive statistics
- Creates visualization plot with FPGA constraints
- Ranks models by accuracy, size, and speed

**Outputs:**
- `resource_utilization.csv` - Complete data table
- `resource_utilization.png` - Visualization (if --plot used)
- Terminal summary with statistics and rankings

---

## Key Features of Updated Workflow

### 🎯 Model-Agnostic Parameter Extraction
- No longer hardcoded for Model2.5
- Works with any model architecture (Model2, Model2.5, Model3, future models)
- Loads actual H5 files and extracts real parameter counts
- More accurate than manual calculations

### 📊 Automatic File Format Detection
- Handles any trial naming format:
  - Single digit: `model_trial_0.h5`
  - Two digit: `model_trial_01.h5`
  - Three digit: `model_trial_001.h5`
- Automatically adapts based on what files exist

### 🚀 Simplified Workflow
- Only 2 scripts needed for complete workflow
- No intermediate directories or manual file copying
- Everything organized and ready for HLS synthesis

---

## Scripts Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `analyze_and_select_pareto.py` | Select Pareto models | Hyperparameter results directory | HLS-ready directory with models |
| `parallel_hls_synthesis.py` | HLS synthesis | Directory with H5 files | HLS projects + reports |
| `analyze_hls_results.py` | Comprehensive analysis | HLS output directory | CSV + statistics + plots |

---

## Complete Example

```bash
# 1. Select best models (Pareto optimal)
python analyze_and_select_pareto.py \
    --input_dir ../model2.5_quantized_4w0i_hyperparameter_results_20260119_181421 \
    --output_dir ../model2_5_pareto_hls_ready \
    --min_accuracy 0.80

# Output: Selected 5 Pareto optimal models
#   - 3 primary Pareto models
#   - 2 secondary Pareto models (redundancy)

# 2. Run HLS synthesis on selected models
python parallel_hls_synthesis.py \
    --input_dir ../model2_5_pareto_hls_ready \
    --num_workers 4

# Synthesis runs for ~2-5 minutes per model

# 3. Analyze results (extract metrics, statistics, and visualization)
python analyze_hls_results.py \
    --results_dir ../model2_5_pareto_hls_ready/hls_outputs \
    --plot

# Done! You now have:
#   ✓ Pareto optimal models
#   ✓ HLS implementations
#   ✓ Resource utilization CSV with all metrics
#   ✓ Visualization plot with FPGA constraints
#   ✓ Comprehensive statistics and rankings
```

---

## Archived Scripts

Old scripts have been moved to `_archived_scripts/` for reference:
- `select_pareto_models.py` - Old standalone Pareto selection
- `plot_and_select_models.py` - Old plotting utility
- `select_top_models.py` - Simple top-N selection
- Outdated documentation files

These are kept for historical reference but are no longer needed for the workflow.

---

## Requirements

```bash
# Python environment
conda activate mlgpu_qkeras  # or mlproj_qkeras, qk-tf214-gpu

# Python packages
pip install tensorflow qkeras hls4ml pandas matplotlib numpy

# System requirements
Xilinx Vitis HLS 2024.1 or later
```

---

## Troubleshooting

### Issue: "TensorFlow not available"
**Solution:** Activate the correct conda environment with TensorFlow and QKeras

### Issue: "No valid trials found"
**Solution:** Check that your input directory contains:
- `model_trial_*.h5` files
- `hyperparams_trial_*.json` files
- `trials_summary.json` file

### Issue: "Could not load model"
**Solution:** Ensure QKeras is installed and models were saved correctly

### Issue: Synthesis failures
**Solution:** Check individual log files in `hls_outputs/hls_model_trial_*/vitis_hls.log`

---

## Author

Eric  
Updated: January 2026
