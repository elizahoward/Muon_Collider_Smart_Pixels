# Complete HLS Synthesis Workflow Summary

## Overview

This document summarizes the complete workflow from hyperparameter tuning results to HLS synthesis-ready models.

---

## Scripts Available

### 1. `analyze_hyperparameter_complexity.py`
**Purpose**: Analyze model complexity from hyperparameter tuning  
**Input**: Hyperparameter search directory (Keras Tuner or flat format)  
**Output**: CSV files + plots in `complexity_analysis/` directory  
**Features**:
- Supports Model2, Model2.5, and Model3
- Calculates nodes and parameters
- Auto-detects model type
- Can extract from H5 files (with fallback)

### 2. `select_pareto_models.py`
**Purpose**: Select Pareto optimal models from complexity analysis  
**Input**: Complexity analysis CSV files  
**Output**: Pareto model CSVs + plots + optional H5 copies  
**Features**:
- Two-tier Pareto selection (primary + secondary)
- Works with parameters or nodes metrics
- Plots saved in input directory
- Model files saved in output directory

### 3. `analyze_and_select_pareto.py` ⭐ **NEW - RECOMMENDED**
**Purpose**: Combined analysis + selection in one step  
**Input**: Hyperparameter results directory (H5 files + JSON)  
**Output**: Single HLS-ready directory with everything  
**Features**:
- All-in-one solution
- Two-tier Pareto for both parameters AND nodes
- Union of all Pareto selections
- Everything organized for HLS synthesis

### 4. `parallel_hls_synthesis.py`
**Purpose**: Synthesize H5 models to HLS implementations  
**Input**: Directory with H5 model files  
**Output**: HLS projects + synthesis reports  
**Features**:
- Parallel processing (multiple workers)
- Automatic cleanup (keeps only essential files)
- Creates tarballs
- Comprehensive logging

---

## Workflow Options

### Option A: Step-by-Step (Original)

```bash
# Step 1: Analyze complexity
python analyze_hyperparameter_complexity.py \
    hyperparameter_tuning/model2_5_search

# Creates: complexity_analysis/model2_5_search/

# Step 2: Select Pareto models
python ericHLS/select_pareto_models.py \
    --input_dir complexity_analysis/model2_5_search \
    --output_dir pareto_models/model2_5 \
    --complexity_metric both \
    --models_dir hyperparameter_tuning/model2_5_search

# Creates: pareto_models/model2_5/ (with H5 files)

# Step 3: HLS synthesis
python ericHLS/parallel_hls_synthesis.py \
    --input_dir pareto_models/model2_5 \
    --num_workers 4
```

### Option B: Combined (Recommended) ⭐

```bash
# Step 1: Analyze + Select (all-in-one)
python ericHLS/analyze_and_select_pareto.py \
    --input_dir model2_5_quantized_4w0i_hyperparameter_results_20251205_174921 \
    --output_dir model2_5_pareto_hls_ready \
    --min_accuracy 0.80

# Creates: model2_5_pareto_hls_ready/ (HLS-ready!)

# Step 2: HLS synthesis
python ericHLS/parallel_hls_synthesis.py \
    --input_dir model2_5_pareto_hls_ready \
    --num_workers 4
```

**Advantages of Option B**:
- ✅ Fewer commands (1 instead of 2)
- ✅ No intermediate directories
- ✅ Everything in one HLS-ready folder
- ✅ Faster workflow

---

## Directory Structure

### After Hyperparameter Tuning

```
project/
└── model2_5_quantized_4w0i_hyperparameter_results_20251205_174921/
    ├── model_trial_001.h5
    ├── hyperparams_trial_001.json
    ├── model_trial_002.h5
    ├── hyperparams_trial_002.json
    ├── ...
    └── trials_summary.json
```

### After analyze_and_select_pareto.py

```
project/
├── model2_5_quantized_4w0i_hyperparameter_results_20251205_174921/
│   └── ... (original files)
│
└── model2_5_pareto_hls_ready/  ← NEW!
    ├── model_trial_003.h5  ← Selected Pareto models
    ├── model_trial_023.h5
    ├── model_trial_027.h5
    ├── ... (20-30 H5 files)
    │
    ├── complexity_vs_accuracy_parameters.png
    ├── complexity_vs_accuracy_nodes.png
    ├── pareto_front_parameters_combined.png
    ├── pareto_front_nodes_combined.png
    │
    ├── hyperparameter_complexity_summary.csv
    ├── hyperparameter_detailed_results.csv
    ├── pareto_optimal_models_parameters_primary.csv
    ├── pareto_optimal_models_parameters_secondary.csv
    ├── pareto_optimal_models_parameters_combined.csv
    ├── pareto_optimal_models_nodes_primary.csv
    ├── pareto_optimal_models_nodes_secondary.csv
    ├── pareto_optimal_models_nodes_combined.csv
    └── analysis_summary.json
```

### After parallel_hls_synthesis.py

```
project/
├── model2_5_quantized_4w0i_hyperparameter_results_20251205_174921/
│   └── ...
│
└── model2_5_pareto_hls_ready/
    ├── ... (H5 files and results)
    │
    └── hls_outputs/  ← NEW!
        ├── hls_model_trial_003/
        │   ├── project.tcl
        │   ├── vitis_hls.log
        │   └── vivado_synth.rpt
        ├── hls_model_trial_003.tar.gz
        ├── hls_model_trial_023/
        ├── hls_model_trial_023.tar.gz
        ├── ...
        └── synthesis_results.json
```

---

## Quick Reference

### For Model2.5

```bash
# Complete workflow (recommended)
python ericHLS/analyze_and_select_pareto.py \
    --input_dir model2.5_quantized_4w0i_hyperparameter_results_20251205_174921 \
    --output_dir model2_5_pareto_hls_ready

python ericHLS/parallel_hls_synthesis.py \
    --input_dir model2_5_pareto_hls_ready \
    --num_workers 4
```

### For Model2 or Model3

```bash
# Use step-by-step approach
python analyze_hyperparameter_complexity.py \
    hyperparameter_tuning/model2_search

python ericHLS/select_pareto_models.py \
    --input_dir complexity_analysis/model2_search \
    --output_dir model2_pareto \
    --complexity_metric both \
    --models_dir hyperparameter_tuning/model2_search

python ericHLS/parallel_hls_synthesis.py \
    --input_dir model2_pareto \
    --num_workers 4
```

---

## Key Features of Each Script

### analyze_hyperparameter_complexity.py

| Feature | Status |
|---------|--------|
| Auto-detect model type | ✅ |
| Model2 support | ✅ |
| Model2.5 support | ✅ |
| Model3 support | ✅ |
| Extract from H5 | ✅ (with fallback) |
| Keras Tuner format | ✅ |
| Flat format | ✅ |
| Generate plots | ✅ |
| Save CSV results | ✅ |

### select_pareto_models.py

| Feature | Status |
|---------|--------|
| Two-tier selection | ✅ |
| Parameters metric | ✅ |
| Nodes metric | ✅ |
| Both metrics | ✅ |
| Copy H5 files | ✅ |
| Generate plots | ✅ |
| Plots in input_dir | ✅ |
| Files in output_dir | ✅ |

### analyze_and_select_pareto.py

| Feature | Status |
|---------|--------|
| Combined workflow | ✅ |
| Model2.5 support | ✅ |
| Two-tier Pareto | ✅ |
| Both metrics | ✅ |
| Union selection | ✅ |
| HLS-ready output | ✅ |
| All-in-one directory | ✅ |
| Complete plots | ✅ (6 plots) |

### parallel_hls_synthesis.py

| Feature | Status |
|---------|--------|
| Parallel processing | ✅ |
| Configurable workers | ✅ |
| FPGA part selection | ✅ |
| Automatic cleanup | ✅ |
| Keep essential files | ✅ |
| Create tarballs | ✅ |
| Synthesis reports | ✅ |
| Error handling | ✅ |

---

## Typical Results

### Model2.5 Example (149 models)

**Input**: 149 hyperparameter trial models  
**After filtering** (min_accuracy=0.80): 149 models (all passed)  

**Pareto Selection**:
- Parameters (primary): 12 models
- Parameters (secondary): 12 models
- Nodes (primary): 12 models
- Nodes (secondary): 12 models
- **Total unique**: 24 models

**Output**:
- 24 H5 files
- 6 plots
- 8 CSV files
- 1 JSON summary

**HLS Synthesis** (24 models, 4 workers):
- Estimated time: 30-60 minutes per model
- Total time: ~6-15 hours (with 4 workers in parallel)

---

## Documentation Files

| File | Purpose |
|------|---------|
| `SELECT_PARETO_MODELS_README.md` | Full guide for Pareto selection |
| `QUICKSTART_PARETO.md` | Quick reference for Pareto selection |
| `COMPARISON_SELECTION_METHODS.md` | Pareto vs percentile comparison |
| `UPDATES_TWO_TIER_PARETO.md` | Two-tier changes documentation |
| `ANALYZE_AND_SELECT_PARETO_README.md` | Combined script guide |
| `WORKFLOW_SUMMARY.md` | This file |

---

## Best Practices

### 1. Start with Combined Script

For Model2.5, always use `analyze_and_select_pareto.py`:
- Simpler workflow
- Faster
- Better organized
- HLS-ready immediately

### 2. Filter by Accuracy

Use `--min_accuracy` to exclude poor models:
```bash
--min_accuracy 0.80  # For Model2.5
--min_accuracy 0.85  # For more selective
```

### 3. Check Outputs

Before HLS synthesis, review:
- Analysis summary JSON
- Pareto front plots
- CSV files with selected models

### 4. Parallel Workers

Choose workers based on your system:
```bash
--num_workers 4   # Good for most systems
--num_workers 8   # If you have many cores
--num_workers 2   # For limited resources
```

### 5. Save Space

HLS synthesis generates large files. The script:
- Keeps only essential files
- Creates tarballs for archiving
- Saves ~90% disk space

---

## Troubleshooting

### Issue: Script hangs during H5 loading

**Solution**: H5 loading warnings are normal. Script falls back to manual calculation.

### Issue: No Pareto models found

**Solution**: Check if models have varying complexity. All identical models won't have a Pareto front.

### Issue: HLS synthesis fails

**Solution**: 
1. Check FPGA part number
2. Verify QKeras is installed
3. Check HLS4ML version compatibility

### Issue: Out of memory during parallel synthesis

**Solution**: Reduce `--num_workers` to 2 or 1

---

## Version History

### v1.0 (December 2025)
- ✅ Initial release of combined workflow
- ✅ Two-tier Pareto selection
- ✅ Support for Model2.5
- ✅ H5 file extraction capability
- ✅ Parallel HLS synthesis

---

## Author

Eric - December 2025

---

## Quick Command Reference

```bash
# RECOMMENDED: Combined workflow for Model2.5
python ericHLS/analyze_and_select_pareto.py \
    --input_dir <hyperparam_results_dir> \
    --output_dir <hls_ready_dir> \
    --min_accuracy 0.80

# HLS synthesis
python ericHLS/parallel_hls_synthesis.py \
    --input_dir <hls_ready_dir> \
    --num_workers 4 \
    --fpga_part xc7z020clg400-1

# ALTERNATIVE: Step-by-step (for Model2/Model3)
python analyze_hyperparameter_complexity.py <search_dir>
python ericHLS/select_pareto_models.py \
    --input_dir complexity_analysis/<search_dir> \
    --output_dir <output_dir> \
    --complexity_metric both \
    --models_dir <search_dir>
python ericHLS/parallel_hls_synthesis.py \
    --input_dir <output_dir> \
    --num_workers 4
```

---

**End of Workflow Summary**

