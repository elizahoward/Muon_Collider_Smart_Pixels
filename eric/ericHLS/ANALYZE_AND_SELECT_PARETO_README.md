# Combined Analysis and Pareto Selection for HLS

## Overview

The `analyze_and_select_pareto.py` script is a **one-stop solution** that combines hyperparameter complexity analysis with Pareto optimal model selection, preparing models for HLS synthesis in a single step.

### What It Does

1. **Analyzes** hyperparameter tuning results (complexity vs accuracy)
2. **Plots** complexity visualizations (parameters and nodes vs accuracy)
3. **Selects** Pareto optimal models (two-tier with redundancy)
4. **Copies** selected H5 model files to output directory
5. **Organizes** all plots, CSVs, and models in one HLS-ready folder

### Why Use This Script?

**Before**: Multiple manual steps
```bash
python analyze_hyperparameter_complexity.py input_dir
python select_pareto_models.py --input_dir complexity_analysis/... --output_dir ...
# Manually copy H5 files
# Manually organize plots
```

**Now**: Single command
```bash
python analyze_and_select_pareto.py --input_dir input_dir --output_dir output_dir
# Done! Ready for HLS synthesis
```

---

## Features

✅ **Integrated workflow** - Analyzes + selects + prepares in one command  
✅ **Two-tier Pareto selection** - Primary + secondary for redundancy  
✅ **Both complexity metrics** - Analyzes parameters AND nodes  
✅ **HLS-ready output** - All files organized for `parallel_hls_synthesis.py`  
✅ **Comprehensive plots** - 6 visualizations (complexity + Pareto fronts)  
✅ **Detailed summaries** - CSV, JSON with complete statistics  
✅ **Smart deduplication** - Union of parameter and node Pareto selections  

---

## Requirements

```bash
pip install numpy pandas matplotlib tensorflow
```

---

## Usage

### Basic Usage

```bash
python ericHLS/analyze_and_select_pareto.py \
    --input_dir model2.5_quantized_4w0i_hyperparameter_results_20251205_174921 \
    --output_dir model2_5_pareto_hls_ready
```

### With Minimum Accuracy Filter

```bash
python ericHLS/analyze_and_select_pareto.py \
    --input_dir model2.5_results \
    --output_dir model2_5_pareto \
    --min_accuracy 0.85
```

### Complete Workflow

```bash
# Step 1: Run this script
python ericHLS/analyze_and_select_pareto.py \
    --input_dir model2_5_quantized_4w0i_hyperparameter_results_20251205_174921 \
    --output_dir model2_5_pareto_hls_ready

# Step 2: HLS synthesis (ready to go!)
python ericHLS/parallel_hls_synthesis.py \
    --input_dir model2_5_pareto_hls_ready \
    --num_workers 4
```

---

## Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input_dir` | str | Yes | Directory with hyperparameter results (H5 files + JSON) |
| `--output_dir` | str | Yes | Output directory for HLS-ready files |
| `--min_accuracy` | float | No | Minimum accuracy threshold (default: 0.55) |

---

## Input Format

The script expects a directory with:

```
input_dir/
├── model_trial_001.h5
├── hyperparams_trial_001.json
├── model_trial_002.h5
├── hyperparams_trial_002.json
├── ...
└── trials_summary.json
```

This is the standard format from hyperparameter tuning scripts.

---

## Output Structure

```
output_dir/
├── H5 MODEL FILES (Pareto optimal, ready for HLS)
│   ├── model_trial_003.h5
│   ├── model_trial_023.h5
│   ├── model_trial_027.h5
│   └── ... (typically 20-30 models)
│
├── COMPLEXITY ANALYSIS PLOTS
│   ├── complexity_vs_accuracy_parameters.png
│   └── complexity_vs_accuracy_nodes.png
│
├── PARETO FRONT PLOTS (Two-Tier)
│   ├── pareto_front_parameters_combined.png
│   └── pareto_front_nodes_combined.png
│
├── PARETO SELECTION RESULTS
│   ├── pareto_optimal_models_parameters_primary.csv
│   ├── pareto_optimal_models_parameters_secondary.csv
│   ├── pareto_optimal_models_parameters_combined.csv
│   ├── pareto_optimal_models_nodes_primary.csv
│   ├── pareto_optimal_models_nodes_secondary.csv
│   └── pareto_optimal_models_nodes_combined.csv
│
├── COMPLEXITY ANALYSIS RESULTS
│   ├── hyperparameter_complexity_summary.csv
│   └── hyperparameter_detailed_results.csv
│
└── SUMMARY
    └── analysis_summary.json
```

---

## Example Run

### Command

```bash
python ericHLS/analyze_and_select_pareto.py \
    --input_dir ../model2.5_quantized_4w0i_hyperparameter_results_20251205_174921 \
    --output_dir ../model2_5_pareto_hls_ready \
    --min_accuracy 0.80
```

### Output

```
================================================================================
COMBINED ANALYSIS AND PARETO SELECTION FOR HLS SYNTHESIS
================================================================================

Input directory: model2.5_quantized_4w0i_hyperparameter_results_20251205_174921
Output directory: model2_5_pareto_hls_ready
Model name: model2.5_quantized_4w0i_hyperparameter_results_20251205_174921
Minimum accuracy: 0.8

================================================================================
STEP 1: COMPLEXITY ANALYSIS
================================================================================

Found 149 H5 models
Analyzing complexity...
  Auto-detected model type: MODEL2.5

  Completed trials: 149
  Valid trials after filtering: 149
  Nodes range: 85 - 285
  Parameters range: 2889 - 15129
  Accuracy range: 0.8072 - 0.9077

================================================================================
STEP 2: GENERATING PLOTS
================================================================================

Creating complexity vs accuracy plots...
  ✓ Saved: complexity_vs_accuracy_parameters.png
  ✓ Saved: complexity_vs_accuracy_nodes.png

================================================================================
PARETO SELECTION: PARAMETERS
================================================================================

--- Tier 1: Primary Pareto Front (parameters) ---
  ✓ Found 12 primary Pareto optimal models

--- Tier 2: Secondary Pareto Front (Redundancy) ---
  ✓ Found 12 secondary Pareto optimal models

================================================================================
PARETO SELECTION: NODES
================================================================================

--- Tier 1: Primary Pareto Front (nodes) ---
  ✓ Found 12 primary Pareto optimal models

--- Tier 2: Secondary Pareto Front (Redundancy) ---
  ✓ Found 12 secondary Pareto optimal models

================================================================================
COMBINING PARETO SELECTIONS
================================================================================

Total unique Pareto models: 24
  From parameters (primary): 12
  From parameters (secondary): 12
  From nodes (primary): 12
  From nodes (secondary): 12

================================================================================
STEP 3: COPYING MODEL FILES
================================================================================

Copying 24 primary Pareto models...
  ✓ model_trial_003.h5
  ✓ model_trial_023.h5
  ✓ model_trial_027.h5
  ... (21 more files)

Total copied: 24 model files

================================================================================
COMPLETE - HLS SYNTHESIS READY!
================================================================================

Output directory: model2_5_pareto_hls_ready

Contents:
  - 24 H5 model files (Pareto optimal)
  - 4 complexity analysis plots
  - 2 Pareto front plots (parameters + nodes)
  - Multiple CSV files with results
  - JSON summary file

Ready for HLS synthesis! Run:
  python parallel_hls_synthesis.py \
      --input_dir model2_5_pareto_hls_ready \
      --num_workers 4
```

---

## Understanding the Results

### Pareto Selection Logic

The script performs **two-tier Pareto selection** for both parameters and nodes:

1. **Primary Pareto Front**: Optimal accuracy/complexity trade-offs
2. **Secondary Pareto Front**: Next-best alternatives (redundancy/backup)

Then it takes the **union** of all selected models:
- Models optimal for parameters
- Models optimal for nodes  
- Both primary and secondary tiers

This ensures you get a diverse set covering all trade-off perspectives.

### Example: Model Selection

**Scenario**: 149 models analyzed

**Parameters Pareto**:
- Primary: 12 models (e.g., trials 3, 23, 27, 32, 64, 71, 81, 87, 108, 110, 134, 140)
- Secondary: 12 models (e.g., trials 22, 26, 42, 68, 70, 72, 79, 92, 104, 116, 118, 144)

**Nodes Pareto**:
- Primary: 12 models (similar but may differ slightly)
- Secondary: 12 models

**Union**: 24 unique models (some overlap between parameters and nodes)

### Why Union?

Models can be Pareto optimal for:
- **Parameters only**: Efficient in parameter count
- **Nodes only**: Efficient in node count
- **Both**: Optimal in both metrics (most efficient)

Taking the union ensures we don't miss any efficient model!

---

## Analysis Summary JSON

```json
{
  "timestamp": "2025-12-07T17:21:58.895495",
  "input_directory": "model2_5",
  "total_models": 149,
  "primary_pareto_parameters": 12,
  "secondary_pareto_parameters": 12,
  "primary_pareto_nodes": 12,
  "secondary_pareto_nodes": 12,
  "accuracy_range": {
    "min": 0.8072,
    "max": 0.9077,
    "mean": 0.8928
  },
  "parameters_range": {
    "min": 2889,
    "max": 15129,
    "mean": 9407.87
  },
  "nodes_range": {
    "min": 85,
    "max": 285,
    "mean": 191.47
  }
}
```

---

## Plots Generated

### 1. Complexity vs Accuracy (2 plots)

Standard scatter plots showing all models:
- `complexity_vs_accuracy_parameters.png`
- `complexity_vs_accuracy_nodes.png`

### 2. Pareto Front Visualizations (2 plots)

Two-tier Pareto fronts:
- **Gray dots**: All models
- **Red diamonds**: Primary Pareto optimal
- **Orange squares**: Secondary Pareto optimal (redundancy)
- **Lines**: Connecting Pareto frontiers
- **Labels**: Trial IDs

Files:
- `pareto_front_parameters_combined.png`
- `pareto_front_nodes_combined.png`

---

## Next Steps: HLS Synthesis

After running this script, your output directory is ready for HLS synthesis:

```bash
python ericHLS/parallel_hls_synthesis.py \
    --input_dir model2_5_pareto_hls_ready \
    --num_workers 4 \
    --fpga_part xc7z020clg400-1
```

The `parallel_hls_synthesis.py` script will:
1. Find all H5 files in the directory
2. Synthesize them to HLS implementations in parallel
3. Generate synthesis reports
4. Create tarballs of results

---

## Comparison with Individual Scripts

### analyze_hyperparameter_complexity.py

**What it does**: Analyzes complexity, generates plots and CSVs  
**Output**: Separate `complexity_analysis` directory

### select_pareto_models.py

**What it does**: Reads CSVs, selects Pareto models, copies files  
**Input**: Complexity analysis results  
**Output**: Separate output directory

### analyze_and_select_pareto.py (This Script)

**What it does**: **BOTH** - analysis + selection + organization  
**Input**: Hyperparameter results directory (H5 files)  
**Output**: Single HLS-ready directory with everything

**Advantages**:
- ✅ One command instead of two
- ✅ No intermediate directories needed
- ✅ All outputs in one place
- ✅ Ready for HLS synthesis immediately
- ✅ Cleaner workflow

---

## Troubleshooting

### No H5 files found

**Error**: `No H5 model files found`

**Solution**: Make sure input directory contains `model_trial_*.h5` files

### Model type not detected

**Error**: `Cannot detect model type from hyperparameters`

**Solution**: Currently supports Model2.5. For other models, use the individual scripts.

### Missing trials_summary.json

**Error**: Cannot extract validation accuracy

**Solution**: Ensure `trials_summary.json` exists in input directory

---

## Supported Model Types

Currently supports:
- ✅ Model2.5

To add support for Model2 or Model3, you would need to add their complexity calculation functions to the script.

---

## Performance

**Typical runtime** (149 models):
- Complexity analysis: ~10 seconds
- Pareto selection: <1 second
- File copying: <1 second
- Plotting: ~5 seconds

**Total**: ~15-20 seconds

---

## Files Overview

| File Type | Purpose | Count |
|-----------|---------|-------|
| H5 models | Selected Pareto optimal models for HLS | 20-30 |
| PNG plots | Visualizations (complexity + Pareto) | 4 |
| CSV files | Results (complexity + Pareto selections) | 8 |
| JSON file | Analysis summary | 1 |

**Total output**: ~35-40 files, all organized and HLS-ready!

---

## Author

Eric - December 2025

## See Also

- `analyze_hyperparameter_complexity.py` - Standalone complexity analysis
- `select_pareto_models.py` - Standalone Pareto selection
- `parallel_hls_synthesis.py` - Next step: HLS synthesis
- `COMPARISON_SELECTION_METHODS.md` - Pareto vs percentile methods






