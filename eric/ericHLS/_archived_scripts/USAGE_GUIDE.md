# HLS Synthesis Workflow - Complete Usage Guide

This guide shows you how to use all the tools in the `ericHLS` directory for efficient HLS synthesis of neural network models.

## Overview of Tools

1. **`select_top_models.py`** - Filter best models from hyperparameter search
2. **`parallel_hls_synthesis.py`** - Synthesize multiple models in parallel  
3. **`check_progress.py`** - Monitor synthesis progress
4. **`analyze_synthesis_results.py`** - Extract resource utilization data

## Complete Workflow

### Step 1: Select Top Models

Filter your hyperparameter search to get only the best models (e.g., top 25%):

```bash
cd /home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/ericHLS

# Select top 25% by validation accuracy
python select_top_models.py \
    --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \
    --search_dir ../hyperparameter_tuning/model2_quantized_4w0i_hyperparameter_search \
    --output_dir ../model2_top_25_for_hls

# Or select specific number (e.g., top 10)
python select_top_models.py \
    --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \
    --search_dir ../hyperparameter_tuning/model2_quantized_4w0i_hyperparameter_search \
    --output_dir ../model2_top_10_for_hls \
    --top_n 10
```

**Arguments:**
- `--input_dir`: Directory with H5 model files
- `--search_dir`: Keras Tuner directory with trial_XXX folders (contains validation accuracies)
- `--output_dir`: Where to copy selected models
- `--percentile`: Threshold percentile (default: 75 for top 25%)
- `--top_n`: Select exactly N top models instead of using percentile
- `--dry_run`: Preview what would be selected without copying

### Step 2: Run Parallel HLS Synthesis

Synthesize the selected models in parallel:

```bash
# Basic synthesis with 4 workers
python parallel_hls_synthesis.py \
    --input_dir ../model2_top_25_for_hls \
    --num_workers 4

# With custom output directory
python parallel_hls_synthesis.py \
    --input_dir ../model2_top_25_for_hls \
    --output_dir ../hls_synthesis_results \
    --num_workers 4

# Different FPGA target
python parallel_hls_synthesis.py \
    --input_dir ../model2_top_25_for_hls \
    --num_workers 4 \
    --fpga_part xcu250-figd2104-2L-e
```

**Arguments:**
- `--input_dir`: Directory with H5 files to synthesize
- `--output_dir`: Base directory for HLS outputs (default: input_dir/hls_outputs)
- `--num_workers`: Number of parallel synthesis jobs (default: 4)
- `--fpga_part`: FPGA part number (default: xc7z020clg400-1)
- `--no_tarball`: Skip creating compressed tarballs
- `--limit`: Process only first N models (for testing)

### Step 3: Monitor Progress

While synthesis is running, monitor in a separate terminal:

```bash
# One-time check
python check_progress.py \
    --results_dir ../model2_top_25_for_hls/hls_outputs

# Continuous monitoring (updates every 30 seconds)
python check_progress.py \
    --results_dir ../model2_top_25_for_hls/hls_outputs \
    --watch \
    --interval 30
```

**Progress output shows:**
- Total/completed/failed counts
- Progress bar
- Recent completions
- Failed models with error messages

### Step 4: Analyze Results

After synthesis completes, extract resource utilization:

```bash
python analyze_synthesis_results.py \
    --results_dir ../model2_top_25_for_hls/hls_outputs
```

**This creates:**
- `resource_analysis.csv` - Detailed resource usage for all models
- Console output with statistics:
  - LUTs, Registers, BRAM, DSP usage
  - Min/Max/Mean/Median values
  - Top models by size and speed

## Quick Start Example

Complete workflow from start to finish:

```bash
cd /home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/ericHLS

# Step 1: Select top 10 models
python select_top_models.py \
    --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \
    --search_dir ../hyperparameter_tuning/model2_quantized_4w0i_hyperparameter_search \
    --output_dir ../model2_top_10_for_hls \
    --top_n 10

# Step 2: Synthesize them (2 at a time for testing)
python parallel_hls_synthesis.py \
    --input_dir ../model2_top_10_for_hls \
    --num_workers 2

# Step 3: Check progress (in another terminal)
python check_progress.py \
    --results_dir ../model2_top_10_for_hls/hls_outputs \
    --watch

# Step 4: Analyze results (after completion)
python analyze_synthesis_results.py \
    --results_dir ../model2_top_10_for_hls/hls_outputs
```

## Understanding the Output

### After select_top_models.py

```
model2_top_10_for_hls/
├── model_trial_033.h5
├── model_trial_128.h5
├── model_trial_047.h5
├── ...
├── hyperparams_trial_033.json
├── hyperparams_trial_128.json
├── ...
└── filtered_trials_summary.json
```

### After parallel_hls_synthesis.py

```
model2_top_10_for_hls/hls_outputs/
├── hls_model_trial_033/
│   ├── project.tcl          (~4 KB)
│   ├── vitis_hls.log        (~450 KB)
│   ├── vivado_synth.rpt     (~12 KB) ⭐ Resource usage here!
│   ├── vivado_synth.tcl     (~4 KB)
│   ├── vivado.jou           (~4 KB)
│   └── vivado.log           (~340 KB)
├── hls_model_trial_033.tar.gz  (~46-423 KB)
├── hls_model_trial_128/
├── hls_model_trial_128.tar.gz
├── ...
├── synthesis_results.json     ⭐ Overall status
└── resource_analysis.csv      ⭐ (after analysis step)
```

## Tips and Best Practices

### Worker Count
- **Start small**: Use 2 workers for testing
- **Scale up**: 4-8 workers for production runs
- **Consider resources**: Each worker uses 2-4 GB RAM and significant CPU

### Disk Space
- **Original HLS project**: ~215 MB per model
- **After cleanup**: ~800 KB per model (uncompressed)
- **Compressed**: ~46-423 KB per model
- **Space savings**: 99.6%!

### Synthesis Time
- **Per model**: 2-5 minutes typically
- **10 models with 2 workers**: ~15-25 minutes
- **100 models with 8 workers**: ~2-6 hours

### Error Handling
- Check `synthesis_results.json` for failed models
- Individual logs in each `hls_model_trial_XXX/` directory
- Failed models don't stop other syntheses

## Troubleshooting

### Issue: "No validation accuracies found"
**Solution**: Make sure to provide `--search_dir` pointing to the Keras Tuner directory (the one with `trial_000`, `trial_001`, etc. folders)

### Issue: Synthesis failures
**Solution**: 
1. Check individual `vitis_hls.log` files
2. Review `synthesis_results.json` for error messages
3. Some models may have unsupported layers

### Issue: Out of disk space
**Solution**:
1. Reduce number of workers
2. Process in smaller batches using `--limit`
3. Check you have at least 10 GB free per worker

### Issue: Slow synthesis
**Solution**:
1. Reduce worker count if CPU is maxed out
2. Check if system is swapping (reduce workers if so)
3. Each synthesis is independent - no benefit beyond available cores

## Advanced Usage

### Process Specific Trials Only

```bash
# Copy specific trials manually to a folder, then synthesize
mkdir ../specific_models
cp ../model2_top_25_for_hls/model_trial_{033,128,047}.h5 ../specific_models/
python parallel_hls_synthesis.py --input_dir ../specific_models --num_workers 2
```

### Different FPGA Targets

```bash
# For larger FPGAs
python parallel_hls_synthesis.py \
    --input_dir ../model2_top_10_for_hls \
    --num_workers 4 \
    --fpga_part xcu250-figd2104-2L-e
```

### Skip Tarball Creation (Faster)

```bash
python parallel_hls_synthesis.py \
    --input_dir ../model2_top_10_for_hls \
    --num_workers 4 \
    --no_tarball
```

## File Reference

### Essential Output Files

| File | Size | Description |
|------|------|-------------|
| `vivado_synth.rpt` | ~12 KB | **⭐ Most important** - Resource utilization (LUTs, FFs, BRAM, DSP) |
| `vitis_hls.log` | ~450 KB | Full HLS synthesis log, latency, Fmax estimates |
| `vivado.log` | ~340 KB | Vivado synthesis log, timing info |
| `project.tcl` | ~4 KB | Project configuration, can recreate project |
| `synthesis_results.json` | varies | Overall status of all syntheses |
| `resource_analysis.csv` | varies | Extracted resource usage for all models |

## Getting Help

Each script has built-in help:

```bash
python select_top_models.py --help
python parallel_hls_synthesis.py --help
python check_progress.py --help
python analyze_synthesis_results.py --help
```

## Author
Eric  
November 2025

