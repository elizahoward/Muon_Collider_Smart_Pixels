# HLS Workflow - Quick Reference Card

## Complete Workflow (3 Simple Steps)

### Step 1: Select Pareto Optimal Models
```bash
python analyze_and_select_pareto.py \
    --input_dir ../model2.5_quantized_4w0i_hyperparameter_results_20260119_181421 \
    --output_dir ../model2_5_pareto_hls_ready \
    --min_accuracy 0.80
```

**Outputs:** Selected models + plots + CSVs

---

### Step 2: Run HLS Synthesis
```bash
python parallel_hls_synthesis.py \
    --input_dir ../model2_5_pareto_hls_ready \
    --num_workers 4
```

**Outputs:** HLS projects + synthesis reports + tarballs

---

### Step 3: Analyze Results
```bash
python analyze_hls_results.py \
    --results_dir ../model2_5_pareto_hls_ready/hls_outputs \
    --plot
```

**Outputs:** CSV + statistics + visualization plot

---

## Common Options

### analyze_and_select_pareto.py
- `--min_accuracy 0.80` - Filter models by minimum accuracy
- Models automatically loaded from H5 files (model-agnostic!)

### parallel_hls_synthesis.py
- `--num_workers 8` - Use more parallel workers
- `--limit 5` - Test with only 5 models
- `--fpga_part xcu250-figd2104-2L-e` - Different FPGA target

### analyze_hls_results.py
- `--plot` - Generate visualization
- `--output_csv custom.csv` - Custom CSV path
- `--pink-luts 10000 --pink-ffs 20000` - Custom FPGA constraints
- `--xilinx-luts 53200 --xilinx-ffs 106400` - Xilinx constraints

---

## One-Liner Examples

**Quick test (2 models only):**
```bash
cd ericHLS && \
python analyze_and_select_pareto.py --input_dir ../results --output_dir ../test_out && \
python parallel_hls_synthesis.py --input_dir ../test_out --num_workers 2 --limit 2 && \
python analyze_hls_results.py --results_dir ../test_out/hls_outputs --plot
```

**Production run:**
```bash
cd ericHLS && \
python analyze_and_select_pareto.py --input_dir ../model2.5_results --output_dir ../pareto_models --min_accuracy 0.85 && \
python parallel_hls_synthesis.py --input_dir ../pareto_models --num_workers 8 && \
python analyze_hls_results.py --results_dir ../pareto_models/hls_outputs --plot
```

---

## Outputs Summary

### After Step 1 (analyze_and_select_pareto.py):
```
model2_5_pareto_hls_ready/
├── model_trial_*.h5                              ← Selected models
├── complexity_vs_accuracy_parameters.png         ← Scatter plot
├── pareto_front_parameters_combined.png          ← Pareto visualization
├── pareto_optimal_models_parameters_*.csv        ← Model lists
└── analysis_summary.json                         ← Metadata
```

### After Step 2 (parallel_hls_synthesis.py):
```
model2_5_pareto_hls_ready/hls_outputs/
├── hls_model_trial_*/                            ← HLS projects
│   ├── vivado_synth.rpt                          ← Resource report
│   ├── vitis_hls.log                             ← Synthesis log
│   └── project.tcl                               ← HLS config
├── hls_model_trial_*.tar.gz                      ← Compressed results
└── synthesis_results.json                        ← Summary
```

### After Step 3 (analyze_hls_results.py):
```
model2_5_pareto_hls_ready/
├── resource_utilization.csv                      ← All metrics
└── resource_utilization.png                      ← Visualization
```

---

## Key Features

### ✨ Model-Agnostic
- Works with any model (Model2, Model2.5, Model3, future models)
- Loads actual H5 files, no hardcoded calculations
- Auto-detects trial naming formats

### 📊 Comprehensive Analysis
- All resources: LUTs, FFs, BRAM, DSP, Fmax
- Utilization percentages
- Links to validation accuracy
- Statistical summaries (min/max/mean/median)

### 🎯 Smart Selection
- Two-tier Pareto front (primary + secondary)
- Optimizes accuracy vs. parameters trade-off
- Automatic redundancy selection

### 🚀 Parallel Processing
- Multiple models synthesized simultaneously
- Automatic cleanup of large files
- Progress tracking

### 📈 Visualization
- Scatter plot with FPGA constraints
- Color-coded by validation accuracy
- Shows which models fit on target FPGAs

---

## Requirements

```bash
# Conda environment
conda activate mlgpu_qkeras  # or mlproj_qkeras, qk-tf214-gpu

# Python packages
pip install tensorflow qkeras hls4ml pandas matplotlib numpy

# System
Xilinx Vitis HLS 2024.1+
```

---

## Troubleshooting

**"TensorFlow not available"**
→ Activate conda environment with TensorFlow

**"No valid trials found"**
→ Check input directory has model_trial_*.h5 and hyperparams_trial_*.json

**Synthesis failures**
→ Check vitis_hls.log in individual hls_model_trial_* directories

**Plot not generated**
→ Install matplotlib: `pip install matplotlib`

---

## Quick Help

```bash
python analyze_and_select_pareto.py --help
python parallel_hls_synthesis.py --help
python analyze_hls_results.py --help
```

---

**Last updated:** January 2026  
**See also:** `STREAMLINED_WORKFLOW.md` for detailed documentation
