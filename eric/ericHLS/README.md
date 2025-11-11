# Parallel HLS Synthesis Tool

This directory contains a tool for parallel synthesis of neural network models to HLS (High-Level Synthesis) implementations.

## Overview

The `parallel_hls_synthesis.py` script processes multiple H5 model files in parallel, converting them to HLS implementations using hls4ml and Xilinx Vitis HLS. It automatically:

- Loads quantized neural network models (QKeras)
- Converts them to HLS implementations
- Runs full synthesis (C synthesis, co-simulation, export, and Vivado synthesis)
- Keeps only essential output files to save disk space
- Creates compressed tarballs of the results
- Generates a comprehensive results summary

## Features

- **Parallel Processing**: Use multiple workers to process several models simultaneously
- **Space Efficient**: Automatically deletes large intermediate files, keeping only:
  - `project.tcl` - Project configuration
  - `vitis_hls.log` - HLS synthesis log
  - `vivado_synth.rpt` - Vivado synthesis utilization report
  - `vivado_synth.tcl` - Vivado synthesis script
  - `vivado.jou` - Vivado journal
  - `vivado.log` - Vivado log
- **Progress Tracking**: Real-time progress updates for each synthesis
- **Error Handling**: Robust error handling with detailed error logs
- **Results Summary**: JSON summary of all synthesis results

## Requirements

```bash
# Python packages
tensorflow
qkeras
hls4ml

# System requirements
Xilinx Vitis HLS 2024.1 or later
```

## Usage

### Basic Usage

Process all model files in a directory with 4 parallel workers:

```bash
python parallel_hls_synthesis.py \
    --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \
    --num_workers 4
```

### Advanced Options

```bash
python parallel_hls_synthesis.py \
    --input_dir <path_to_h5_files> \
    --output_dir <custom_output_dir> \
    --num_workers <number_of_workers> \
    --pattern "model_trial_*.h5" \
    --fpga_part xc7z020clg400-1 \
    --limit 10
```

### Command Line Arguments

- `--input_dir` (required): Directory containing H5 model files
- `--output_dir` (optional): Base directory for HLS outputs (default: input_dir/hls_outputs)
- `--num_workers` (optional): Number of parallel workers (default: 4)
- `--pattern` (optional): Pattern to match H5 files (default: model_trial_*.h5)
- `--fpga_part` (optional): FPGA part number (default: xc7z020clg400-1)
- `--no_tarball`: Skip creating tarball of output files
- `--limit` (optional): Limit number of models to process (useful for testing)

## Examples

### Example 1: Process all models with 8 workers

```bash
python parallel_hls_synthesis.py \
    --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \
    --num_workers 8
```

### Example 2: Process specific trials only

```bash
python parallel_hls_synthesis.py \
    --input_dir ../model2_results \
    --pattern "model_trial_00*.h5" \
    --num_workers 2
```

### Example 3: Use different FPGA part

```bash
python parallel_hls_synthesis.py \
    --input_dir ../model2_results \
    --num_workers 4 \
    --fpga_part xcu250-figd2104-2L-e
```

### Example 4: Test run (process only 2 models)

```bash
python parallel_hls_synthesis.py \
    --input_dir ../model2_results \
    --num_workers 2 \
    --limit 2
```

## Output Structure

After running the script, the output directory will contain:

```
hls_outputs/
├── hls_model_trial_000/
│   ├── project.tcl
│   ├── vitis_hls.log
│   ├── vivado_synth.rpt
│   ├── vivado_synth.tcl
│   ├── vivado.jou
│   └── vivado.log
├── hls_model_trial_000.tar.gz
├── hls_model_trial_001/
│   └── ...
├── hls_model_trial_001.tar.gz
├── ...
└── synthesis_results.json
```

## Results Summary

The `synthesis_results.json` file contains:

```json
{
  "total": 100,
  "successful": 98,
  "failed": 2,
  "results": [
    {
      "h5_file": "/path/to/model_trial_000.h5",
      "status": "success",
      "output_dir": "/path/to/hls_model_trial_000",
      "kept_files": [...],
      "tarball": "/path/to/hls_model_trial_000.tar.gz",
      "start_time": "2025-11-09T...",
      "end_time": "2025-11-09T..."
    },
    ...
  ]
}
```

## Performance Considerations

- **Synthesis Time**: Each model typically takes 2-5 minutes to synthesize
- **Disk Space**: Intermediate files can be very large (>1 GB per model). The script automatically cleans these up, keeping only ~10-20 MB per model
- **CPU Usage**: Each worker uses significant CPU resources during synthesis
- **Memory**: Each worker may use 2-4 GB of RAM

### Recommended Worker Counts

- **4-core CPU**: 2-4 workers
- **8-core CPU**: 4-8 workers  
- **16+ core CPU**: 8-16 workers

## Troubleshooting

### Common Issues

1. **Out of Disk Space**
   - Ensure you have sufficient disk space (at least 10 GB free per worker)
   - Reduce the number of workers

2. **Synthesis Failures**
   - Check the individual log files in the output directories
   - Review `synthesis_results.json` for error messages
   - Some models may fail due to unsupported layers or configurations

3. **Vitis HLS Not Found**
   - Ensure Vitis HLS is properly installed and sourced
   - Check that `vitis_hls` command is available in your PATH

4. **Memory Issues**
   - Reduce the number of workers
   - Process models in smaller batches using `--limit`

## Notes

- The script disables GPU usage by default to avoid conflicts in parallel processing
- Models are loaded without compilation (`compile=False`) for faster loading
- The script creates a comprehensive log for each synthesis in the respective output directories

## Author

Eric  
Date: November 2025

