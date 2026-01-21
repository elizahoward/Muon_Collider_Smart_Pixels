# Extract HLS Resources Script

## Overview

`extract_hls_resources.py` is a Python script that extracts resource utilization metrics (LUTs and FFs) from Vivado synthesis reports generated during HLS (High-Level Synthesis) compilation. It processes multiple model trials, generates a CSV summary, and optionally creates a visualization plot with FPGA constraint regions.

## What It Does

The script:
1. Scans an `hls_outputs` directory for `hls_model_trial_*` folders
2. Reads the `vivado_synth.rpt` file from each folder
3. Extracts LUT (Look-Up Table) and FF (Flip Flop) utilization numbers
4. Creates a CSV file with the results in the parent directory
5. Optionally generates a scatter plot showing resource utilization with FPGA constraint regions

## Usage

### Basic Usage (CSV Only)

```bash
python extract_hls_resources.py <path_to_hls_outputs_folder>
```

### Example

```bash
python extract_hls_resources.py /home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/model2.5_quantized_4w0i_hyperparameter_results_20260119_181421/hls_outputs
```

This will create `resource_utilization.csv` in the parent directory (i.e., `model2.5_quantized_4w0i_hyperparameter_results_20260119_181421/`).

### Generate Plot with Default FPGA Constraints

```bash
python extract_hls_resources.py <hls_outputs_folder> --plot
```

This creates both a CSV file and a PNG plot showing:
- Red scatter points for each model trial
- **Pink box**: Pink Board constraints (10,000 LUTs × 20,000 FFs)
- **Blue box**: Xilinx Zynq xc7z020 constraints (53,200 LUTs × 106,400 FFs)

### Custom Output Paths

```bash
python extract_hls_resources.py <hls_outputs_folder> -o <output_csv_path> --plot --plot-output <output_plot_path>
```

### Custom FPGA Constraints

```bash
python extract_hls_resources.py <hls_outputs_folder> --plot \
    --pink-luts 10000 --pink-ffs 20000 \
    --xilinx-luts 53200 --xilinx-ffs 106400
```

## Input Requirements

The script expects the following directory structure:

```
hls_outputs/
├── hls_model_trial_0/
│   └── vivado_synth.rpt
├── hls_model_trial_1/
│   └── vivado_synth.rpt
├── hls_model_trial_2/
│   └── vivado_synth.rpt
└── hls_model_trial_3/
    └── vivado_synth.rpt
```

Each `vivado_synth.rpt` should be a standard Vivado utilization report containing the "Slice Logic" table.

## Output Format

### CSV Output

The generated CSV file has the following columns:

- **Model**: Model identifier (e.g., "trial_0", "trial_1")
- **Trial_Number**: Numeric trial number (0, 1, 2, ...)
- **LUTs**: Number of Look-Up Tables used
- **FFs**: Number of Flip Flops used

#### Example CSV Output

```csv
Model,Trial_Number,LUTs,FFs
trial_0,0,111368,127915
trial_1,1,76128,86914
trial_2,2,74811,85456
trial_3,3,71459,84847
```

### Plot Output

When using `--plot`, a high-resolution PNG image (300 DPI) is generated with:
- **Filename**: `resource_utilization.png` (default)
- **Format**: PNG image
- **Resolution**: 300 DPI (suitable for papers/presentations)
- **Size**: Typically 250-300 KB
- **Dimensions**: 12×8 inches (3600×2400 pixels)

## Command-Line Options

```
positional arguments:
  hls_outputs_dir       Path to the hls_outputs directory containing 
                        hls_model_trial_* folders

optional arguments:
  -h, --help            Show help message and exit
  -o OUTPUT, --output OUTPUT
                        Output CSV file path (default: <parent_dir>/resource_utilization.csv)
  --plot                Generate a visualization plot of the resource utilization
  --plot-output PLOT_OUTPUT
                        Output plot file path (default: <parent_dir>/resource_utilization.png)
  --pink-luts PINK_LUTS
                        Maximum LUTs for Pink Board (default: 10000)
  --pink-ffs PINK_FFS   Maximum FFs for Pink Board (default: 20000)
  --xilinx-luts XILINX_LUTS
                        Maximum LUTs for Xilinx FPGA (default: 53200)
  --xilinx-ffs XILINX_FFS
                        Maximum FFs for Xilinx FPGA (default: 106400)
```

## Dependencies

### Required
- Python 3.6+
- Standard library modules: `os`, `re`, `csv`, `argparse`, `pathlib`, `typing`

### Optional (for plotting)
- `matplotlib` (install with `pip install matplotlib` or `conda install matplotlib`)

If matplotlib is not installed and `--plot` is used, the script will generate the CSV but skip the plot with a warning.

## Error Handling

The script handles several error conditions gracefully:

- **Missing directory**: Exits with an error if the specified directory doesn't exist
- **Missing report files**: Warns and skips trials with missing `vivado_synth.rpt` files
- **Parse failures**: Warns if resource data cannot be extracted from a report
- **No valid data**: Exits with an error if no valid data is extracted
- **Missing matplotlib**: Warns and skips plot generation if matplotlib is not available

## Integration with Other Scripts

This script complements the existing HLS workflow scripts in this directory:

- **`parallel_hls_synthesis.py`**: Runs HLS synthesis to generate the reports
- **`analyze_synthesis_results.py`**: Analyzes synthesis results
- **`select_pareto_models.py`**: Performs Pareto optimization on models
- **`extract_hls_resources.py`**: (This script) Extracts resource metrics for analysis

## Technical Details

### Parsing Logic

The script uses regular expressions to extract data from the "Slice Logic" section of the Vivado report:

- **LUTs**: Extracted from the "Slice LUTs*" row
- **FFs**: Extracted from the "Slice Registers" row

### Pattern Matching

```python
lut_pattern = r'\|\s*Slice LUTs\*?\s*\|\s*(\d+)\s*\|'
ff_pattern = r'\|\s*Slice Registers\s*\|\s*(\d+)\s*\|'
```

These patterns match the standard Vivado report format and extract the "Used" column value.

## Troubleshooting

### "No hls_model_trial_* directories found"

- Verify you're pointing to the correct `hls_outputs` directory
- Check that subdirectories follow the naming convention `hls_model_trial_N`

### "Could not parse resource data"

- Verify that `vivado_synth.rpt` files exist and are complete
- Check that the reports are from Vivado (not other tools)
- Ensure the synthesis completed successfully

### "No data was extracted"

- Check that at least one trial has a valid `vivado_synth.rpt` file
- Verify the report format matches the expected Vivado format

## Visualization Details

The generated plot includes:

### Plot Elements
1. **Scatter Points**: Red circles representing each model trial
2. **Annotations**: Yellow boxes labeling each model (trial_0, trial_1, etc.)
3. **Pink Box**: Pink Board FPGA constraint region (semi-transparent pink)
4. **Blue Box**: Xilinx FPGA constraint region (semi-transparent blue)
5. **Grid**: Dashed gridlines for easier reading
6. **Legend**: Shows FPGA constraint specifications
7. **Statistics Box**: Summary of model count and resource ranges

### Interpreting the Plot
- **Points inside a colored box**: Model fits within that FPGA's constraints
- **Points outside all boxes**: Model exceeds all FPGA constraints
- **Closer to origin (0,0)**: More efficient design with fewer resources

## Example Workflows

### Workflow 1: Quick Analysis
```bash
# 1. Run HLS synthesis (if not already done)
python parallel_hls_synthesis.py --input-dir ./model_results

# 2. Extract resource utilization with visualization
python extract_hls_resources.py ./model_results/hls_outputs --plot

# 3. View the results
cat ./model_results/resource_utilization.csv
# Open resource_utilization.png to view the plot
```

### Workflow 2: Custom FPGA Targets
```bash
# For different FPGA targets
python extract_hls_resources.py ./model_results/hls_outputs --plot \
    --pink-luts 15000 --pink-ffs 30000 \
    --xilinx-luts 63400 --xilinx-ffs 126800
```

### Workflow 3: Using with mlproj Environment
```bash
# Activate the conda environment with matplotlib
conda activate mlproj

# Run with plotting enabled
python extract_hls_resources.py ./model_results/hls_outputs --plot
```

## See Also

- `WORKFLOW_SUMMARY.md` - Overview of the complete HLS workflow
- `parallel_hls_synthesis.py` - HLS synthesis script
- `analyze_synthesis_results.py` - Results analysis script
- `select_pareto_models.py` - Pareto optimization script
