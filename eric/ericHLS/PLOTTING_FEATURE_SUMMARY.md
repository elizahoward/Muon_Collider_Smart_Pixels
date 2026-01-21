# Plotting Feature Summary

## Overview

The `extract_hls_resources.py` script has been enhanced with visualization capabilities to plot LUT and FF utilization with FPGA constraint regions.

## What Was Added

### 1. Plotting Function
A comprehensive `plot_resource_utilization()` function that creates a professional scatter plot showing:
- Model trials as red scatter points
- FPGA constraint regions as semi-transparent colored boxes
- Annotations for each model
- Grid, legend, and statistics

### 2. Command-Line Options
New flags for controlling plot generation:
- `--plot`: Enable plot generation
- `--plot-output`: Specify output plot path
- `--pink-luts`, `--pink-ffs`: Set Pink Board constraints
- `--xilinx-luts`, `--xilinx-ffs`: Set Xilinx FPGA constraints

### 3. Flexible FPGA Constraints
Default values provided but fully customizable:
- **Pink Board**: 10,000 LUTs × 20,000 FFs (pink region)
- **Xilinx Zynq xc7z020**: 53,200 LUTs × 106,400 FFs (blue region)

## Usage Examples

### Basic Plot Generation
```bash
python extract_hls_resources.py /path/to/hls_outputs --plot
```

### Custom FPGA Constraints
```bash
python extract_hls_resources.py /path/to/hls_outputs --plot \
    --pink-luts 15000 --pink-ffs 30000 \
    --xilinx-luts 63400 --xilinx-ffs 126800
```

### Custom Output Paths
```bash
python extract_hls_resources.py /path/to/hls_outputs \
    -o results.csv \
    --plot \
    --plot-output results_plot.png
```

## Plot Features

### Visual Elements
1. **Red Scatter Points**: Each model trial
2. **Yellow Annotation Boxes**: Model labels (trial_0, trial_1, etc.)
3. **Pink Box**: Pink Board FPGA constraints (semi-transparent)
4. **Blue Box**: Xilinx FPGA constraints (semi-transparent)
5. **Black Borders**: Clear boundary lines for constraint regions
6. **Dashed Grid**: For easier value reading
7. **Legend**: Shows FPGA specifications
8. **Statistics Box**: Model count and resource ranges

### Plot Specifications
- **Resolution**: 300 DPI (publication quality)
- **Size**: 12×8 inches (3600×2400 pixels)
- **Format**: PNG
- **File Size**: ~250-300 KB

## Test Results

Successfully tested with sample data:
```
Model       LUTs      FFs
trial_0    111,368   127,915
trial_1     76,128    86,914
trial_2     74,811    85,456
trial_3     71,459    84,847
```

Generated outputs:
- ✅ CSV file: `resource_utilization.csv` (123 bytes)
- ✅ PNG plot: `resource_utilization.png` (257 KB)

## Dependencies

### Required
- Python 3.6+
- Standard library only (for CSV generation)

### Optional (for plotting)
- `matplotlib` 3.0+

If matplotlib is not installed:
- Script still generates CSV
- Prints warning about missing matplotlib
- Continues without error

## Key Design Decisions

### 1. Optional Dependency
Matplotlib is optional so the script works in minimal environments for CSV-only extraction.

### 2. Configurable Constraints
All FPGA limits can be customized via command-line arguments to support different hardware targets.

### 3. Professional Visualization
- High DPI for publication quality
- Color-blind friendly colors (red, pink, blue)
- Clear annotations and labels
- Statistics box for quick reference

### 4. Automatic Axis Scaling
Plot automatically scales to show all data points and constraint regions with appropriate padding.

### 5. Thousands Separators
Axis labels use comma separators (e.g., "100,000") for readability with large numbers.

## Integration with Workflow

The plotting feature integrates seamlessly with the existing HLS workflow:

```bash
# Step 1: Run HLS synthesis
python parallel_hls_synthesis.py --input-dir ./models

# Step 2: Extract and visualize
python extract_hls_resources.py ./models/hls_outputs --plot

# Step 3: Analyze results
# - Open resource_utilization.csv for numerical data
# - View resource_utilization.png for visual analysis
```

## Future Enhancements (Possible)

- Support for additional FPGA types
- Interactive plots (using plotly)
- Pareto frontier overlay
- Resource efficiency metrics
- Multiple plot formats (PDF, SVG)
- Comparison across multiple runs

## Files Modified/Created

### Modified
- `extract_hls_resources.py` - Added plotting functionality
- `EXTRACT_HLS_RESOURCES_README.md` - Updated documentation

### Created
- `PLOTTING_FEATURE_SUMMARY.md` - This document
- `resource_utilization.png` - Example output (in test directory)

## Conda Environment

Recommended environment for full functionality:
```bash
conda activate mlproj  # Has matplotlib installed
python extract_hls_resources.py <args> --plot
```

For CSV-only (no matplotlib needed):
```bash
python extract_hls_resources.py <args>  # No --plot flag
```
