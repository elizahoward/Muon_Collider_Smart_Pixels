# Archived Scripts

These scripts have been superseded by the streamlined workflow and are kept for historical reference only.

## What's Archived

### Scripts:
- `select_pareto_models.py` - Old standalone Pareto selection (replaced by `analyze_and_select_pareto.py`)
- `plot_and_select_models.py` - Old plotting utility (functionality integrated into `analyze_and_select_pareto.py`)
- `select_top_models.py` - Simple top-N selection (Pareto selection is more sophisticated)

### Documentation:
- `SELECT_PARETO_MODELS_README.md` - Documentation for old select_pareto_models.py
- `QUICKSTART_PARETO.md` - Old quickstart guide
- `UPDATES_TWO_TIER_PARETO.md` - Documentation of two-tier Pareto feature (now standard)
- `COMPARISON_SELECTION_METHODS.md` - Comparison of old selection methods
- `USAGE_GUIDE.md` - Old usage guide

## Why Archived?

The workflow has been streamlined to use a single combined script (`analyze_and_select_pareto.py`) that:
- Loads models directly from H5 files (model-agnostic)
- Automatically detects trial naming formats
- Performs two-tier Pareto selection
- Generates all plots and summaries
- Prepares models for HLS synthesis

See `../STREAMLINED_WORKFLOW.md` for the current workflow.

## Can I Delete These?

Yes, these files are kept only for reference. The current workflow does not use them.
If you're confident in the new workflow, you can safely delete this entire directory.

---
Archived: January 2026

## January 2026 Update

Additional scripts archived:
- `analyze_synthesis_results.py` - Superseded by `analyze_hls_results.py`
- `extract_hls_resources.py` - Superseded by `analyze_hls_results.py`  
- `EXTRACT_HLS_RESOURCES_README.md` - Documentation for old extract script

The new `analyze_hls_results.py` combines the functionality of both scripts into one comprehensive tool that:
- Extracts all resource metrics (LUTs, FFs, BRAM, DSP, Fmax, utilization %)
- Provides comprehensive statistics (min/max/mean/median)
- Ranks models by multiple criteria (accuracy, size, speed)
- Creates visualization plots with FPGA constraints
- Links to validation accuracy data automatically

See `../STREAMLINED_WORKFLOW.md` for updated usage.
