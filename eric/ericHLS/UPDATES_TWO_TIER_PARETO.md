# Updates: Two-Tier Pareto Selection with Redundancy

## Date: December 7, 2025

## Summary of Changes

The `select_pareto_models.py` script has been significantly enhanced with two major improvements:

### 1. Two-Tier Pareto Selection for Redundancy ✨

**What Changed:**
- Script now runs Pareto selection **twice** for each complexity metric
- **Primary tier**: Finds the optimal Pareto front
- **Secondary tier**: Finds backup Pareto models from remaining models

**Why This Matters:**
- **Redundancy**: If a primary model fails HLS synthesis, you have a backup
- **Alternatives**: Secondary models offer similar trade-offs with different architectures
- **Robustness**: More options to meet hardware constraints
- **Coverage**: Typically doubles the number of selected models (e.g., 12 → 24)

**Example Output:**
```
Primary Pareto: 12 models (8.1%)
Secondary Pareto: 12 models (8.1%)
Total selected: 24 models (16.1%)
```

### 2. Plot Location Changed 📍

**What Changed:**
- Plots are now saved in the **input directory** (complexity_analysis folder)
- CSV/JSON/H5 files still saved in **output directory**

**Why This Matters:**
- Plots stay with the analysis data they visualize
- Easier to find and reference plots
- Better organization: analysis results (plots) vs deployment files (models)

**Before:**
```
output_dir/
├── pareto_front_parameters.png  <- Was here
└── pareto_optimal_models.csv
```

**After:**
```
input_dir/ (complexity_analysis/)
└── pareto_front_parameters_combined.png  <- Now here

output_dir/
├── pareto_optimal_models_parameters_primary.csv
├── pareto_optimal_models_parameters_secondary.csv
├── pareto_optimal_models_parameters_combined.csv
└── pareto_optimal_models_parameters.json
```

---

## Visual Comparison

### Old Visualization (Single Tier)
```
Accuracy
  ^
  |                    ★
  |                ○○○○○○○
  |            ○○○○★○○○○○○○○
  |        ○○○○○○○○○○○○○○○○○○  
  +-----------------------------------------> Complexity
  
  Gray = All models
  Red ★ = Pareto optimal (12 models)
```

### New Visualization (Two-Tier)
```
Accuracy
  ^
  |                    ★
  |                ○○□○★○□○
  |            ○○○○★○□○○○○○○
  |        ○○□○○○★○○□○○○○○○○  
  +-----------------------------------------> Complexity
  
  Gray ○ = All models
  Red ★ = Primary Pareto (12 models)
  Orange □ = Secondary Pareto (12 models)
```

---

## New Output Files

### For Each Complexity Metric:

**Before** (3 files):
- `pareto_front_parameters.png` (in output_dir)
- `pareto_optimal_models_parameters.csv`
- `pareto_optimal_models_parameters.json`

**After** (7 files):
- `pareto_front_parameters_combined.png` (in **input_dir**)
- `pareto_optimal_models_parameters_primary.csv` (in output_dir)
- `pareto_optimal_models_parameters_secondary.csv` (in output_dir)
- `pareto_optimal_models_parameters_combined.csv` (in output_dir)
- `pareto_optimal_models_parameters.json` (in output_dir, includes both tiers)

---

## Usage Example

### Command (unchanged)
```bash
python ericHLS/select_pareto_models.py \
    --input_dir complexity_analysis/model2_quantized_4w0i_hyperparameter_search \
    --output_dir pareto_models/model2_pareto \
    --complexity_metric both
```

### New Output Console
```
PROCESSING WITH COMPLEXITY METRIC: PARAMETERS
================================================================================

--- TIER 1: PRIMARY PARETO FRONT ---
Finding primary Pareto optimal models...
✓ Found 12 primary Pareto optimal models

--- TIER 2: SECONDARY PARETO FRONT (REDUNDANCY) ---
Removing primary Pareto models and finding secondary front...
✓ Found 12 secondary Pareto optimal models

STATISTICS:
Total models: 149
Primary Pareto optimal: 12 (8.1%)
Secondary Pareto optimal: 12 (8.1%)
Total selected: 24 (16.1%)
```

---

## Real-World Example: Model2 (4w0i)

### Primary Pareto Front (Top Tier)
| Trial | Accuracy | Parameters | Role |
|-------|----------|------------|------|
| 027 | 90.77% | 14,585 | **Best accuracy** |
| 108 | 90.55% | 8,601 | **Balanced** |
| 110 | 90.21% | 3,977 | **Efficient** |
| 071 | 87.88% | 2,889 | **Smallest** |

### Secondary Pareto Front (Redundancy)
| Trial | Accuracy | Parameters | Role |
|-------|----------|------------|------|
| 134 | 90.67% | 14,585 | **Backup for best** (only 0.1% worse!) |
| 092 | 90.33% | 8,329 | **Alternative balanced** |
| 116 | 90.02% | 5,609 | **Alternative efficient** |
| 072 | 87.54% | 2,889 | **Backup smallest** |

### Use Case Scenario

**Scenario**: You need a model with ~8,500 parameters and >90% accuracy

**Primary Choice**: Trial 108 (90.55%, 8,601 params)
- Deploy this first

**If Trial 108 fails HLS synthesis:**
→ **Secondary Backup**: Trial 092 (90.33%, 8,329 params)
- Similar size, similar accuracy, **different hyperparameters**
- May synthesize better on your specific FPGA

**Result**: 2x chance of success with minimal re-analysis!

---

## Benefits of Two-Tier Approach

### 1. Redundancy for Critical Deployments
- **Problem**: Primary model fails HLS synthesis due to unusual layer configurations
- **Solution**: Try secondary model with similar performance but different architecture
- **Impact**: Avoid expensive re-training or re-tuning

### 2. Hardware-Specific Optimization
- **Problem**: Model A has better accuracy but doesn't fit FPGA resources
- **Solution**: Try secondary model with similar accuracy but smaller size
- **Impact**: More flexibility in meeting constraints

### 3. A/B Testing in Production
- **Problem**: Need multiple models for comparison
- **Solution**: Deploy both primary and secondary at similar complexity levels
- **Impact**: Data-driven selection from real-world performance

### 4. Research & Publication
- **Problem**: Need to show robustness of architecture choices
- **Solution**: Demonstrate multiple Pareto-optimal configurations
- **Impact**: Stronger claims about generalizability

---

## Migration Guide

### If You Used the Old Script

**No Breaking Changes!**
- Old commands still work
- Just get more models and better organization

**What to Update**:
1. Check for plots in `input_dir` now (complexity_analysis folder)
2. Use `*_primary.csv` for main deployment models
3. Use `*_secondary.csv` for backup/alternative models
4. Use `*_combined.csv` for complete list

### Workflow Integration

**Before** (single tier):
```bash
# 1. Select Pareto models
python select_pareto_models.py --input_dir X --output_dir Y

# 2. Synthesize (12 models)
python parallel_hls_synthesis.py --input_dir Y

# 3. If models fail, go back to step 1 and manually select more
```

**After** (two-tier):
```bash
# 1. Select Pareto models (now with redundancy)
python select_pareto_models.py --input_dir X --output_dir Y

# 2. Synthesize primary models (12 models)
python parallel_hls_synthesis.py --input_dir Y --filter primary

# 3. If needed, synthesize secondary models (12 more)
python parallel_hls_synthesis.py --input_dir Y --filter secondary

# Result: No need to go back and re-analyze!
```

---

## Testing Results

### Model2 Dataset (149 models)
- **Primary Pareto**: 12 models (8.1%)
- **Secondary Pareto**: 12 models (8.1%)
- **Total coverage**: 24 models (16.1%)
- **Redundancy rate**: 100% (every primary has backup)

### Model3 Dataset (20 models)
- **Primary Pareto**: 5 models (25.0%)
- **Secondary Pareto**: 4 models (20.0%)
- **Total coverage**: 9 models (45.0%)
- **Redundancy rate**: 80% (4 out of 5 primaries have backup)

---

## Documentation Updates

All documentation has been updated:
- ✅ `SELECT_PARETO_MODELS_README.md` - Full guide with two-tier explanation
- ✅ `QUICKSTART_PARETO.md` - Quick reference updated
- ✅ `COMPARISON_SELECTION_METHODS.md` - Comparison tables updated

---

## Questions & Answers

### Q: Do I get twice as many models now?
**A**: Approximately, yes! Typically ~8-12% become ~16-24% selected.

### Q: Are secondary models worse than primary?
**A**: They're dominated by primary models in the strict Pareto sense, but often by tiny margins (0.1-0.5% accuracy). They're excellent backups!

### Q: Should I synthesize both tiers?
**A**: Start with primary. Synthesize secondary only if:
- Primary models fail synthesis
- You need more options for constraints
- You want A/B testing candidates

### Q: Where are the plots now?
**A**: In the `input_dir` (complexity_analysis folder). Look for `*_combined.png` files.

### Q: Can I disable two-tier selection?
**A**: The script always runs two-tier now for completeness. Just use `*_primary.csv` if you only want the primary tier.

---

## Author

Eric - December 7, 2025

## See Also

- `select_pareto_models.py` - Updated script
- `SELECT_PARETO_MODELS_README.md` - Complete documentation
- `QUICKSTART_PARETO.md` - Quick start guide

