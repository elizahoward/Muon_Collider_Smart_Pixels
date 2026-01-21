# Quick Start: Pareto Model Selection (Two-Tier with Redundancy)

## TL;DR

```bash
# Select Pareto optimal models from complexity analysis (two-tier for redundancy)
python ericHLS/select_pareto_models.py \
    --input_dir complexity_analysis/model2_quantized_4w0i_hyperparameter_search \
    --output_dir pareto_models/model2_pareto \
    --complexity_metric both
```

## What You Get

### 📊 Visualizations (saved in input_dir)
- **Two-tier Pareto front plots** showing:
  - All models (gray dots)
  - Primary Pareto optimal models (red diamonds)
  - Secondary Pareto optimal models (orange squares - redundancy)
- Models annotated with trial IDs
- Statistics box with counts for both tiers
- Lines connecting both Pareto frontiers

### 📁 Data Files (saved in output_dir)
- **CSV**: Separate files for primary, secondary, and combined Pareto models
- **JSON**: Structured data with both tiers for programmatic access
- **H5**: Model files for both tiers (optional, if `--models_dir` specified)

### 📈 Statistics
- Total vs Primary vs Secondary Pareto model counts
- Accuracy and complexity ranges
- Ranked lists of both Pareto tiers
- Redundancy analysis

## Real Example Output

```
PARETO OPTIMAL SELECTION STATISTICS
================================================================================
Complexity metric: parameters
Total models: 149
Primary Pareto optimal: 12 (8.1%)
Secondary Pareto optimal: 12 (8.1%)
Total selected: 24 (16.1%)

Primary Pareto Optimal Models:
Trial ID     Accuracy     Parameters     
----------------------------------------
027          0.9077       14585          <- BEST accuracy (primary)
140          0.9057       12409          
023          0.9057       10505          
108          0.9055       8601           <- BALANCED (90.5% acc, 8.6k params)
003          0.9043       8329           
087          0.9033       6153           
032          0.9023       5881           
110          0.9021       3977           <- EFFICIENT (90.2% acc, 4k params)
064          0.9014       3705           
081          0.8989       3433           
144          0.8866       3161           
071          0.8788       2889           <- SMALLEST (primary)

Secondary Pareto Optimal Models (Redundancy):
Trial ID     Accuracy     Parameters     
----------------------------------------
134          0.9067       14585          <- BACKUP for best accuracy
118          0.9057       12409          
068          0.9054       10505          
092          0.9033       8329           <- ALTERNATIVE balanced
042          0.9033       8057           
022          0.9018       7785           
026          0.9017       6153           
116          0.9002       5609           
079          0.8980       3705           
070          0.8948       3433           
104          0.8828       3161           
072          0.8754       2889           <- BACKUP smallest
```

### Interpretation

**Primary Tier** (Best Trade-offs):
- **Best Accuracy** (Trial 027): 90.77% accuracy, 14,585 parameters
  - Use if accuracy is paramount
- **Balanced** (Trial 108): 90.55% accuracy, 8,601 parameters  
  - Only 0.22% accuracy loss, but 41% smaller!
- **Efficient** (Trial 110): 90.21% accuracy, 3,977 parameters
  - 0.56% accuracy loss, but 73% smaller!
- **Smallest** (Trial 071): 87.88% accuracy, 2,889 parameters
  - 5x smaller, 2.89% accuracy loss

**Secondary Tier** (Redundancy/Alternatives):
- **Backup for Best** (Trial 134): 90.67% accuracy, 14,585 parameters
  - Only 0.1% worse than primary, same size - perfect backup!
- **Alternative Balanced** (Trial 092): 90.33% accuracy, 8,329 parameters
  - Similar trade-off, different hyperparameter configuration
- **Backup Smallest** (Trial 072): 87.54% accuracy, 2,889 parameters
  - Similar size to smallest primary, alternative architecture

**Why Secondary Matters**:
- If primary model fails synthesis → try secondary
- Different hyperparameters → may synthesize better on FPGA
- More options → better chance of meeting all constraints

## Common Use Cases

### 1. FPGA Deployment (Resource Constrained)

```bash
python select_pareto_models.py \
    --input_dir complexity_analysis/model2/ \
    --output_dir fpga_candidates/ \
    --complexity_metric parameters \
    --min_accuracy 0.88
```

Pick from Pareto models that fit your FPGA budget.

### 2. Maximum Accuracy (Cloud)

```bash
# Pareto still useful to avoid unnecessarily large models
python select_pareto_models.py \
    --input_dir complexity_analysis/model3/ \
    --output_dir best_models/ \
    --complexity_metric parameters
```

Choose the highest-accuracy Pareto model (first in list).

### 3. Quick Analysis (Plots Only)

```bash
python select_pareto_models.py \
    --input_dir complexity_analysis/model2/ \
    --output_dir analysis/ \
    --plot_only
```

Visualize trade-offs without copying files.

### 4. Complete Analysis (Both Metrics)

```bash
python select_pareto_models.py \
    --input_dir complexity_analysis/model2_5/ \
    --output_dir pareto_complete/ \
    --complexity_metric both
```

Get Pareto fronts for both parameters and nodes.

## Files Generated

```
input_dir/ (complexity_analysis folder - PLOTS HERE)
├── pareto_front_parameters_combined.png # 🎨 Two-tier scatter plot
└── pareto_front_nodes_combined.png      # 🎨 (if --complexity_metric both)

output_dir/ (specified directory - DATA HERE)
├── pareto_optimal_models_parameters_primary.csv   # 📊 Primary Pareto models
├── pareto_optimal_models_parameters_secondary.csv # 📊 Secondary (redundancy)
├── pareto_optimal_models_parameters_combined.csv  # 📊 All Pareto models
├── pareto_optimal_models_parameters.json          # 💾 Structured data (both tiers)
├── pareto_optimal_models_nodes_primary.csv        # 📊 (if --complexity_metric both)
├── pareto_optimal_models_nodes_secondary.csv      # 📊 (if --complexity_metric both)
├── pareto_optimal_models_nodes_combined.csv       # 📊 (if --complexity_metric both)
├── pareto_optimal_models_nodes.json               # 💾 (if --complexity_metric both)
└── model_trial_*.h5                               # 🤖 Both tiers (if model files found)
```

## Integration with Workflow

### Full Pipeline

```bash
# 1. Hyperparameter tuning (already done)
# Creates: hyperparameter_tuning/model2_quantized_4w0i_hyperparameter_search/

# 2. Complexity analysis
python analyze_hyperparameter_complexity.py \
    hyperparameter_tuning/model2_quantized_4w0i_hyperparameter_search

# Creates: complexity_analysis/model2_quantized_4w0i_hyperparameter_search/

# 3. Pareto selection (NEW!)
python ericHLS/select_pareto_models.py \
    --input_dir complexity_analysis/model2_quantized_4w0i_hyperparameter_search \
    --output_dir pareto_models/model2_pareto \
    --complexity_metric both

# Creates: pareto_models/model2_pareto/

# 4. HLS synthesis on Pareto models
python ericHLS/parallel_hls_synthesis.py \
    --input_dir pareto_models/model2_pareto \
    --num_workers 4

# 5. Final selection
python ericHLS/plot_and_select_models.py \
    --results_dir hls_results/model2_pareto
```

## Command-Line Options

### Essential
- `--input_dir`: Complexity analysis directory (required)
- `--output_dir`: Where to save results (required)

### Optional
- `--complexity_metric {parameters,nodes,both}`: Which metric (default: parameters)
- `--plot_only`: Don't copy model files, just visualize
- `--min_accuracy 0.XX`: Filter out low-accuracy models first
- `--models_dir PATH`: Where to find H5 files (auto-detected usually)

## Tips

### ✅ DO

- Use `--complexity_metric both` to see both perspectives
- Filter with `--min_accuracy` to exclude failed models (e.g., 0.55)
- Start with `--plot_only` for quick exploration
- Check the plots before deciding on final model

### ❌ DON'T

- Don't worry if you get fewer models than expected - that's normal!
- Don't expect all models to be Pareto optimal (typically 5-15%)
- Don't ignore the smallest Pareto models - they might surprise you!

## Troubleshooting

### "No CSV file found"
→ Run `analyze_hyperparameter_complexity.py` first

### "No Pareto models found"
→ Check your data has multiple models with varying complexity

### "Model files not copied"
→ Either use `--models_dir` or just use `--plot_only`

## Questions?

See the full documentation:
- `SELECT_PARETO_MODELS_README.md` - Complete guide
- `COMPARISON_SELECTION_METHODS.md` - vs percentile selection

## Author

Eric - 2025

