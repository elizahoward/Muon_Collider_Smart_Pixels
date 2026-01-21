# Model Selection Methods: Comparison

## Overview

Two complementary approaches for selecting models from hyperparameter tuning results:

| Method | Script | Best For |
|--------|--------|----------|
| **Percentile-based** | `select_top_models.py` | Selecting top N% performers by accuracy |
| **Pareto optimal** | `select_pareto_models.py` | Finding optimal accuracy/complexity trade-offs |

---

## Method 1: Percentile-Based Selection (`select_top_models.py`)

### How It Works

Selects models based on accuracy percentiles (e.g., top 25% or top N models).

### Pros

- ✅ Simple and intuitive
- ✅ Guarantees a specific number of models
- ✅ Focuses purely on accuracy
- ✅ Good for initial filtering

### Cons

- ❌ May include many similar high-accuracy models
- ❌ Ignores model complexity entirely
- ❌ Can result in unnecessarily complex models
- ❌ No diversity in complexity range

### Example Results

```
Top 25% by accuracy (37 models selected from 149):
Trial 032: 94.96% accuracy, 35777 params
Trial 029: 94.94% accuracy, 42057 params
Trial 009: 94.92% accuracy, 34385 params
...
Trial 048: 93.50% accuracy, 45123 params  <- High complexity, marginal benefit
```

### When to Use

- You need a specific number of top models
- Accuracy is the only priority
- Complexity constraints are not a concern
- Initial broad filtering before further analysis

---

## Method 2: Pareto Optimal Selection (`select_pareto_models.py`)

### How It Works

Identifies models where improving accuracy requires accepting higher complexity (and vice versa).

### Pros

- ✅ Provides diverse range of complexity options
- ✅ Highlights optimal trade-offs
- ✅ Eliminates dominated (suboptimal) models
- ✅ Better for resource-constrained deployment

### Cons

- ❌ Variable number of models selected
- ❌ May exclude some high-accuracy models
- ❌ Requires understanding of trade-offs
- ❌ More complex to interpret

### Example Results

```
Pareto optimal (12 models selected from 149):
Trial 027: 90.77% accuracy, 14585 params  <- Best accuracy
Trial 140: 90.57% accuracy, 12409 params  <- 0.2% loss, 15% smaller
Trial 023: 90.57% accuracy, 10505 params  <- Same acc, even smaller
Trial 108: 90.55% accuracy,  8601 params  <- Good balance
...
Trial 071: 87.88% accuracy,  2889 params  <- Smallest (5x smaller!)
```

### When to Use

- Resource constraints matter (FPGA, memory, latency)
- Want to visualize accuracy/complexity trade-off
- Need to justify model choice to stakeholders
- Want optimal frontier of solutions

---

## Visual Comparison

### Percentile Selection (Top 25%)

```
Accuracy
  ^
  |                    ████████
  |                ████████████████
  |            ████████████████████████  <- Selected
  |        ████████████████████████████████
  |    ████████████████████████████████████
  |████████████████████████████████████████
  +-----------------------------------------> Complexity
  
  Selected: All models above 75th percentile
  Issue: Many redundant high-complexity models
```

### Pareto Optimal Selection

```
Accuracy
  ^
  |                    ★
  |                ○○○★○○○
  |            ○○○○★○○○○○○○○
  |        ○○○○○○★○○○○○○○○○○○  <- ★ = Pareto front
  |    ○○○○○○○○★○○○○○○○○○○○○○
  |○○○○○○○○○○○○★○○○○○○○○○○○○○
  +-----------------------------------------> Complexity
  
  Selected: Only ★ (Pareto optimal) models
  Benefit: Maximum diversity, no redundancy
```

---

## Decision Matrix

### Choose **Percentile-Based** if:

- [ ] You need exactly N models
- [ ] Accuracy is paramount
- [ ] No resource constraints
- [ ] Quick filtering needed
- [ ] Hardware can handle any complexity

### Choose **Pareto Optimal** if:

- [x] Resource constraints exist (FPGA, latency, memory)
- [x] Want to visualize trade-offs
- [x] Need to justify model selection
- [x] Want efficient model diversity
- [x] Deploying to constrained hardware

---

## Recommended Workflow

### Option A: Sequential (Conservative)

```bash
# 1. Start with percentile to filter out poor models
python select_top_models.py \
    --input_dir results/ \
    --output_dir filtered/ \
    --percentile 75

# 2. Find Pareto models among the filtered set
python select_pareto_models.py \
    --input_dir filtered/ \
    --output_dir pareto/ \
    --complexity_metric both
```

### Option B: Direct (Recommended)

```bash
# Directly find Pareto models from complexity analysis
python select_pareto_models.py \
    --input_dir complexity_analysis/model2/ \
    --output_dir pareto_models/ \
    --complexity_metric both \
    --min_accuracy 0.85  # Optional: filter poor models first
```

---

## Real-World Example

### Scenario: FPGA Deployment with 10,000 Parameter Budget

**Dataset**: 149 models, accuracy range 80.7% - 90.8%

#### Percentile Method (Top 25%)

```
Selected: 37 models
Problem: Only 3 models under 10,000 params
All others exceed budget -> Manual filtering needed
```

#### Pareto Method

```
Selected: 12 models
Result: 8 models under 10,000 params
Range: 2,889 to 8,601 params (87.9% to 90.6% acc)
Clear choices for different budgets!
```

**Winner**: Pareto method gives immediate actionable options.

---

## Combining Both Methods

### Workflow for Large-Scale Tuning

```bash
# 1. Analyze complexity
python analyze_hyperparameter_complexity.py \
    hyperparameter_tuning/model2_search/

# 2. Get Pareto optimal models (for trade-off analysis)
python select_pareto_models.py \
    --input_dir complexity_analysis/model2_search/ \
    --output_dir pareto_analysis/

# 3. Get top 10 models (for maximum accuracy)
python select_top_models.py \
    --search_dir hyperparameter_tuning/model2_search/ \
    --output_dir top10_models/ \
    --top_n 10

# 4. Compare and choose based on requirements
```

---

## Summary Table

| Aspect | Percentile-Based | Pareto Optimal |
|--------|------------------|----------------|
| **Selection Criterion** | Accuracy threshold | Multi-objective trade-off |
| **Number of Models** | Fixed (user-defined) | Variable (data-driven) |
| **Complexity Awareness** | No | Yes |
| **Visualization** | No (just filtering) | Yes (scatter plots) |
| **Redundancy** | High (similar models) | Low (diverse frontier) |
| **Best For** | Accuracy-focused | Resource-constrained |
| **Use Case** | Cloud deployment | FPGA/Edge deployment |
| **Interpretability** | Simple | Requires understanding |

---

## Recommendations by Scenario

### Cloud/Server Deployment
→ Use **percentile** (top 10-20 models) for maximum accuracy

### FPGA Deployment
→ Use **Pareto** to visualize complexity/accuracy trade-offs

### Edge Devices
→ Use **Pareto** with `--min_accuracy` filter for viable options

### Research/Publication
→ Use **Pareto** for comprehensive trade-off analysis

### Production A/B Testing
→ Use **percentile** (top N) for multiple candidates

### Hardware Synthesis Exploration
→ Use **Pareto** to minimize synthesis attempts

---

## Author

Eric - 2025

## See Also

- `select_top_models.py`: Percentile-based selection
- `select_pareto_models.py`: Pareto optimal selection  
- `analyze_hyperparameter_complexity.py`: Prerequisite analysis

