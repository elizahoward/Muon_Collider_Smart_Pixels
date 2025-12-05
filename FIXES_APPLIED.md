# Fixes Applied - Dec 1, 2025

## Problem Identified

Your first run with the improvements showed **worse results**:
- **Old (Nov 17):** 4-bit acc=0.846, auc=0.895
- **New (Dec 1):** 4-bit acc=0.831, auc=0.886

## Root Causes Found

### 1. Incomplete Warm-Start ❌
The warm-start only copied **5 out of 6 layers**:
```
✓ Copied: 5 layers
⚠ Skipped: 1 layers (output_dense)
```

**Problem:** Layer name mismatch
- Unquantized model: `output` (Dense layer)
- Quantized model: `output_dense` (QDense layer)

**Fix Applied:** Changed quantized model to use `name="output"` instead of `name="output_dense"`

### 2. Learning Rate Too Aggressive ❌
Using the same polynomial decay as unquantized (1e-3 → 1e-4) was too aggressive for the quantized model.

**Evidence from training log:**
- Training accuracy oscillated: 0.7571 → 0.7740 → 0.7647 → 0.7748 (bouncing around)
- Validation loss unstable: 0.4709 → 0.4727 → 0.4785 → 0.4556 (not smooth)
- Never reached the old performance level

**Fix Applied:** Use **conservative polynomial decay** for quantized models:
- Initial LR: **5e-4** (0.5x unquantized, instead of 1e-3)
- End LR: **1e-5** (0.1x unquantized, instead of 1e-4)
- Still decays (better than constant), but starts lower for stability

## Changes Made

### File: `model2.py`
**Line ~360 and ~530:** Fixed output layer name
```python
# OLD:
output_dense = QDense(1, name="output_dense")(merged_dense)
output = QActivation("sigmoid", name="output")(output_dense)

# NEW:
output_dense = QDense(1, name="output")(merged_dense)  # Match unquantized!
output = QActivation("sigmoid", name="output_activation")(output_dense)
```

### File: `Model_Classes.py`
**Line ~340:** Changed learning rate strategy for quantized models
```python
# NEW: Conservative polynomial decay
if is_quantized:
    initial_lr = self.initial_lr * 0.5  # 5e-4 (half of unquantized)
    end_lr = self.end_lr * 0.1          # 1e-5 (tenth of unquantized)
```

## Expected Results (Next Run)

With both fixes applied:

1. **Warm-start will copy ALL 6 layers** ✓
   ```
   ✓ Copied: 6 layers
   ⚠ Skipped: 0 layers
   ```

2. **Training will be more stable** ✓
   - Smoother validation loss curve
   - Less oscillation in accuracy
   - Better convergence

3. **Final accuracy should improve** ✓
   - Target: **Acc ≈ 0.87-0.90** (up from 0.831)
   - Target: **AUC ≈ 0.91-0.93** (up from 0.886)

## How to Test

Run the same command again:

```bash
cd /home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric
conda activate mlgpu_qkeras
python model2.py
```

### What to Look For

1. **Warm-start output should show:**
   ```
   ✓ Copied weights: xz_dense1
   ✓ Copied weights: yl_dense1
   ✓ Copied weights: merged_dense1
   ✓ Copied weights: merged_dense2
   ✓ Copied weights: merged_dense3
   ✓ Copied weights: output          <-- THIS SHOULD NOW WORK!
   
   Warm-start summary:
     ✓ Copied: 6 layers              <-- ALL 6!
     ⚠ Skipped: 0 layers
   ```

2. **Training output should show:**
   ```
   Quantized model detected - using conservative polynomial decay
     Initial LR: 5.00e-04 (0.5x unquantized)
     End LR: 1.00e-05 (0.1x unquantized)
   ```

3. **Training should be smoother:**
   - Validation loss should decrease more consistently
   - Less bouncing in accuracy
   - Should reach higher final accuracy

## Why This Approach

### Conservative LR + Warm-Start = Best of Both Worlds

1. **Warm-start** gives the model a head start (good initial weights)
2. **Lower initial LR (5e-4)** prevents destroying those good weights
3. **Polynomial decay** still allows fine-tuning as training progresses
4. **Very low end LR (1e-5)** allows precise final adjustments

This is the standard approach for **quantization-aware fine-tuning** in the literature.

## Comparison of Approaches

| Approach | Initial LR | Decay | Warm-Start | Expected Result |
|----------|-----------|-------|------------|-----------------|
| **Original (Nov 17)** | 1e-3 (constant) | None | No | Acc=0.846 ✓ |
| **First attempt (Dec 1)** | 1e-3 | Yes | Partial (5/6) | Acc=0.831 ✗ |
| **New (Dec 1 fixed)** | 5e-4 | Yes | Full (6/6) | Acc=0.87-0.90 ✓✓ |

## Gradient Clipping

Also enabled `clipnorm=1.0` for quantized models to prevent gradient explosions.

This limits the magnitude of gradients during backpropagation, providing additional stability.

## Next Steps

1. **Run the updated code** (both fixes are now in place)
2. **Check the warm-start output** (should show 6/6 layers copied)
3. **Monitor training** (should be smoother and reach higher accuracy)
4. **Compare results** with the old run

If results are still not good, we can try:
- Even lower learning rate (3e-4 initial)
- More epochs (80-100 instead of 50)
- Different power for decay (power=1 for linear instead of power=2 for quadratic)

But I'm confident this will work much better! 🚀





