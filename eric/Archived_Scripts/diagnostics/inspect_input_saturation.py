"""
inspect_input_saturation.py

Empirical check for the cluster-saturation hypothesis behind the Model 3 vs
Model 2.5 weighted-bkg-rej gap on the April 2026 dataset.

What it does
------------
Loads ONE validation batch from the April 2026 TFRecord set and reports, for
each input feature the model sees, the raw value distribution and what happens
to it after `QActivation(quantized_bits(N, 0))` (the input quantizer used by
both Model 2.5 and Model 3 with `int_bits=0`, which clips to [-1, 1]).

It also reports the same statistics for `cluster` after applying the diagnostic
log1p Lambda so we can confirm the fix removes saturation BEFORE retraining.

Usage
-----
  cd .../eric
  python diagnostics/inspect_input_saturation.py \
      --bits 10 \
      --data_folder /local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized

Flags
-----
  --bits N            input_bits to quantize at (default 10, matches the run we
                      are diagnosing)
  --data_folder PATH  TFRecord root (default = April 2026 dataset)
  --no_profiles       skip the Model 2.5 profile comparison (use if profiles
                      can't be loaded from this TFRecord set)

Outputs
-------
  Prints a side-by-side report for cluster (raw + after log1p), x_profile,
  y_profile. The "saturation_before" half (cluster, raw) and
  "saturation_after" half (cluster, after log1p) are both produced in a single
  run so we don't have to flip a flag and re-run.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Make sibling model modules importable when run from eric/
HERE = os.path.dirname(os.path.abspath(__file__))
ERIC_DIR = os.path.dirname(HERE)
sys.path.insert(0, ERIC_DIR)
sys.path.insert(0, os.path.join(ERIC_DIR, "..", "MuC_Smartpix_ML"))

DEFAULT_DATA_FOLDER = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData_normalized"
)


def fmt_report(name: str, raw: np.ndarray, q: np.ndarray | None = None,
               bits: int = 10, int_bits: int = 0) -> str:
    """Build a multi-line report string for one feature tensor.

    `raw` is the value the model sees as input; `q` (optional) is the result of
    passing it through QActivation.

    For QKeras `quantized_bits(bits, int_bits, alpha=1)` the maximum
    representable magnitude is `2**int_bits - 2**(int_bits - bits + 1)`.
    With (10, 0) that's `1 - 2**-9 ≈ 0.99805`. Saturation = pinned at that
    extremum.
    """
    raw = np.asarray(raw)
    nz_mask = np.abs(raw) > 0
    nz = raw[nz_mask]

    # Actual saturation threshold for QKeras quantized_bits(bits, int_bits).
    qmax = (2.0 ** int_bits) - (2.0 ** (int_bits - bits + 1))
    # Use a tolerance one quantization step below qmax.
    qstep = 2.0 ** (int_bits - bits + 1)
    sat_thresh = qmax - 0.5 * qstep

    lines = [f"=== {name} ==="]
    lines.append(f"  shape={raw.shape}  total entries={raw.size}")
    lines.append(f"  nonzero count={nz.size}  ({nz.size / raw.size:.2%} of entries)")

    if nz.size > 0:
        # Percentile breakdown of raw NONZERO values.
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_vals = np.quantile(np.abs(nz), [p / 100.0 for p in pcts])
        pct_str = "  raw |.| percentiles (over nonzero):  " + "  ".join(
            f"p{p}={v:.3f}" for p, v in zip(pcts, pct_vals)
        )
        lines.append(
            f"  raw min/max/mean = "
            f"{raw.min():.3f} / {raw.max():.3f} / {nz.mean():.3f}"
        )
        lines.append(pct_str)

        # Bucket counts: how many nonzero values fall below qmax (i.e. survive
        # quantization with real precision) vs at-or-above (pinned at ±qmax)?
        below_thresh = qmax  # any |v| < qmax is sub-saturation
        below = int((np.abs(nz) < below_thresh).sum())
        atabove = int((np.abs(nz) >= below_thresh).sum())
        lines.append(
            f"  raw bucket vs qmax({qmax:.4f}):  |v|<qmax={below}  "
            f"({below/nz.size:.3%} of nonzero)  ;  |v|>=qmax={atabove}  "
            f"({atabove/nz.size:.3%} of nonzero)"
        )

        # Finer breakdown of the nonzero distribution against the quantizer's
        # dynamic range. Useful when profiles vs cluster have similar maxes
        # but very different shapes underneath.
        if qmax > 0:
            edges = np.array([0, 0.05*qmax, 0.1*qmax, 0.25*qmax, 0.5*qmax,
                              0.75*qmax, qmax, np.inf])
            labels = ["[0,5%]", "(5,10%]", "(10,25%]", "(25,50%]",
                      "(50,75%]", "(75,100%]", ">=qmax (saturates)"]
            counts, _ = np.histogram(np.abs(nz), bins=edges)
            lines.append("  raw |.| distribution within quantizer range:")
            for lab, c in zip(labels, counts):
                lines.append(f"    {lab:>22s} : {c:8d}  ({c/nz.size:.2%})")

    if q is not None:
        q = np.asarray(q)
        sat_mask = np.abs(q) >= sat_thresh
        sat_count = int(sat_mask.sum())
        lines.append(
            f"  quantized saturated count (|q|>={sat_thresh:.4f}) = {sat_count}  "
            f"({sat_count / q.size:.3%} of all entries, "
            f"{(sat_count / max(nz.size, 1)):.3%} of nonzero entries)"
        )
        lines.append(
            f"  quantized min/max = {q.min():.4f} / {q.max():.4f}  "
            f"(qmax={qmax:.4f}, qstep={qstep:.5f})"
        )
        # How many distinct quantized levels are actually used? If the input is
        # mostly clipped, this collapses to a tiny number (effectively a binary
        # mask).
        unique_q = np.unique(q)
        lines.append(
            f"  distinct quantized levels used = {unique_q.size}  "
            f"(of 2^{bits} = {2**bits} representable)"
        )
        # Show the most common quantized values and their counts to see whether
        # the quantizer collapses everything to <a few> levels.
        if unique_q.size <= 20:
            uvals, ucounts = np.unique(q, return_counts=True)
            order = np.argsort(-ucounts)
            top = ", ".join(
                f"{uvals[i]:+.4f}({ucounts[i]})"
                for i in order[:min(10, uvals.size)]
            )
            lines.append(f"  quantized level histogram (val(count)): {top}")

    return "\n".join(lines)


def make_log1p(t: np.ndarray) -> np.ndarray:
    """Mirror the Lambda we'll insert in Model 3: sign(x) * log2(1 + |x|)."""
    import tensorflow as tf

    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)
    out = tf.sign(t_tf) * tf.math.log1p(tf.abs(t_tf)) / tf.math.log(
        tf.constant(2.0, dtype=tf.float32)
    )
    return out.numpy()


def quantize(t: np.ndarray, bits: int) -> np.ndarray:
    """Apply the same input quantization Model 3 / Model 2.5 use:
    QActivation(quantized_bits(bits, 0)).
    """
    return quantize_with_intbits(t, bits, 0)


def quantize_with_intbits(t: np.ndarray, bits: int, int_bits: int) -> np.ndarray:
    """Quantize via QActivation(quantized_bits(bits, int_bits))."""
    from qkeras import QActivation
    import tensorflow as tf

    qact = QActivation(f"quantized_bits({bits},{int_bits})")
    return qact(tf.convert_to_tensor(t, dtype=tf.float32)).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, default=10,
                        help="input_bits passed to quantized_bits (default 10)")
    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER,
                        help="TFRecord root (default: April 2026 dataset)")
    parser.add_argument("--no_profiles", action="store_true",
                        help="Skip x_profile/y_profile inspection (use if the "
                             "TFRecord set was built without those features)")
    args = parser.parse_args()

    print("=" * 72)
    print(f"Input saturation inspection")
    print(f"  data_folder : {args.data_folder}")
    print(f"  bits        : quantized_bits({args.bits},0)  -> clips to [-1, 1]")
    print("=" * 72)

    # --- Load Model 3's batch (cluster + scalar features) ---
    from model3_quantized_inputs import Model3_QuantizedInputs

    print("\n[load] instantiating Model3_QuantizedInputs and loading TFRecords...")
    m3 = Model3_QuantizedInputs(
        tfRecordFolder=args.data_folder,
        input_bits=args.bits,
        input_int_bits=0,
    )
    m3.loadTfRecords()
    X3, _ = m3.validation_generator[0]
    cluster_raw = X3["cluster"].numpy() if hasattr(X3["cluster"], "numpy") else np.asarray(X3["cluster"])
    print(f"[load] cluster batch shape: {cluster_raw.shape}")

    print("\n--- Pre-fix: cluster as Model 3 currently sees it ---")
    cluster_q_pre = quantize(cluster_raw, args.bits)
    print(fmt_report("cluster (raw -> QActivation)", cluster_raw, cluster_q_pre,
                     bits=args.bits, int_bits=0))

    print("\n--- Post-fix variant A: cluster -> Lambda(log1p) -> QActivation(N,0) ---")
    cluster_logged = make_log1p(cluster_raw)
    cluster_q_post = quantize(cluster_logged, args.bits)
    print(fmt_report("cluster (log1p -> QActivation)", cluster_logged, cluster_q_post,
                     bits=args.bits, int_bits=0))

    # Variant B: same input but more integer bits in the quantizer (no Lambda).
    print("\n--- Post-fix variant B: cluster -> QActivation(N, 4)  [int_bits=4] ---")
    cluster_q_b = quantize_with_intbits(cluster_raw, args.bits, 4)
    print(fmt_report("cluster (raw -> QActivation int4)", cluster_raw, cluster_q_b,
                     bits=args.bits, int_bits=4))

    # --- Optional: load profiles via Model 2.5's loader for side-by-side ---
    if not args.no_profiles:
        try:
            from model2_5_quantized_inputs import Model2_5_QuantizedInputs

            print("\n[load] instantiating Model2_5_QuantizedInputs (for x/y profiles)...")
            m25 = Model2_5_QuantizedInputs(
                tfRecordFolder=args.data_folder,
                input_bits=args.bits,
                input_int_bits=0,
            )
            m25.loadTfRecords()
            X25, _ = m25.validation_generator[0]
            xprof_raw = X25["x_profile"].numpy() if hasattr(X25["x_profile"], "numpy") else np.asarray(X25["x_profile"])
            yprof_raw = X25["y_profile"].numpy() if hasattr(X25["y_profile"], "numpy") else np.asarray(X25["y_profile"])

            print("\n--- Reference: x_profile / y_profile as Model 2.5 sees them ---")
            xprof_q = quantize(xprof_raw, args.bits)
            yprof_q = quantize(yprof_raw, args.bits)
            print(fmt_report("x_profile (raw -> QActivation)", xprof_raw, xprof_q,
                             bits=args.bits, int_bits=0))
            print()
            print(fmt_report("y_profile (raw -> QActivation)", yprof_raw, yprof_q,
                             bits=args.bits, int_bits=0))
        except Exception as e:
            print(f"\n[warn] Skipped profile inspection ({type(e).__name__}: {e}). "
                  f"Re-run with --no_profiles to silence.")

    print("\n" + "=" * 72)
    print("Decision rules (from plan)")
    print("=" * 72)
    print("  Step 0 hypothesis confirmed if:")
    print("    - cluster (raw -> QActivation) saturated fraction is >> 0")
    print("    - x_profile / y_profile saturated fractions are ~0")
    print("  Step 2 fix verified if:")
    print("    - cluster (log1p -> QActivation) saturated fraction is ~0")
    print("=" * 72)


if __name__ == "__main__":
    main()
