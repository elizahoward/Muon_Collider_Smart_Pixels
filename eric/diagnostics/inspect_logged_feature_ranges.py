"""
inspect_logged_feature_ranges.py

Tiny diagnostic to inspect real value ranges in TFRecord inputs and explain why
"double-log" profile features can still be > 1.

What this checks
----------------
1) Loads one validation batch using the same Model2.5 TFRecord path as training.
2) Reads stored tensors exactly as the model receives them:
   - cluster
   - x_profile
   - y_profile
3) Recomputes profile tensors from stored cluster:
   - sum across axes (same as generator)
   - apply signed log2(1 + |x|) (same as generator's second profile log)
4) Compares recomputed vs stored profiles.
5) Reports min/max/percentiles and the fraction above thresholds (1, 2, 3, 4).

Interpretation note
-------------------
For signed log2(1 + |v|), output > 1 whenever |v| > 1.
So if the profile sum before the second log exceeds 1 (very common), the
"double-log" profile can still be > 1.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ERIC_DIR = os.path.dirname(HERE)
sys.path.insert(0, ERIC_DIR)
sys.path.insert(0, os.path.join(ERIC_DIR, "..", "MuC_Smartpix_ML"))
sys.path.insert(0, os.path.join(ERIC_DIR, "..", "MuC_Smartpix_Data_Production", "tfRecords"))

DEFAULT_DATA_FOLDER = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData_normalized"
)


def signed_log2_1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * (np.log1p(np.abs(x)) / np.log(2.0))


def stat_block(name: str, arr: np.ndarray, thresholds: Iterable[float] = (1.0, 2.0, 3.0, 4.0)) -> str:
    a = np.asarray(arr, dtype=np.float32).ravel()
    abs_a = np.abs(a)
    nz = abs_a[abs_a > 0]
    lines = [f"=== {name} ==="]
    lines.append(f"  shape={arr.shape}  total={a.size}")
    lines.append(f"  min/max/mean={a.min():.4f} / {a.max():.4f} / {a.mean():.4f}")
    lines.append(f"  nonzero={nz.size} ({(nz.size / a.size):.2%} of all)")
    if nz.size:
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        vals = np.quantile(nz, [p / 100.0 for p in pcts])
        lines.append(
            "  |v| percentiles over nonzero: " +
            "  ".join(f"p{p}={v:.4f}" for p, v in zip(pcts, vals))
        )
        for t in thresholds:
            c = int((nz > t).sum())
            lines.append(f"  |v|>{t:.1f}: {c} ({(c / nz.size):.2%} of nonzero)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--batch-index", type=int, default=0)
    args = parser.parse_args()

    from model3_quantized_inputs import Model3_QuantizedInputs
    from model2_5_quantized_inputs import Model2_5_QuantizedInputs

    print("=" * 78)
    print("Inspect logged feature ranges from TFRecords")
    print(f"  data_folder : {args.data_folder}")
    print(f"  batch_index : {args.batch_index}")
    print("=" * 78)

    # Cluster path via Model3 loader (contains "cluster" feature).
    m3 = Model3_QuantizedInputs(tfRecordFolder=args.data_folder, input_bits=10, input_int_bits=0)
    m3.loadTfRecords()
    X3, _ = m3.validation_generator[args.batch_index]
    cluster = np.asarray(X3["cluster"])

    print("\n--- Stored TFRecord tensors ---")
    print(stat_block("cluster (stored)", cluster))

    # Recompute profile pipeline from stored cluster:
    # 1) sum over axes, 2) signed log2(1+|x|).
    y_sum_from_cluster = np.sum(cluster, axis=2)  # shape: (batch, 13)
    x_sum_from_cluster = np.sum(cluster, axis=1)  # shape: (batch, 21)
    y_prof_rebuilt = signed_log2_1p(y_sum_from_cluster)
    x_prof_rebuilt = signed_log2_1p(x_sum_from_cluster)

    print("\n--- Recomputed profiles from stored cluster ---")
    print(stat_block("x_profile rebuilt", x_prof_rebuilt))
    print()
    print(stat_block("y_profile rebuilt", y_prof_rebuilt))

    # Load stored profiles directly via Model2.5 loader.
    m25 = Model2_5_QuantizedInputs(tfRecordFolder=args.data_folder, input_bits=10, input_int_bits=0)
    m25.loadTfRecords()
    X25, _ = m25.validation_generator[args.batch_index]
    x_profile_stored = np.asarray(X25["x_profile"])
    y_profile_stored = np.asarray(X25["y_profile"])

    print("\n--- Stored TFRecord profiles (Model2.5 view) ---")
    print(stat_block("x_profile stored", x_profile_stored))
    print()
    print(stat_block("y_profile stored", y_profile_stored))

    # Consistency check.
    x_diff = np.abs(x_profile_stored - x_prof_rebuilt)
    y_diff = np.abs(y_profile_stored - y_prof_rebuilt)
    print("\n--- Consistency checks (stored vs rebuilt) ---")
    print(
        "x_profile max|diff|={:.6g}, mean|diff|={:.6g}".format(
            float(x_diff.max()), float(x_diff.mean())
        )
    )
    print(
        "y_profile max|diff|={:.6g}, mean|diff|={:.6g}".format(
            float(y_diff.max()), float(y_diff.mean())
        )
    )

    print("\n--- Why values can still be > 1 after 'double log' ---")
    print("For signed log2(1+|v|): output > 1 iff |v| > 1.")
    x_pre_second = np.abs(x_sum_from_cluster)
    y_pre_second = np.abs(y_sum_from_cluster)
    x_gt1 = int((x_pre_second > 1.0).sum())
    y_gt1 = int((y_pre_second > 1.0).sum())
    print(
        f"x_profile pre-second-log |sum(cluster)|>1: {x_gt1}/{x_pre_second.size} "
        f"({(x_gt1 / x_pre_second.size):.2%})"
    )
    print(
        f"y_profile pre-second-log |sum(cluster)|>1: {y_gt1}/{y_pre_second.size} "
        f"({(y_gt1 / y_pre_second.size):.2%})"
    )
    print(
        "If profile sums are usually above 1 before the second log, then post-log "
        "profiles being >1 is expected."
    )


if __name__ == "__main__":
    main()

