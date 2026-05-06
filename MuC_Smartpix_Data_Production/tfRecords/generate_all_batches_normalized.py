#!/usr/bin/env python3
"""
Generate TFRecords with cluster, x_profile, and y_profile normalized to [0, 1].

This script generates a parallel set of TFRecords where those three features
are divided by fixed physical-maximum divisors and clipped to [0, 1]:

    cluster   /= 18.025  (observed max ~17.5 + 3% headroom)
    x_profile /=  8.071  (theoretical max log2(1+13×17.5)≈7.836 + 3% headroom)
    y_profile /=  8.781  (theoretical max log2(1+21×17.5)≈8.525 + 3% headroom)

All scalar features (y_local, x_local, nModule, z_global, total_charge) are
already normalized to [0, 1] in the parent class and are left unchanged.

Output directory naming: filtering_records{batch_size}_data_shuffled_{ts}_bigData_normalized

Usage
-----
    cd .../MuC_Smartpix_Data_Production/tfRecords
    python generate_all_batches_normalized.py
"""

import os
import sys
import glob
import shutil
import time
from pathlib import Path

import numpy as np

# Make the sibling ODG module importable when run from this directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import OptimizedDataGenerator4_data_shuffled_bigData_NewFormat as ODG


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------
CLUSTER_NORM   = 18.025  # observed max ~17.5 × 1.03 (3% headroom)
X_PROFILE_NORM =  8.071  # log2(1+13×17.5)≈7.836 × 1.03 (3% headroom); x_profile has 21 entries, each summed over 13 rows
Y_PROFILE_NORM =  8.781  # log2(1+21×17.5)≈8.525 × 1.03 (3% headroom); y_profile has 13 entries, each summed over 21 cols


class OptimizedDataGeneratorNormalized(ODG.OptimizedDataGeneratorDataShuffledBigData):
    """
    Drop-in replacement for OptimizedDataGeneratorDataShuffledBigData that
    normalizes cluster, x_profile, and y_profile to [0, 1] before writing
    TFRecord files.

    Implementation note
    -------------------
    The parent __init__ calls self.save_batches_parallel() immediately after
    building self.x_features.  We intercept that call via the _defer_save flag
    so we can normalize x_features first, then save.
    """

    def __init__(self, **kwargs):
        self._defer_save = True          # suppress the auto-save in parent __init__
        super().__init__(**kwargs)
        # At this point self.x_features is fully populated but nothing is
        # written to disk yet (save_batches_parallel was a no-op above).
        if not kwargs.get('load_records', False):
            self._normalize_features()
            self._defer_save = False
            self.save_batches_parallel()   # now write the normalized data
            # Parent set tfrecord_filenames to [] because the save was deferred;
            # refresh it now that the files actually exist.
            import tensorflow as tf
            self.tfrecord_filenames = np.sort(np.array(
                tf.io.gfile.glob(os.path.join(self.tf_records_dir, "*.tfrecord"))
            ))

    # Override to honour the defer flag.
    def save_batches_parallel(self):
        if getattr(self, '_defer_save', False):
            return   # called from parent __init__ before normalization; skip
        super().save_batches_parallel()

    def _normalize_features(self):
        """Divide spatial features by their fixed-maximum constants, clip to [0,1]."""
        if 'cluster' in self.x_features:
            self.x_features['cluster'] = np.clip(
                self.x_features['cluster'] / CLUSTER_NORM, 0.0, 1.0
            )
            print(f"[normalize] cluster  / {CLUSTER_NORM}  -> "
                  f"range [{self.x_features['cluster'].min():.4f}, "
                  f"{self.x_features['cluster'].max():.4f}]")

        if 'x_profile' in self.x_features:
            self.x_features['x_profile'] = np.clip(
                self.x_features['x_profile'] / X_PROFILE_NORM, 0.0, 1.0
            )
            print(f"[normalize] x_profile / {X_PROFILE_NORM}  -> "
                  f"range [{self.x_features['x_profile'].min():.4f}, "
                  f"{self.x_features['x_profile'].max():.4f}]")

        if 'y_profile' in self.x_features:
            self.x_features['y_profile'] = np.clip(
                self.x_features['y_profile'] / Y_PROFILE_NORM, 0.0, 1.0
            )
            print(f"[normalize] y_profile / {Y_PROFILE_NORM}  -> "
                  f"range [{self.x_features['y_profile'].min():.4f}, "
                  f"{self.x_features['y_profile'].max():.4f}]")


# ---------------------------------------------------------------------------
# Runner 

def create_normalized_tf_records(batch_size, is_single_timestamp,
                                 base_output_dir, data_directory_path,
                                 random_seed=42):
    """Generate one train+validation pair of normalized TFRecord directories."""

    is_directory_recursive = False
    file_type      = "parquet"
    data_format    = "3D"
    normalization  = 1
    file_fraction  = 0.8
    to_standardize = False
    transpose      = None
    x_feature_description = "all"
    filteringBIB   = True
    shuffle_data   = True

    if is_single_timestamp:
        input_shape     = (1, 13, 21)
        time_stamps     = [19]
        timestamp_suffix = "single"
    else:
        input_shape     = (20, 13, 21)
        time_stamps     = list(range(20))
        timestamp_suffix = "all"

    directory_name = (f"filtering_records{batch_size}_data_shuffled_"
                      f"{timestamp_suffix}_bigData_normalized")
    records_dir = os.path.join(base_output_dir, directory_name)

    print(f"\n{'='*60}")
    print(f"CREATING: {directory_name}")
    print(f"Batch size       : {batch_size}")
    print(f"Timestamps       : {'Single (19)' if is_single_timestamp else 'All (0-19)'}")
    print(f"Random seed      : {random_seed}")
    print(f"Normalization    : cluster/{CLUSTER_NORM}, "
          f"x_profile/{X_PROFILE_NORM}, y_profile/{Y_PROFILE_NORM}")
    print(f"{'='*60}")

    if not os.path.exists(records_dir):
        os.makedirs(records_dir)
    else:
        print(f"Directory {records_dir} already exists. Cleaning...")
        for fname in os.listdir(records_dir):
            fpath = os.path.join(records_dir, fname)
            try:
                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f"  Failed to delete {fpath}: {e}")

    tf_dir_train      = Path(records_dir, "tfrecords_train").resolve()
    tf_dir_validation = Path(records_dir, "tfrecords_validation").resolve()
    os.makedirs(tf_dir_train,      exist_ok=True)
    os.makedirs(tf_dir_validation, exist_ok=True)

    # ---- file counts --------------------------------------------------------
    def count_files(pattern):
        return len(glob.glob(
            data_directory_path + pattern + data_format + "_*." + file_type,
            recursive=is_directory_recursive
        ))

    total_mm  = count_files("bib_mm_recon")
    total_mp  = count_files("bib_mp_recon")
    total_sig = count_files("signal_recon")

    count_mm  = round(file_fraction * total_mm)
    count_mp  = round(file_fraction * total_mp)
    count_sig = round(file_fraction * total_sig)

    print(f"Files  mm : total={total_mm}  train={count_mm}  val={total_mm-count_mm}")
    print(f"Files  mp : total={total_mp}  train={count_mp}  val={total_mp-count_mp}")
    print(f"Files sig : total={total_sig}  train={count_sig}  val={total_sig-count_sig}")

    # ---- training generator -------------------------------------------------
    print("\nCreating training generator...")
    t0 = time.time()
    OptimizedDataGeneratorNormalized(
        data_directory_path=data_directory_path,
        is_directory_recursive=is_directory_recursive,
        file_type=file_type,
        data_format=data_format,
        batch_size=batch_size,
        to_standardize=to_standardize,
        normalization=normalization,
        file_count_mm=count_mm,
        file_count_mp=count_mp,
        file_count_sig=count_sig,
        input_shape=input_shape,
        transpose=transpose,
        time_stamps=time_stamps,
        tf_records_dir=str(tf_dir_train),
        x_feature_description=x_feature_description,
        filteringBIB=filteringBIB,
        shuffle_data=shuffle_data,
        random_seed=random_seed,
    )
    print(f"Training generator done in {time.time() - t0:.1f}s")

    # ---- validation generator -----------------------------------------------
    print("\nCreating validation generator...")
    t0 = time.time()
    OptimizedDataGeneratorNormalized(
        data_directory_path=data_directory_path,
        is_directory_recursive=is_directory_recursive,
        file_type=file_type,
        data_format=data_format,
        batch_size=batch_size,
        to_standardize=to_standardize,
        normalization=normalization,
        file_count_mm=total_mm - count_mm,
        file_count_mp=total_mp - count_mp,
        file_count_sig=total_sig - count_sig,
        files_from_end=True,
        input_shape=input_shape,
        transpose=transpose,
        time_stamps=time_stamps,
        tf_records_dir=str(tf_dir_validation),
        x_feature_description=x_feature_description,
        filteringBIB=filteringBIB,
        shuffle_data=shuffle_data,
        random_seed=random_seed + 1,
    )
    print(f"Validation generator done in {time.time() - t0:.1f}s")

    print(f"\n✓ Normalized TFRecords written to:")
    print(f"    train : {tf_dir_train}")
    print(f"    val   : {tf_dir_validation}")
    return records_dir


def main():
    datasetName = "Data_Set_2026V2_Apr"
    data_root   = Path("/local/d1/smartpixML/2026Datasets/Data_Files/") / datasetName
    base_output_dir      = str(data_root / "TF_Records")
    data_directory_path  = str(data_root / "Parquet_Files") + "/"

    os.makedirs(base_output_dir, exist_ok=True)

    batch_sizes       = [16384]
    timestamp_configs = [(True, "single timestamp (19)")]

    print(f"\n{'='*80}")
    print("GENERATING NORMALIZED TF RECORDS")
    print(f"  Output dir        : {base_output_dir}")
    print(f"  Batch sizes       : {batch_sizes}")
    print(f"  cluster  / {CLUSTER_NORM}  (3% headroom over observed max ~17.5)")
    print(f"  x_profile / {X_PROFILE_NORM}  (3% headroom over theoretical max ~7.836)")
    print(f"  y_profile / {Y_PROFILE_NORM}  (3% headroom over theoretical max ~8.525)")
    print(f"{'='*80}")

    t_total = time.time()
    created = []

    for batch_size in batch_sizes:
        for is_single_ts, ts_desc in timestamp_configs:
            try:
                d = create_normalized_tf_records(
                    batch_size=batch_size,
                    is_single_timestamp=is_single_ts,
                    base_output_dir=base_output_dir,
                    data_directory_path=data_directory_path,
                    random_seed=42,
                )
                created.append(d)
            except Exception as e:
                import traceback
                print(f"ERROR for batch_size={batch_size} {ts_desc}: {e}")
                traceback.print_exc()

    elapsed = time.time() - t_total
    print(f"\n{'='*80}")
    print(f"DONE — {len(created)} directory(ies) created in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")
    for d in created:
        print(f"  {os.path.basename(d)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
