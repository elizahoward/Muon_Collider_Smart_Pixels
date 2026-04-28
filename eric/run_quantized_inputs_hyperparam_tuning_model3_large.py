"""
Large-capacity hyperparameter tuning for Model3 — 10-bit weights & bias.

Uses Model3_QuantizedInputs_Large, which has a much bigger HP search space
than the original (conv filters up to 256, dense heads up to 2048 units).

Hypothesis: Model3 underperforms Model2 because it is capacity-limited.
This search lets the tuner find much larger architectures to test that.

Example
-------
    cd .../eric
    python run_quantized_inputs_hyperparam_tuning_model3_large.py
"""

import gc
import os
import sys

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from model3_quantized_inputs_large import Model3_QuantizedInputs_Large

DATA_FOLDER = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData_normalized"
)

BIT_CONFIGS         = [(10, 0)]
MAX_TRIALS          = 10
EXECUTIONS_PER_TRIAL = 1
EPOCHS              = 35
TAG                 = "large_10bit"

print("=" * 70)
print("Model3_QuantizedInputs_Large — 10-bit HP tuning (large search space)")
print("=" * 70)
print(f"  Bit configs  : {BIT_CONFIGS}")
print(f"  Max trials   : {MAX_TRIALS}")
print(f"  Epochs/trial : {EPOCHS}")
print()
print("  Search space:")
print("    conv_filters       : 16 / 32 / 48 / 64 / 96 / 128 / 192 / 256")
print("    kernel_rows/cols   : 3 / 5 / 7")
print("    scalar_dense_units : 32 / 64 / 96 / 128 / 192 / 256 / 384 / 512")
print("    merged_dense_1     : 128 / 192 / 256 / 384 / 512 / 640 / 768 / 1024 / 1280 / 1536 / 2048")
print("    merged_dense_2     : 64 / 96 / 128 / 192 / 256 / 384 / 512 / 640 / 768 / 1024")
print("    dropout_rate       : 0.0 – 0.4 (step 0.1)")
print("    learning_rate      : 5e-5 – 5e-2 (log)")
print("    input_bits         : 4 – 12 (step 2)")
print("=" * 70)

all_results = {}

for w, i in BIT_CONFIGS:
    tf.keras.backend.clear_session()
    gc.collect()

    input_bits = w + 2   # 12 for 10-bit weights

    print(f"\n--- bit_config=({w},{i})  input_bits={input_bits} ---")

    model = Model3_QuantizedInputs_Large(
        tfRecordFolder=DATA_FOLDER,
        # baseline values (not used during HP tuning — tuner overrides these)
        conv_filters=64,
        kernel_rows=3,
        kernel_cols=3,
        scalar_dense_units=128,
        merged_dense_1=512,
        merged_dense_2=256,
        dropout_rate=0.1,
        initial_lr=0.000871145,
        end_lr=5.3e-05,
        power=2,
        input_bits=input_bits,
        input_int_bits=i,
        hp_input_bits_min=4,
        hp_input_bits_max=12,
        hp_input_bits_step=2,
        hp_input_int_bits_min=0,
        hp_input_int_bits_max=0,
        hp_input_int_bits_step=1,
    )

    results = model.runQuantizedHyperparameterTuning(
        bit_configs=[(w, i)],
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        numEpochs=EPOCHS,
        use_weighted_bkg_rej=False,
        hyperparameter_name_tag=TAG,
    )
    all_results.update(results)

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)
for config_name, result in all_results.items():
    print(f"\n{config_name}:")
    print(f"  Results directory: {result['config_dir']}/")
    print(f"  Number of trials:  {result['num_trials']}")
    print(f"  Best model:        {result['model_files'][0]}")
    print(f"  Summary file:      {result['summary_file']}")
    print("  Best hyperparameters:")
    for param, value in result["best_hyperparameters"].items():
        print(f"    {param}: {value}")
print("\n" + "=" * 70)
