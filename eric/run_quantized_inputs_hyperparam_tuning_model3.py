"""
Quantized hyperparameter tuning for Model3 with input quantization.

Mirrors run_quantized_hyperparam_tuning_model3.py but uses Model3_QuantizedInputs
so that every raw input tensor is passed through a QActivation(quantized_bits)
layer before any Conv/Dense computation, and the tanh output is rescaled from
[-1, 1] to [0, 1] via Lambda((x + 1) / 2).

input_bits is set to weight_bits + 2 per config (same convention as Model2.5).
input_bits / input_int_bits are also included as searchable hyperparameters
inside the tuner.

Example
-------
    cd .../eric
    python run_quantized_inputs_hyperparam_tuning_model3.py \\
        --bit-config 4,0 --bit-config 6,0 \\
        --max-trials 10 --epochs 40 --tag qi

Author: Eric
Date: 2026
"""

import argparse
import sys

sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from model3_quantized_inputs import Model3_QuantizedInputs

DEFAULT_DATA_FOLDER = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData"
)


def _parse_bit_config(s: str):
    s = s.strip()
    if "," not in s:
        raise argparse.ArgumentTypeError(f"Expected W,I pair, got {s!r}")
    w, i = s.split(",", 1)
    return int(w.strip()), int(i.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Model3 quantized HP tuning with quantized inputs (tagged tuner/output dirs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER,
                        help="TFRecords root")
    parser.add_argument(
        "--bit-config",
        type=_parse_bit_config,
        action="append",
        dest="bit_configs",
        metavar="W,I",
        help="Weight quantizer (total bits, int bits). Repeat for multiple sweeps. Default: 4,0 6,0",
    )
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--executions-per-trial", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per tuner trial")
    parser.add_argument(
        "--tag", type=str, default="qi",
        help="Suffix for Keras Tuner project_name and results folder",
    )
    parser.add_argument(
        "--no-results-tag", action="store_true",
        help="Omit tag (same naming as baseline run_quantized_hyperparam_tuning_model3.py)",
    )
    parser.add_argument(
        "--weighted-br", action="store_true",
        help="Optimize val_weighted_bkg_rej instead of val_binary_accuracy",
    )
    parser.add_argument("--br95", type=float, default=0.1, help="Weight for BR @ 95%% sig. eff.")
    parser.add_argument("--br98", type=float, default=0.7, help="Weight for BR @ 98%% sig. eff.")
    parser.add_argument("--br99", type=float, default=0.2, help="Weight for BR @ 99%% sig. eff.")
    args = parser.parse_args()

    bit_configs = args.bit_configs if args.bit_configs else [(4, 0), (6, 0)]
    bkg_rej_weights = {0.95: args.br95, 0.98: args.br98, 0.99: args.br99}
    tag = None if args.no_results_tag else args.tag

    print("=" * 70)
    print("Model3_QuantizedInputs — quantized hyperparameter tuning")
    print("=" * 70)
    print(f"  Data folder:     {args.data_folder}")
    print(f"  Bit configs:     {bit_configs}")
    print(f"  Max trials:      {args.max_trials}")
    print(f"  Exec / trial:    {args.executions_per_trial}")
    print(f"  Epochs / trial:  {args.epochs}")
    print(f"  Results tag:     {tag!r}")
    print(f"  Objective:       {'val_weighted_bkg_rej' if args.weighted_br else 'val_binary_accuracy'}")
    if args.weighted_br:
        print(f"  BR weights:      95={args.br95} 98={args.br98} 99={args.br99}")
    print(f"  Input bits:      weight_bits + 2 per config (also a tunable HP)")
    print("=" * 70)

    all_results = {}

    for w, i in bit_configs:
        input_bits     = w + 2
        input_int_bits = i

        print(f"\n--- bit_config=({w},{i})  input_bits={input_bits} ---")

        model = Model3_QuantizedInputs(
            tfRecordFolder=args.data_folder,
            conv_filters=8,
            kernel_rows=3,
            kernel_cols=3,
            scalar_dense_units=32,
            merged_dense_1=200,
            merged_dense_2=100,
            dropout_rate=0.1,
            initial_lr=0.000871145,
            end_lr=5.3e-05,
            power=2,
            input_bits=input_bits,
            input_int_bits=input_int_bits,
            hp_input_bits_min=4,
            hp_input_bits_max=10,
            hp_input_bits_step=2,
            hp_input_int_bits_min=0,
            hp_input_int_bits_max=0,
            hp_input_int_bits_step=1,
        )

        results = model.runQuantizedHyperparameterTuning(
            bit_configs=[(w, i)],
            max_trials=args.max_trials,
            executions_per_trial=args.executions_per_trial,
            numEpochs=args.epochs,
            use_weighted_bkg_rej=args.weighted_br,
            bkg_rej_weights=bkg_rej_weights,
            hyperparameter_name_tag=tag,
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


if __name__ == "__main__":
    main()
