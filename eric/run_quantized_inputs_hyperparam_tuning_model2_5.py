"""
Quantized hyperparameter tuning for Model2.5 with **input quantization** (QActivation on every input).

This mirrors the workflow in ``run_quantized_hyperparam_tuning_model2_5.py`` (Keras Tuner RandomSearch on
layer sizes, dropout, learning rate, and — via ``Model2_5_QuantizedInputs`` — ``input_bits`` /
``input_int_bits``), but uses ``Model2_5_QuantizedInputs`` and passes ``hyperparameter_name_tag`` so
tuner project folders and on-disk result directories are suffixed (default: ``qi``), avoiding clashes
with baseline Model2.5 quantized HP runs.

Example
-------
    cd .../eric
    python run_quantized_inputs_hyperparam_tuning_model2_5.py \\
        --data_folder /home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026V2_Apr/TF_Records/filtering_records16384_data_shuffled_single_bigData \\
        --bit-config 4,0 --bit-config 6,0 --bit-config 8,0 --bit-config 10,0 \\
        --max-trials 1 \\
        --executions-per-trial 2 \\
        --epochs 20 \\
        --tag qi

Author: Eric
Date: 2026
"""

import argparse
import sys

sys.path.append("/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/")
sys.path.append("../MuC_Smartpix_ML/")

from model2_5_quantized_inputs import Model2_5_QuantizedInputs

DEFAULT_DATA_FOLDER = (
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/"
    "Data_Files/Data_Set_2026V2_Apr/TF_Records/filtering_records16384_data_shuffled_single_bigData"
)


def _parse_bit_config(s: str):
    s = s.strip()
    if "," not in s:
        raise argparse.ArgumentTypeError(f"Expected W,I pair, got {s!r}")
    w, i = s.split(",", 1)
    return int(w.strip()), int(i.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Model2.5 quantized HP tuning with quantized inputs (tagged tuner/output dirs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER, help="TFRecords root")
    parser.add_argument(
        "--bit-config",
        type=_parse_bit_config,
        action="append",
        dest="bit_configs",
        metavar="W,I",
        help="Weight quantizer (total bits, int bits). Repeat for multiple sweeps. Default: 6,0",
    )
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--executions-per-trial", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per tuner trial")
    parser.add_argument(
        "--tag",
        type=str,
        default="qi",
        help="Suffix for Keras Tuner project_name and results folder",
    )
    parser.add_argument(
        "--no-results-tag",
        action="store_true",
        help="Omit tag (same naming as legacy Model2 runQuantizedHyperparameterTuning)",
    )
    parser.add_argument(
        "--weighted-br",
        action="store_true",
        help="Optimize val_weighted_bkg_rej instead of val_binary_accuracy",
    )
    parser.add_argument("--br95", type=float, default=0.1, help="Weight for BR @ 95%% sig. eff.")
    parser.add_argument("--br98", type=float, default=0.7, help="Weight for BR @ 98%% sig. eff.")
    parser.add_argument("--br99", type=float, default=0.2, help="Weight for BR @ 99%% sig. eff.")
    parser.add_argument("--nmodule-weight-bits", type=int, default=None, help="nModule-xlocal branch weight bits")
    parser.add_argument("--nmodule-int-bits", type=int, default=0)
    parser.add_argument("--hp-input-bits-min", type=int, default=4)
    parser.add_argument("--hp-input-bits-max", type=int, default=16)
    parser.add_argument("--hp-input-bits-step", type=int, default=2)
    parser.add_argument("--hp-input-int-bits-min", type=int, default=0)
    parser.add_argument("--hp-input-int-bits-max", type=int, default=4)
    parser.add_argument("--hp-input-int-bits-step", type=int, default=1)
    args = parser.parse_args()

    bit_configs = args.bit_configs if args.bit_configs else [(4, 0), (6, 0), (8, 0), (10, 0)]

    if args.nmodule_weight_bits is not None:
        nm_wb = args.nmodule_weight_bits
        nm_ib = args.nmodule_int_bits
    else:
        first_w, first_i = bit_configs[0]
        nm_wb, nm_ib = first_w, first_i

    model = Model2_5_QuantizedInputs(
        tfRecordFolder=args.data_folder,
        dense_units=128,
        nmodule_xlocal_units=32,
        dense2_units=128,
        dense3_units=64,
        dropout_rate=0.1,
        initial_lr=1e-3,
        end_lr=1e-4,
        power=2,
        bit_configs=bit_configs,
        nmodule_xlocal_weight_bits=nm_wb,
        nmodule_xlocal_int_bits=nm_ib,
        hp_input_bits_min=args.hp_input_bits_min,
        hp_input_bits_max=args.hp_input_bits_max,
        hp_input_bits_step=args.hp_input_bits_step,
        hp_input_int_bits_min=args.hp_input_int_bits_min,
        hp_input_int_bits_max=args.hp_input_int_bits_max,
        hp_input_int_bits_step=args.hp_input_int_bits_step,
    )

    bkg_rej_weights = {0.95: args.br95, 0.98: args.br98, 0.99: args.br99}

    print("=" * 70)
    print("Model2.5_QuantizedInputs — quantized hyperparameter tuning")
    print("=" * 70)
    print(f"  Data folder:     {args.data_folder}")
    print(f"  Bit configs:     {bit_configs}")
    print(f"  nModule-xlocal:  {nm_wb} weight bits, {nm_ib} int bits")
    print(f"  Max trials:      {args.max_trials}")
    print(f"  Exec / trial:    {args.executions_per_trial}")
    print(f"  Epochs / trial:  {args.epochs}")
    tag = None if args.no_results_tag else args.tag
    print(f"  Results tag:     {tag!r}")
    print(f"  Objective:       {'val_weighted_bkg_rej' if args.weighted_br else 'val_binary_accuracy'}")
    if args.weighted_br:
        print(f"  BR weights:      95={args.br95} 98={args.br98} 99={args.br99}")
    print(f"  Input HP bits:   [{args.hp_input_bits_min}, {args.hp_input_bits_max}] step {args.hp_input_bits_step}")
    print("=" * 70)

    results = model.runQuantizedHyperparameterTuning(
        bit_configs=bit_configs,
        max_trials=args.max_trials,
        executions_per_trial=args.executions_per_trial,
        numEpochs=args.epochs,
        use_weighted_bkg_rej=args.weighted_br,
        bkg_rej_weights=bkg_rej_weights,
        hyperparameter_name_tag=tag,
    )

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 70)
    for config_name, result in results.items():
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
