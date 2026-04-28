"""
Input Quantization Sweep for Model2.5

Tests the effect of fixing input quantization to 4, 6, 8, or 10 bits on Model2.5
with 6-bit weight/bias quantization and 8-bit activations (quantized_relu / quantized_sigmoid).

For each input bit-width, a separate Keras Tuner RandomSearch is run over architecture
hyperparameters (layer sizes, dropout, learning rate) — but NOT over input_bits.
This cleanly isolates the effect of input precision on best achievable performance.

Default: 5 HP trials per input bit-width × 4 bit-widths = 20 models total.

Example
-------
    cd .../eric
    python run_input_bits_sweep_model2_5.py \\
        --data_folder /path/to/tfrecords \\
        --max-trials 5 \\
        --epochs 20 \\
        --tag inputsweep

Dry-run sanity check (fast, 1 trial, 2 epochs):
    python run_input_bits_sweep_model2_5.py \\
        --data_folder /path/to/tfrecords \\
        --max-trials 1 \\
        --epochs 2 \\
        --tag dryrun

Author: Eric
Date: 2026
"""

import argparse
import sys

sys.path.append("/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/")
sys.path.append("../MuC_Smartpix_ML/")

from model2_5_fixed_input_bits import Model2_5_FixedInputBits

DEFAULT_DATA_FOLDER = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData_normalized"
)

# Fixed weight/bias quantization for all runs in this sweep
WEIGHT_BITS = 6
INT_BITS    = 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Model2.5 input quantization sweep: tests 4/6/8/10-bit inputs with "
            "fixed 6-bit weights. Runs HP search (architecture only) at each level."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER,
                        help="TFRecords root directory")
    parser.add_argument("--input-bits", type=int, nargs="+", default=[4, 6, 8, 10],
                        metavar="BITS",
                        help="Input bit-widths to sweep over")
    parser.add_argument("--max-trials", type=int, default=15,
                        help="HP trials per input bit-width (default 15 → 60 models total; space has exactly 15 unique combos)")
    parser.add_argument("--executions-per-trial", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per tuner trial")
    parser.add_argument("--tag", type=str, default="inputsweep",
                        help="Base suffix for output directory names")
    parser.add_argument("--weighted-br", action="store_true",
                        help="Optimize val_weighted_bkg_rej instead of val_binary_accuracy")
    parser.add_argument("--br95", type=float, default=0.1,
                        help="Weight for BR @ 95%% signal efficiency")
    parser.add_argument("--br98", type=float, default=0.7,
                        help="Weight for BR @ 98%% signal efficiency")
    parser.add_argument("--br99", type=float, default=0.2,
                        help="Weight for BR @ 99%% signal efficiency")
    parser.add_argument("--nmodule-weight-bits", type=int, default=None,
                        help="nModule-xlocal branch weight bits (defaults to WEIGHT_BITS=6)")
    parser.add_argument("--nmodule-int-bits", type=int, default=0)
    args = parser.parse_args()

    bkg_rej_weights = {0.95: args.br95, 0.98: args.br98, 0.99: args.br99}
    nm_wb = args.nmodule_weight_bits if args.nmodule_weight_bits is not None else WEIGHT_BITS
    nm_ib = args.nmodule_int_bits if args.nmodule_weight_bits is not None else INT_BITS

    total_models = len(args.input_bits) * args.max_trials

    print("=" * 70)
    print("Model2.5 — Input Quantization Sweep")
    print("=" * 70)
    print(f"  Data folder:      {args.data_folder}")
    print(f"  Input bit-widths: {args.input_bits}")
    print(f"  Weight bits:      {WEIGHT_BITS}w{INT_BITS}i  (fixed for all runs)")
    print(f"  nModule weights:  {nm_wb}w{nm_ib}i")
    print(f"  Activations:      quantized_relu(8,0) / quantized_sigmoid(8,0)  (fixed)")
    print(f"  Max trials:       {args.max_trials} per bit-width  ({total_models} models total, HP space = 15 unique combos)")
    print(f"  Exec / trial:     {args.executions_per_trial}")
    print(f"  Epochs / trial:   {args.epochs}")
    print(f"  Base tag:         {args.tag!r}")
    print(f"  Objective:        {'val_weighted_bkg_rej' if args.weighted_br else 'val_binary_accuracy'}")
    if args.weighted_br:
        print(f"  BR weights:       95={args.br95}  98={args.br98}  99={args.br99}")
    print("=" * 70)

    all_results = {}

    for input_bits in args.input_bits:
        # Each input bit-width gets a unique tag so output dirs don't collide.
        run_tag = f"ib{input_bits}b_{args.tag}"

        print(f"\n{'─' * 70}")
        print(f"  Input bits = {input_bits}  (tag: {run_tag!r})")
        print(f"{'─' * 70}")

        model = Model2_5_FixedInputBits(
            tfRecordFolder=args.data_folder,
            dense_units=128,
            nmodule_xlocal_units=32,
            dense2_units=128,
            dense3_units=64,
            dropout_rate=0.1,
            initial_lr=1e-3,
            end_lr=1e-4,
            power=2,
            bit_configs=[(WEIGHT_BITS, INT_BITS)],
            nmodule_xlocal_weight_bits=nm_wb,
            nmodule_xlocal_int_bits=nm_ib,
            input_bits=input_bits,
            input_int_bits=0,
        )

        results = model.runQuantizedHyperparameterTuning(
            bit_configs=[(WEIGHT_BITS, INT_BITS)],
            max_trials=args.max_trials,
            executions_per_trial=args.executions_per_trial,
            numEpochs=args.epochs,
            use_weighted_bkg_rej=args.weighted_br,
            bkg_rej_weights=bkg_rej_weights,
            hyperparameter_name_tag=run_tag,
        )
        all_results.update(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INPUT BITS SWEEP COMPLETED")
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
    print("Compare roc_metrics_summary.csv across the result directories to")
    print("see AUC / BR95 / BR98 / BR99 vs. input bit-width.")
    print("=" * 70)


if __name__ == "__main__":
    main()
