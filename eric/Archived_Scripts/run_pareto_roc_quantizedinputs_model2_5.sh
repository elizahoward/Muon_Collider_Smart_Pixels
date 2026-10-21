#!/bin/bash
# Run Pareto ROC selection sequentially for all model2.5 quantized-input bit-width sweeps.

set -e

BASE=/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric
SCRIPT=$BASE/ericHLS/analyze_and_select_pareto_roc.py
DATA=/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026V2_Apr/TF_Records/filtering_records16384_data_shuffled_single_bigData

# Control TensorFlow parallelism without modifying the Python script
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export OMP_NUM_THREADS=8

declare -a INPUT_DIRS=(
    "$BASE/model2.5_quantizedinputs_quantized_3w0i_qi2_hyperparameter_results_20260422_174247"
    "$BASE/model2.5_quantizedinputs_quantized_4w0i_qi2_hyperparameter_results_20260422_172940"
    "$BASE/model2.5_quantizedinputs_quantized_6w0i_qi2_hyperparameter_results_20260422_171733"
    "$BASE/model2.5_quantizedinputs_quantized_8w0i_qi2_hyperparameter_results_20260422_170559"
    "$BASE/model2.5_quantizedinputs_quantized_10w0i_qi2_hyperparameter_results_20260422_165541"
)

declare -a OUTPUT_DIRS=(
    "$BASE/model2.5_quantizedinputs_3w0i_pareto_roc_selected"
    "$BASE/model2.5_quantizedinputs_4w0i_pareto_roc_selected"
    "$BASE/model2.5_quantizedinputs_6w0i_pareto_roc_selected"
    "$BASE/model2.5_quantizedinputs_8w0i_pareto_roc_selected"
    "$BASE/model2.5_quantizedinputs_10w0i_pareto_roc_selected"
)

for i in "${!INPUT_DIRS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "[$((i+1))/${#INPUT_DIRS[@]}] $(basename ${INPUT_DIRS[$i]})"
    echo "========================================================================"
    python "$SCRIPT" \
        --input_dir  "${INPUT_DIRS[$i]}" \
        --data_dir   "$DATA" \
        --output_dir "${OUTPUT_DIRS[$i]}" \
        --use_weighted \
        --no_separate_folders
done

echo ""
echo "All done."
