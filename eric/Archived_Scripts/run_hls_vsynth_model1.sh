#!/bin/bash
# Run Vivado synthesis (vsynth) sequentially across all model1 Pareto-selected bit-width folders.
# Within each folder, models are synthesised in parallel (--num_workers N).
# Folders are processed one at a time to avoid overloading the machine.
#
# Usage:
#   bash run_hls_vsynth_model1.sh [num_workers]
#   e.g. bash run_hls_vsynth_model1.sh 6

set -e

SCRIPT=/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/ericHLS/parallel_hls_synthesis_resource.py
BASE=/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/model1_pareto_selected

NUM_WORKERS="${1:-6}"
MAX_THREADS=4   # per-worker thread cap (NUM_WORKERS × MAX_THREADS total)

PATTERN="hp*q_model_trial_*.h5"

declare -a INPUT_DIRS=(
    "$BASE/3w0i_i5_sigmoid"
    "$BASE/4w0i_i6_sigmoid"
    "$BASE/6w0i_i8_sigmoid"
    "$BASE/8w0i_i10_sigmoid"
    "$BASE/10w0i_i12_sigmoid"
)

echo "NUM_WORKERS : $NUM_WORKERS"
echo "MAX_THREADS : $MAX_THREADS  (peak threads ≈ $((NUM_WORKERS * MAX_THREADS)))"
echo ""

for i in "${!INPUT_DIRS[@]}"; do
    INPUT_DIR="${INPUT_DIRS[$i]}"
    echo ""
    echo "========================================================================"
    echo "[$((i+1))/${#INPUT_DIRS[@]}] $(basename $INPUT_DIR) [vsynth]"
    echo "========================================================================"
    python "$SCRIPT" \
        --input_dir   "$INPUT_DIR" \
        --output_dir  "$INPUT_DIR/hls_outputs_vsynth" \
        --num_workers "$NUM_WORKERS" \
        --max_threads "$MAX_THREADS" \
        --pattern     "$PATTERN" \
        --vsynth
done

echo ""
echo "All vsynth runs complete."
