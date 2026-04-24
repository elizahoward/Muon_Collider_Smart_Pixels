#!/bin/bash
# Run HLS synthesis sequentially across all model2.5 quantized-input bit-width folders.
# Within each folder, up to 6 models are synthesized in parallel (--num_workers 6).
# Folders are processed one at a time to avoid overloading the machine.

set -e

SCRIPT=/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/ericHLS/parallel_hls_synthesis_resource.py
BASE=/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric

NUM_WORKERS=6
MAX_THREADS=4   # per-worker host+HLS thread cap (6 workers × 4 threads = 24 max)

declare -a INPUT_DIRS=(
    "$BASE/model2.5_quantizedinputs_3w0i_pareto_roc_selected"
    "$BASE/model2.5_quantizedinputs_4w0i_pareto_roc_selected"
    "$BASE/model2.5_quantizedinputs_6w0i_pareto_roc_selected"
    "$BASE/model2.5_quantizedinputs_8w0i_pareto_roc_selected"
    "$BASE/model2.5_quantizedinputs_10w0i_pareto_roc_selected"
)

for i in "${!INPUT_DIRS[@]}"; do
    INPUT_DIR="${INPUT_DIRS[$i]}"
    echo ""
    echo "========================================================================"
    echo "[$((i+1))/${#INPUT_DIRS[@]}] $(basename $INPUT_DIR)"
    echo "========================================================================"
    python "$SCRIPT" \
        --input_dir  "$INPUT_DIR" \
        --num_workers "$NUM_WORKERS" \
        --max_threads "$MAX_THREADS"
done

echo ""
echo "All HLS synthesis runs complete."
