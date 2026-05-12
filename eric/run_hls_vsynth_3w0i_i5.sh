#!/bin/bash
# Vivado synthesis (vsynth) run for 3w0i_i5_sigmoid — for comparison against HLS estimates.

set -e

SCRIPT=/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/ericHLS/parallel_hls_synthesis_resource.py
INPUT_DIR=/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/model1_pareto_selected/3w0i_i5_sigmoid

python "$SCRIPT" \
    --input_dir   "$INPUT_DIR" \
    --output_dir  "$INPUT_DIR/hls_outputs_vsynth" \
    --num_workers 6 \
    --max_threads 4 \
    --pattern     "hp*q_model_trial_*.h5" \
    --vsynth

echo ""
echo "vsynth run complete. Results in $INPUT_DIR/hls_outputs_vsynth"
