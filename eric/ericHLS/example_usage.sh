#!/bin/bash
# Example usage scripts for parallel HLS synthesis

echo "=========================================="
echo "Parallel HLS Synthesis - Example Usage"
echo "=========================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "1. Basic Usage - Process all models with 4 workers"
echo "---------------------------------------------------"
echo "python parallel_hls_synthesis.py \\"
echo "    --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\"
echo "    --num_workers 4"
echo ""

echo "2. Test Run - Process only 2 models for testing"
echo "------------------------------------------------"
echo "python parallel_hls_synthesis.py \\"
echo "    --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\"
echo "    --num_workers 2 \\"
echo "    --limit 2"
echo ""

echo "3. Custom Output Directory"
echo "--------------------------"
echo "python parallel_hls_synthesis.py \\"
echo "    --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\"
echo "    --output_dir ../hls_synthesis_outputs \\"
echo "    --num_workers 4"
echo ""

echo "4. Different FPGA Target"
echo "------------------------"
echo "python parallel_hls_synthesis.py \\"
echo "    --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \\"
echo "    --num_workers 4 \\"
echo "    --fpga_part xcu250-figd2104-2L-e"
echo ""

echo "5. Check Progress (one-time check)"
echo "-----------------------------------"
echo "python check_progress.py \\"
echo "    --results_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140/hls_outputs"
echo ""

echo "6. Monitor Progress Continuously"
echo "--------------------------------"
echo "python check_progress.py \\"
echo "    --results_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140/hls_outputs \\"
echo "    --watch \\"
echo "    --interval 60"
echo ""

echo "7. Analyze Results After Completion"
echo "------------------------------------"
echo "python analyze_synthesis_results.py \\"
echo "    --results_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140/hls_outputs"
echo ""

echo ""
echo "=========================================="
echo "Would you like to run a test synthesis?"
echo "=========================================="
echo ""
echo "This will process the first 2 models to verify everything works."
echo "Estimated time: 5-10 minutes"
echo ""
read -p "Run test? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "Starting test synthesis..."
    echo ""
    python parallel_hls_synthesis.py \
        --input_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140 \
        --num_workers 2 \
        --limit 2
    
    echo ""
    echo "Test complete! Check the results:"
    python check_progress.py \
        --results_dir ../model2_quantized_4w0i_hyperparameter_results_20251105_232140/hls_outputs
fi

