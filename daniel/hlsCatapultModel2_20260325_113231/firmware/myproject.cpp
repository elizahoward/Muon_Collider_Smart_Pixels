#include <iostream>

#include "myproject.h"
#include <mc_scverify.h>

#include <ac_shared.h>
#include <ac_sync.h>

#include "parameters.h"




#pragma hls_design top
// hls-fpga-machine-learning insert IFSynPragmas
#pragma hls_resource y_profile:rsc variables="y_profile" map_to_module="ccs_ioport.ccs_in_wait"
#pragma hls_resource layer6_out:rsc variables="layer6_out" map_to_module="ccs_ioport.ccs_out_wait"
void CCS_BLOCK(myproject)(
    ac_channel<input_t> &y_profile,
    ac_channel<result_t> &layer6_out
) {

// hls-fpga-machine-learning insert weights
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w6.h"
#include "weights/b6.h"

    // hls-fpga-machine-learning insert IO

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight4_t, 520>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 10>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight6_t, 10>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 1>(b6, "b6.txt");
        loaded_weights = true;
    }
#endif



    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    #pragma hls_fifo_depth 1 
    static ac_channel<input_t> layer8_cpy1/*("layer8_cpy1")*/; 
    #pragma hls_resource layer8_cpy1:cns variables="layer8_cpy1" map_to_module="hls4ml_lib.mgc_pipe_mem" fifo_depth="1"
    #pragma hls_fifo_depth 1 
    static ac_channel<input_t> layer8_cpy2/*("layer8_cpy2")*/; 
    #pragma hls_resource layer8_cpy2:cns variables="layer8_cpy2" map_to_module="hls4ml_lib.mgc_pipe_mem" fifo_depth="1"
    nnet::clone_stream<input_t, input_t, 8>(y_profile, layer8_cpy1, layer8_cpy2); // clone_y_profile

    #pragma hls_fifo_depth 1 
    static ac_channel<concatenate_24_result_t> layer2_out/*("layer2_out")*/; 
    #pragma hls_resource layer2_out:cns variables="layer2_out" map_to_module="hls4ml_lib.mgc_pipe_mem" fifo_depth="1"
    nnet::concatenate1d<input_t, input_t, concatenate_24_result_t, config2>(layer8_cpy1, layer8_cpy2, layer2_out); // concatenate_24

    #pragma hls_fifo_depth 1 
    static ac_channel<concatenate_24_result_t> layer9_cpy1/*("layer9_cpy1")*/; 
    #pragma hls_resource layer9_cpy1:cns variables="layer9_cpy1" map_to_module="hls4ml_lib.mgc_pipe_mem" fifo_depth="1"
    #pragma hls_fifo_depth 1 
    static ac_channel<concatenate_24_result_t> layer9_cpy2/*("layer9_cpy2")*/; 
    #pragma hls_resource layer9_cpy2:cns variables="layer9_cpy2" map_to_module="hls4ml_lib.mgc_pipe_mem" fifo_depth="1"
    nnet::clone_stream<concatenate_24_result_t, concatenate_24_result_t, 9>(layer2_out, layer9_cpy1, layer9_cpy2); // clone_concatenate_24

    #pragma hls_fifo_depth 1 
    static ac_channel<concatenate_25_result_t> layer3_out/*("layer3_out")*/; 
    #pragma hls_resource layer3_out:cns variables="layer3_out" map_to_module="hls4ml_lib.mgc_pipe_mem" fifo_depth="1"
    nnet::concatenate1d<concatenate_24_result_t, concatenate_24_result_t, concatenate_25_result_t, config3>(layer9_cpy1, layer9_cpy2, layer3_out); // concatenate_25

    #pragma hls_fifo_depth 1 
    static ac_channel<hidden_dense_result_t> layer4_out/*("layer4_out")*/; 
    #pragma hls_resource layer4_out:cns variables="layer4_out" map_to_module="hls4ml_lib.mgc_pipe_mem" fifo_depth="1"
    nnet::dense<concatenate_25_result_t, hidden_dense_result_t, config4>(layer3_out, layer4_out, w4, b4); // hidden_dense

    nnet::dense<hidden_dense_result_t, result_t, config6>(layer4_out, layer6_out, w6, b6); // output_dense

}

