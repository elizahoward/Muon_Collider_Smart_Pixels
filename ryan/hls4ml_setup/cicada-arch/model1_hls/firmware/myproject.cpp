#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t z_global[N_INPUT_1_1], input2_t x_size[N_INPUT_1_2], input4_t y_size[N_INPUT_1_4], input6_t y_local[N_INPUT_1_6],
    result_t layer13_out[N_LAYER_12]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=z_global complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=x_size complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_size complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_local complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=z_global,x_size,y_size,y_local,layer13_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<dense_3_weight_t, 28>(w8, "w8.txt");
        nnet::load_weights_from_txt<dense_3_bias_t, 7>(b8, "b8.txt");
        nnet::load_weights_from_txt<dense_4_weight_t, 49>(w10, "w10.txt");
        nnet::load_weights_from_txt<dense_4_bias_t, 7>(b10, "b10.txt");
        nnet::load_weights_from_txt<dense_5_weight_t, 7>(w12, "w12.txt");
        nnet::load_weights_from_txt<dense_5_bias_t, 1>(b12, "b12.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer3_t layer3_out[OUT_CONCAT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::concatenate1d<input_t, input2_t, layer3_t, config3>(z_global, x_size, layer3_out); // concatenate_3

    layer5_t layer5_out[OUT_CONCAT_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::concatenate1d<layer3_t, input4_t, layer5_t, config5>(layer3_out, y_size, layer5_out); // concatenate_4

    layer7_t layer7_out[OUT_CONCAT_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::concatenate1d<layer5_t, input6_t, layer7_t, config7>(layer5_out, y_local, layer7_out); // concatenate_5

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // dense_3

    layer9_t layer9_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // dense_3_relu

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // dense_4

    layer11_t layer11_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<layer10_t, layer11_t, relu_config11>(layer10_out, layer11_out); // dense_4_relu

    layer12_t layer12_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::dense<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // dense_5

    nnet::sigmoid<layer12_t, result_t, sigmoid_config13>(layer12_out, layer13_out); // dense_5_sigmoid

}
