#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t cluster[N_INPUT_1_1*N_INPUT_2_1], input5_t z_global[N_INPUT_1_5], input6_t y_local[N_INPUT_1_6],
    result_t layer23_out[N_LAYER_21]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=cluster complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=z_global complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_local complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=cluster,z_global,y_local,layer23_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight3_t, 432>(w3, "w3.txt");
        nnet::load_weights_from_txt<bias3_t, 48>(b3, "b3.txt");
        nnet::load_weights_from_txt<weight10_t, 96>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 48>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight15_t, 190320>(w15, "w15.txt");
        nnet::load_weights_from_txt<bias15_t, 65>(b15, "b15.txt");
        nnet::load_weights_from_txt<weight18_t, 2535>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 39>(b18, "b18.txt");
        nnet::load_weights_from_txt<weight21_t, 39>(w21, "w21.txt");
        nnet::load_weights_from_txt<bias21_t, 1>(b21, "b21.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    auto& layer2_out = cluster;
    layer3_t layer3_out[OUT_HEIGHT_3*OUT_WIDTH_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::conv_2d_cl<input_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // conv2d

    layer7_t layer7_out[OUT_HEIGHT_3*OUT_WIDTH_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer3_t, layer7_t, relu_config7>(layer3_out, layer7_out); // conv2d_act

    layer8_t layer8_out[OUT_CONCAT_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::concatenate1d<input5_t, input6_t, layer8_t, config8>(z_global, y_local, layer8_out); // concat_scalars

    layer9_t layer9_out[OUT_HEIGHT_9*OUT_WIDTH_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::pooling2d_cl<layer7_t, layer9_t, config9>(layer7_out, layer9_out); // pool2d_1

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer8_t, layer10_t, config10>(layer8_out, layer10_out, w10, b10); // dense_scalars

    auto& layer12_out = layer9_out;
    layer13_t layer13_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::relu<layer10_t, layer13_t, relu_config13>(layer10_out, layer13_out); // dense_scalars_act

    layer14_t layer14_out[OUT_CONCAT_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::concatenate1d<layer9_t, layer13_t, layer14_t, config14>(layer12_out, layer13_out, layer14_out); // concat_all

    layer15_t layer15_out[N_LAYER_15];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::dense<layer14_t, layer15_t, config15>(layer14_out, layer15_out, w15, b15); // merged_dense1

    layer17_t layer17_out[N_LAYER_15];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::relu<layer15_t, layer17_t, relu_config17>(layer15_out, layer17_out); // merged_dense1_act

    layer18_t layer18_out[N_LAYER_18];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::dense<layer17_t, layer18_t, config18>(layer17_out, layer18_out, w18, b18); // merged_dense2

    layer20_t layer20_out[N_LAYER_18];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::relu<layer18_t, layer20_t, relu_config20>(layer18_out, layer20_out); // merged_dense2_act

    layer21_t layer21_out[N_LAYER_21];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::dense<layer20_t, layer21_t, config21>(layer20_out, layer21_out, w21, b21); // output_dense

    nnet::hard_tanh<layer21_t, result_t, hard_tanh_config23>(layer21_out, layer23_out); // output

}
