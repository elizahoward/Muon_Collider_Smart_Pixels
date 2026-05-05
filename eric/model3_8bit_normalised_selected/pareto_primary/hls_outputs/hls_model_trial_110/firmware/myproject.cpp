#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t cluster[N_INPUT_1_1*N_INPUT_2_1], input3_t nModule[N_INPUT_1_3], input4_t x_local[N_INPUT_1_4], input8_t y_local[N_INPUT_1_8],
    result_t layer29_out[N_LAYER_27]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=cluster complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=nModule complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=x_local complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_local complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer29_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=cluster,nModule,x_local,y_local,layer29_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight9_t, 72>(w9, "w9.txt");
        nnet::load_weights_from_txt<bias9_t, 8>(b9, "b9.txt");
        nnet::load_weights_from_txt<weight16_t, 72>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 24>(b16, "b16.txt");
        nnet::load_weights_from_txt<weight21_t, 36288>(w21, "w21.txt");
        nnet::load_weights_from_txt<bias21_t, 72>(b21, "b21.txt");
        nnet::load_weights_from_txt<weight24_t, 2592>(w24, "w24.txt");
        nnet::load_weights_from_txt<bias24_t, 36>(b24, "b24.txt");
        nnet::load_weights_from_txt<weight27_t, 36>(w27, "w27.txt");
        nnet::load_weights_from_txt<bias27_t, 1>(b27, "b27.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::linear<input_t, layer2_t, linear_config2>(cluster, layer2_out); // q_input_cluster

    auto& layer5_out = layer2_out;
    layer6_t layer6_out[N_INPUT_1_3];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::linear<input3_t, layer6_t, linear_config6>(nModule, layer6_out); // q_input_nModule

    layer7_t layer7_out[N_INPUT_1_4];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::linear<input4_t, layer7_t, linear_config7>(x_local, layer7_out); // q_input_x_local

    layer9_t layer9_out[OUT_HEIGHT_9*OUT_WIDTH_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::conv_2d_cl<layer2_t, layer9_t, config9>(layer5_out, layer9_out, w9, b9); // conv2d

    layer11_t layer11_out[OUT_CONCAT_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::concatenate1d<layer6_t, layer7_t, layer11_t, config11>(layer6_out, layer7_out, layer11_out); // concat_scalars_1

    layer12_t layer12_out[N_INPUT_1_8];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::linear<input8_t, layer12_t, linear_config12>(y_local, layer12_out); // q_input_y_local

    layer13_t layer13_out[OUT_HEIGHT_9*OUT_WIDTH_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::relu<layer9_t, layer13_t, relu_config13>(layer9_out, layer13_out); // conv2d_act

    layer14_t layer14_out[OUT_CONCAT_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::concatenate1d<layer11_t, layer12_t, layer14_t, config14>(layer11_out, layer12_out, layer14_out); // concat_scalars_2

    layer15_t layer15_out[OUT_HEIGHT_15*OUT_WIDTH_15*N_FILT_15];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::pooling2d_cl<layer13_t, layer15_t, config15>(layer13_out, layer15_out); // pool2d_1

    layer16_t layer16_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::dense<layer14_t, layer16_t, config16>(layer14_out, layer16_out, w16, b16); // dense_scalars

    auto& layer18_out = layer15_out;
    layer19_t layer19_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::relu<layer16_t, layer19_t, relu_config19>(layer16_out, layer19_out); // dense_scalars_act

    layer20_t layer20_out[OUT_CONCAT_20];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::concatenate1d<layer15_t, layer19_t, layer20_t, config20>(layer18_out, layer19_out, layer20_out); // concat_all

    layer21_t layer21_out[N_LAYER_21];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::dense<layer20_t, layer21_t, config21>(layer20_out, layer21_out, w21, b21); // merged_dense1

    layer23_t layer23_out[N_LAYER_21];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::relu<layer21_t, layer23_t, relu_config23>(layer21_out, layer23_out); // merged_dense1_act

    layer24_t layer24_out[N_LAYER_24];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    nnet::dense<layer23_t, layer24_t, config24>(layer23_out, layer24_out, w24, b24); // merged_dense2

    layer26_t layer26_out[N_LAYER_24];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    nnet::relu<layer24_t, layer26_t, relu_config26>(layer24_out, layer26_out); // merged_dense2_act

    layer27_t layer27_out[N_LAYER_27];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    nnet::dense<layer26_t, layer27_t, config27>(layer26_out, layer27_out, w27, b27); // output_dense

    nnet::hard_sigmoid<layer27_t, result_t, hard_sigmoid_config29>(layer27_out, layer29_out); // output_activation

}
