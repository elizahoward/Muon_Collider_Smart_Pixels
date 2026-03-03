#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t x_profile[N_INPUT_1_1], input5_t nModule[N_INPUT_1_5], input6_t x_local[N_INPUT_1_6], input2_t y_profile[N_INPUT_1_2], input4_t y_local[N_INPUT_1_4],
    result_t layer24_out[N_LAYER_22]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x_profile complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=nModule complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=x_local complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_profile complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_local complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x_profile,nModule,x_local,y_profile,y_local,layer24_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight9_t, 280>(w9, "w9.txt");
        nnet::load_weights_from_txt<bias9_t, 8>(b9, "b9.txt");
        nnet::load_weights_from_txt<weight11_t, 4>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 2>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight16_t, 40>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 4>(b16, "b16.txt");
        nnet::load_weights_from_txt<weight19_t, 16>(w19, "w19.txt");
        nnet::load_weights_from_txt<bias19_t, 4>(b19, "b19.txt");
        nnet::load_weights_from_txt<weight22_t, 4>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 1>(b22, "b22.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer3_t layer3_out[OUT_CONCAT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::concatenate1d<input_t, input2_t, layer3_t, config3>(x_profile, y_profile, layer3_out); // xy_concat

    layer7_t layer7_out[OUT_CONCAT_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::concatenate1d<layer3_t, input4_t, layer7_t, config7>(layer3_out, y_local, layer7_out); // other_features

    layer8_t layer8_out[OUT_CONCAT_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::concatenate1d<input5_t, input6_t, layer8_t, config8>(nModule, x_local, layer8_out); // nmodule_xlocal_concat

    layer9_t layer9_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::dense<layer7_t, layer9_t, config9>(layer7_out, layer9_out, w9, b9); // other_dense

    layer11_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense<layer8_t, layer11_t, config11>(layer8_out, layer11_out, w11, b11); // nmodule_xlocal_dense

    layer13_t layer13_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::relu<layer9_t, layer13_t, relu_config13>(layer9_out, layer13_out); // other_activation

    layer14_t layer14_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::relu<layer11_t, layer14_t, relu_config14>(layer11_out, layer14_out); // nmodule_xlocal_activation

    layer15_t layer15_out[OUT_CONCAT_15];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::concatenate1d<layer13_t, layer14_t, layer15_t, config15>(layer13_out, layer14_out, layer15_out); // merged_features

    layer16_t layer16_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::dense<layer15_t, layer16_t, config16>(layer15_out, layer16_out, w16, b16); // dense2

    layer18_t layer18_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::relu<layer16_t, layer18_t, relu_config18>(layer16_out, layer18_out); // dense2_activation

    layer19_t layer19_out[N_LAYER_19];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::dense<layer18_t, layer19_t, config19>(layer18_out, layer19_out, w19, b19); // dense3

    layer21_t layer21_out[N_LAYER_19];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::relu<layer19_t, layer21_t, relu_config21>(layer19_out, layer21_out); // dense3_activation

    layer22_t layer22_out[N_LAYER_22];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::dense<layer21_t, layer22_t, config22>(layer21_out, layer22_out, w22, b22); // output

    nnet::hard_tanh<layer22_t, result_t, hard_tanh_config24>(layer22_out, layer24_out); // output_activation

}
