#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t cluster[N_INPUT_1_1*N_INPUT_2_1], input3_t nModule[N_INPUT_1_3], input4_t x_local[N_INPUT_1_4], input8_t y_local[N_INPUT_1_8],
    result_t layer25_out[N_LAYER_23]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=cluster complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=nModule complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=x_local complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_local complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=cluster,nModule,x_local,y_local,layer25_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight5_t, 216>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 24>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight12_t, 192>(w12, "w12.txt");
        nnet::load_weights_from_txt<bias12_t, 64>(b12, "b12.txt");
        nnet::load_weights_from_txt<weight17_t, 165440>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 110>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight20_t, 9680>(w20, "w20.txt");
        nnet::load_weights_from_txt<bias20_t, 88>(b20, "b20.txt");
        nnet::load_weights_from_txt<weight23_t, 88>(w23, "w23.txt");
        nnet::load_weights_from_txt<bias23_t, 1>(b23, "b23.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    auto& layer2_out = cluster;
    layer5_t layer5_out[OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::conv_2d_cl<input_t, layer5_t, config5>(layer2_out, layer5_out, w5, b5); // conv2d

    layer7_t layer7_out[OUT_CONCAT_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::concatenate1d<input3_t, input4_t, layer7_t, config7>(nModule, x_local, layer7_out); // concat_scalars_1

    layer9_t layer9_out[OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<layer5_t, layer9_t, relu_config9>(layer5_out, layer9_out); // conv2d_act

    layer10_t layer10_out[OUT_CONCAT_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::concatenate1d<layer7_t, input8_t, layer10_t, config10>(layer7_out, y_local, layer10_out); // concat_scalars_2

    layer11_t layer11_out[OUT_HEIGHT_11*OUT_WIDTH_11*N_FILT_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::pooling2d_cl<layer9_t, layer11_t, config11>(layer9_out, layer11_out); // pool2d_1

    layer12_t layer12_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::dense<layer10_t, layer12_t, config12>(layer10_out, layer12_out, w12, b12); // dense_scalars

    auto& layer14_out = layer11_out;
    layer15_t layer15_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::relu<layer12_t, layer15_t, relu_config15>(layer12_out, layer15_out); // dense_scalars_act

    layer16_t layer16_out[OUT_CONCAT_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::concatenate1d<layer11_t, layer15_t, layer16_t, config16>(layer14_out, layer15_out, layer16_out); // concat_all

    layer17_t layer17_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer16_t, layer17_t, config17>(layer16_out, layer17_out, w17, b17); // merged_dense1

    layer19_t layer19_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::relu<layer17_t, layer19_t, relu_config19>(layer17_out, layer19_out); // merged_dense1_act

    layer20_t layer20_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::dense<layer19_t, layer20_t, config20>(layer19_out, layer20_out, w20, b20); // merged_dense2

    layer22_t layer22_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::relu<layer20_t, layer22_t, relu_config22>(layer20_out, layer22_out); // merged_dense2_act

    layer23_t layer23_out[N_LAYER_23];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::dense<layer22_t, layer23_t, config23>(layer22_out, layer23_out, w23, b23); // output_dense

    nnet::hard_tanh<layer23_t, result_t, hard_tanh_config25>(layer23_out, layer25_out); // output

}
