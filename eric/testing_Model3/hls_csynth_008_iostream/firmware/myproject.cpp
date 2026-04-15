#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &cluster, hls::stream<input3_t> &nModule, hls::stream<input4_t> &x_local, hls::stream<input8_t> &y_local,
    hls::stream<result_t> &layer25_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=cluster,nModule,x_local,y_local,layer25_out 
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

    hls::stream<layer2_t> layer26_out("layer26_out");
    #pragma HLS STREAM variable=layer26_out depth=273
    nnet::repack_stream<input_t, layer2_t, 273>(cluster, layer26_out); // repack_add_channel

    hls::stream<layer27_t> layer27_out("layer27_out");
    #pragma HLS STREAM variable=layer27_out depth=345
    nnet::zeropad2d_cl<layer2_t, layer27_t, config27>(layer26_out, layer27_out); // zp2d_conv2d

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=273
    nnet::conv_2d_cl<layer27_t, layer5_t, config5>(layer27_out, layer5_out, w5, b5); // conv2d

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=1
    nnet::concatenate1d<input3_t, input4_t, layer7_t, config7>(nModule, x_local, layer7_out); // concat_scalars_1

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=273
    nnet::relu<layer5_t, layer9_t, relu_config9>(layer5_out, layer9_out); // conv2d_act

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1
    nnet::concatenate1d<layer7_t, input8_t, layer10_t, config10>(layer7_out, y_local, layer10_out); // concat_scalars_2

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=60
    nnet::pooling2d_cl<layer9_t, layer11_t, config11>(layer9_out, layer11_out); // pool2d_1

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::dense<layer10_t, layer12_t, config12>(layer10_out, layer12_out, w12, b12); // dense_scalars

    auto& layer14_out = layer11_out;
    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=1
    nnet::relu<layer12_t, layer15_t, relu_config15>(layer12_out, layer15_out); // dense_scalars_act

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=1
    nnet::concatenate1d<layer11_t, layer15_t, layer16_t, config16>(layer14_out, layer15_out, layer16_out); // concat_all

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=1
    nnet::dense<layer16_t, layer17_t, config17>(layer16_out, layer17_out, w17, b17); // merged_dense1

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=1
    nnet::relu<layer17_t, layer19_t, relu_config19>(layer17_out, layer19_out); // merged_dense1_act

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=1
    nnet::dense<layer19_t, layer20_t, config20>(layer19_out, layer20_out, w20, b20); // merged_dense2

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=1
    nnet::relu<layer20_t, layer22_t, relu_config22>(layer20_out, layer22_out); // merged_dense2_act

    hls::stream<layer23_t> layer23_out("layer23_out");
    #pragma HLS STREAM variable=layer23_out depth=1
    nnet::dense<layer22_t, layer23_t, config23>(layer22_out, layer23_out, w23, b23); // output_dense

    nnet::hard_tanh<layer23_t, result_t, hard_tanh_config25>(layer23_out, layer25_out); // output

}
