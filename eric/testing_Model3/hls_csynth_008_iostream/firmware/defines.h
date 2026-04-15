#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 13
#define N_INPUT_2_1 21
#define N_SIZE_1_26 13
#define N_SIZE_2_26 21
#define N_SIZE_3_26 1
#define OUT_HEIGHT_27 15
#define OUT_WIDTH_27 23
#define N_CHAN_27 1
#define N_INPUT_1_3 1
#define N_INPUT_1_4 1
#define OUT_HEIGHT_5 13
#define OUT_WIDTH_5 21
#define N_FILT_5 24
#define OUT_CONCAT_7 2
#define N_INPUT_1_8 1
#define OUT_HEIGHT_5 13
#define OUT_WIDTH_5 21
#define N_FILT_5 24
#define OUT_CONCAT_10 3
#define OUT_HEIGHT_11 6
#define OUT_WIDTH_11 10
#define N_FILT_11 24
#define N_LAYER_12 64
#define N_SIZE_0_14 1440
#define N_LAYER_12 64
#define OUT_CONCAT_16 1504
#define N_LAYER_17 110
#define N_LAYER_17 110
#define N_LAYER_20 88
#define N_LAYER_20 88
#define N_LAYER_23 1
#define N_LAYER_23 1

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 21*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer2_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer27_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> input3_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> input4_t;
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 24*1> layer5_t;
typedef ap_fixed<4,1> weight5_t;
typedef ap_fixed<4,1> bias5_t;
typedef nnet::array<ap_fixed<16,6>, 2*1> layer7_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> input8_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT>, 24*1> layer9_t;
typedef ap_fixed<18,8> conv2d_act_table_t;
typedef nnet::array<ap_fixed<16,6>, 3*1> layer10_t;
typedef nnet::array<ap_fixed<16,6>, 24*1> layer11_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer12_t;
typedef ap_fixed<4,1> weight12_t;
typedef ap_fixed<4,1> bias12_t;
typedef ap_uint<1> layer12_index;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT>, 64*1> layer15_t;
typedef ap_fixed<18,8> dense_scalars_act_table_t;
typedef nnet::array<ap_fixed<16,6>, 1504*1> layer16_t;
typedef nnet::array<ap_fixed<16,6>, 110*1> layer17_t;
typedef ap_fixed<4,1> weight17_t;
typedef ap_fixed<4,1> bias17_t;
typedef ap_uint<1> layer17_index;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT>, 110*1> layer19_t;
typedef ap_fixed<18,8> merged_dense1_act_table_t;
typedef nnet::array<ap_fixed<16,6>, 88*1> layer20_t;
typedef ap_fixed<4,1> weight20_t;
typedef ap_fixed<4,1> bias20_t;
typedef ap_uint<1> layer20_index;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT>, 88*1> layer22_t;
typedef ap_fixed<18,8> merged_dense2_act_table_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer23_t;
typedef ap_fixed<4,1> weight23_t;
typedef ap_fixed<4,1> bias23_t;
typedef ap_uint<1> layer23_index;
typedef nnet::array<ap_fixed<8,1,AP_RND_CONV,AP_SAT>, 1*1> result_t;
typedef ap_ufixed<2,0> slope25_t;
typedef ap_ufixed<2,0> shift25_t;
typedef ap_fixed<18,8> output_table_t;

#endif
