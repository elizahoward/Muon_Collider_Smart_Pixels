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
#define N_INPUT_1_1 13
#define N_INPUT_2_1 21
#define N_INPUT_1_3 1
#define N_INPUT_1_4 1
#define N_SIZE_0_5 13
#define N_SIZE_1_5 21
#define N_SIZE_2_5 1
#define N_INPUT_1_3 1
#define N_INPUT_1_4 1
#define N_INPUT_1_8 1
#define OUT_HEIGHT_9 13
#define OUT_WIDTH_9 21
#define N_FILT_9 8
#define OUT_CONCAT_11 2
#define N_INPUT_1_8 1
#define OUT_HEIGHT_9 13
#define OUT_WIDTH_9 21
#define N_FILT_9 8
#define OUT_CONCAT_14 3
#define OUT_HEIGHT_15 6
#define OUT_WIDTH_15 10
#define N_FILT_15 8
#define N_LAYER_16 24
#define N_SIZE_0_18 480
#define N_LAYER_16 24
#define OUT_CONCAT_20 504
#define N_LAYER_21 72
#define N_LAYER_21 72
#define N_LAYER_24 36
#define N_LAYER_24 36
#define N_LAYER_27 1
#define N_LAYER_27 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<10,1,AP_RND_CONV,AP_SAT> layer2_t;
typedef ap_fixed<18,8> q_input_cluster_table_t;
typedef ap_fixed<16,6> input3_t;
typedef ap_fixed<16,6> input4_t;
typedef ap_fixed<10,1,AP_RND_CONV,AP_SAT> layer6_t;
typedef ap_fixed<18,8> q_input_nModule_table_t;
typedef ap_fixed<10,1,AP_RND_CONV,AP_SAT> layer7_t;
typedef ap_fixed<18,8> q_input_x_local_table_t;
typedef ap_fixed<16,6> input8_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<8,1> weight9_t;
typedef ap_fixed<8,1> bias9_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<10,1,AP_RND_CONV,AP_SAT> layer12_t;
typedef ap_fixed<18,8> q_input_y_local_table_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer13_t;
typedef ap_fixed<18,8> conv2d_act_table_t;
typedef ap_fixed<16,6> layer14_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<16,6> layer16_t;
typedef ap_fixed<8,1> weight16_t;
typedef ap_fixed<8,1> bias16_t;
typedef ap_uint<1> layer16_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer19_t;
typedef ap_fixed<18,8> dense_scalars_act_table_t;
typedef ap_fixed<16,6> layer20_t;
typedef ap_fixed<16,6> layer21_t;
typedef ap_fixed<8,1> weight21_t;
typedef ap_fixed<8,1> bias21_t;
typedef ap_uint<1> layer21_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer23_t;
typedef ap_fixed<18,8> merged_dense1_act_table_t;
typedef ap_fixed<16,6> layer24_t;
typedef ap_fixed<8,1> weight24_t;
typedef ap_fixed<8,1> bias24_t;
typedef ap_uint<1> layer24_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer26_t;
typedef ap_fixed<18,8> merged_dense2_act_table_t;
typedef ap_fixed<16,6> layer27_t;
typedef ap_fixed<8,1> weight27_t;
typedef ap_fixed<8,1> bias27_t;
typedef ap_uint<1> layer27_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> result_t;
typedef ap_ufixed<2,0> slope29_t;
typedef ap_ufixed<2,0> shift29_t;
typedef ap_fixed<18,8> output_activation_table_t;

#endif
