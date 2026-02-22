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
#define N_SIZE_0_2 13
#define N_SIZE_1_2 21
#define N_SIZE_2_2 1
#define OUT_HEIGHT_3 13
#define OUT_WIDTH_3 21
#define N_FILT_3 48
#define N_INPUT_1_5 1
#define N_INPUT_1_6 1
#define OUT_HEIGHT_3 13
#define OUT_WIDTH_3 21
#define N_FILT_3 48
#define OUT_CONCAT_8 2
#define OUT_HEIGHT_9 6
#define OUT_WIDTH_9 10
#define N_FILT_9 48
#define N_LAYER_10 16
#define N_SIZE_0_12 2880
#define N_LAYER_10 16
#define OUT_CONCAT_14 2896
#define N_LAYER_15 80
#define N_LAYER_15 80
#define N_LAYER_18 32
#define N_LAYER_18 32
#define N_LAYER_21 1
#define N_LAYER_21 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<4,1> weight3_t;
typedef ap_fixed<4,1> bias3_t;
typedef ap_fixed<16,6> input5_t;
typedef ap_fixed<16,6> input6_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer7_t;
typedef ap_fixed<18,8> conv2d_act_table_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<4,1> weight10_t;
typedef ap_fixed<4,1> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer13_t;
typedef ap_fixed<18,8> dense_scalars_act_table_t;
typedef ap_fixed<16,6> layer14_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<4,1> weight15_t;
typedef ap_fixed<4,1> bias15_t;
typedef ap_uint<1> layer15_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer17_t;
typedef ap_fixed<18,8> merged_dense1_act_table_t;
typedef ap_fixed<16,6> layer18_t;
typedef ap_fixed<4,1> weight18_t;
typedef ap_fixed<4,1> bias18_t;
typedef ap_uint<1> layer18_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer20_t;
typedef ap_fixed<18,8> merged_dense2_act_table_t;
typedef ap_fixed<16,6> layer21_t;
typedef ap_fixed<4,1> weight21_t;
typedef ap_fixed<4,1> bias21_t;
typedef ap_uint<1> layer21_index;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> result_t;
typedef ap_ufixed<2,0> slope23_t;
typedef ap_ufixed<2,0> shift23_t;
typedef ap_fixed<18,8> output_table_t;

#endif
