#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 1
#define N_INPUT_1_2 1
#define OUT_CONCAT_3 2
#define N_INPUT_1_4 1
#define OUT_CONCAT_5 3
#define N_INPUT_1_6 1
#define OUT_CONCAT_7 4
#define N_LAYER_8 7
#define N_LAYER_8 7
#define N_LAYER_10 7
#define N_LAYER_10 7
#define N_LAYER_12 1
#define N_LAYER_12 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> input2_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> input4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> input6_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<16,6> dense_3_weight_t;
typedef ap_fixed<16,6> dense_3_bias_t;
typedef ap_uint<1> layer8_index;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<18,8> dense_3_relu_table_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<16,6> dense_4_weight_t;
typedef ap_fixed<16,6> dense_4_bias_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<18,8> dense_4_relu_table_t;
typedef ap_fixed<16,6> layer12_t;
typedef ap_fixed<16,6> dense_5_weight_t;
typedef ap_fixed<16,6> dense_5_bias_t;
typedef ap_uint<1> layer12_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> dense_5_sigmoid_table_t;

#endif
