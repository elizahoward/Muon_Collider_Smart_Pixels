#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 21
#define N_INPUT_1_2 13
#define OUT_CONCAT_3 34
#define N_INPUT_1_4 1
#define N_INPUT_1_5 1
#define N_INPUT_1_6 1
#define OUT_CONCAT_7 35
#define OUT_CONCAT_8 2
#define N_LAYER_9 8
#define N_LAYER_11 2
#define N_LAYER_9 8
#define N_LAYER_11 2
#define OUT_CONCAT_15 10
#define N_LAYER_16 4
#define N_LAYER_16 4
#define N_LAYER_19 4
#define N_LAYER_19 4
#define N_LAYER_22 1
#define N_LAYER_22 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> input2_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> input4_t;
typedef ap_fixed<16,6> input5_t;
typedef ap_fixed<16,6> input6_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<4,1> weight9_t;
typedef ap_fixed<4,1> bias9_t;
typedef ap_uint<1> layer9_index;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<6,1> weight11_t;
typedef ap_fixed<6,1> bias11_t;
typedef ap_uint<1> layer11_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer13_t;
typedef ap_fixed<18,8> other_activation_table_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer14_t;
typedef ap_fixed<18,8> nmodule_xlocal_activation_table_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<16,6> layer16_t;
typedef ap_fixed<4,1> weight16_t;
typedef ap_fixed<4,1> bias16_t;
typedef ap_uint<1> layer16_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer18_t;
typedef ap_fixed<18,8> dense2_activation_table_t;
typedef ap_fixed<16,6> layer19_t;
typedef ap_fixed<4,1> weight19_t;
typedef ap_fixed<4,1> bias19_t;
typedef ap_uint<1> layer19_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer21_t;
typedef ap_fixed<18,8> dense3_activation_table_t;
typedef ap_fixed<16,6> layer22_t;
typedef ap_fixed<4,1> weight22_t;
typedef ap_fixed<4,1> bias22_t;
typedef ap_uint<1> layer22_index;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> result_t;
typedef ap_ufixed<2,0> slope24_t;
typedef ap_ufixed<2,0> shift24_t;
typedef ap_fixed<18,8> output_activation_table_t;

#endif
