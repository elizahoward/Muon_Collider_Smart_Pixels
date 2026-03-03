#ifndef DEFINES_H_
#define DEFINES_H_

#include "nnet_utils/nnet_types.h"
#include <ac_channel.h>
#include <ac_fixed.h>
#include <ac_int.h>
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
enum {
  AC_BUS_WORDS = 1,
};


// hls-fpga-machine-learning insert layer-precision
typedef ac_fixed<16,6,true> input_t;
typedef ac_fixed<16,6,true> input2_t;
typedef ac_fixed<16,6,true> layer3_t;
typedef ac_fixed<16,6,true> input4_t;
typedef ac_fixed<16,6,true> input5_t;
typedef ac_fixed<16,6,true> input6_t;
typedef ac_fixed<16,6,true> layer7_t;
typedef ac_fixed<16,6,true> layer8_t;
typedef ac_fixed<16,6,true> model_default_t;
typedef ac_fixed<16,6,true> layer9_t;
typedef ac_fixed<4,1,true> weight9_t;
typedef ac_fixed<4,1,true> bias9_t;
typedef ac_int<1, false> layer9_index;
typedef ac_fixed<16,6,true> layer11_t;
typedef ac_fixed<6,1,true> weight11_t;
typedef ac_fixed<6,1,true> bias11_t;
typedef ac_int<1, false> layer11_index;
typedef ac_fixed<16,6,true> layer13_t;
typedef ac_fixed<18,8,true> other_activation_table_t;
typedef ac_fixed<16,6,true> layer14_t;
typedef ac_fixed<18,8,true> nmodule_xlocal_activation_table_t;
typedef ac_fixed<16,6,true> layer15_t;
typedef ac_fixed<16,6,true> layer16_t;
typedef ac_fixed<4,1,true> weight16_t;
typedef ac_fixed<4,1,true> bias16_t;
typedef ac_int<1, false> layer16_index;
typedef ac_fixed<16,6,true> layer18_t;
typedef ac_fixed<18,8,true> dense2_activation_table_t;
typedef ac_fixed<16,6,true> layer19_t;
typedef ac_fixed<4,1,true> weight19_t;
typedef ac_fixed<4,1,true> bias19_t;
typedef ac_int<1, false> layer19_index;
typedef ac_fixed<16,6,true> layer21_t;
typedef ac_fixed<18,8,true> dense3_activation_table_t;
typedef ac_fixed<16,6,true> layer22_t;
typedef ac_fixed<4,1,true> weight22_t;
typedef ac_fixed<4,1,true> bias22_t;
typedef ac_int<1, false> layer22_index;
typedef ac_fixed<2,0,false> output_activation_slope_prec;
typedef ac_fixed<2,0,false> output_activation_shift_prec;
typedef ac_fixed<16,6,true> result_t;
typedef ac_fixed<2,0,false> slope24_t;
typedef ac_fixed<2,0,false> shift24_t;
typedef ac_fixed<18,8,true> output_activation_table_t;

#endif
