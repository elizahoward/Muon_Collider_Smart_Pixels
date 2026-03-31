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
typedef nnet::array<ac_fixed<16,6,true>, 13> input_t;
typedef nnet::array<ac_fixed<16,6,true>, 26> concatenate_24_result_t;
typedef nnet::array<ac_fixed<16,6,true>, 52> concatenate_25_result_t;
typedef ac_fixed<16,6,true> model_default_t;
typedef nnet::array<ac_fixed<16,6,true>, 10> hidden_dense_result_t;
typedef ac_fixed<4,1,true> weight4_t;
typedef ac_fixed<4,1,true> bias4_t;
typedef ac_int<1, false> layer4_index;
typedef nnet::array<ac_fixed<16,6,true>, 1> result_t;
typedef ac_fixed<4,1,true> weight6_t;
typedef ac_fixed<4,1,true> bias6_t;
typedef ac_int<1, false> layer6_index;

#endif
