#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t x_profile[N_INPUT_1_1], input5_t nModule[N_INPUT_1_5], input6_t x_local[N_INPUT_1_6], input2_t y_profile[N_INPUT_1_2], input4_t y_local[N_INPUT_1_4],
    result_t layer24_out[N_LAYER_22]
);

#endif
