#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t cluster[N_INPUT_1_1*N_INPUT_2_1], input3_t nModule[N_INPUT_1_3], input4_t x_local[N_INPUT_1_4], input8_t y_local[N_INPUT_1_8],
    result_t layer25_out[N_LAYER_23]
);

#endif
