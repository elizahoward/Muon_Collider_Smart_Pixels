#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t cluster[N_INPUT_1_1*N_INPUT_2_1], input5_t z_global[N_INPUT_1_5], input6_t y_local[N_INPUT_1_6],
    result_t layer23_out[N_LAYER_21]
);

#endif
