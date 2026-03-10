#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t z_global[N_INPUT_1_1], input2_t x_size[N_INPUT_1_2], input4_t y_size[N_INPUT_1_4], input6_t y_local[N_INPUT_1_6],
    result_t layer13_out[N_LAYER_12]
);

#endif
