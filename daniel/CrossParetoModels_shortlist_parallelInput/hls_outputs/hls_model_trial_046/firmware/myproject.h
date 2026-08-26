#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t cluster[N_INPUT_1_1*N_INPUT_2_1], input3_t nModule[N_INPUT_1_3], input4_t x_local[N_INPUT_1_4], input8_t y_local[N_INPUT_1_8],
    result_t layer29_out[N_LAYER_27],
    weight9_t w9[18], 
    bias9_t b9[2], 
    weight16_t w16[48], 
    bias16_t b16[16], 
    weight21_t w21[9792], 
    bias21_t b21[72], 
    weight24_t w24[4176], 
    bias24_t b24[58], 
    weight27_t w27[58], 
    bias27_t b27[1]
);

#endif
