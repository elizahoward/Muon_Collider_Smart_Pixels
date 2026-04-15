#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    hls::stream<input_t> &cluster, hls::stream<input3_t> &nModule, hls::stream<input4_t> &x_local, hls::stream<input8_t> &y_local,
    hls::stream<result_t> &layer25_out
);

#endif
