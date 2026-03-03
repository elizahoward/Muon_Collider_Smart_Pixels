#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include <ac_channel.h>
#include <ac_fixed.h>
#include <ac_int.h>
#include <ac_sync.h>

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t x_profile[21] /* reshape */, ac_sync &x_profile_sync, input5_t nModule[1] /* reshape */, ac_sync &nModule_sync, input6_t x_local[1] /* reshape */, ac_sync &x_local_sync, input2_t y_profile[13] /* reshape */, ac_sync &y_profile_sync, input4_t y_local[1] /* reshape */, ac_sync &y_local_sync,
    result_t layer24_out[1] /* partition */, ac_sync &layer24_out_sync
);

#endif
