#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include <ac_channel.h>
#include <ac_fixed.h>
#include <ac_int.h>
#include <ac_sync.h>

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    ac_channel<input_t> &y_profile,
    ac_channel<result_t> &layer6_out
);

#endif
