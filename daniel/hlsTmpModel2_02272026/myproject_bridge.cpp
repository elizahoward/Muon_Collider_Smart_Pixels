#ifndef MYPROJECT_BRIDGE_H_
#define MYPROJECT_BRIDGE_H_

#include "firmware/myproject.h"
#include "nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>
#include <ac_shared.h>
#include <ac_sync.h>

static std::string s_weights_dir = "weights";

const char *get_weights_dir() { return s_weights_dir.c_str(); }

// hls-fpga-machine-learning insert bram

// hls-fpga-machine-learning insert declare weights
weight9_t w9[4480];
bias9_t b9[128];
weight11_t w11[28];
bias11_t b11[14];
weight16_t w16[7952];
bias16_t b16[56];
weight19_t w19[2352];
bias19_t b19[42];
weight22_t w22[42];
bias22_t b22[1];

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// Wrapper of top level function for Python bridge
void myproject_float(
    float x_profile[21], float nModule[1], float x_local[1], float y_profile[13], float y_local[1],
    float layer24_out[1]
) {

    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        loaded_weights = true;
    }

    ac_shared<input_t[21] > x_profile_ap /* reshape */;
    ac_sync x_profile_sync; 
    x_profile_sync.sync_out(); 
    nnet::convert_data<float, input_t, 21>(x_profile, x_profile_ap);
    ac_shared<input5_t[1] > nModule_ap /* reshape */;
    ac_sync nModule_sync; 
    nModule_sync.sync_out(); 
    nnet::convert_data<float, input5_t, 1>(nModule, nModule_ap);
    ac_shared<input6_t[1] > x_local_ap /* reshape */;
    ac_sync x_local_sync; 
    x_local_sync.sync_out(); 
    nnet::convert_data<float, input6_t, 1>(x_local, x_local_ap);
    ac_shared<input2_t[13] > y_profile_ap /* reshape */;
    ac_sync y_profile_sync; 
    y_profile_sync.sync_out(); 
    nnet::convert_data<float, input2_t, 13>(y_profile, y_profile_ap);
    ac_shared<input4_t[1] > y_local_ap /* reshape */;
    ac_sync y_local_sync; 
    y_local_sync.sync_out(); 
    nnet::convert_data<float, input4_t, 1>(y_local, y_local_ap);

    ac_shared<result_t[1] > layer24_out_ap /* partition */;
    ac_sync layer24_out_sync;

    myproject(x_profile_ap, x_profile_sync, nModule_ap, nModule_sync, x_local_ap, x_local_sync, y_profile_ap, y_profile_sync, y_local_ap, y_local_sync,layer24_out_ap, layer24_out_sync);

    nnet::convert_data<result_t, float, 1>(layer24_out_ap, layer24_out);
}

void myproject_double(
    double x_profile[21], double nModule[1], double x_local[1], double y_profile[13], double y_local[1],
    double layer24_out[1]
) {

    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        loaded_weights = true;
    }

    ac_shared<input_t[21] > x_profile_ap /* reshape */;
    ac_sync x_profile_sync; 
    x_profile_sync.sync_out(); 
    nnet::convert_data<double, input_t, 21>(x_profile, x_profile_ap);
    ac_shared<input5_t[1] > nModule_ap /* reshape */;
    ac_sync nModule_sync; 
    nModule_sync.sync_out(); 
    nnet::convert_data<double, input5_t, 1>(nModule, nModule_ap);
    ac_shared<input6_t[1] > x_local_ap /* reshape */;
    ac_sync x_local_sync; 
    x_local_sync.sync_out(); 
    nnet::convert_data<double, input6_t, 1>(x_local, x_local_ap);
    ac_shared<input2_t[13] > y_profile_ap /* reshape */;
    ac_sync y_profile_sync; 
    y_profile_sync.sync_out(); 
    nnet::convert_data<double, input2_t, 13>(y_profile, y_profile_ap);
    ac_shared<input4_t[1] > y_local_ap /* reshape */;
    ac_sync y_local_sync; 
    y_local_sync.sync_out(); 
    nnet::convert_data<double, input4_t, 1>(y_local, y_local_ap);

    ac_shared<result_t[1] > layer24_out_ap /* partition */;
    ac_sync layer24_out_sync;

    myproject(x_profile_ap, x_profile_sync, nModule_ap, nModule_sync, x_local_ap, x_local_sync, y_profile_ap, y_profile_sync, y_local_ap, y_local_sync,layer24_out_ap, layer24_out_sync);

    nnet::convert_data<result_t, double, 1>(layer24_out_ap, layer24_out);
}
}

#endif
