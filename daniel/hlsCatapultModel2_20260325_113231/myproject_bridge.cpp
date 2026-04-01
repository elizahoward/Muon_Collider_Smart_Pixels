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
weight4_t w4[520];
bias4_t b4[10];
weight6_t w6[10];
bias6_t b6[1];

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
    float y_profile[13],
    float layer6_out[1]
) {

    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        loaded_weights = true;
    }

    ac_channel<input_t> y_profile_ap/*("y_profile")*/;
    nnet::convert_data<float, input_t, 13>(y_profile, y_profile_ap);

    ac_channel<result_t> layer6_out_ap/*("layer6_out")*/;

    myproject(y_profile_ap,layer6_out_ap);

    nnet::convert_data<result_t, float, 1>(layer6_out_ap, layer6_out);
}

void myproject_double(
    double y_profile[13],
    double layer6_out[1]
) {

    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        loaded_weights = true;
    }

    ac_channel<input_t> y_profile_ap/*("y_profile")*/;
    nnet::convert_data<double, input_t, 13>(y_profile, y_profile_ap);

    ac_channel<result_t> layer6_out_ap/*("layer6_out")*/;

    myproject(y_profile_ap,layer6_out_ap);

    nnet::convert_data<result_t, double, 1>(layer6_out_ap, layer6_out);
}
}

#endif
