#ifndef MYPROJECT_BRIDGE_H_
#define MYPROJECT_BRIDGE_H_

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// hls-fpga-machine-learning insert bram

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
    float cluster[N_INPUT_1_1*N_INPUT_2_1], float z_global[N_INPUT_1_5], float y_local[N_INPUT_1_6],
    float layer23_out[N_LAYER_21]
) {

    input_t cluster_ap[N_INPUT_1_1*N_INPUT_2_1];
    nnet::convert_data<float, input_t, N_INPUT_1_1*N_INPUT_2_1>(cluster, cluster_ap);
    input5_t z_global_ap[N_INPUT_1_5];
    nnet::convert_data<float, input5_t, N_INPUT_1_5>(z_global, z_global_ap);
    input6_t y_local_ap[N_INPUT_1_6];
    nnet::convert_data<float, input6_t, N_INPUT_1_6>(y_local, y_local_ap);

    result_t layer23_out_ap[N_LAYER_21];

    myproject(cluster_ap,z_global_ap,y_local_ap,layer23_out_ap);

    nnet::convert_data<result_t, float, N_LAYER_21>(layer23_out_ap, layer23_out);
}

void myproject_double(
    double cluster[N_INPUT_1_1*N_INPUT_2_1], double z_global[N_INPUT_1_5], double y_local[N_INPUT_1_6],
    double layer23_out[N_LAYER_21]
) {
    input_t cluster_ap[N_INPUT_1_1*N_INPUT_2_1];
    nnet::convert_data<double, input_t, N_INPUT_1_1*N_INPUT_2_1>(cluster, cluster_ap);
    input5_t z_global_ap[N_INPUT_1_5];
    nnet::convert_data<double, input5_t, N_INPUT_1_5>(z_global, z_global_ap);
    input6_t y_local_ap[N_INPUT_1_6];
    nnet::convert_data<double, input6_t, N_INPUT_1_6>(y_local, y_local_ap);

    result_t layer23_out_ap[N_LAYER_21];

    myproject(cluster_ap,z_global_ap,y_local_ap,layer23_out_ap);

    nnet::convert_data<result_t, double, N_LAYER_21>(layer23_out_ap, layer23_out);
}
}

#endif
