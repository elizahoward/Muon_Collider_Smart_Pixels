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
    float cluster[N_INPUT_1_1*N_INPUT_2_1], float nModule[N_INPUT_1_3], float x_local[N_INPUT_1_4], float y_local[N_INPUT_1_8],
    float layer25_out[N_LAYER_23]
) {

    hls::stream<input_t> cluster_ap("cluster");
    nnet::convert_data<float, input_t, N_INPUT_1_1*N_INPUT_2_1>(cluster, cluster_ap);
    hls::stream<input3_t> nModule_ap("nModule");
    nnet::convert_data<float, input3_t, N_INPUT_1_3>(nModule, nModule_ap);
    hls::stream<input4_t> x_local_ap("x_local");
    nnet::convert_data<float, input4_t, N_INPUT_1_4>(x_local, x_local_ap);
    hls::stream<input8_t> y_local_ap("y_local");
    nnet::convert_data<float, input8_t, N_INPUT_1_8>(y_local, y_local_ap);

    hls::stream<result_t> layer25_out_ap("layer25_out");

    myproject(cluster_ap,nModule_ap,x_local_ap,y_local_ap,layer25_out_ap);

    nnet::convert_data<result_t, float, N_LAYER_23>(layer25_out_ap, layer25_out);
}

void myproject_double(
    double cluster[N_INPUT_1_1*N_INPUT_2_1], double nModule[N_INPUT_1_3], double x_local[N_INPUT_1_4], double y_local[N_INPUT_1_8],
    double layer25_out[N_LAYER_23]
) {
    hls::stream<input_t> cluster_ap("cluster");
    nnet::convert_data<double, input_t, N_INPUT_1_1*N_INPUT_2_1>(cluster, cluster_ap);
    hls::stream<input3_t> nModule_ap("nModule");
    nnet::convert_data<double, input3_t, N_INPUT_1_3>(nModule, nModule_ap);
    hls::stream<input4_t> x_local_ap("x_local");
    nnet::convert_data<double, input4_t, N_INPUT_1_4>(x_local, x_local_ap);
    hls::stream<input8_t> y_local_ap("y_local");
    nnet::convert_data<double, input8_t, N_INPUT_1_8>(y_local, y_local_ap);

    hls::stream<result_t> layer25_out_ap("layer25_out");

    myproject(cluster_ap,nModule_ap,x_local_ap,y_local_ap,layer25_out_ap);

    nnet::convert_data<result_t, double, N_LAYER_23>(layer25_out_ap, layer25_out);
}
}

#endif
