#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <ac_fixed.h>
#include <ac_int.h>

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_stream.h"

// hls-fpga-machine-learning insert weights


// hls-fpga-machine-learning insert layer-config
// concatenate_24
struct config2 : nnet::concat_config {
    static const unsigned n_elem1_0 = 13;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 13;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// concatenate_25
struct config3 : nnet::concat_config {
    static const unsigned n_elem1_0 = 26;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 26;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// hidden_dense
struct config4 : nnet::dense_config {
    static const unsigned n_in = 52;
    static const unsigned n_out = 10;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 116;
    static const unsigned n_nonzeros = 404;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    typedef layer4_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// output_dense
struct config6 : nnet::dense_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 2;
    static const unsigned n_nonzeros = 8;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias6_t bias_t;
    typedef weight6_t weight_t;
    typedef layer6_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



#endif
