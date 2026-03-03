#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <ac_fixed.h>
#include <ac_int.h>

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"

// hls-fpga-machine-learning insert weights


// hls-fpga-machine-learning insert layer-config
// xy_concat
struct config3 : nnet::concat_config {
    static const unsigned n_elem1_0 = 21;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 13;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// other_features
struct config7 : nnet::concat_config {
    static const unsigned n_elem1_0 = 34;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 1;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// nmodule_xlocal_concat
struct config8 : nnet::concat_config {
    static const unsigned n_elem1_0 = 1;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 1;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// other_dense
struct config9 : nnet::dense_config {
    static const unsigned n_in = 35;
    static const unsigned n_out = 128;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 825;
    static const unsigned n_nonzeros = 3655;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias9_t bias_t;
    typedef weight9_t weight_t;
    typedef layer9_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// nmodule_xlocal_dense
struct config11 : nnet::dense_config {
    static const unsigned n_in = 2;
    static const unsigned n_out = 14;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 2;
    static const unsigned n_nonzeros = 26;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    typedef layer11_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// other_activation
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef other_activation_table_t table_t;
};

// nmodule_xlocal_activation
struct relu_config14 : nnet::activ_config {
    static const unsigned n_in = 14;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef nmodule_xlocal_activation_table_t table_t;
};

// merged_features
struct config15 : nnet::concat_config {
    static const unsigned n_elem1_0 = 128;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 14;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// dense2
struct config16 : nnet::dense_config {
    static const unsigned n_in = 142;
    static const unsigned n_out = 56;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 2720;
    static const unsigned n_nonzeros = 5232;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias16_t bias_t;
    typedef weight16_t weight_t;
    typedef layer16_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense2_activation
struct relu_config18 : nnet::activ_config {
    static const unsigned n_in = 56;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense2_activation_table_t table_t;
};

// dense3
struct config19 : nnet::dense_config {
    static const unsigned n_in = 56;
    static const unsigned n_out = 42;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 533;
    static const unsigned n_nonzeros = 1819;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias19_t bias_t;
    typedef weight19_t weight_t;
    typedef layer19_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense3_activation
struct relu_config21 : nnet::activ_config {
    static const unsigned n_in = 42;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense3_activation_table_t table_t;
};

// output
struct config22 : nnet::dense_config {
    static const unsigned n_in = 42;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 11;
    static const unsigned n_nonzeros = 31;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias22_t bias_t;
    typedef weight22_t weight_t;
    typedef layer22_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// output_activation
struct hard_tanh_config24 {
    static const unsigned n_in = 1;
    static const slope24_t slope;
    static const shift24_t shift;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
};
// really this allocation of pixels array ought to be in a .cpp file
#ifndef INCLUDED_MC_TESTBENCH_H
const slope24_t hard_tanh_config24::slope = 0.5;
const shift24_t hard_tanh_config24::shift = 0.5;
#endif



#endif
