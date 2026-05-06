#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w16.h"
#include "weights/b16.h"
#include "weights/w21.h"
#include "weights/b21.h"
#include "weights/w24.h"
#include "weights/b24.h"
#include "weights/w27.h"
#include "weights/b27.h"

// hls-fpga-machine-learning insert layer-config
// q_input_cluster
struct linear_config2 : nnet::activ_config {
    static const unsigned n_in = 273;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    typedef q_input_cluster_table_t table_t;
};

// q_input_nModule
struct linear_config6 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    typedef q_input_nModule_table_t table_t;
};

// q_input_x_local
struct linear_config7 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    typedef q_input_x_local_table_t table_t;
};

// conv2d
struct config9_mult : nnet::dense_config {
    static const unsigned n_in = 9;
    static const unsigned n_out = 8;
    static const unsigned reuse_factor = 72;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 3;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef bias9_t bias_t;
    typedef weight9_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config9 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = 13;
    static const unsigned in_width = 21;
    static const unsigned n_chan = 1;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 8;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 13;
    static const unsigned out_width = 21;
    static const unsigned reuse_factor = 72;
    static const unsigned n_zeros = 3;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 13;
    static const unsigned min_width = 21;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 273;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_9<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef bias9_t bias_t;
    typedef weight9_t weight_t;
    typedef config9_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_unscaled<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config9::filt_height * config9::filt_width> config9::pixels[] = {0};

// concat_scalars_1
struct config11 : nnet::concat_config {
    static const unsigned n_elem1_0 = 1;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 1;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// q_input_y_local
struct linear_config12 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    typedef q_input_y_local_table_t table_t;
};

// conv2d_act
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 2184;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 128;
    typedef conv2d_act_table_t table_t;
};

// concat_scalars_2
struct config14 : nnet::concat_config {
    static const unsigned n_elem1_0 = 2;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 1;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// pool2d_1
struct config15 : nnet::pooling2d_config {
    static const unsigned in_height = 13;
    static const unsigned in_width = 21;
    static const unsigned n_filt = 8;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 6;
    static const unsigned out_width = 10;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 16;
    typedef model_default_t accum_t;
};

// dense_scalars
struct config16 : nnet::dense_config {
    static const unsigned n_in = 3;
    static const unsigned n_out = 24;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 18;
    static const unsigned n_zeros = 1;
    static const unsigned n_nonzeros = 71;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias16_t bias_t;
    typedef weight16_t weight_t;
    typedef layer16_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_scalars_act
struct relu_config19 : nnet::activ_config {
    static const unsigned n_in = 24;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    typedef dense_scalars_act_table_t table_t;
};

// concat_all
struct config20 : nnet::concat_config {
    static const unsigned n_elem1_0 = 480;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 24;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// merged_dense1
struct config21 : nnet::dense_config {
    static const unsigned n_in = 504;
    static const unsigned n_out = 72;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 14;
    static const unsigned n_zeros = 845;
    static const unsigned n_nonzeros = 35443;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias21_t bias_t;
    typedef weight21_t weight_t;
    typedef layer21_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// merged_dense1_act
struct relu_config23 : nnet::activ_config {
    static const unsigned n_in = 72;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    typedef merged_dense1_act_table_t table_t;
};

// merged_dense2
struct config24 : nnet::dense_config {
    static const unsigned n_in = 72;
    static const unsigned n_out = 36;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 18;
    static const unsigned n_zeros = 38;
    static const unsigned n_nonzeros = 2554;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias24_t bias_t;
    typedef weight24_t weight_t;
    typedef layer24_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// merged_dense2_act
struct relu_config26 : nnet::activ_config {
    static const unsigned n_in = 36;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
    typedef merged_dense2_act_table_t table_t;
};

// output_dense
struct config27 : nnet::dense_config {
    static const unsigned n_in = 36;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 18;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 36;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias27_t bias_t;
    typedef weight27_t weight_t;
    typedef layer27_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// output_activation
struct hard_sigmoid_config29 {
    static const unsigned n_in = 1;
    static const slope29_t slope;
    static const shift29_t shift;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 16;
};
const slope29_t hard_sigmoid_config29::slope = 0.5;
const shift29_t hard_sigmoid_config29::shift = 0.5;


#endif
