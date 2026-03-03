#include <iostream>

#include "myproject.h"
#include <mc_scverify.h>

#include <ac_shared.h>
#include <ac_sync.h>

#include "parameters.h"




#pragma hls_design top
// hls-fpga-machine-learning insert IFSynPragmas
void CCS_BLOCK(myproject)(
    input_t x_profile[21] /* reshape */, ac_sync &x_profile_sync, input5_t nModule[1] /* reshape */, ac_sync &nModule_sync, input6_t x_local[1] /* reshape */, ac_sync &x_local_sync, input2_t y_profile[13] /* reshape */, ac_sync &y_profile_sync, input4_t y_local[1] /* reshape */, ac_sync &y_local_sync,
    result_t layer24_out[1] /* partition */, ac_sync &layer24_out_sync
) {

// hls-fpga-machine-learning insert weights
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w16.h"
#include "weights/b16.h"
#include "weights/w19.h"
#include "weights/b19.h"
#include "weights/w22.h"
#include "weights/b22.h"

    // hls-fpga-machine-learning insert IO
    // RESHAPE variable=x_profile complete dim=0
    // RESHAPE variable=nModule complete dim=0
    // RESHAPE variable=x_local complete dim=0
    // RESHAPE variable=y_profile complete dim=0
    // RESHAPE variable=y_local complete dim=0
    // PARTITION variable=layer24_out complete dim=0

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight9_t, 4480>(w9, "w9.txt");
        nnet::load_weights_from_txt<bias9_t, 128>(b9, "b9.txt");
        nnet::load_weights_from_txt<weight11_t, 28>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 14>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight16_t, 7952>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 56>(b16, "b16.txt");
        nnet::load_weights_from_txt<weight19_t, 2352>(w19, "w19.txt");
        nnet::load_weights_from_txt<bias19_t, 42>(b19, "b19.txt");
        nnet::load_weights_from_txt<weight22_t, 42>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 1>(b22, "b22.txt");
        loaded_weights = true;
    }
#endif



    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    
    static ac_shared<layer3_t[34] > layer3_out /* partition */; 
    static bool layer3_out_init = ac::init_array<AC_VAL_DC>(layer3_out, 34); 
    static ac_sync layer3_out_sync;
    // PARTITION variable=layer3_out complete dim=0
    nnet::concatenate1d<input_t, input2_t, layer3_t, config3>(x_profile, x_profile_sync, y_profile, y_profile_sync, layer3_out, layer3_out_sync); // xy_concat

    
    static ac_shared<layer7_t[35] > layer7_out /* partition */; 
    static bool layer7_out_init = ac::init_array<AC_VAL_DC>(layer7_out, 35); 
    static ac_sync layer7_out_sync;
    // PARTITION variable=layer7_out complete dim=0
    nnet::concatenate1d<layer3_t, input4_t, layer7_t, config7>(layer3_out, layer3_out_sync, y_local, y_local_sync, layer7_out, layer7_out_sync); // other_features

    
    static ac_shared<layer8_t[2] > layer8_out /* partition */; 
    static bool layer8_out_init = ac::init_array<AC_VAL_DC>(layer8_out, 2); 
    static ac_sync layer8_out_sync;
    // PARTITION variable=layer8_out complete dim=0
    nnet::concatenate1d<input5_t, input6_t, layer8_t, config8>(nModule, nModule_sync, x_local, x_local_sync, layer8_out, layer8_out_sync); // nmodule_xlocal_concat

    
    static ac_shared<layer9_t[128] > layer9_out /* partition */; 
    static bool layer9_out_init = ac::init_array<AC_VAL_DC>(layer9_out, 128); 
    static ac_sync layer9_out_sync;
    // PARTITION variable=layer9_out complete dim=0
    nnet::dense<layer7_t, layer9_t, config9>(layer7_out, layer7_out_sync, layer9_out, layer9_out_sync, w9, b9); // other_dense

    
    static ac_shared<layer11_t[14] > layer11_out /* partition */; 
    static bool layer11_out_init = ac::init_array<AC_VAL_DC>(layer11_out, 14); 
    static ac_sync layer11_out_sync;
    // PARTITION variable=layer11_out complete dim=0
    nnet::dense<layer8_t, layer11_t, config11>(layer8_out, layer8_out_sync, layer11_out, layer11_out_sync, w11, b11); // nmodule_xlocal_dense

    
    static ac_shared<layer13_t[128] > layer13_out /* partition */; 
    static bool layer13_out_init = ac::init_array<AC_VAL_DC>(layer13_out, 128); 
    static ac_sync layer13_out_sync;
    // PARTITION variable=layer13_out complete dim=0
    nnet::relu<layer9_t, layer13_t, relu_config13>(layer9_out, layer9_out_sync, layer13_out, layer13_out_sync); // other_activation

    
    static ac_shared<layer14_t[14] > layer14_out /* partition */; 
    static bool layer14_out_init = ac::init_array<AC_VAL_DC>(layer14_out, 14); 
    static ac_sync layer14_out_sync;
    // PARTITION variable=layer14_out complete dim=0
    nnet::relu<layer11_t, layer14_t, relu_config14>(layer11_out, layer11_out_sync, layer14_out, layer14_out_sync); // nmodule_xlocal_activation

    
    static ac_shared<layer15_t[142] > layer15_out /* partition */; 
    static bool layer15_out_init = ac::init_array<AC_VAL_DC>(layer15_out, 142); 
    static ac_sync layer15_out_sync;
    // PARTITION variable=layer15_out complete dim=0
    nnet::concatenate1d<layer13_t, layer14_t, layer15_t, config15>(layer13_out, layer13_out_sync, layer14_out, layer14_out_sync, layer15_out, layer15_out_sync); // merged_features

    
    static ac_shared<layer16_t[56] > layer16_out /* partition */; 
    static bool layer16_out_init = ac::init_array<AC_VAL_DC>(layer16_out, 56); 
    static ac_sync layer16_out_sync;
    // PARTITION variable=layer16_out complete dim=0
    nnet::dense<layer15_t, layer16_t, config16>(layer15_out, layer15_out_sync, layer16_out, layer16_out_sync, w16, b16); // dense2

    
    static ac_shared<layer18_t[56] > layer18_out /* partition */; 
    static bool layer18_out_init = ac::init_array<AC_VAL_DC>(layer18_out, 56); 
    static ac_sync layer18_out_sync;
    // PARTITION variable=layer18_out complete dim=0
    nnet::relu<layer16_t, layer18_t, relu_config18>(layer16_out, layer16_out_sync, layer18_out, layer18_out_sync); // dense2_activation

    
    static ac_shared<layer19_t[42] > layer19_out /* partition */; 
    static bool layer19_out_init = ac::init_array<AC_VAL_DC>(layer19_out, 42); 
    static ac_sync layer19_out_sync;
    // PARTITION variable=layer19_out complete dim=0
    nnet::dense<layer18_t, layer19_t, config19>(layer18_out, layer18_out_sync, layer19_out, layer19_out_sync, w19, b19); // dense3

    
    static ac_shared<layer21_t[42] > layer21_out /* partition */; 
    static bool layer21_out_init = ac::init_array<AC_VAL_DC>(layer21_out, 42); 
    static ac_sync layer21_out_sync;
    // PARTITION variable=layer21_out complete dim=0
    nnet::relu<layer19_t, layer21_t, relu_config21>(layer19_out, layer19_out_sync, layer21_out, layer21_out_sync); // dense3_activation

    
    static ac_shared<layer22_t[1] > layer22_out /* partition */; 
    static bool layer22_out_init = ac::init_array<AC_VAL_DC>(layer22_out, 1); 
    static ac_sync layer22_out_sync;
    // PARTITION variable=layer22_out complete dim=0
    nnet::dense<layer21_t, layer22_t, config22>(layer21_out, layer21_out_sync, layer22_out, layer22_out_sync, w22, b22); // output

    nnet::hard_tanh<layer22_t, result_t, hard_tanh_config24>(layer22_out, layer22_out_sync, layer24_out, layer24_out_sync); // output_activation

}

