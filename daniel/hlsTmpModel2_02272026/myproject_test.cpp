#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static std::string s_weights_dir;

const char *get_weights_dir() { return s_weights_dir.c_str(); }

#include "firmware/myproject.h"
#include <ac_shared.h>
#include <ac_sync.h>
#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_scverify.h"
// #include "firmware/parameters.h"

#include <mc_scverify.h>

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

#ifndef RANDOM_FRAMES
#define RANDOM_FRAMES 1
#endif

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
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

CCS_MAIN(int argc, char *argv[]) {
    if ((argc < 2) || (argc == 3)) {
        std::cerr << "Error - too few arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <weights_dir> <tb_input_features> <tb_output_predictions> <threshold>"
                  << std::endl;
        std::cerr << "Where: <weights_dir>           - string pathname to directory containing wN.txt and bN.txt files"
                  << std::endl;
        std::cerr << "       <tb_input_features>     - string pathname to tb_input_features.dat (optional)" << std::endl;
        std::cerr << "       <tb_output_predictions> - string pathname to tb_output_predictions.dat (optional)" << std::endl;
        std::cerr << "       <threshold>             - Python vs C++ prediction comparison threshold." << std::endl;
        std::cerr << "                                 Set to 0.0 to disable. (optional)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "If no testbench input/prediction data provided, random input data will be generated" << std::endl;
        CCS_RETURN(-1);
    }
    s_weights_dir = argv[1];
    std::cout << "  Weights directory: " << s_weights_dir << std::endl;

    std::string tb_in;
    std::string tb_out;
    float threshold = 0.0;
    std::ifstream fin;
    std::ifstream fpr;
    bool use_random = false;
    if (argc == 2) {
        std::cout << "No testbench files provided - Using random input data" << std::endl;
        use_random = true;
    } else if (argc > 3) {
        tb_in = argv[2];
        tb_out = argv[3];
        std::cout << "  Test Feature Data: " << tb_in << std::endl;
        std::cout << "  Test Predictions : " << tb_out << std::endl;

        // load input data from text file
        fin.open(tb_in);
        // load predictions from text file
        fpr.open(tb_out);
        if (!fin.is_open() || !fpr.is_open()) {
            use_random = true;
        }
    }
    if (argc == 5) {
        threshold = atof(argv[4]);
    }

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);

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
    std::string iline;
    std::string pline;
    int e = 0;
    unsigned int total_err_cnt = 0;
    (void)total_err_cnt; // to prevent unused-variable warnings when tb feature is not enabled
    (void)threshold; // to prevent unused-variable warnings when tb feature is not enabled



    if (!use_random) {
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<float> in; // variable's name must be 'in' to work with inserted code below
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr; // variable's name must be 'pr' to work with inserted code below
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            //    std::cout << "    Input feature map size = " << in.size() << " Output predictions size = " << pr.size() <<
            //    std::endl;

            // hls-fpga-machine-learning insert data
            ac_shared<input_t[21] > x_profile /* reshape */;
            static bool x_profile_init = ac::init_array<AC_VAL_0>(x_profile, 21); 
            ac_sync x_profile_sync; 
            x_profile_sync.sync_out(); 
            nnet::copy_data<float, input_t, 0, 21>(in, x_profile);
            ac_shared<input5_t[1] > nModule /* reshape */;
            static bool nModule_init = ac::init_array<AC_VAL_0>(nModule, 1); 
            ac_sync nModule_sync; 
            nModule_sync.sync_out(); 
            nnet::copy_data<float, input5_t, 21, 1>(in, nModule);
            ac_shared<input6_t[1] > x_local /* reshape */;
            static bool x_local_init = ac::init_array<AC_VAL_0>(x_local, 1); 
            ac_sync x_local_sync; 
            x_local_sync.sync_out(); 
            nnet::copy_data<float, input6_t, 22, 1>(in, x_local);
            ac_shared<input2_t[13] > y_profile /* reshape */;
            static bool y_profile_init = ac::init_array<AC_VAL_0>(y_profile, 13); 
            ac_sync y_profile_sync; 
            y_profile_sync.sync_out(); 
            nnet::copy_data<float, input2_t, 23, 13>(in, y_profile);
            ac_shared<input4_t[1] > y_local /* reshape */;
            static bool y_local_init = ac::init_array<AC_VAL_0>(y_local, 1); 
            ac_sync y_local_sync; 
            y_local_sync.sync_out(); 
            nnet::copy_data<float, input4_t, 36, 1>(in, y_local);
            ac_shared<result_t[1] > layer24_out /* partition */;
            ac_sync layer24_out_sync;

            // hls-fpga-machine-learning insert top-level-function
            myproject(x_profile, x_profile_sync, nModule, nModule_sync, x_local, x_local_sync, y_profile, y_profile_sync, y_local, y_local_sync,layer24_out, layer24_out_sync);
            layer24_out_sync.sync_in();

            if (threshold > 0.0) {
                // hls-fpga-machine-learning insert output-compare
                total_err_cnt += nnet::compare_data<float, result_t, 0, 1>(pr, layer24_out, threshold);
            }

            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                for(int i = 0; i < 1; i++) {
                  std::cout << pr[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                nnet::print_result<result_t, 1>(layer24_out, fout);
            }
            e++;

            // hls-fpga-machine-learning insert tb-output
            nnet::print_result<result_t, 1>(layer24_out, fout);
        }
        if (fin.is_open()) {
            fin.close();
        }
        if (fpr.is_open()) {
            fpr.close();
        }
    } else {
        std::cout << "INFO: Unable to open input/predictions file(s) so feeding random values" << std::endl;
        std::cout << "Number of Frames Passed from the tcl= " << RANDOM_FRAMES << std::endl;

        if (RANDOM_FRAMES > 0) {
            for (unsigned int k = 0; k < RANDOM_FRAMES-1; k++) {
                // hls-fpga-machine-learning insert random
                ac_shared<input_t[21] > x_profile /* reshape */;
                static bool x_profile_init = ac::init_array<AC_VAL_DC>(x_profile, 21); 
                ac_sync x_profile_sync; 
                x_profile_sync.sync_out(); 
                ac_shared<input5_t[1] > nModule /* reshape */;
                static bool nModule_init = ac::init_array<AC_VAL_DC>(nModule, 1); 
                ac_sync nModule_sync; 
                nModule_sync.sync_out(); 
                ac_shared<input6_t[1] > x_local /* reshape */;
                static bool x_local_init = ac::init_array<AC_VAL_DC>(x_local, 1); 
                ac_sync x_local_sync; 
                x_local_sync.sync_out(); 
                ac_shared<input2_t[13] > y_profile /* reshape */;
                static bool y_profile_init = ac::init_array<AC_VAL_DC>(y_profile, 13); 
                ac_sync y_profile_sync; 
                y_profile_sync.sync_out(); 
                ac_shared<input4_t[1] > y_local /* reshape */;
                static bool y_local_init = ac::init_array<AC_VAL_DC>(y_local, 1); 
                ac_sync y_local_sync; 
                y_local_sync.sync_out(); 
                ac_shared<result_t[1] > layer24_out /* partition */;
                ac_sync layer24_out_sync;

                // hls-fpga-machine-learning insert top-level-function
                myproject(x_profile, x_profile_sync, nModule, nModule_sync, x_local, x_local_sync, y_profile, y_profile_sync, y_local, y_local_sync,layer24_out, layer24_out_sync);
                layer24_out_sync.sync_in();

                // hls-fpga-machine-learning insert output
                nnet::print_result<result_t, 1>(layer24_out, fout);

                // hls-fpga-machine-learning insert tb-output
                nnet::print_result<result_t, 1>(layer24_out, fout);
            }
        } else {
            // hls-fpga-machine-learning insert zero
            ac_shared<input_t[21] > x_profile /* reshape */;
            static bool x_profile_init = ac::init_array<AC_VAL_0>(x_profile, 21); 
            ac_sync x_profile_sync; 
            x_profile_sync.sync_out(); 
            ac_shared<input5_t[1] > nModule /* reshape */;
            static bool nModule_init = ac::init_array<AC_VAL_0>(nModule, 1); 
            ac_sync nModule_sync; 
            nModule_sync.sync_out(); 
            ac_shared<input6_t[1] > x_local /* reshape */;
            static bool x_local_init = ac::init_array<AC_VAL_0>(x_local, 1); 
            ac_sync x_local_sync; 
            x_local_sync.sync_out(); 
            ac_shared<input2_t[13] > y_profile /* reshape */;
            static bool y_profile_init = ac::init_array<AC_VAL_0>(y_profile, 13); 
            ac_sync y_profile_sync; 
            y_profile_sync.sync_out(); 
            ac_shared<input4_t[1] > y_local /* reshape */;
            static bool y_local_init = ac::init_array<AC_VAL_0>(y_local, 1); 
            ac_sync y_local_sync; 
            y_local_sync.sync_out(); 
            ac_shared<result_t[1] > layer24_out /* partition */;
            ac_sync layer24_out_sync;

            // hls-fpga-machine-learning insert top-level-function
            myproject(x_profile, x_profile_sync, nModule, nModule_sync, x_local, x_local_sync, y_profile, y_profile_sync, y_local, y_local_sync,layer24_out, layer24_out_sync);
            layer24_out_sync.sync_in();

            // hls-fpga-machine-learning insert output
            nnet::print_result<result_t, 1>(layer24_out, fout);

            // hls-fpga-machine-learning insert tb-output
            nnet::print_result<result_t, 1>(layer24_out, fout);
        }
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    if (!use_random) {
        if (total_err_cnt) {
            std::cerr << "Error: A total of " << total_err_cnt
                      << " differences detected between golden Python prediction and C++ prediction using threshold of "
                      << threshold << std::endl;
        } else {
            if (threshold > 0.0) {
                std::cout << "Python predictions and C++ predictions are within threshold of " << threshold << std::endl;
            }
        }
    }
    return total_err_cnt;
}
