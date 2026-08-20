#include "hls_signal_handler.h"
#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include "ap_fixed.h"
#include "ap_int.h"
#include "autopilot_cbe.h"
#include "hls_half.h"
#include "hls_directio.h"
#include "hls_stream.h"

using namespace std;

// wrapc file define:
#define AUTOTB_TVIN_cluster "../tv/cdatafile/c.myproject.autotvin_cluster.dat"
#define AUTOTB_TVOUT_cluster "../tv/cdatafile/c.myproject.autotvout_cluster.dat"
#define AUTOTB_TVIN_nModule "../tv/cdatafile/c.myproject.autotvin_nModule.dat"
#define AUTOTB_TVOUT_nModule "../tv/cdatafile/c.myproject.autotvout_nModule.dat"
#define AUTOTB_TVIN_x_local "../tv/cdatafile/c.myproject.autotvin_x_local.dat"
#define AUTOTB_TVOUT_x_local "../tv/cdatafile/c.myproject.autotvout_x_local.dat"
#define AUTOTB_TVIN_y_local "../tv/cdatafile/c.myproject.autotvin_y_local.dat"
#define AUTOTB_TVOUT_y_local "../tv/cdatafile/c.myproject.autotvout_y_local.dat"
#define AUTOTB_TVIN_layer29_out "../tv/cdatafile/c.myproject.autotvin_layer29_out.dat"
#define AUTOTB_TVOUT_layer29_out "../tv/cdatafile/c.myproject.autotvout_layer29_out.dat"
#define AUTOTB_TVIN_w9 "../tv/cdatafile/c.myproject.autotvin_w9.dat"
#define AUTOTB_TVOUT_w9 "../tv/cdatafile/c.myproject.autotvout_w9.dat"
#define AUTOTB_TVIN_b9_0 "../tv/cdatafile/c.myproject.autotvin_b9_0.dat"
#define AUTOTB_TVOUT_b9_0 "../tv/cdatafile/c.myproject.autotvout_b9_0.dat"
#define AUTOTB_TVIN_b9_1 "../tv/cdatafile/c.myproject.autotvin_b9_1.dat"
#define AUTOTB_TVOUT_b9_1 "../tv/cdatafile/c.myproject.autotvout_b9_1.dat"
#define AUTOTB_TVIN_w16 "../tv/cdatafile/c.myproject.autotvin_w16.dat"
#define AUTOTB_TVOUT_w16 "../tv/cdatafile/c.myproject.autotvout_w16.dat"
#define AUTOTB_TVIN_b16_0 "../tv/cdatafile/c.myproject.autotvin_b16_0.dat"
#define AUTOTB_TVOUT_b16_0 "../tv/cdatafile/c.myproject.autotvout_b16_0.dat"
#define AUTOTB_TVIN_b16_1 "../tv/cdatafile/c.myproject.autotvin_b16_1.dat"
#define AUTOTB_TVOUT_b16_1 "../tv/cdatafile/c.myproject.autotvout_b16_1.dat"
#define AUTOTB_TVIN_b16_2 "../tv/cdatafile/c.myproject.autotvin_b16_2.dat"
#define AUTOTB_TVOUT_b16_2 "../tv/cdatafile/c.myproject.autotvout_b16_2.dat"
#define AUTOTB_TVIN_b16_3 "../tv/cdatafile/c.myproject.autotvin_b16_3.dat"
#define AUTOTB_TVOUT_b16_3 "../tv/cdatafile/c.myproject.autotvout_b16_3.dat"
#define AUTOTB_TVIN_b16_4 "../tv/cdatafile/c.myproject.autotvin_b16_4.dat"
#define AUTOTB_TVOUT_b16_4 "../tv/cdatafile/c.myproject.autotvout_b16_4.dat"
#define AUTOTB_TVIN_b16_5 "../tv/cdatafile/c.myproject.autotvin_b16_5.dat"
#define AUTOTB_TVOUT_b16_5 "../tv/cdatafile/c.myproject.autotvout_b16_5.dat"
#define AUTOTB_TVIN_b16_6 "../tv/cdatafile/c.myproject.autotvin_b16_6.dat"
#define AUTOTB_TVOUT_b16_6 "../tv/cdatafile/c.myproject.autotvout_b16_6.dat"
#define AUTOTB_TVIN_b16_7 "../tv/cdatafile/c.myproject.autotvin_b16_7.dat"
#define AUTOTB_TVOUT_b16_7 "../tv/cdatafile/c.myproject.autotvout_b16_7.dat"
#define AUTOTB_TVIN_b16_8 "../tv/cdatafile/c.myproject.autotvin_b16_8.dat"
#define AUTOTB_TVOUT_b16_8 "../tv/cdatafile/c.myproject.autotvout_b16_8.dat"
#define AUTOTB_TVIN_b16_9 "../tv/cdatafile/c.myproject.autotvin_b16_9.dat"
#define AUTOTB_TVOUT_b16_9 "../tv/cdatafile/c.myproject.autotvout_b16_9.dat"
#define AUTOTB_TVIN_b16_10 "../tv/cdatafile/c.myproject.autotvin_b16_10.dat"
#define AUTOTB_TVOUT_b16_10 "../tv/cdatafile/c.myproject.autotvout_b16_10.dat"
#define AUTOTB_TVIN_b16_11 "../tv/cdatafile/c.myproject.autotvin_b16_11.dat"
#define AUTOTB_TVOUT_b16_11 "../tv/cdatafile/c.myproject.autotvout_b16_11.dat"
#define AUTOTB_TVIN_b16_12 "../tv/cdatafile/c.myproject.autotvin_b16_12.dat"
#define AUTOTB_TVOUT_b16_12 "../tv/cdatafile/c.myproject.autotvout_b16_12.dat"
#define AUTOTB_TVIN_b16_13 "../tv/cdatafile/c.myproject.autotvin_b16_13.dat"
#define AUTOTB_TVOUT_b16_13 "../tv/cdatafile/c.myproject.autotvout_b16_13.dat"
#define AUTOTB_TVIN_b16_14 "../tv/cdatafile/c.myproject.autotvin_b16_14.dat"
#define AUTOTB_TVOUT_b16_14 "../tv/cdatafile/c.myproject.autotvout_b16_14.dat"
#define AUTOTB_TVIN_b16_15 "../tv/cdatafile/c.myproject.autotvin_b16_15.dat"
#define AUTOTB_TVOUT_b16_15 "../tv/cdatafile/c.myproject.autotvout_b16_15.dat"
#define AUTOTB_TVIN_w21 "../tv/cdatafile/c.myproject.autotvin_w21.dat"
#define AUTOTB_TVOUT_w21 "../tv/cdatafile/c.myproject.autotvout_w21.dat"
#define AUTOTB_TVIN_b21_0 "../tv/cdatafile/c.myproject.autotvin_b21_0.dat"
#define AUTOTB_TVOUT_b21_0 "../tv/cdatafile/c.myproject.autotvout_b21_0.dat"
#define AUTOTB_TVIN_b21_1 "../tv/cdatafile/c.myproject.autotvin_b21_1.dat"
#define AUTOTB_TVOUT_b21_1 "../tv/cdatafile/c.myproject.autotvout_b21_1.dat"
#define AUTOTB_TVIN_b21_2 "../tv/cdatafile/c.myproject.autotvin_b21_2.dat"
#define AUTOTB_TVOUT_b21_2 "../tv/cdatafile/c.myproject.autotvout_b21_2.dat"
#define AUTOTB_TVIN_b21_3 "../tv/cdatafile/c.myproject.autotvin_b21_3.dat"
#define AUTOTB_TVOUT_b21_3 "../tv/cdatafile/c.myproject.autotvout_b21_3.dat"
#define AUTOTB_TVIN_b21_4 "../tv/cdatafile/c.myproject.autotvin_b21_4.dat"
#define AUTOTB_TVOUT_b21_4 "../tv/cdatafile/c.myproject.autotvout_b21_4.dat"
#define AUTOTB_TVIN_b21_5 "../tv/cdatafile/c.myproject.autotvin_b21_5.dat"
#define AUTOTB_TVOUT_b21_5 "../tv/cdatafile/c.myproject.autotvout_b21_5.dat"
#define AUTOTB_TVIN_b21_6 "../tv/cdatafile/c.myproject.autotvin_b21_6.dat"
#define AUTOTB_TVOUT_b21_6 "../tv/cdatafile/c.myproject.autotvout_b21_6.dat"
#define AUTOTB_TVIN_b21_7 "../tv/cdatafile/c.myproject.autotvin_b21_7.dat"
#define AUTOTB_TVOUT_b21_7 "../tv/cdatafile/c.myproject.autotvout_b21_7.dat"
#define AUTOTB_TVIN_b21_8 "../tv/cdatafile/c.myproject.autotvin_b21_8.dat"
#define AUTOTB_TVOUT_b21_8 "../tv/cdatafile/c.myproject.autotvout_b21_8.dat"
#define AUTOTB_TVIN_b21_9 "../tv/cdatafile/c.myproject.autotvin_b21_9.dat"
#define AUTOTB_TVOUT_b21_9 "../tv/cdatafile/c.myproject.autotvout_b21_9.dat"
#define AUTOTB_TVIN_b21_10 "../tv/cdatafile/c.myproject.autotvin_b21_10.dat"
#define AUTOTB_TVOUT_b21_10 "../tv/cdatafile/c.myproject.autotvout_b21_10.dat"
#define AUTOTB_TVIN_b21_11 "../tv/cdatafile/c.myproject.autotvin_b21_11.dat"
#define AUTOTB_TVOUT_b21_11 "../tv/cdatafile/c.myproject.autotvout_b21_11.dat"
#define AUTOTB_TVIN_b21_12 "../tv/cdatafile/c.myproject.autotvin_b21_12.dat"
#define AUTOTB_TVOUT_b21_12 "../tv/cdatafile/c.myproject.autotvout_b21_12.dat"
#define AUTOTB_TVIN_b21_13 "../tv/cdatafile/c.myproject.autotvin_b21_13.dat"
#define AUTOTB_TVOUT_b21_13 "../tv/cdatafile/c.myproject.autotvout_b21_13.dat"
#define AUTOTB_TVIN_b21_14 "../tv/cdatafile/c.myproject.autotvin_b21_14.dat"
#define AUTOTB_TVOUT_b21_14 "../tv/cdatafile/c.myproject.autotvout_b21_14.dat"
#define AUTOTB_TVIN_b21_15 "../tv/cdatafile/c.myproject.autotvin_b21_15.dat"
#define AUTOTB_TVOUT_b21_15 "../tv/cdatafile/c.myproject.autotvout_b21_15.dat"
#define AUTOTB_TVIN_b21_16 "../tv/cdatafile/c.myproject.autotvin_b21_16.dat"
#define AUTOTB_TVOUT_b21_16 "../tv/cdatafile/c.myproject.autotvout_b21_16.dat"
#define AUTOTB_TVIN_b21_17 "../tv/cdatafile/c.myproject.autotvin_b21_17.dat"
#define AUTOTB_TVOUT_b21_17 "../tv/cdatafile/c.myproject.autotvout_b21_17.dat"
#define AUTOTB_TVIN_b21_18 "../tv/cdatafile/c.myproject.autotvin_b21_18.dat"
#define AUTOTB_TVOUT_b21_18 "../tv/cdatafile/c.myproject.autotvout_b21_18.dat"
#define AUTOTB_TVIN_b21_19 "../tv/cdatafile/c.myproject.autotvin_b21_19.dat"
#define AUTOTB_TVOUT_b21_19 "../tv/cdatafile/c.myproject.autotvout_b21_19.dat"
#define AUTOTB_TVIN_b21_20 "../tv/cdatafile/c.myproject.autotvin_b21_20.dat"
#define AUTOTB_TVOUT_b21_20 "../tv/cdatafile/c.myproject.autotvout_b21_20.dat"
#define AUTOTB_TVIN_b21_21 "../tv/cdatafile/c.myproject.autotvin_b21_21.dat"
#define AUTOTB_TVOUT_b21_21 "../tv/cdatafile/c.myproject.autotvout_b21_21.dat"
#define AUTOTB_TVIN_b21_22 "../tv/cdatafile/c.myproject.autotvin_b21_22.dat"
#define AUTOTB_TVOUT_b21_22 "../tv/cdatafile/c.myproject.autotvout_b21_22.dat"
#define AUTOTB_TVIN_b21_23 "../tv/cdatafile/c.myproject.autotvin_b21_23.dat"
#define AUTOTB_TVOUT_b21_23 "../tv/cdatafile/c.myproject.autotvout_b21_23.dat"
#define AUTOTB_TVIN_b21_24 "../tv/cdatafile/c.myproject.autotvin_b21_24.dat"
#define AUTOTB_TVOUT_b21_24 "../tv/cdatafile/c.myproject.autotvout_b21_24.dat"
#define AUTOTB_TVIN_b21_25 "../tv/cdatafile/c.myproject.autotvin_b21_25.dat"
#define AUTOTB_TVOUT_b21_25 "../tv/cdatafile/c.myproject.autotvout_b21_25.dat"
#define AUTOTB_TVIN_b21_26 "../tv/cdatafile/c.myproject.autotvin_b21_26.dat"
#define AUTOTB_TVOUT_b21_26 "../tv/cdatafile/c.myproject.autotvout_b21_26.dat"
#define AUTOTB_TVIN_b21_27 "../tv/cdatafile/c.myproject.autotvin_b21_27.dat"
#define AUTOTB_TVOUT_b21_27 "../tv/cdatafile/c.myproject.autotvout_b21_27.dat"
#define AUTOTB_TVIN_b21_28 "../tv/cdatafile/c.myproject.autotvin_b21_28.dat"
#define AUTOTB_TVOUT_b21_28 "../tv/cdatafile/c.myproject.autotvout_b21_28.dat"
#define AUTOTB_TVIN_b21_29 "../tv/cdatafile/c.myproject.autotvin_b21_29.dat"
#define AUTOTB_TVOUT_b21_29 "../tv/cdatafile/c.myproject.autotvout_b21_29.dat"
#define AUTOTB_TVIN_b21_30 "../tv/cdatafile/c.myproject.autotvin_b21_30.dat"
#define AUTOTB_TVOUT_b21_30 "../tv/cdatafile/c.myproject.autotvout_b21_30.dat"
#define AUTOTB_TVIN_b21_31 "../tv/cdatafile/c.myproject.autotvin_b21_31.dat"
#define AUTOTB_TVOUT_b21_31 "../tv/cdatafile/c.myproject.autotvout_b21_31.dat"
#define AUTOTB_TVIN_b21_32 "../tv/cdatafile/c.myproject.autotvin_b21_32.dat"
#define AUTOTB_TVOUT_b21_32 "../tv/cdatafile/c.myproject.autotvout_b21_32.dat"
#define AUTOTB_TVIN_b21_33 "../tv/cdatafile/c.myproject.autotvin_b21_33.dat"
#define AUTOTB_TVOUT_b21_33 "../tv/cdatafile/c.myproject.autotvout_b21_33.dat"
#define AUTOTB_TVIN_b21_34 "../tv/cdatafile/c.myproject.autotvin_b21_34.dat"
#define AUTOTB_TVOUT_b21_34 "../tv/cdatafile/c.myproject.autotvout_b21_34.dat"
#define AUTOTB_TVIN_b21_35 "../tv/cdatafile/c.myproject.autotvin_b21_35.dat"
#define AUTOTB_TVOUT_b21_35 "../tv/cdatafile/c.myproject.autotvout_b21_35.dat"
#define AUTOTB_TVIN_b21_36 "../tv/cdatafile/c.myproject.autotvin_b21_36.dat"
#define AUTOTB_TVOUT_b21_36 "../tv/cdatafile/c.myproject.autotvout_b21_36.dat"
#define AUTOTB_TVIN_b21_37 "../tv/cdatafile/c.myproject.autotvin_b21_37.dat"
#define AUTOTB_TVOUT_b21_37 "../tv/cdatafile/c.myproject.autotvout_b21_37.dat"
#define AUTOTB_TVIN_b21_38 "../tv/cdatafile/c.myproject.autotvin_b21_38.dat"
#define AUTOTB_TVOUT_b21_38 "../tv/cdatafile/c.myproject.autotvout_b21_38.dat"
#define AUTOTB_TVIN_b21_39 "../tv/cdatafile/c.myproject.autotvin_b21_39.dat"
#define AUTOTB_TVOUT_b21_39 "../tv/cdatafile/c.myproject.autotvout_b21_39.dat"
#define AUTOTB_TVIN_b21_40 "../tv/cdatafile/c.myproject.autotvin_b21_40.dat"
#define AUTOTB_TVOUT_b21_40 "../tv/cdatafile/c.myproject.autotvout_b21_40.dat"
#define AUTOTB_TVIN_b21_41 "../tv/cdatafile/c.myproject.autotvin_b21_41.dat"
#define AUTOTB_TVOUT_b21_41 "../tv/cdatafile/c.myproject.autotvout_b21_41.dat"
#define AUTOTB_TVIN_b21_42 "../tv/cdatafile/c.myproject.autotvin_b21_42.dat"
#define AUTOTB_TVOUT_b21_42 "../tv/cdatafile/c.myproject.autotvout_b21_42.dat"
#define AUTOTB_TVIN_b21_43 "../tv/cdatafile/c.myproject.autotvin_b21_43.dat"
#define AUTOTB_TVOUT_b21_43 "../tv/cdatafile/c.myproject.autotvout_b21_43.dat"
#define AUTOTB_TVIN_b21_44 "../tv/cdatafile/c.myproject.autotvin_b21_44.dat"
#define AUTOTB_TVOUT_b21_44 "../tv/cdatafile/c.myproject.autotvout_b21_44.dat"
#define AUTOTB_TVIN_b21_45 "../tv/cdatafile/c.myproject.autotvin_b21_45.dat"
#define AUTOTB_TVOUT_b21_45 "../tv/cdatafile/c.myproject.autotvout_b21_45.dat"
#define AUTOTB_TVIN_b21_46 "../tv/cdatafile/c.myproject.autotvin_b21_46.dat"
#define AUTOTB_TVOUT_b21_46 "../tv/cdatafile/c.myproject.autotvout_b21_46.dat"
#define AUTOTB_TVIN_b21_47 "../tv/cdatafile/c.myproject.autotvin_b21_47.dat"
#define AUTOTB_TVOUT_b21_47 "../tv/cdatafile/c.myproject.autotvout_b21_47.dat"
#define AUTOTB_TVIN_b21_48 "../tv/cdatafile/c.myproject.autotvin_b21_48.dat"
#define AUTOTB_TVOUT_b21_48 "../tv/cdatafile/c.myproject.autotvout_b21_48.dat"
#define AUTOTB_TVIN_b21_49 "../tv/cdatafile/c.myproject.autotvin_b21_49.dat"
#define AUTOTB_TVOUT_b21_49 "../tv/cdatafile/c.myproject.autotvout_b21_49.dat"
#define AUTOTB_TVIN_b21_50 "../tv/cdatafile/c.myproject.autotvin_b21_50.dat"
#define AUTOTB_TVOUT_b21_50 "../tv/cdatafile/c.myproject.autotvout_b21_50.dat"
#define AUTOTB_TVIN_b21_51 "../tv/cdatafile/c.myproject.autotvin_b21_51.dat"
#define AUTOTB_TVOUT_b21_51 "../tv/cdatafile/c.myproject.autotvout_b21_51.dat"
#define AUTOTB_TVIN_b21_52 "../tv/cdatafile/c.myproject.autotvin_b21_52.dat"
#define AUTOTB_TVOUT_b21_52 "../tv/cdatafile/c.myproject.autotvout_b21_52.dat"
#define AUTOTB_TVIN_b21_53 "../tv/cdatafile/c.myproject.autotvin_b21_53.dat"
#define AUTOTB_TVOUT_b21_53 "../tv/cdatafile/c.myproject.autotvout_b21_53.dat"
#define AUTOTB_TVIN_b21_54 "../tv/cdatafile/c.myproject.autotvin_b21_54.dat"
#define AUTOTB_TVOUT_b21_54 "../tv/cdatafile/c.myproject.autotvout_b21_54.dat"
#define AUTOTB_TVIN_b21_55 "../tv/cdatafile/c.myproject.autotvin_b21_55.dat"
#define AUTOTB_TVOUT_b21_55 "../tv/cdatafile/c.myproject.autotvout_b21_55.dat"
#define AUTOTB_TVIN_b21_56 "../tv/cdatafile/c.myproject.autotvin_b21_56.dat"
#define AUTOTB_TVOUT_b21_56 "../tv/cdatafile/c.myproject.autotvout_b21_56.dat"
#define AUTOTB_TVIN_b21_57 "../tv/cdatafile/c.myproject.autotvin_b21_57.dat"
#define AUTOTB_TVOUT_b21_57 "../tv/cdatafile/c.myproject.autotvout_b21_57.dat"
#define AUTOTB_TVIN_b21_58 "../tv/cdatafile/c.myproject.autotvin_b21_58.dat"
#define AUTOTB_TVOUT_b21_58 "../tv/cdatafile/c.myproject.autotvout_b21_58.dat"
#define AUTOTB_TVIN_b21_59 "../tv/cdatafile/c.myproject.autotvin_b21_59.dat"
#define AUTOTB_TVOUT_b21_59 "../tv/cdatafile/c.myproject.autotvout_b21_59.dat"
#define AUTOTB_TVIN_b21_60 "../tv/cdatafile/c.myproject.autotvin_b21_60.dat"
#define AUTOTB_TVOUT_b21_60 "../tv/cdatafile/c.myproject.autotvout_b21_60.dat"
#define AUTOTB_TVIN_b21_61 "../tv/cdatafile/c.myproject.autotvin_b21_61.dat"
#define AUTOTB_TVOUT_b21_61 "../tv/cdatafile/c.myproject.autotvout_b21_61.dat"
#define AUTOTB_TVIN_b21_62 "../tv/cdatafile/c.myproject.autotvin_b21_62.dat"
#define AUTOTB_TVOUT_b21_62 "../tv/cdatafile/c.myproject.autotvout_b21_62.dat"
#define AUTOTB_TVIN_b21_63 "../tv/cdatafile/c.myproject.autotvin_b21_63.dat"
#define AUTOTB_TVOUT_b21_63 "../tv/cdatafile/c.myproject.autotvout_b21_63.dat"
#define AUTOTB_TVIN_b21_64 "../tv/cdatafile/c.myproject.autotvin_b21_64.dat"
#define AUTOTB_TVOUT_b21_64 "../tv/cdatafile/c.myproject.autotvout_b21_64.dat"
#define AUTOTB_TVIN_b21_65 "../tv/cdatafile/c.myproject.autotvin_b21_65.dat"
#define AUTOTB_TVOUT_b21_65 "../tv/cdatafile/c.myproject.autotvout_b21_65.dat"
#define AUTOTB_TVIN_b21_66 "../tv/cdatafile/c.myproject.autotvin_b21_66.dat"
#define AUTOTB_TVOUT_b21_66 "../tv/cdatafile/c.myproject.autotvout_b21_66.dat"
#define AUTOTB_TVIN_b21_67 "../tv/cdatafile/c.myproject.autotvin_b21_67.dat"
#define AUTOTB_TVOUT_b21_67 "../tv/cdatafile/c.myproject.autotvout_b21_67.dat"
#define AUTOTB_TVIN_b21_68 "../tv/cdatafile/c.myproject.autotvin_b21_68.dat"
#define AUTOTB_TVOUT_b21_68 "../tv/cdatafile/c.myproject.autotvout_b21_68.dat"
#define AUTOTB_TVIN_b21_69 "../tv/cdatafile/c.myproject.autotvin_b21_69.dat"
#define AUTOTB_TVOUT_b21_69 "../tv/cdatafile/c.myproject.autotvout_b21_69.dat"
#define AUTOTB_TVIN_b21_70 "../tv/cdatafile/c.myproject.autotvin_b21_70.dat"
#define AUTOTB_TVOUT_b21_70 "../tv/cdatafile/c.myproject.autotvout_b21_70.dat"
#define AUTOTB_TVIN_b21_71 "../tv/cdatafile/c.myproject.autotvin_b21_71.dat"
#define AUTOTB_TVOUT_b21_71 "../tv/cdatafile/c.myproject.autotvout_b21_71.dat"
#define AUTOTB_TVIN_w24 "../tv/cdatafile/c.myproject.autotvin_w24.dat"
#define AUTOTB_TVOUT_w24 "../tv/cdatafile/c.myproject.autotvout_w24.dat"
#define AUTOTB_TVIN_b24_0 "../tv/cdatafile/c.myproject.autotvin_b24_0.dat"
#define AUTOTB_TVOUT_b24_0 "../tv/cdatafile/c.myproject.autotvout_b24_0.dat"
#define AUTOTB_TVIN_b24_1 "../tv/cdatafile/c.myproject.autotvin_b24_1.dat"
#define AUTOTB_TVOUT_b24_1 "../tv/cdatafile/c.myproject.autotvout_b24_1.dat"
#define AUTOTB_TVIN_b24_2 "../tv/cdatafile/c.myproject.autotvin_b24_2.dat"
#define AUTOTB_TVOUT_b24_2 "../tv/cdatafile/c.myproject.autotvout_b24_2.dat"
#define AUTOTB_TVIN_b24_3 "../tv/cdatafile/c.myproject.autotvin_b24_3.dat"
#define AUTOTB_TVOUT_b24_3 "../tv/cdatafile/c.myproject.autotvout_b24_3.dat"
#define AUTOTB_TVIN_b24_4 "../tv/cdatafile/c.myproject.autotvin_b24_4.dat"
#define AUTOTB_TVOUT_b24_4 "../tv/cdatafile/c.myproject.autotvout_b24_4.dat"
#define AUTOTB_TVIN_b24_5 "../tv/cdatafile/c.myproject.autotvin_b24_5.dat"
#define AUTOTB_TVOUT_b24_5 "../tv/cdatafile/c.myproject.autotvout_b24_5.dat"
#define AUTOTB_TVIN_b24_6 "../tv/cdatafile/c.myproject.autotvin_b24_6.dat"
#define AUTOTB_TVOUT_b24_6 "../tv/cdatafile/c.myproject.autotvout_b24_6.dat"
#define AUTOTB_TVIN_b24_7 "../tv/cdatafile/c.myproject.autotvin_b24_7.dat"
#define AUTOTB_TVOUT_b24_7 "../tv/cdatafile/c.myproject.autotvout_b24_7.dat"
#define AUTOTB_TVIN_b24_8 "../tv/cdatafile/c.myproject.autotvin_b24_8.dat"
#define AUTOTB_TVOUT_b24_8 "../tv/cdatafile/c.myproject.autotvout_b24_8.dat"
#define AUTOTB_TVIN_b24_9 "../tv/cdatafile/c.myproject.autotvin_b24_9.dat"
#define AUTOTB_TVOUT_b24_9 "../tv/cdatafile/c.myproject.autotvout_b24_9.dat"
#define AUTOTB_TVIN_b24_10 "../tv/cdatafile/c.myproject.autotvin_b24_10.dat"
#define AUTOTB_TVOUT_b24_10 "../tv/cdatafile/c.myproject.autotvout_b24_10.dat"
#define AUTOTB_TVIN_b24_11 "../tv/cdatafile/c.myproject.autotvin_b24_11.dat"
#define AUTOTB_TVOUT_b24_11 "../tv/cdatafile/c.myproject.autotvout_b24_11.dat"
#define AUTOTB_TVIN_b24_12 "../tv/cdatafile/c.myproject.autotvin_b24_12.dat"
#define AUTOTB_TVOUT_b24_12 "../tv/cdatafile/c.myproject.autotvout_b24_12.dat"
#define AUTOTB_TVIN_b24_13 "../tv/cdatafile/c.myproject.autotvin_b24_13.dat"
#define AUTOTB_TVOUT_b24_13 "../tv/cdatafile/c.myproject.autotvout_b24_13.dat"
#define AUTOTB_TVIN_b24_14 "../tv/cdatafile/c.myproject.autotvin_b24_14.dat"
#define AUTOTB_TVOUT_b24_14 "../tv/cdatafile/c.myproject.autotvout_b24_14.dat"
#define AUTOTB_TVIN_b24_15 "../tv/cdatafile/c.myproject.autotvin_b24_15.dat"
#define AUTOTB_TVOUT_b24_15 "../tv/cdatafile/c.myproject.autotvout_b24_15.dat"
#define AUTOTB_TVIN_b24_16 "../tv/cdatafile/c.myproject.autotvin_b24_16.dat"
#define AUTOTB_TVOUT_b24_16 "../tv/cdatafile/c.myproject.autotvout_b24_16.dat"
#define AUTOTB_TVIN_b24_17 "../tv/cdatafile/c.myproject.autotvin_b24_17.dat"
#define AUTOTB_TVOUT_b24_17 "../tv/cdatafile/c.myproject.autotvout_b24_17.dat"
#define AUTOTB_TVIN_b24_18 "../tv/cdatafile/c.myproject.autotvin_b24_18.dat"
#define AUTOTB_TVOUT_b24_18 "../tv/cdatafile/c.myproject.autotvout_b24_18.dat"
#define AUTOTB_TVIN_b24_19 "../tv/cdatafile/c.myproject.autotvin_b24_19.dat"
#define AUTOTB_TVOUT_b24_19 "../tv/cdatafile/c.myproject.autotvout_b24_19.dat"
#define AUTOTB_TVIN_b24_20 "../tv/cdatafile/c.myproject.autotvin_b24_20.dat"
#define AUTOTB_TVOUT_b24_20 "../tv/cdatafile/c.myproject.autotvout_b24_20.dat"
#define AUTOTB_TVIN_b24_21 "../tv/cdatafile/c.myproject.autotvin_b24_21.dat"
#define AUTOTB_TVOUT_b24_21 "../tv/cdatafile/c.myproject.autotvout_b24_21.dat"
#define AUTOTB_TVIN_b24_22 "../tv/cdatafile/c.myproject.autotvin_b24_22.dat"
#define AUTOTB_TVOUT_b24_22 "../tv/cdatafile/c.myproject.autotvout_b24_22.dat"
#define AUTOTB_TVIN_b24_23 "../tv/cdatafile/c.myproject.autotvin_b24_23.dat"
#define AUTOTB_TVOUT_b24_23 "../tv/cdatafile/c.myproject.autotvout_b24_23.dat"
#define AUTOTB_TVIN_b24_24 "../tv/cdatafile/c.myproject.autotvin_b24_24.dat"
#define AUTOTB_TVOUT_b24_24 "../tv/cdatafile/c.myproject.autotvout_b24_24.dat"
#define AUTOTB_TVIN_b24_25 "../tv/cdatafile/c.myproject.autotvin_b24_25.dat"
#define AUTOTB_TVOUT_b24_25 "../tv/cdatafile/c.myproject.autotvout_b24_25.dat"
#define AUTOTB_TVIN_b24_26 "../tv/cdatafile/c.myproject.autotvin_b24_26.dat"
#define AUTOTB_TVOUT_b24_26 "../tv/cdatafile/c.myproject.autotvout_b24_26.dat"
#define AUTOTB_TVIN_b24_27 "../tv/cdatafile/c.myproject.autotvin_b24_27.dat"
#define AUTOTB_TVOUT_b24_27 "../tv/cdatafile/c.myproject.autotvout_b24_27.dat"
#define AUTOTB_TVIN_b24_28 "../tv/cdatafile/c.myproject.autotvin_b24_28.dat"
#define AUTOTB_TVOUT_b24_28 "../tv/cdatafile/c.myproject.autotvout_b24_28.dat"
#define AUTOTB_TVIN_b24_29 "../tv/cdatafile/c.myproject.autotvin_b24_29.dat"
#define AUTOTB_TVOUT_b24_29 "../tv/cdatafile/c.myproject.autotvout_b24_29.dat"
#define AUTOTB_TVIN_b24_30 "../tv/cdatafile/c.myproject.autotvin_b24_30.dat"
#define AUTOTB_TVOUT_b24_30 "../tv/cdatafile/c.myproject.autotvout_b24_30.dat"
#define AUTOTB_TVIN_b24_31 "../tv/cdatafile/c.myproject.autotvin_b24_31.dat"
#define AUTOTB_TVOUT_b24_31 "../tv/cdatafile/c.myproject.autotvout_b24_31.dat"
#define AUTOTB_TVIN_b24_32 "../tv/cdatafile/c.myproject.autotvin_b24_32.dat"
#define AUTOTB_TVOUT_b24_32 "../tv/cdatafile/c.myproject.autotvout_b24_32.dat"
#define AUTOTB_TVIN_b24_33 "../tv/cdatafile/c.myproject.autotvin_b24_33.dat"
#define AUTOTB_TVOUT_b24_33 "../tv/cdatafile/c.myproject.autotvout_b24_33.dat"
#define AUTOTB_TVIN_b24_34 "../tv/cdatafile/c.myproject.autotvin_b24_34.dat"
#define AUTOTB_TVOUT_b24_34 "../tv/cdatafile/c.myproject.autotvout_b24_34.dat"
#define AUTOTB_TVIN_b24_35 "../tv/cdatafile/c.myproject.autotvin_b24_35.dat"
#define AUTOTB_TVOUT_b24_35 "../tv/cdatafile/c.myproject.autotvout_b24_35.dat"
#define AUTOTB_TVIN_b24_36 "../tv/cdatafile/c.myproject.autotvin_b24_36.dat"
#define AUTOTB_TVOUT_b24_36 "../tv/cdatafile/c.myproject.autotvout_b24_36.dat"
#define AUTOTB_TVIN_b24_37 "../tv/cdatafile/c.myproject.autotvin_b24_37.dat"
#define AUTOTB_TVOUT_b24_37 "../tv/cdatafile/c.myproject.autotvout_b24_37.dat"
#define AUTOTB_TVIN_b24_38 "../tv/cdatafile/c.myproject.autotvin_b24_38.dat"
#define AUTOTB_TVOUT_b24_38 "../tv/cdatafile/c.myproject.autotvout_b24_38.dat"
#define AUTOTB_TVIN_b24_39 "../tv/cdatafile/c.myproject.autotvin_b24_39.dat"
#define AUTOTB_TVOUT_b24_39 "../tv/cdatafile/c.myproject.autotvout_b24_39.dat"
#define AUTOTB_TVIN_b24_40 "../tv/cdatafile/c.myproject.autotvin_b24_40.dat"
#define AUTOTB_TVOUT_b24_40 "../tv/cdatafile/c.myproject.autotvout_b24_40.dat"
#define AUTOTB_TVIN_b24_41 "../tv/cdatafile/c.myproject.autotvin_b24_41.dat"
#define AUTOTB_TVOUT_b24_41 "../tv/cdatafile/c.myproject.autotvout_b24_41.dat"
#define AUTOTB_TVIN_b24_42 "../tv/cdatafile/c.myproject.autotvin_b24_42.dat"
#define AUTOTB_TVOUT_b24_42 "../tv/cdatafile/c.myproject.autotvout_b24_42.dat"
#define AUTOTB_TVIN_b24_43 "../tv/cdatafile/c.myproject.autotvin_b24_43.dat"
#define AUTOTB_TVOUT_b24_43 "../tv/cdatafile/c.myproject.autotvout_b24_43.dat"
#define AUTOTB_TVIN_b24_44 "../tv/cdatafile/c.myproject.autotvin_b24_44.dat"
#define AUTOTB_TVOUT_b24_44 "../tv/cdatafile/c.myproject.autotvout_b24_44.dat"
#define AUTOTB_TVIN_b24_45 "../tv/cdatafile/c.myproject.autotvin_b24_45.dat"
#define AUTOTB_TVOUT_b24_45 "../tv/cdatafile/c.myproject.autotvout_b24_45.dat"
#define AUTOTB_TVIN_b24_46 "../tv/cdatafile/c.myproject.autotvin_b24_46.dat"
#define AUTOTB_TVOUT_b24_46 "../tv/cdatafile/c.myproject.autotvout_b24_46.dat"
#define AUTOTB_TVIN_b24_47 "../tv/cdatafile/c.myproject.autotvin_b24_47.dat"
#define AUTOTB_TVOUT_b24_47 "../tv/cdatafile/c.myproject.autotvout_b24_47.dat"
#define AUTOTB_TVIN_b24_48 "../tv/cdatafile/c.myproject.autotvin_b24_48.dat"
#define AUTOTB_TVOUT_b24_48 "../tv/cdatafile/c.myproject.autotvout_b24_48.dat"
#define AUTOTB_TVIN_b24_49 "../tv/cdatafile/c.myproject.autotvin_b24_49.dat"
#define AUTOTB_TVOUT_b24_49 "../tv/cdatafile/c.myproject.autotvout_b24_49.dat"
#define AUTOTB_TVIN_b24_50 "../tv/cdatafile/c.myproject.autotvin_b24_50.dat"
#define AUTOTB_TVOUT_b24_50 "../tv/cdatafile/c.myproject.autotvout_b24_50.dat"
#define AUTOTB_TVIN_b24_51 "../tv/cdatafile/c.myproject.autotvin_b24_51.dat"
#define AUTOTB_TVOUT_b24_51 "../tv/cdatafile/c.myproject.autotvout_b24_51.dat"
#define AUTOTB_TVIN_b24_52 "../tv/cdatafile/c.myproject.autotvin_b24_52.dat"
#define AUTOTB_TVOUT_b24_52 "../tv/cdatafile/c.myproject.autotvout_b24_52.dat"
#define AUTOTB_TVIN_b24_53 "../tv/cdatafile/c.myproject.autotvin_b24_53.dat"
#define AUTOTB_TVOUT_b24_53 "../tv/cdatafile/c.myproject.autotvout_b24_53.dat"
#define AUTOTB_TVIN_b24_54 "../tv/cdatafile/c.myproject.autotvin_b24_54.dat"
#define AUTOTB_TVOUT_b24_54 "../tv/cdatafile/c.myproject.autotvout_b24_54.dat"
#define AUTOTB_TVIN_b24_55 "../tv/cdatafile/c.myproject.autotvin_b24_55.dat"
#define AUTOTB_TVOUT_b24_55 "../tv/cdatafile/c.myproject.autotvout_b24_55.dat"
#define AUTOTB_TVIN_b24_56 "../tv/cdatafile/c.myproject.autotvin_b24_56.dat"
#define AUTOTB_TVOUT_b24_56 "../tv/cdatafile/c.myproject.autotvout_b24_56.dat"
#define AUTOTB_TVIN_b24_57 "../tv/cdatafile/c.myproject.autotvin_b24_57.dat"
#define AUTOTB_TVOUT_b24_57 "../tv/cdatafile/c.myproject.autotvout_b24_57.dat"
#define AUTOTB_TVIN_w27 "../tv/cdatafile/c.myproject.autotvin_w27.dat"
#define AUTOTB_TVOUT_w27 "../tv/cdatafile/c.myproject.autotvout_w27.dat"
#define AUTOTB_TVIN_b27 "../tv/cdatafile/c.myproject.autotvin_b27.dat"
#define AUTOTB_TVOUT_b27 "../tv/cdatafile/c.myproject.autotvout_b27.dat"


// tvout file define:
#define AUTOTB_TVOUT_PC_layer29_out "../tv/rtldatafile/rtl.myproject.autotvout_layer29_out.dat"


namespace hls::sim
{
  template<size_t n>
  struct Byte {
    unsigned char a[n];

    Byte()
    {
      for (size_t i = 0; i < n; ++i) {
        a[i] = 0;
      }
    }

    template<typename T>
    Byte<n>& operator= (const T &val)
    {
      std::memcpy(a, &val, n);
      return *this;
    }
  };

  struct SimException : public std::exception {
    const std::string msg;
    const size_t line;
    SimException(const std::string &msg, const size_t line)
      : msg(msg), line(line)
    {
    }
  };

  void errExit(const size_t line, const std::string &msg)
  {
    std::string s;
    s += "ERROR";
//  s += '(';
//  s += __FILE__;
//  s += ":";
//  s += std::to_string(line);
//  s += ')';
    s += ": ";
    s += msg;
    s += "\n";
    fputs(s.c_str(), stderr);
    exit(1);
  }
}

namespace hls::sim
{
  size_t divide_ceil(size_t a, size_t b)
  {
    return (a + b - 1) / b;
  }

  const bool little_endian()
  {
    int a = 1;
    return *(char*)&a == 1;
  }

  inline void rev_endian(unsigned char *p, size_t nbytes)
  {
    std::reverse(p, p+nbytes);
  }

  const bool LE = little_endian();

  inline size_t least_nbyte(size_t width)
  {
    return (width+7)>>3;
  }

  std::string formatData(unsigned char *pos, size_t wbits)
  {
    size_t wbytes = least_nbyte(wbits);
    size_t i = LE ? wbytes-1 : 0;
    auto next = [&] () {
      auto c = pos[i];
      LE ? --i : ++i;
      return c;
    };
    std::ostringstream ss;
    ss << "0x";
    if (int t = (wbits & 0x7)) {
      if (t <= 4) {
        unsigned char mask = (1<<t)-1;
        ss << std::hex << std::setfill('0') << std::setw(1)
           << (int) (next() & mask);
        wbytes -= 1;
      }
    }
    for (size_t i = 0; i < wbytes; ++i) {
      ss << std::hex << std::setfill('0') << std::setw(2) << (int)next();
    }
    return ss.str();
  }

  char ord(char c)
  {
    if (c >= 'a' && c <= 'f') {
      return c-'a'+10;
    } else if (c >= 'A' && c <= 'F') {
      return c-'A'+10;
    } else if (c >= '0' && c <= '9') {
      return c-'0';
    } else {
      throw SimException("Not Hexdecimal Digit", __LINE__);
    }
  }

  void unformatData(const char *data, unsigned char *put, size_t pbytes = 0)
  {
    size_t nchars = strlen(data+2);
    size_t nbytes = (nchars+1)>>1;
    if (pbytes == 0) {
      pbytes = nbytes;
    } else if (pbytes > nbytes) {
      throw SimException("Wrong size specified", __LINE__);
    }
    put = LE ? put : put+pbytes-1;
    auto nextp = [&] () {
      return LE ? put++ : put--;
    };
    const char *c = data + (nchars + 2) - 1;
    auto next = [&] () {
      char res { *c == 'x' ? (char)0 : ord(*c) };
      --c;
      return res;
    };
    for (size_t i = 0; i < pbytes; ++i) {
      char l = next();
      char h = next();
      *nextp() = (h<<4)+l;
    }
  }

  char* strip(char *s)
  {
    while (isspace(*s)) {
      ++s;
    }
    for (char *p = s+strlen(s)-1; p >= s; --p) {
      if (isspace(*p)) {
        *p = 0;
      } else {
        return s;
      }
    }
    return s;
  }

  size_t sum(const std::vector<size_t> &v)
  {
    size_t res = 0;
    for (const auto &e : v) {
      res += e;
    }
    return res;
  }

  const char* bad = "Bad TV file";
  const char* err = "Error on TV file";

  const unsigned char bmark[] = {
    0x5a, 0x5a, 0xa5, 0xa5, 0x0f, 0x0f, 0xf0, 0xf0
  };

#ifdef USE_BINARY_TV_FILE
  class Input {
    FILE *fp;
    long pos;

    void read(unsigned char *buf, size_t size)
    {
      if (fread(buf, size, 1, fp) != 1) {
        throw SimException(bad, __LINE__);
      }
      if (LE) {
        rev_endian(buf, size);
      }
    }

  public:
    void advance(size_t nbytes)
    {
      if (fseek(fp, nbytes, SEEK_CUR) == -1) {
        throw SimException(bad, __LINE__);
      }
    }

    Input(const char *path) : fp(nullptr)
    {
      fp = fopen(path, "rb");
      if (fp == nullptr) {
        errExit(__LINE__, err);
      }
    }

    void begin()
    {
      advance(8);
      pos = ftell(fp);
    }

    void reset()
    {
      fseek(fp, pos, SEEK_SET);
    }

    void into(unsigned char *param, size_t wbytes, size_t asize, size_t nbytes)
    {
      size_t n = nbytes / asize;
      size_t r = nbytes % asize;
      for (size_t i = 0; i < n; ++i) {
        read(param, wbytes);
        param += asize;
      }
      if (r > 0) {
        advance(asize-r);
        read(param, r);
      }
    }

    ~Input()
    {
      unsigned char buf[8];
      size_t res = fread(buf, 8, 1, fp);
      fclose(fp);
      if (res != 1) {
        errExit(__LINE__, bad);
      }
      if (std::memcmp(buf, bmark, 8) != 0) {
        errExit(__LINE__, bad);
      }
    }
  };

  class Output {
    FILE *fp;

    void write(unsigned char *buf, size_t size)
    {
      if (LE) {
        rev_endian(buf, size);
      }
      if (fwrite(buf, size, 1, fp) != 1) {
        throw SimException(err, __LINE__);
      }
      if (LE) {
        rev_endian(buf, size);
      }
    }

  public:
    Output(const char *path) : fp(nullptr)
    {
      fp = fopen(path, "wb");
      if (fp == nullptr) {
        errExit(__LINE__, err);
      }
    }

    void begin(size_t total)
    {
      unsigned char buf[8] = {0};
      std::memcpy(buf, &total, sizeof(buf));
      write(buf, sizeof(buf));
    }

    void from(unsigned char *param, size_t wbytes, size_t asize, size_t nbytes, size_t skip)
    {
      param -= asize*skip;
      size_t n = divide_ceil(nbytes, asize);
      for (size_t i = 0; i < n; ++i) {
        write(param, wbytes);
        param += asize;
      }
    }

    ~Output()
    {
      size_t res = fwrite(bmark, 8, 1, fp);
      fclose(fp);
      if (res != 1) {
        errExit(__LINE__, err);
      }
    }
  };
#endif

  class Reader {
    FILE *fp;
    long pos;
    int size;
    char *s;

    void readline()
    {
      s = fgets(s, size, fp);
      if (s == nullptr) {
        throw SimException(bad, __LINE__);
      }
    }

  public:
    Reader(const char *path) : fp(nullptr), size(1<<12), s(new char[size])
    {
      try {
        fp = fopen(path, "r");
        if (fp == nullptr) {
          throw SimException(err, __LINE__);
        } else {
          readline();
          static const char mark[] = "[[[runtime]]]\n";
          if (strcmp(s, mark) != 0) {
            throw SimException(bad, __LINE__);
          }
        }
      } catch (const hls::sim::SimException &e) {
        errExit(e.line, e.msg);
      }
    }

    ~Reader()
    {
      fclose(fp);
      delete[] s;
    }

    void begin()
    {
      readline();
      static const char mark[] = "[[transaction]]";
      if (strncmp(s, mark, strlen(mark)) != 0) {
        throw SimException(bad, __LINE__);
      }
      pos = ftell(fp);
    }

    void reset()
    {
      fseek(fp, pos, SEEK_SET);
    }

    void skip(size_t n)
    {
      for (size_t i = 0; i < n; ++i) {
        readline();
      }
    }

    char* next()
    {
      long pos = ftell(fp);
      readline();
      if (*s == '[') {
        fseek(fp, pos, SEEK_SET);
        return nullptr;
      }
      return strip(s);
    }

    void end()
    {
      do {
        readline();
      } while (strcmp(s, "[[/transaction]]\n") != 0);
    }
  };

  class Writer {
    FILE *fp;

    void write(const char *s)
    {
      if (fputs(s, fp) == EOF) {
        throw SimException(err, __LINE__);
      }
    }

  public:
    Writer(const char *path) : fp(nullptr)
    {
      try {
        fp = fopen(path, "w");
        if (fp == nullptr) {
          throw SimException(err, __LINE__);
        } else {
          static const char mark[] = "[[[runtime]]]\n";
          write(mark);
        }
      } catch (const hls::sim::SimException &e) {
        errExit(e.line, e.msg);
      }
    }

    virtual ~Writer()
    {
      try {
        static const char mark[] = "[[[/runtime]]]\n";
        write(mark);
      } catch (const hls::sim::SimException &e) {
        errExit(e.line, e.msg);
      }
      fclose(fp);
    }

    void begin(size_t AESL_transaction)
    {
      static const char mark[] = "[[transaction]]           ";
      write(mark);
      auto buf = std::to_string(AESL_transaction);
      buf.push_back('\n');
      buf.push_back('\0');
      write(buf.data());
    }

    void next(const char *s)
    {
      write(s);
      write("\n");
    }

    void end()
    {
      static const char mark[] = "[[/transaction]]\n";
      write(mark);
    }
  };

  bool RTLOutputCheckAndReplacement(char *data)
  {
    bool changed = false;
    for (size_t i = 2; i < strlen(data); ++i) {
      if (data[i] == 'X' || data[i] == 'x') {
        data[i] = '0';
        changed = true;
      }
    }
    return changed;
  }

  void warnOnX()
  {
    static const char msg[] =
      "WARNING: [SIM 212-201] RTL produces unknown value "
      "'x' or 'X' on some port, possible cause: "
      "There are uninitialized variables in the design.\n";
    fprintf(stderr, msg);
  }

#ifndef POST_CHECK
  class RefTCL {
    FILE *fp;
    std::ostringstream ss;

    void formatDepth()
    {
      ss << "set depth_list {\n";
      for (auto &p : depth) {
        ss << "  {" << p.first << " " << p.second << "}\n";
      }
      if (nameHBM != "") {
        ss << "  {" << nameHBM << " " << depthHBM << "}\n";
      }
      ss << "}\n";
    }

    void formatTransNum()
    {
      ss << "set trans_num " << AESL_transaction << "\n";
    }

    void formatHBM()
    {
      ss << "set HBM_ArgDict {\n"
         << "  Name " << nameHBM << "\n"
         << "  Port " << portHBM << "\n"
         << "  BitWidth " << widthHBM << "\n"
         << "}\n";
    }

    void close()
    {
      formatDepth();
      formatTransNum();
      if (nameHBM != "") {
        formatHBM();
      }
      std::string &&s { ss.str() };
      size_t res = fwrite(s.data(), s.size(), 1, fp);
      fclose(fp);
      if (res != 1) {
        errExit(__LINE__, err);
      }
    }

  public:
    std::map<const std::string, size_t> depth;
    std::string nameHBM;
    size_t depthHBM;
    std::string portHBM;
    unsigned widthHBM;
    size_t AESL_transaction;
    std::mutex mut;

    RefTCL(const char *path)
    {
      fp = fopen(path, "w");
      if (fp == nullptr) {
        errExit(__LINE__, err);
      }
    }

    void set(const char* name, size_t dep)
    {
      std::lock_guard<std::mutex> guard(mut);
      if (depth[name] < dep) {
        depth[name] = dep;
      }
    }

    ~RefTCL()
    {
      close();
    }
  };

#endif

  struct Register {
    const char* name;
    unsigned width;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* owriter;
    Writer* iwriter;
#endif
    void* param;

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      if (strcmp(name, "return") == 0) {
        tcl.set("ap_return", 1);
      } else {
        tcl.set(name, 1);
      }
    }
#endif
    ~Register()
    {
#ifdef POST_CHECK
      delete reader;
#else
      delete owriter;
      delete iwriter;
#endif
    }
  };

  template<typename E>
  struct DirectIO {
    unsigned width;
    const char* name;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* writer;
    Writer* swriter;
    Writer* gwriter;
#endif
    hls::directio<E>* param;
    std::vector<E> buf;
    size_t initSize;
    size_t depth;
    bool hasWrite;

    void markSize()
    {
      initSize = param->size();
    }

    void buffer()
    {
      buf.clear();
      while (param->valid()) {
        buf.push_back(param->read());
      }
      for (auto &e : buf) {
        param->write(e);
      }
    }

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      tcl.set(name, depth);
    }
#endif

    ~DirectIO()
    {
#ifdef POST_CHECK
      delete reader;
#else
      delete writer;
      delete swriter;
      delete gwriter;
#endif
    }
  };

  template<typename Reader, typename Writer>
  struct Memory {
    unsigned width;
    unsigned asize;
    bool hbm;
    std::vector<const char*> name;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* owriter;
    Writer* iwriter;
#endif
    std::vector<void*> param;
    std::vector<size_t> nbytes;
    std::vector<size_t> offset;
    std::vector<bool> hasWrite;

    size_t depth()
    {
      size_t depth = 0;
      for (size_t n : nbytes) {
        depth += divide_ceil(n, asize);
      }
      return depth;
    }

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      if (hbm) {
        tcl.nameHBM.append(name[0]);
        tcl.portHBM.append("{").append(name[0]);
        for (size_t i = 1; i < name.size(); ++i) {
          tcl.nameHBM.append("_").append(name[i]);
          tcl.portHBM.append(" ").append(name[i]);
        }
        tcl.nameHBM.append("_HBM");
        tcl.portHBM.append("}");
        tcl.widthHBM = width;
        tcl.depthHBM = divide_ceil(nbytes[0], asize);
      } else {
        tcl.set(name[0], depth());
      }
    }
#endif

    ~Memory()
    {
#ifdef POST_CHECK
      delete reader;
#else
      delete owriter;
      delete iwriter;
#endif
    }
  };

  struct A2Stream {
    unsigned width;
    unsigned asize;
    const char* name;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* owriter;
    Writer* iwriter;
#endif
    void* param;
    size_t nbytes;
    bool hasWrite;

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      tcl.set(name, divide_ceil(nbytes, asize));
    }
#endif

    ~A2Stream()
    {
#ifdef POST_CHECK
      delete reader;
#else
      delete owriter;
      delete iwriter;
#endif
    }
  };

  template<typename E>
  struct Stream {
    unsigned width;
    const char* name;
#ifdef POST_CHECK
    Reader* reader;
#else
    Writer* writer;
    Writer* swriter;
    Writer* gwriter;
#endif
    hls::stream<E>* param;
    std::vector<E> buf;
    size_t initSize;
    size_t depth;
    bool hasWrite;

    void markSize()
    {
      initSize = param->size();
    }

    void buffer()
    {
      buf.clear();
      while (!param->empty()) {
        buf.push_back(param->read());
      }
      for (auto &e : buf) {
        param->write(e);
      }
    }

#ifndef POST_CHECK
    void doTCL(RefTCL &tcl)
    {
      tcl.set(name, depth);
    }
#endif

    ~Stream()
    {
#ifdef POST_CHECK
      delete reader;
#else
      delete writer;
      delete swriter;
      delete gwriter;
#endif
    }
  };

#ifdef POST_CHECK
  void check(Register &port)
  {
    port.reader->begin();
    bool foundX = false;
    if (char *s = port.reader->next()) {
      foundX |= RTLOutputCheckAndReplacement(s);
      unformatData(s, (unsigned char*)port.param);
    }
    port.reader->end();
    if (foundX) {
      warnOnX();
    }
  }

  template<typename E>
  void check(DirectIO<E> &port)
  {
    if (port.hasWrite) {
      port.reader->begin();
      bool foundX = false;
      E *p = new E;
      while (char *s = port.reader->next()) {
        foundX |= RTLOutputCheckAndReplacement(s);
        unformatData(s, (unsigned char*)p);
        port.param->write(*p);
      }
      delete p;
      port.reader->end();
      if (foundX) {
        warnOnX();
      }
    } else {
      port.reader->begin();
      size_t n = 0;
      if (char *s = port.reader->next()) {
        std::istringstream ss(s);
        ss >> n;
      } else {
        throw SimException(bad, __LINE__);
      }
      port.reader->end();
      for (size_t j = 0; j < n; ++j) {
        port.param->read();
      }
    }
  }

#ifdef USE_BINARY_TV_FILE
  void checkHBM(Memory<Input, Output> &port)
  {
    port.reader->begin();
    size_t wbytes = least_nbyte(port.width);
    for (size_t i = 0; i < port.param.size(); ++i) {
      if (port.hasWrite[i]) {
        port.reader->reset();
        size_t skip = wbytes * port.offset[i];
        port.reader->advance(skip);
        port.reader->into((unsigned char*)port.param[i], wbytes,
                           port.asize, port.nbytes[i] - skip);
      }
    }
  }

  void check(Memory<Input, Output> &port)
  {
    if (port.hbm) {
      return checkHBM(port);
    } else {
      port.reader->begin();
      size_t wbytes = least_nbyte(port.width);
      for (size_t i = 0; i < port.param.size(); ++i) {
        if (port.hasWrite[i]) {
          port.reader->into((unsigned char*)port.param[i], wbytes,
                             port.asize, port.nbytes[i]);
        } else {
          size_t n = divide_ceil(port.nbytes[i], port.asize);
          port.reader->advance(port.asize*n);
        }
      }
    }
  }
#endif
  void transfer(Reader *reader, size_t nbytes, unsigned char *put, bool &foundX)
  {
    if (char *s = reader->next()) {
      foundX |= RTLOutputCheckAndReplacement(s);
      unformatData(s, put, nbytes);
    } else {
      throw SimException("No more data", __LINE__);
    }
  }

  void checkHBM(Memory<Reader, Writer> &port)
  {
    port.reader->begin();
    bool foundX = false;
    size_t wbytes = least_nbyte(port.width);
    for (size_t i = 0, last = port.param.size()-1; i <= last; ++i) {
      if (port.hasWrite[i]) {
        port.reader->skip(port.offset[i]);
        size_t n = port.nbytes[i] / port.asize - port.offset[i];
        unsigned char *put = (unsigned char*)port.param[i];
        for (size_t j = 0; j < n; ++j) {
          transfer(port.reader, wbytes, put, foundX);
          put += port.asize;
        }
        if (i < last) {
          port.reader->reset();
        }
      }
    }
    port.reader->end();
    if (foundX) {
      warnOnX();
    }
  }

  void check(Memory<Reader, Writer> &port)
  {
    if (port.hbm) {
      return checkHBM(port);
    } else {
      port.reader->begin();
      bool foundX = false;
      size_t wbytes = least_nbyte(port.width);
      for (size_t i = 0; i < port.param.size(); ++i) {
        if (port.hasWrite[i]) {
          size_t n = port.nbytes[i] / port.asize;
          size_t r = port.nbytes[i] % port.asize;
          unsigned char *put = (unsigned char*)port.param[i];
          for (size_t j = 0; j < n; ++j) {
            transfer(port.reader, wbytes, put, foundX);
            put += port.asize;
          }
          if (r > 0) {
            transfer(port.reader, r, put, foundX);
          }
        } else {
          size_t n = divide_ceil(port.nbytes[i], port.asize);
          port.reader->skip(n);
        }
      }
      port.reader->end();
      if (foundX) {
        warnOnX();
      }
    }
  }

  void check(A2Stream &port)
  {
    port.reader->begin();
    bool foundX = false;
    if (port.hasWrite) {
      size_t wbytes = least_nbyte(port.width);
      size_t n = port.nbytes / port.asize;
      size_t r = port.nbytes % port.asize;
      unsigned char *put = (unsigned char*)port.param;
      for (size_t j = 0; j < n; ++j) {
        if (char *s = port.reader->next()) {
          foundX |= RTLOutputCheckAndReplacement(s);
          unformatData(s, put, wbytes);
        }
        put += port.asize;
      }
      if (r > 0) {
        if (char *s = port.reader->next()) {
          foundX |= RTLOutputCheckAndReplacement(s);
          unformatData(s, put, r);
        }
      }
    }
    port.reader->end();
    if (foundX) {
      warnOnX();
    }
  }

  template<typename E>
  void check(Stream<E> &port)
  {
    if (port.hasWrite) {
      port.reader->begin();
      bool foundX = false;
      E *p = new E;
      while (char *s = port.reader->next()) {
        foundX |= RTLOutputCheckAndReplacement(s);
        unformatData(s, (unsigned char*)p);
        port.param->write(*p);
      }
      delete p;
      port.reader->end();
      if (foundX) {
        warnOnX();
      }
    } else {
      port.reader->begin();
      size_t n = 0;
      if (char *s = port.reader->next()) {
        std::istringstream ss(s);
        ss >> n;
      } else {
        throw SimException(bad, __LINE__);
      }
      port.reader->end();
      for (size_t j = 0; j < n; ++j) {
        port.param->read();
      }
    }
  }
#else
  void dump(Register &port, Writer *writer, size_t AESL_transaction)
  {
    writer->begin(AESL_transaction);
    std::string &&s { formatData((unsigned char*)port.param, port.width) };
    writer->next(s.data());
    writer->end();
  }

  template<typename E>
  void dump(DirectIO<E> &port, size_t AESL_transaction)
  {
    if (port.hasWrite) {
      port.writer->begin(AESL_transaction);
      port.depth = port.param->size()-port.initSize;
      for (size_t j = 0; j < port.depth; ++j) {
        std::string &&s {
          formatData((unsigned char*)&port.buf[port.initSize+j], port.width)
        };
        port.writer->next(s.c_str());
      }
      port.writer->end();

      port.swriter->begin(AESL_transaction);
      port.swriter->next(std::to_string(port.depth).c_str());
      port.swriter->end();
    } else {
      port.writer->begin(AESL_transaction);
      port.depth = port.initSize-port.param->size();
      for (size_t j = 0; j < port.depth; ++j) {
        std::string &&s {
          formatData((unsigned char*)&port.buf[j], port.width)
        };
        port.writer->next(s.c_str());
      }
      port.writer->end();

      port.swriter->begin(AESL_transaction);
      port.swriter->next(std::to_string(port.depth).c_str());
      port.swriter->end();

      port.gwriter->begin(AESL_transaction);
      size_t n = (port.depth ? port.initSize : port.depth);
      size_t d = port.depth;
      do {
        port.gwriter->next(std::to_string(n--).c_str());
      } while (d--);
      port.gwriter->end();
    }
  }

  void error_on_depth_unspecified(const char *portName)
  {
    std::string msg {"A depth specification is required for interface port "};
    msg.append("'");
    msg.append(portName);
    msg.append("'");
    msg.append(" for cosimulation.");
    throw SimException(msg, __LINE__);
  }

#ifdef USE_BINARY_TV_FILE
  void dump(Memory<Input, Output> &port, Output *writer, size_t AESL_transaction)
  {
    writer->begin(port.depth());
    size_t wbytes = least_nbyte(port.width);
    for (size_t i = 0; i < port.param.size(); ++i) {
      if (port.nbytes[i] == 0) {
        error_on_depth_unspecified(port.hbm ? port.name[i] : port.name[0]);
      }
      writer->from((unsigned char*)port.param[i], wbytes, port.asize,
                   port.nbytes[i], 0);
    }
  }

#endif
  void dump(Memory<Reader, Writer> &port, Writer *writer, size_t AESL_transaction)
  {
    writer->begin(AESL_transaction);
    for (size_t i = 0; i < port.param.size(); ++i) {
      if (port.nbytes[i] == 0) {
        error_on_depth_unspecified(port.hbm ? port.name[i] : port.name[0]);
      }
      size_t n = divide_ceil(port.nbytes[i], port.asize);
      unsigned char *put = (unsigned char*)port.param[i];
      for (size_t j = 0; j < n; ++j) {
        std::string &&s {
          formatData(put, port.width)
        };
        writer->next(s.data());
        put += port.asize;
      }
      if (port.hbm) {
        break;
      }
    }
    writer->end();
  }

  void dump(A2Stream &port, Writer *writer, size_t AESL_transaction)
  {
    writer->begin(AESL_transaction);
    if (port.nbytes == 0) {
      error_on_depth_unspecified(port.name);
    }
    size_t n = divide_ceil(port.nbytes, port.asize);
    unsigned char *put = (unsigned char*)port.param;
    for (size_t j = 0; j < n; ++j) {
      std::string &&s { formatData(put, port.width) };
      writer->next(s.data());
      put += port.asize;
    }
    writer->end();
  }

  template<typename E>
  void dump(Stream<E> &port, size_t AESL_transaction)
  {
    if (port.hasWrite) {
      port.writer->begin(AESL_transaction);
      port.depth = port.param->size()-port.initSize;
      for (size_t j = 0; j < port.depth; ++j) {
        std::string &&s {
          formatData((unsigned char*)&port.buf[port.initSize+j], port.width)
        };
        port.writer->next(s.c_str());
      }
      port.writer->end();

      port.swriter->begin(AESL_transaction);
      port.swriter->next(std::to_string(port.depth).c_str());
      port.swriter->end();
    } else {
      port.writer->begin(AESL_transaction);
      port.depth = port.initSize-port.param->size();
      for (size_t j = 0; j < port.depth; ++j) {
        std::string &&s {
          formatData((unsigned char*)&port.buf[j], port.width)
        };
        port.writer->next(s.c_str());
      }
      port.writer->end();

      port.swriter->begin(AESL_transaction);
      port.swriter->next(std::to_string(port.depth).c_str());
      port.swriter->end();

      port.gwriter->begin(AESL_transaction);
      size_t n = (port.depth ? port.initSize : port.depth);
      size_t d = port.depth;
      do {
        port.gwriter->next(std::to_string(n--).c_str());
      } while (d--);
      port.gwriter->end();
    }
  }
#endif
}



extern "C"
void myproject_hw_stub_wrapper(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);

extern "C"
void apatb_myproject_hw(void* __xlx_apatb_param_cluster, void* __xlx_apatb_param_nModule, void* __xlx_apatb_param_x_local, void* __xlx_apatb_param_y_local, void* __xlx_apatb_param_layer29_out, void* __xlx_apatb_param_w9, void* __xlx_apatb_param_b9_0, void* __xlx_apatb_param_b9_1, void* __xlx_apatb_param_w16, void* __xlx_apatb_param_b16_0, void* __xlx_apatb_param_b16_1, void* __xlx_apatb_param_b16_2, void* __xlx_apatb_param_b16_3, void* __xlx_apatb_param_b16_4, void* __xlx_apatb_param_b16_5, void* __xlx_apatb_param_b16_6, void* __xlx_apatb_param_b16_7, void* __xlx_apatb_param_b16_8, void* __xlx_apatb_param_b16_9, void* __xlx_apatb_param_b16_10, void* __xlx_apatb_param_b16_11, void* __xlx_apatb_param_b16_12, void* __xlx_apatb_param_b16_13, void* __xlx_apatb_param_b16_14, void* __xlx_apatb_param_b16_15, void* __xlx_apatb_param_w21, void* __xlx_apatb_param_b21_0, void* __xlx_apatb_param_b21_1, void* __xlx_apatb_param_b21_2, void* __xlx_apatb_param_b21_3, void* __xlx_apatb_param_b21_4, void* __xlx_apatb_param_b21_5, void* __xlx_apatb_param_b21_6, void* __xlx_apatb_param_b21_7, void* __xlx_apatb_param_b21_8, void* __xlx_apatb_param_b21_9, void* __xlx_apatb_param_b21_10, void* __xlx_apatb_param_b21_11, void* __xlx_apatb_param_b21_12, void* __xlx_apatb_param_b21_13, void* __xlx_apatb_param_b21_14, void* __xlx_apatb_param_b21_15, void* __xlx_apatb_param_b21_16, void* __xlx_apatb_param_b21_17, void* __xlx_apatb_param_b21_18, void* __xlx_apatb_param_b21_19, void* __xlx_apatb_param_b21_20, void* __xlx_apatb_param_b21_21, void* __xlx_apatb_param_b21_22, void* __xlx_apatb_param_b21_23, void* __xlx_apatb_param_b21_24, void* __xlx_apatb_param_b21_25, void* __xlx_apatb_param_b21_26, void* __xlx_apatb_param_b21_27, void* __xlx_apatb_param_b21_28, void* __xlx_apatb_param_b21_29, void* __xlx_apatb_param_b21_30, void* __xlx_apatb_param_b21_31, void* __xlx_apatb_param_b21_32, void* __xlx_apatb_param_b21_33, void* __xlx_apatb_param_b21_34, void* __xlx_apatb_param_b21_35, void* __xlx_apatb_param_b21_36, void* __xlx_apatb_param_b21_37, void* __xlx_apatb_param_b21_38, void* __xlx_apatb_param_b21_39, void* __xlx_apatb_param_b21_40, void* __xlx_apatb_param_b21_41, void* __xlx_apatb_param_b21_42, void* __xlx_apatb_param_b21_43, void* __xlx_apatb_param_b21_44, void* __xlx_apatb_param_b21_45, void* __xlx_apatb_param_b21_46, void* __xlx_apatb_param_b21_47, void* __xlx_apatb_param_b21_48, void* __xlx_apatb_param_b21_49, void* __xlx_apatb_param_b21_50, void* __xlx_apatb_param_b21_51, void* __xlx_apatb_param_b21_52, void* __xlx_apatb_param_b21_53, void* __xlx_apatb_param_b21_54, void* __xlx_apatb_param_b21_55, void* __xlx_apatb_param_b21_56, void* __xlx_apatb_param_b21_57, void* __xlx_apatb_param_b21_58, void* __xlx_apatb_param_b21_59, void* __xlx_apatb_param_b21_60, void* __xlx_apatb_param_b21_61, void* __xlx_apatb_param_b21_62, void* __xlx_apatb_param_b21_63, void* __xlx_apatb_param_b21_64, void* __xlx_apatb_param_b21_65, void* __xlx_apatb_param_b21_66, void* __xlx_apatb_param_b21_67, void* __xlx_apatb_param_b21_68, void* __xlx_apatb_param_b21_69, void* __xlx_apatb_param_b21_70, void* __xlx_apatb_param_b21_71, void* __xlx_apatb_param_w24, void* __xlx_apatb_param_b24_0, void* __xlx_apatb_param_b24_1, void* __xlx_apatb_param_b24_2, void* __xlx_apatb_param_b24_3, void* __xlx_apatb_param_b24_4, void* __xlx_apatb_param_b24_5, void* __xlx_apatb_param_b24_6, void* __xlx_apatb_param_b24_7, void* __xlx_apatb_param_b24_8, void* __xlx_apatb_param_b24_9, void* __xlx_apatb_param_b24_10, void* __xlx_apatb_param_b24_11, void* __xlx_apatb_param_b24_12, void* __xlx_apatb_param_b24_13, void* __xlx_apatb_param_b24_14, void* __xlx_apatb_param_b24_15, void* __xlx_apatb_param_b24_16, void* __xlx_apatb_param_b24_17, void* __xlx_apatb_param_b24_18, void* __xlx_apatb_param_b24_19, void* __xlx_apatb_param_b24_20, void* __xlx_apatb_param_b24_21, void* __xlx_apatb_param_b24_22, void* __xlx_apatb_param_b24_23, void* __xlx_apatb_param_b24_24, void* __xlx_apatb_param_b24_25, void* __xlx_apatb_param_b24_26, void* __xlx_apatb_param_b24_27, void* __xlx_apatb_param_b24_28, void* __xlx_apatb_param_b24_29, void* __xlx_apatb_param_b24_30, void* __xlx_apatb_param_b24_31, void* __xlx_apatb_param_b24_32, void* __xlx_apatb_param_b24_33, void* __xlx_apatb_param_b24_34, void* __xlx_apatb_param_b24_35, void* __xlx_apatb_param_b24_36, void* __xlx_apatb_param_b24_37, void* __xlx_apatb_param_b24_38, void* __xlx_apatb_param_b24_39, void* __xlx_apatb_param_b24_40, void* __xlx_apatb_param_b24_41, void* __xlx_apatb_param_b24_42, void* __xlx_apatb_param_b24_43, void* __xlx_apatb_param_b24_44, void* __xlx_apatb_param_b24_45, void* __xlx_apatb_param_b24_46, void* __xlx_apatb_param_b24_47, void* __xlx_apatb_param_b24_48, void* __xlx_apatb_param_b24_49, void* __xlx_apatb_param_b24_50, void* __xlx_apatb_param_b24_51, void* __xlx_apatb_param_b24_52, void* __xlx_apatb_param_b24_53, void* __xlx_apatb_param_b24_54, void* __xlx_apatb_param_b24_55, void* __xlx_apatb_param_b24_56, void* __xlx_apatb_param_b24_57, void* __xlx_apatb_param_w27, void* __xlx_apatb_param_b27)
{
  static hls::sim::Register port0 {
    .name = "cluster",
    .width = 4368,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_cluster),
#endif
  };
  port0.param = __xlx_apatb_param_cluster;

  static hls::sim::Register port1 {
    .name = "nModule",
    .width = 16,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_nModule),
#endif
  };
  port1.param = __xlx_apatb_param_nModule;

  static hls::sim::Register port2 {
    .name = "x_local",
    .width = 16,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_x_local),
#endif
  };
  port2.param = __xlx_apatb_param_x_local;

  static hls::sim::Register port3 {
    .name = "y_local",
    .width = 16,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_y_local),
#endif
  };
  port3.param = __xlx_apatb_param_y_local;

  static hls::sim::Register port4 {
    .name = "layer29_out",
    .width = 8,
#ifdef POST_CHECK
    .reader = new hls::sim::Reader(AUTOTB_TVOUT_PC_layer29_out),
#else
    .owriter = new hls::sim::Writer(AUTOTB_TVOUT_layer29_out),
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_layer29_out),
#endif
  };
  port4.param = __xlx_apatb_param_layer29_out;

  static hls::sim::Register port5 {
    .name = "b9_0",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b9_0),
#endif
  };
  port5.param = __xlx_apatb_param_b9_0;

  static hls::sim::Register port6 {
    .name = "b9_1",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b9_1),
#endif
  };
  port6.param = __xlx_apatb_param_b9_1;

  static hls::sim::Register port7 {
    .name = "b16_0",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_0),
#endif
  };
  port7.param = __xlx_apatb_param_b16_0;

  static hls::sim::Register port8 {
    .name = "b16_1",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_1),
#endif
  };
  port8.param = __xlx_apatb_param_b16_1;

  static hls::sim::Register port9 {
    .name = "b16_2",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_2),
#endif
  };
  port9.param = __xlx_apatb_param_b16_2;

  static hls::sim::Register port10 {
    .name = "b16_3",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_3),
#endif
  };
  port10.param = __xlx_apatb_param_b16_3;

  static hls::sim::Register port11 {
    .name = "b16_4",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_4),
#endif
  };
  port11.param = __xlx_apatb_param_b16_4;

  static hls::sim::Register port12 {
    .name = "b16_5",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_5),
#endif
  };
  port12.param = __xlx_apatb_param_b16_5;

  static hls::sim::Register port13 {
    .name = "b16_6",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_6),
#endif
  };
  port13.param = __xlx_apatb_param_b16_6;

  static hls::sim::Register port14 {
    .name = "b16_7",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_7),
#endif
  };
  port14.param = __xlx_apatb_param_b16_7;

  static hls::sim::Register port15 {
    .name = "b16_8",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_8),
#endif
  };
  port15.param = __xlx_apatb_param_b16_8;

  static hls::sim::Register port16 {
    .name = "b16_9",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_9),
#endif
  };
  port16.param = __xlx_apatb_param_b16_9;

  static hls::sim::Register port17 {
    .name = "b16_10",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_10),
#endif
  };
  port17.param = __xlx_apatb_param_b16_10;

  static hls::sim::Register port18 {
    .name = "b16_11",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_11),
#endif
  };
  port18.param = __xlx_apatb_param_b16_11;

  static hls::sim::Register port19 {
    .name = "b16_12",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_12),
#endif
  };
  port19.param = __xlx_apatb_param_b16_12;

  static hls::sim::Register port20 {
    .name = "b16_13",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_13),
#endif
  };
  port20.param = __xlx_apatb_param_b16_13;

  static hls::sim::Register port21 {
    .name = "b16_14",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_14),
#endif
  };
  port21.param = __xlx_apatb_param_b16_14;

  static hls::sim::Register port22 {
    .name = "b16_15",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b16_15),
#endif
  };
  port22.param = __xlx_apatb_param_b16_15;

  static hls::sim::Register port23 {
    .name = "b21_0",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_0),
#endif
  };
  port23.param = __xlx_apatb_param_b21_0;

  static hls::sim::Register port24 {
    .name = "b21_1",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_1),
#endif
  };
  port24.param = __xlx_apatb_param_b21_1;

  static hls::sim::Register port25 {
    .name = "b21_2",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_2),
#endif
  };
  port25.param = __xlx_apatb_param_b21_2;

  static hls::sim::Register port26 {
    .name = "b21_3",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_3),
#endif
  };
  port26.param = __xlx_apatb_param_b21_3;

  static hls::sim::Register port27 {
    .name = "b21_4",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_4),
#endif
  };
  port27.param = __xlx_apatb_param_b21_4;

  static hls::sim::Register port28 {
    .name = "b21_5",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_5),
#endif
  };
  port28.param = __xlx_apatb_param_b21_5;

  static hls::sim::Register port29 {
    .name = "b21_6",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_6),
#endif
  };
  port29.param = __xlx_apatb_param_b21_6;

  static hls::sim::Register port30 {
    .name = "b21_7",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_7),
#endif
  };
  port30.param = __xlx_apatb_param_b21_7;

  static hls::sim::Register port31 {
    .name = "b21_8",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_8),
#endif
  };
  port31.param = __xlx_apatb_param_b21_8;

  static hls::sim::Register port32 {
    .name = "b21_9",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_9),
#endif
  };
  port32.param = __xlx_apatb_param_b21_9;

  static hls::sim::Register port33 {
    .name = "b21_10",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_10),
#endif
  };
  port33.param = __xlx_apatb_param_b21_10;

  static hls::sim::Register port34 {
    .name = "b21_11",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_11),
#endif
  };
  port34.param = __xlx_apatb_param_b21_11;

  static hls::sim::Register port35 {
    .name = "b21_12",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_12),
#endif
  };
  port35.param = __xlx_apatb_param_b21_12;

  static hls::sim::Register port36 {
    .name = "b21_13",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_13),
#endif
  };
  port36.param = __xlx_apatb_param_b21_13;

  static hls::sim::Register port37 {
    .name = "b21_14",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_14),
#endif
  };
  port37.param = __xlx_apatb_param_b21_14;

  static hls::sim::Register port38 {
    .name = "b21_15",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_15),
#endif
  };
  port38.param = __xlx_apatb_param_b21_15;

  static hls::sim::Register port39 {
    .name = "b21_16",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_16),
#endif
  };
  port39.param = __xlx_apatb_param_b21_16;

  static hls::sim::Register port40 {
    .name = "b21_17",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_17),
#endif
  };
  port40.param = __xlx_apatb_param_b21_17;

  static hls::sim::Register port41 {
    .name = "b21_18",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_18),
#endif
  };
  port41.param = __xlx_apatb_param_b21_18;

  static hls::sim::Register port42 {
    .name = "b21_19",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_19),
#endif
  };
  port42.param = __xlx_apatb_param_b21_19;

  static hls::sim::Register port43 {
    .name = "b21_20",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_20),
#endif
  };
  port43.param = __xlx_apatb_param_b21_20;

  static hls::sim::Register port44 {
    .name = "b21_21",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_21),
#endif
  };
  port44.param = __xlx_apatb_param_b21_21;

  static hls::sim::Register port45 {
    .name = "b21_22",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_22),
#endif
  };
  port45.param = __xlx_apatb_param_b21_22;

  static hls::sim::Register port46 {
    .name = "b21_23",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_23),
#endif
  };
  port46.param = __xlx_apatb_param_b21_23;

  static hls::sim::Register port47 {
    .name = "b21_24",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_24),
#endif
  };
  port47.param = __xlx_apatb_param_b21_24;

  static hls::sim::Register port48 {
    .name = "b21_25",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_25),
#endif
  };
  port48.param = __xlx_apatb_param_b21_25;

  static hls::sim::Register port49 {
    .name = "b21_26",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_26),
#endif
  };
  port49.param = __xlx_apatb_param_b21_26;

  static hls::sim::Register port50 {
    .name = "b21_27",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_27),
#endif
  };
  port50.param = __xlx_apatb_param_b21_27;

  static hls::sim::Register port51 {
    .name = "b21_28",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_28),
#endif
  };
  port51.param = __xlx_apatb_param_b21_28;

  static hls::sim::Register port52 {
    .name = "b21_29",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_29),
#endif
  };
  port52.param = __xlx_apatb_param_b21_29;

  static hls::sim::Register port53 {
    .name = "b21_30",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_30),
#endif
  };
  port53.param = __xlx_apatb_param_b21_30;

  static hls::sim::Register port54 {
    .name = "b21_31",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_31),
#endif
  };
  port54.param = __xlx_apatb_param_b21_31;

  static hls::sim::Register port55 {
    .name = "b21_32",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_32),
#endif
  };
  port55.param = __xlx_apatb_param_b21_32;

  static hls::sim::Register port56 {
    .name = "b21_33",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_33),
#endif
  };
  port56.param = __xlx_apatb_param_b21_33;

  static hls::sim::Register port57 {
    .name = "b21_34",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_34),
#endif
  };
  port57.param = __xlx_apatb_param_b21_34;

  static hls::sim::Register port58 {
    .name = "b21_35",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_35),
#endif
  };
  port58.param = __xlx_apatb_param_b21_35;

  static hls::sim::Register port59 {
    .name = "b21_36",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_36),
#endif
  };
  port59.param = __xlx_apatb_param_b21_36;

  static hls::sim::Register port60 {
    .name = "b21_37",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_37),
#endif
  };
  port60.param = __xlx_apatb_param_b21_37;

  static hls::sim::Register port61 {
    .name = "b21_38",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_38),
#endif
  };
  port61.param = __xlx_apatb_param_b21_38;

  static hls::sim::Register port62 {
    .name = "b21_39",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_39),
#endif
  };
  port62.param = __xlx_apatb_param_b21_39;

  static hls::sim::Register port63 {
    .name = "b21_40",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_40),
#endif
  };
  port63.param = __xlx_apatb_param_b21_40;

  static hls::sim::Register port64 {
    .name = "b21_41",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_41),
#endif
  };
  port64.param = __xlx_apatb_param_b21_41;

  static hls::sim::Register port65 {
    .name = "b21_42",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_42),
#endif
  };
  port65.param = __xlx_apatb_param_b21_42;

  static hls::sim::Register port66 {
    .name = "b21_43",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_43),
#endif
  };
  port66.param = __xlx_apatb_param_b21_43;

  static hls::sim::Register port67 {
    .name = "b21_44",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_44),
#endif
  };
  port67.param = __xlx_apatb_param_b21_44;

  static hls::sim::Register port68 {
    .name = "b21_45",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_45),
#endif
  };
  port68.param = __xlx_apatb_param_b21_45;

  static hls::sim::Register port69 {
    .name = "b21_46",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_46),
#endif
  };
  port69.param = __xlx_apatb_param_b21_46;

  static hls::sim::Register port70 {
    .name = "b21_47",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_47),
#endif
  };
  port70.param = __xlx_apatb_param_b21_47;

  static hls::sim::Register port71 {
    .name = "b21_48",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_48),
#endif
  };
  port71.param = __xlx_apatb_param_b21_48;

  static hls::sim::Register port72 {
    .name = "b21_49",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_49),
#endif
  };
  port72.param = __xlx_apatb_param_b21_49;

  static hls::sim::Register port73 {
    .name = "b21_50",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_50),
#endif
  };
  port73.param = __xlx_apatb_param_b21_50;

  static hls::sim::Register port74 {
    .name = "b21_51",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_51),
#endif
  };
  port74.param = __xlx_apatb_param_b21_51;

  static hls::sim::Register port75 {
    .name = "b21_52",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_52),
#endif
  };
  port75.param = __xlx_apatb_param_b21_52;

  static hls::sim::Register port76 {
    .name = "b21_53",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_53),
#endif
  };
  port76.param = __xlx_apatb_param_b21_53;

  static hls::sim::Register port77 {
    .name = "b21_54",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_54),
#endif
  };
  port77.param = __xlx_apatb_param_b21_54;

  static hls::sim::Register port78 {
    .name = "b21_55",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_55),
#endif
  };
  port78.param = __xlx_apatb_param_b21_55;

  static hls::sim::Register port79 {
    .name = "b21_56",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_56),
#endif
  };
  port79.param = __xlx_apatb_param_b21_56;

  static hls::sim::Register port80 {
    .name = "b21_57",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_57),
#endif
  };
  port80.param = __xlx_apatb_param_b21_57;

  static hls::sim::Register port81 {
    .name = "b21_58",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_58),
#endif
  };
  port81.param = __xlx_apatb_param_b21_58;

  static hls::sim::Register port82 {
    .name = "b21_59",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_59),
#endif
  };
  port82.param = __xlx_apatb_param_b21_59;

  static hls::sim::Register port83 {
    .name = "b21_60",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_60),
#endif
  };
  port83.param = __xlx_apatb_param_b21_60;

  static hls::sim::Register port84 {
    .name = "b21_61",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_61),
#endif
  };
  port84.param = __xlx_apatb_param_b21_61;

  static hls::sim::Register port85 {
    .name = "b21_62",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_62),
#endif
  };
  port85.param = __xlx_apatb_param_b21_62;

  static hls::sim::Register port86 {
    .name = "b21_63",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_63),
#endif
  };
  port86.param = __xlx_apatb_param_b21_63;

  static hls::sim::Register port87 {
    .name = "b21_64",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_64),
#endif
  };
  port87.param = __xlx_apatb_param_b21_64;

  static hls::sim::Register port88 {
    .name = "b21_65",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_65),
#endif
  };
  port88.param = __xlx_apatb_param_b21_65;

  static hls::sim::Register port89 {
    .name = "b21_66",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_66),
#endif
  };
  port89.param = __xlx_apatb_param_b21_66;

  static hls::sim::Register port90 {
    .name = "b21_67",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_67),
#endif
  };
  port90.param = __xlx_apatb_param_b21_67;

  static hls::sim::Register port91 {
    .name = "b21_68",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_68),
#endif
  };
  port91.param = __xlx_apatb_param_b21_68;

  static hls::sim::Register port92 {
    .name = "b21_69",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_69),
#endif
  };
  port92.param = __xlx_apatb_param_b21_69;

  static hls::sim::Register port93 {
    .name = "b21_70",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_70),
#endif
  };
  port93.param = __xlx_apatb_param_b21_70;

  static hls::sim::Register port94 {
    .name = "b21_71",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b21_71),
#endif
  };
  port94.param = __xlx_apatb_param_b21_71;

  static hls::sim::Register port95 {
    .name = "b24_0",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_0),
#endif
  };
  port95.param = __xlx_apatb_param_b24_0;

  static hls::sim::Register port96 {
    .name = "b24_1",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_1),
#endif
  };
  port96.param = __xlx_apatb_param_b24_1;

  static hls::sim::Register port97 {
    .name = "b24_2",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_2),
#endif
  };
  port97.param = __xlx_apatb_param_b24_2;

  static hls::sim::Register port98 {
    .name = "b24_3",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_3),
#endif
  };
  port98.param = __xlx_apatb_param_b24_3;

  static hls::sim::Register port99 {
    .name = "b24_4",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_4),
#endif
  };
  port99.param = __xlx_apatb_param_b24_4;

  static hls::sim::Register port100 {
    .name = "b24_5",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_5),
#endif
  };
  port100.param = __xlx_apatb_param_b24_5;

  static hls::sim::Register port101 {
    .name = "b24_6",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_6),
#endif
  };
  port101.param = __xlx_apatb_param_b24_6;

  static hls::sim::Register port102 {
    .name = "b24_7",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_7),
#endif
  };
  port102.param = __xlx_apatb_param_b24_7;

  static hls::sim::Register port103 {
    .name = "b24_8",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_8),
#endif
  };
  port103.param = __xlx_apatb_param_b24_8;

  static hls::sim::Register port104 {
    .name = "b24_9",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_9),
#endif
  };
  port104.param = __xlx_apatb_param_b24_9;

  static hls::sim::Register port105 {
    .name = "b24_10",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_10),
#endif
  };
  port105.param = __xlx_apatb_param_b24_10;

  static hls::sim::Register port106 {
    .name = "b24_11",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_11),
#endif
  };
  port106.param = __xlx_apatb_param_b24_11;

  static hls::sim::Register port107 {
    .name = "b24_12",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_12),
#endif
  };
  port107.param = __xlx_apatb_param_b24_12;

  static hls::sim::Register port108 {
    .name = "b24_13",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_13),
#endif
  };
  port108.param = __xlx_apatb_param_b24_13;

  static hls::sim::Register port109 {
    .name = "b24_14",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_14),
#endif
  };
  port109.param = __xlx_apatb_param_b24_14;

  static hls::sim::Register port110 {
    .name = "b24_15",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_15),
#endif
  };
  port110.param = __xlx_apatb_param_b24_15;

  static hls::sim::Register port111 {
    .name = "b24_16",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_16),
#endif
  };
  port111.param = __xlx_apatb_param_b24_16;

  static hls::sim::Register port112 {
    .name = "b24_17",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_17),
#endif
  };
  port112.param = __xlx_apatb_param_b24_17;

  static hls::sim::Register port113 {
    .name = "b24_18",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_18),
#endif
  };
  port113.param = __xlx_apatb_param_b24_18;

  static hls::sim::Register port114 {
    .name = "b24_19",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_19),
#endif
  };
  port114.param = __xlx_apatb_param_b24_19;

  static hls::sim::Register port115 {
    .name = "b24_20",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_20),
#endif
  };
  port115.param = __xlx_apatb_param_b24_20;

  static hls::sim::Register port116 {
    .name = "b24_21",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_21),
#endif
  };
  port116.param = __xlx_apatb_param_b24_21;

  static hls::sim::Register port117 {
    .name = "b24_22",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_22),
#endif
  };
  port117.param = __xlx_apatb_param_b24_22;

  static hls::sim::Register port118 {
    .name = "b24_23",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_23),
#endif
  };
  port118.param = __xlx_apatb_param_b24_23;

  static hls::sim::Register port119 {
    .name = "b24_24",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_24),
#endif
  };
  port119.param = __xlx_apatb_param_b24_24;

  static hls::sim::Register port120 {
    .name = "b24_25",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_25),
#endif
  };
  port120.param = __xlx_apatb_param_b24_25;

  static hls::sim::Register port121 {
    .name = "b24_26",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_26),
#endif
  };
  port121.param = __xlx_apatb_param_b24_26;

  static hls::sim::Register port122 {
    .name = "b24_27",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_27),
#endif
  };
  port122.param = __xlx_apatb_param_b24_27;

  static hls::sim::Register port123 {
    .name = "b24_28",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_28),
#endif
  };
  port123.param = __xlx_apatb_param_b24_28;

  static hls::sim::Register port124 {
    .name = "b24_29",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_29),
#endif
  };
  port124.param = __xlx_apatb_param_b24_29;

  static hls::sim::Register port125 {
    .name = "b24_30",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_30),
#endif
  };
  port125.param = __xlx_apatb_param_b24_30;

  static hls::sim::Register port126 {
    .name = "b24_31",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_31),
#endif
  };
  port126.param = __xlx_apatb_param_b24_31;

  static hls::sim::Register port127 {
    .name = "b24_32",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_32),
#endif
  };
  port127.param = __xlx_apatb_param_b24_32;

  static hls::sim::Register port128 {
    .name = "b24_33",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_33),
#endif
  };
  port128.param = __xlx_apatb_param_b24_33;

  static hls::sim::Register port129 {
    .name = "b24_34",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_34),
#endif
  };
  port129.param = __xlx_apatb_param_b24_34;

  static hls::sim::Register port130 {
    .name = "b24_35",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_35),
#endif
  };
  port130.param = __xlx_apatb_param_b24_35;

  static hls::sim::Register port131 {
    .name = "b24_36",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_36),
#endif
  };
  port131.param = __xlx_apatb_param_b24_36;

  static hls::sim::Register port132 {
    .name = "b24_37",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_37),
#endif
  };
  port132.param = __xlx_apatb_param_b24_37;

  static hls::sim::Register port133 {
    .name = "b24_38",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_38),
#endif
  };
  port133.param = __xlx_apatb_param_b24_38;

  static hls::sim::Register port134 {
    .name = "b24_39",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_39),
#endif
  };
  port134.param = __xlx_apatb_param_b24_39;

  static hls::sim::Register port135 {
    .name = "b24_40",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_40),
#endif
  };
  port135.param = __xlx_apatb_param_b24_40;

  static hls::sim::Register port136 {
    .name = "b24_41",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_41),
#endif
  };
  port136.param = __xlx_apatb_param_b24_41;

  static hls::sim::Register port137 {
    .name = "b24_42",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_42),
#endif
  };
  port137.param = __xlx_apatb_param_b24_42;

  static hls::sim::Register port138 {
    .name = "b24_43",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_43),
#endif
  };
  port138.param = __xlx_apatb_param_b24_43;

  static hls::sim::Register port139 {
    .name = "b24_44",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_44),
#endif
  };
  port139.param = __xlx_apatb_param_b24_44;

  static hls::sim::Register port140 {
    .name = "b24_45",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_45),
#endif
  };
  port140.param = __xlx_apatb_param_b24_45;

  static hls::sim::Register port141 {
    .name = "b24_46",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_46),
#endif
  };
  port141.param = __xlx_apatb_param_b24_46;

  static hls::sim::Register port142 {
    .name = "b24_47",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_47),
#endif
  };
  port142.param = __xlx_apatb_param_b24_47;

  static hls::sim::Register port143 {
    .name = "b24_48",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_48),
#endif
  };
  port143.param = __xlx_apatb_param_b24_48;

  static hls::sim::Register port144 {
    .name = "b24_49",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_49),
#endif
  };
  port144.param = __xlx_apatb_param_b24_49;

  static hls::sim::Register port145 {
    .name = "b24_50",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_50),
#endif
  };
  port145.param = __xlx_apatb_param_b24_50;

  static hls::sim::Register port146 {
    .name = "b24_51",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_51),
#endif
  };
  port146.param = __xlx_apatb_param_b24_51;

  static hls::sim::Register port147 {
    .name = "b24_52",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_52),
#endif
  };
  port147.param = __xlx_apatb_param_b24_52;

  static hls::sim::Register port148 {
    .name = "b24_53",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_53),
#endif
  };
  port148.param = __xlx_apatb_param_b24_53;

  static hls::sim::Register port149 {
    .name = "b24_54",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_54),
#endif
  };
  port149.param = __xlx_apatb_param_b24_54;

  static hls::sim::Register port150 {
    .name = "b24_55",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_55),
#endif
  };
  port150.param = __xlx_apatb_param_b24_55;

  static hls::sim::Register port151 {
    .name = "b24_56",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_56),
#endif
  };
  port151.param = __xlx_apatb_param_b24_56;

  static hls::sim::Register port152 {
    .name = "b24_57",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b24_57),
#endif
  };
  port152.param = __xlx_apatb_param_b24_57;

  static hls::sim::Register port153 {
    .name = "b27",
    .width = 10,
#ifdef POST_CHECK
#else
    .owriter = nullptr,
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_b27),
#endif
  };
  port153.param = __xlx_apatb_param_b27;

#ifdef USE_BINARY_TV_FILE
  static hls::sim::Memory<hls::sim::Input, hls::sim::Output> port154 {
#else
  static hls::sim::Memory<hls::sim::Reader, hls::sim::Writer> port154 {
#endif
    .width = 10,
    .asize = 2,
    .hbm = false,
    .name = { "w9" },
#ifdef POST_CHECK
#else
    .owriter = nullptr,
#ifdef USE_BINARY_TV_FILE
    .iwriter = new hls::sim::Output(AUTOTB_TVIN_w9),
#else
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_w9),
#endif
#endif
  };
  port154.param = { __xlx_apatb_param_w9 };
  port154.nbytes = { 36 };
  port154.offset = {  };
  port154.hasWrite = { false };

#ifdef USE_BINARY_TV_FILE
  static hls::sim::Memory<hls::sim::Input, hls::sim::Output> port155 {
#else
  static hls::sim::Memory<hls::sim::Reader, hls::sim::Writer> port155 {
#endif
    .width = 16,
    .asize = 2,
    .hbm = false,
    .name = { "w16" },
#ifdef POST_CHECK
#else
    .owriter = nullptr,
#ifdef USE_BINARY_TV_FILE
    .iwriter = new hls::sim::Output(AUTOTB_TVIN_w16),
#else
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_w16),
#endif
#endif
  };
  port155.param = { __xlx_apatb_param_w16 };
  port155.nbytes = { 96 };
  port155.offset = {  };
  port155.hasWrite = { false };

#ifdef USE_BINARY_TV_FILE
  static hls::sim::Memory<hls::sim::Input, hls::sim::Output> port156 {
#else
  static hls::sim::Memory<hls::sim::Reader, hls::sim::Writer> port156 {
#endif
    .width = 16,
    .asize = 2,
    .hbm = false,
    .name = { "w21" },
#ifdef POST_CHECK
#else
    .owriter = nullptr,
#ifdef USE_BINARY_TV_FILE
    .iwriter = new hls::sim::Output(AUTOTB_TVIN_w21),
#else
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_w21),
#endif
#endif
  };
  port156.param = { __xlx_apatb_param_w21 };
  port156.nbytes = { 19584 };
  port156.offset = {  };
  port156.hasWrite = { false };

#ifdef USE_BINARY_TV_FILE
  static hls::sim::Memory<hls::sim::Input, hls::sim::Output> port157 {
#else
  static hls::sim::Memory<hls::sim::Reader, hls::sim::Writer> port157 {
#endif
    .width = 16,
    .asize = 2,
    .hbm = false,
    .name = { "w24" },
#ifdef POST_CHECK
#else
    .owriter = nullptr,
#ifdef USE_BINARY_TV_FILE
    .iwriter = new hls::sim::Output(AUTOTB_TVIN_w24),
#else
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_w24),
#endif
#endif
  };
  port157.param = { __xlx_apatb_param_w24 };
  port157.nbytes = { 8352 };
  port157.offset = {  };
  port157.hasWrite = { false };

#ifdef USE_BINARY_TV_FILE
  static hls::sim::Memory<hls::sim::Input, hls::sim::Output> port158 {
#else
  static hls::sim::Memory<hls::sim::Reader, hls::sim::Writer> port158 {
#endif
    .width = 16,
    .asize = 2,
    .hbm = false,
    .name = { "w27" },
#ifdef POST_CHECK
#else
    .owriter = nullptr,
#ifdef USE_BINARY_TV_FILE
    .iwriter = new hls::sim::Output(AUTOTB_TVIN_w27),
#else
    .iwriter = new hls::sim::Writer(AUTOTB_TVIN_w27),
#endif
#endif
  };
  port158.param = { __xlx_apatb_param_w27 };
  port158.nbytes = { 116 };
  port158.offset = {  };
  port158.hasWrite = { false };

  try {
#ifdef POST_CHECK
    CodeState = ENTER_WRAPC_PC;
    check(port4);
#else
    static hls::sim::RefTCL tcl("../tv/cdatafile/ref.tcl");
    CodeState = DUMP_INPUTS;
    dump(port0, port0.iwriter, tcl.AESL_transaction);
    dump(port1, port1.iwriter, tcl.AESL_transaction);
    dump(port2, port2.iwriter, tcl.AESL_transaction);
    dump(port3, port3.iwriter, tcl.AESL_transaction);
    dump(port4, port4.iwriter, tcl.AESL_transaction);
    dump(port5, port5.iwriter, tcl.AESL_transaction);
    dump(port6, port6.iwriter, tcl.AESL_transaction);
    dump(port7, port7.iwriter, tcl.AESL_transaction);
    dump(port8, port8.iwriter, tcl.AESL_transaction);
    dump(port9, port9.iwriter, tcl.AESL_transaction);
    dump(port10, port10.iwriter, tcl.AESL_transaction);
    dump(port11, port11.iwriter, tcl.AESL_transaction);
    dump(port12, port12.iwriter, tcl.AESL_transaction);
    dump(port13, port13.iwriter, tcl.AESL_transaction);
    dump(port14, port14.iwriter, tcl.AESL_transaction);
    dump(port15, port15.iwriter, tcl.AESL_transaction);
    dump(port16, port16.iwriter, tcl.AESL_transaction);
    dump(port17, port17.iwriter, tcl.AESL_transaction);
    dump(port18, port18.iwriter, tcl.AESL_transaction);
    dump(port19, port19.iwriter, tcl.AESL_transaction);
    dump(port20, port20.iwriter, tcl.AESL_transaction);
    dump(port21, port21.iwriter, tcl.AESL_transaction);
    dump(port22, port22.iwriter, tcl.AESL_transaction);
    dump(port23, port23.iwriter, tcl.AESL_transaction);
    dump(port24, port24.iwriter, tcl.AESL_transaction);
    dump(port25, port25.iwriter, tcl.AESL_transaction);
    dump(port26, port26.iwriter, tcl.AESL_transaction);
    dump(port27, port27.iwriter, tcl.AESL_transaction);
    dump(port28, port28.iwriter, tcl.AESL_transaction);
    dump(port29, port29.iwriter, tcl.AESL_transaction);
    dump(port30, port30.iwriter, tcl.AESL_transaction);
    dump(port31, port31.iwriter, tcl.AESL_transaction);
    dump(port32, port32.iwriter, tcl.AESL_transaction);
    dump(port33, port33.iwriter, tcl.AESL_transaction);
    dump(port34, port34.iwriter, tcl.AESL_transaction);
    dump(port35, port35.iwriter, tcl.AESL_transaction);
    dump(port36, port36.iwriter, tcl.AESL_transaction);
    dump(port37, port37.iwriter, tcl.AESL_transaction);
    dump(port38, port38.iwriter, tcl.AESL_transaction);
    dump(port39, port39.iwriter, tcl.AESL_transaction);
    dump(port40, port40.iwriter, tcl.AESL_transaction);
    dump(port41, port41.iwriter, tcl.AESL_transaction);
    dump(port42, port42.iwriter, tcl.AESL_transaction);
    dump(port43, port43.iwriter, tcl.AESL_transaction);
    dump(port44, port44.iwriter, tcl.AESL_transaction);
    dump(port45, port45.iwriter, tcl.AESL_transaction);
    dump(port46, port46.iwriter, tcl.AESL_transaction);
    dump(port47, port47.iwriter, tcl.AESL_transaction);
    dump(port48, port48.iwriter, tcl.AESL_transaction);
    dump(port49, port49.iwriter, tcl.AESL_transaction);
    dump(port50, port50.iwriter, tcl.AESL_transaction);
    dump(port51, port51.iwriter, tcl.AESL_transaction);
    dump(port52, port52.iwriter, tcl.AESL_transaction);
    dump(port53, port53.iwriter, tcl.AESL_transaction);
    dump(port54, port54.iwriter, tcl.AESL_transaction);
    dump(port55, port55.iwriter, tcl.AESL_transaction);
    dump(port56, port56.iwriter, tcl.AESL_transaction);
    dump(port57, port57.iwriter, tcl.AESL_transaction);
    dump(port58, port58.iwriter, tcl.AESL_transaction);
    dump(port59, port59.iwriter, tcl.AESL_transaction);
    dump(port60, port60.iwriter, tcl.AESL_transaction);
    dump(port61, port61.iwriter, tcl.AESL_transaction);
    dump(port62, port62.iwriter, tcl.AESL_transaction);
    dump(port63, port63.iwriter, tcl.AESL_transaction);
    dump(port64, port64.iwriter, tcl.AESL_transaction);
    dump(port65, port65.iwriter, tcl.AESL_transaction);
    dump(port66, port66.iwriter, tcl.AESL_transaction);
    dump(port67, port67.iwriter, tcl.AESL_transaction);
    dump(port68, port68.iwriter, tcl.AESL_transaction);
    dump(port69, port69.iwriter, tcl.AESL_transaction);
    dump(port70, port70.iwriter, tcl.AESL_transaction);
    dump(port71, port71.iwriter, tcl.AESL_transaction);
    dump(port72, port72.iwriter, tcl.AESL_transaction);
    dump(port73, port73.iwriter, tcl.AESL_transaction);
    dump(port74, port74.iwriter, tcl.AESL_transaction);
    dump(port75, port75.iwriter, tcl.AESL_transaction);
    dump(port76, port76.iwriter, tcl.AESL_transaction);
    dump(port77, port77.iwriter, tcl.AESL_transaction);
    dump(port78, port78.iwriter, tcl.AESL_transaction);
    dump(port79, port79.iwriter, tcl.AESL_transaction);
    dump(port80, port80.iwriter, tcl.AESL_transaction);
    dump(port81, port81.iwriter, tcl.AESL_transaction);
    dump(port82, port82.iwriter, tcl.AESL_transaction);
    dump(port83, port83.iwriter, tcl.AESL_transaction);
    dump(port84, port84.iwriter, tcl.AESL_transaction);
    dump(port85, port85.iwriter, tcl.AESL_transaction);
    dump(port86, port86.iwriter, tcl.AESL_transaction);
    dump(port87, port87.iwriter, tcl.AESL_transaction);
    dump(port88, port88.iwriter, tcl.AESL_transaction);
    dump(port89, port89.iwriter, tcl.AESL_transaction);
    dump(port90, port90.iwriter, tcl.AESL_transaction);
    dump(port91, port91.iwriter, tcl.AESL_transaction);
    dump(port92, port92.iwriter, tcl.AESL_transaction);
    dump(port93, port93.iwriter, tcl.AESL_transaction);
    dump(port94, port94.iwriter, tcl.AESL_transaction);
    dump(port95, port95.iwriter, tcl.AESL_transaction);
    dump(port96, port96.iwriter, tcl.AESL_transaction);
    dump(port97, port97.iwriter, tcl.AESL_transaction);
    dump(port98, port98.iwriter, tcl.AESL_transaction);
    dump(port99, port99.iwriter, tcl.AESL_transaction);
    dump(port100, port100.iwriter, tcl.AESL_transaction);
    dump(port101, port101.iwriter, tcl.AESL_transaction);
    dump(port102, port102.iwriter, tcl.AESL_transaction);
    dump(port103, port103.iwriter, tcl.AESL_transaction);
    dump(port104, port104.iwriter, tcl.AESL_transaction);
    dump(port105, port105.iwriter, tcl.AESL_transaction);
    dump(port106, port106.iwriter, tcl.AESL_transaction);
    dump(port107, port107.iwriter, tcl.AESL_transaction);
    dump(port108, port108.iwriter, tcl.AESL_transaction);
    dump(port109, port109.iwriter, tcl.AESL_transaction);
    dump(port110, port110.iwriter, tcl.AESL_transaction);
    dump(port111, port111.iwriter, tcl.AESL_transaction);
    dump(port112, port112.iwriter, tcl.AESL_transaction);
    dump(port113, port113.iwriter, tcl.AESL_transaction);
    dump(port114, port114.iwriter, tcl.AESL_transaction);
    dump(port115, port115.iwriter, tcl.AESL_transaction);
    dump(port116, port116.iwriter, tcl.AESL_transaction);
    dump(port117, port117.iwriter, tcl.AESL_transaction);
    dump(port118, port118.iwriter, tcl.AESL_transaction);
    dump(port119, port119.iwriter, tcl.AESL_transaction);
    dump(port120, port120.iwriter, tcl.AESL_transaction);
    dump(port121, port121.iwriter, tcl.AESL_transaction);
    dump(port122, port122.iwriter, tcl.AESL_transaction);
    dump(port123, port123.iwriter, tcl.AESL_transaction);
    dump(port124, port124.iwriter, tcl.AESL_transaction);
    dump(port125, port125.iwriter, tcl.AESL_transaction);
    dump(port126, port126.iwriter, tcl.AESL_transaction);
    dump(port127, port127.iwriter, tcl.AESL_transaction);
    dump(port128, port128.iwriter, tcl.AESL_transaction);
    dump(port129, port129.iwriter, tcl.AESL_transaction);
    dump(port130, port130.iwriter, tcl.AESL_transaction);
    dump(port131, port131.iwriter, tcl.AESL_transaction);
    dump(port132, port132.iwriter, tcl.AESL_transaction);
    dump(port133, port133.iwriter, tcl.AESL_transaction);
    dump(port134, port134.iwriter, tcl.AESL_transaction);
    dump(port135, port135.iwriter, tcl.AESL_transaction);
    dump(port136, port136.iwriter, tcl.AESL_transaction);
    dump(port137, port137.iwriter, tcl.AESL_transaction);
    dump(port138, port138.iwriter, tcl.AESL_transaction);
    dump(port139, port139.iwriter, tcl.AESL_transaction);
    dump(port140, port140.iwriter, tcl.AESL_transaction);
    dump(port141, port141.iwriter, tcl.AESL_transaction);
    dump(port142, port142.iwriter, tcl.AESL_transaction);
    dump(port143, port143.iwriter, tcl.AESL_transaction);
    dump(port144, port144.iwriter, tcl.AESL_transaction);
    dump(port145, port145.iwriter, tcl.AESL_transaction);
    dump(port146, port146.iwriter, tcl.AESL_transaction);
    dump(port147, port147.iwriter, tcl.AESL_transaction);
    dump(port148, port148.iwriter, tcl.AESL_transaction);
    dump(port149, port149.iwriter, tcl.AESL_transaction);
    dump(port150, port150.iwriter, tcl.AESL_transaction);
    dump(port151, port151.iwriter, tcl.AESL_transaction);
    dump(port152, port152.iwriter, tcl.AESL_transaction);
    dump(port153, port153.iwriter, tcl.AESL_transaction);
    dump(port154, port154.iwriter, tcl.AESL_transaction);
    dump(port155, port155.iwriter, tcl.AESL_transaction);
    dump(port156, port156.iwriter, tcl.AESL_transaction);
    dump(port157, port157.iwriter, tcl.AESL_transaction);
    dump(port158, port158.iwriter, tcl.AESL_transaction);
    port0.doTCL(tcl);
    port1.doTCL(tcl);
    port2.doTCL(tcl);
    port3.doTCL(tcl);
    port4.doTCL(tcl);
    port5.doTCL(tcl);
    port6.doTCL(tcl);
    port7.doTCL(tcl);
    port8.doTCL(tcl);
    port9.doTCL(tcl);
    port10.doTCL(tcl);
    port11.doTCL(tcl);
    port12.doTCL(tcl);
    port13.doTCL(tcl);
    port14.doTCL(tcl);
    port15.doTCL(tcl);
    port16.doTCL(tcl);
    port17.doTCL(tcl);
    port18.doTCL(tcl);
    port19.doTCL(tcl);
    port20.doTCL(tcl);
    port21.doTCL(tcl);
    port22.doTCL(tcl);
    port23.doTCL(tcl);
    port24.doTCL(tcl);
    port25.doTCL(tcl);
    port26.doTCL(tcl);
    port27.doTCL(tcl);
    port28.doTCL(tcl);
    port29.doTCL(tcl);
    port30.doTCL(tcl);
    port31.doTCL(tcl);
    port32.doTCL(tcl);
    port33.doTCL(tcl);
    port34.doTCL(tcl);
    port35.doTCL(tcl);
    port36.doTCL(tcl);
    port37.doTCL(tcl);
    port38.doTCL(tcl);
    port39.doTCL(tcl);
    port40.doTCL(tcl);
    port41.doTCL(tcl);
    port42.doTCL(tcl);
    port43.doTCL(tcl);
    port44.doTCL(tcl);
    port45.doTCL(tcl);
    port46.doTCL(tcl);
    port47.doTCL(tcl);
    port48.doTCL(tcl);
    port49.doTCL(tcl);
    port50.doTCL(tcl);
    port51.doTCL(tcl);
    port52.doTCL(tcl);
    port53.doTCL(tcl);
    port54.doTCL(tcl);
    port55.doTCL(tcl);
    port56.doTCL(tcl);
    port57.doTCL(tcl);
    port58.doTCL(tcl);
    port59.doTCL(tcl);
    port60.doTCL(tcl);
    port61.doTCL(tcl);
    port62.doTCL(tcl);
    port63.doTCL(tcl);
    port64.doTCL(tcl);
    port65.doTCL(tcl);
    port66.doTCL(tcl);
    port67.doTCL(tcl);
    port68.doTCL(tcl);
    port69.doTCL(tcl);
    port70.doTCL(tcl);
    port71.doTCL(tcl);
    port72.doTCL(tcl);
    port73.doTCL(tcl);
    port74.doTCL(tcl);
    port75.doTCL(tcl);
    port76.doTCL(tcl);
    port77.doTCL(tcl);
    port78.doTCL(tcl);
    port79.doTCL(tcl);
    port80.doTCL(tcl);
    port81.doTCL(tcl);
    port82.doTCL(tcl);
    port83.doTCL(tcl);
    port84.doTCL(tcl);
    port85.doTCL(tcl);
    port86.doTCL(tcl);
    port87.doTCL(tcl);
    port88.doTCL(tcl);
    port89.doTCL(tcl);
    port90.doTCL(tcl);
    port91.doTCL(tcl);
    port92.doTCL(tcl);
    port93.doTCL(tcl);
    port94.doTCL(tcl);
    port95.doTCL(tcl);
    port96.doTCL(tcl);
    port97.doTCL(tcl);
    port98.doTCL(tcl);
    port99.doTCL(tcl);
    port100.doTCL(tcl);
    port101.doTCL(tcl);
    port102.doTCL(tcl);
    port103.doTCL(tcl);
    port104.doTCL(tcl);
    port105.doTCL(tcl);
    port106.doTCL(tcl);
    port107.doTCL(tcl);
    port108.doTCL(tcl);
    port109.doTCL(tcl);
    port110.doTCL(tcl);
    port111.doTCL(tcl);
    port112.doTCL(tcl);
    port113.doTCL(tcl);
    port114.doTCL(tcl);
    port115.doTCL(tcl);
    port116.doTCL(tcl);
    port117.doTCL(tcl);
    port118.doTCL(tcl);
    port119.doTCL(tcl);
    port120.doTCL(tcl);
    port121.doTCL(tcl);
    port122.doTCL(tcl);
    port123.doTCL(tcl);
    port124.doTCL(tcl);
    port125.doTCL(tcl);
    port126.doTCL(tcl);
    port127.doTCL(tcl);
    port128.doTCL(tcl);
    port129.doTCL(tcl);
    port130.doTCL(tcl);
    port131.doTCL(tcl);
    port132.doTCL(tcl);
    port133.doTCL(tcl);
    port134.doTCL(tcl);
    port135.doTCL(tcl);
    port136.doTCL(tcl);
    port137.doTCL(tcl);
    port138.doTCL(tcl);
    port139.doTCL(tcl);
    port140.doTCL(tcl);
    port141.doTCL(tcl);
    port142.doTCL(tcl);
    port143.doTCL(tcl);
    port144.doTCL(tcl);
    port145.doTCL(tcl);
    port146.doTCL(tcl);
    port147.doTCL(tcl);
    port148.doTCL(tcl);
    port149.doTCL(tcl);
    port150.doTCL(tcl);
    port151.doTCL(tcl);
    port152.doTCL(tcl);
    port153.doTCL(tcl);
    port154.doTCL(tcl);
    port155.doTCL(tcl);
    port156.doTCL(tcl);
    port157.doTCL(tcl);
    port158.doTCL(tcl);
    CodeState = CALL_C_DUT;
    myproject_hw_stub_wrapper(__xlx_apatb_param_cluster, __xlx_apatb_param_nModule, __xlx_apatb_param_x_local, __xlx_apatb_param_y_local, __xlx_apatb_param_layer29_out, __xlx_apatb_param_w9, __xlx_apatb_param_b9_0, __xlx_apatb_param_b9_1, __xlx_apatb_param_w16, __xlx_apatb_param_b16_0, __xlx_apatb_param_b16_1, __xlx_apatb_param_b16_2, __xlx_apatb_param_b16_3, __xlx_apatb_param_b16_4, __xlx_apatb_param_b16_5, __xlx_apatb_param_b16_6, __xlx_apatb_param_b16_7, __xlx_apatb_param_b16_8, __xlx_apatb_param_b16_9, __xlx_apatb_param_b16_10, __xlx_apatb_param_b16_11, __xlx_apatb_param_b16_12, __xlx_apatb_param_b16_13, __xlx_apatb_param_b16_14, __xlx_apatb_param_b16_15, __xlx_apatb_param_w21, __xlx_apatb_param_b21_0, __xlx_apatb_param_b21_1, __xlx_apatb_param_b21_2, __xlx_apatb_param_b21_3, __xlx_apatb_param_b21_4, __xlx_apatb_param_b21_5, __xlx_apatb_param_b21_6, __xlx_apatb_param_b21_7, __xlx_apatb_param_b21_8, __xlx_apatb_param_b21_9, __xlx_apatb_param_b21_10, __xlx_apatb_param_b21_11, __xlx_apatb_param_b21_12, __xlx_apatb_param_b21_13, __xlx_apatb_param_b21_14, __xlx_apatb_param_b21_15, __xlx_apatb_param_b21_16, __xlx_apatb_param_b21_17, __xlx_apatb_param_b21_18, __xlx_apatb_param_b21_19, __xlx_apatb_param_b21_20, __xlx_apatb_param_b21_21, __xlx_apatb_param_b21_22, __xlx_apatb_param_b21_23, __xlx_apatb_param_b21_24, __xlx_apatb_param_b21_25, __xlx_apatb_param_b21_26, __xlx_apatb_param_b21_27, __xlx_apatb_param_b21_28, __xlx_apatb_param_b21_29, __xlx_apatb_param_b21_30, __xlx_apatb_param_b21_31, __xlx_apatb_param_b21_32, __xlx_apatb_param_b21_33, __xlx_apatb_param_b21_34, __xlx_apatb_param_b21_35, __xlx_apatb_param_b21_36, __xlx_apatb_param_b21_37, __xlx_apatb_param_b21_38, __xlx_apatb_param_b21_39, __xlx_apatb_param_b21_40, __xlx_apatb_param_b21_41, __xlx_apatb_param_b21_42, __xlx_apatb_param_b21_43, __xlx_apatb_param_b21_44, __xlx_apatb_param_b21_45, __xlx_apatb_param_b21_46, __xlx_apatb_param_b21_47, __xlx_apatb_param_b21_48, __xlx_apatb_param_b21_49, __xlx_apatb_param_b21_50, __xlx_apatb_param_b21_51, __xlx_apatb_param_b21_52, __xlx_apatb_param_b21_53, __xlx_apatb_param_b21_54, __xlx_apatb_param_b21_55, __xlx_apatb_param_b21_56, __xlx_apatb_param_b21_57, __xlx_apatb_param_b21_58, __xlx_apatb_param_b21_59, __xlx_apatb_param_b21_60, __xlx_apatb_param_b21_61, __xlx_apatb_param_b21_62, __xlx_apatb_param_b21_63, __xlx_apatb_param_b21_64, __xlx_apatb_param_b21_65, __xlx_apatb_param_b21_66, __xlx_apatb_param_b21_67, __xlx_apatb_param_b21_68, __xlx_apatb_param_b21_69, __xlx_apatb_param_b21_70, __xlx_apatb_param_b21_71, __xlx_apatb_param_w24, __xlx_apatb_param_b24_0, __xlx_apatb_param_b24_1, __xlx_apatb_param_b24_2, __xlx_apatb_param_b24_3, __xlx_apatb_param_b24_4, __xlx_apatb_param_b24_5, __xlx_apatb_param_b24_6, __xlx_apatb_param_b24_7, __xlx_apatb_param_b24_8, __xlx_apatb_param_b24_9, __xlx_apatb_param_b24_10, __xlx_apatb_param_b24_11, __xlx_apatb_param_b24_12, __xlx_apatb_param_b24_13, __xlx_apatb_param_b24_14, __xlx_apatb_param_b24_15, __xlx_apatb_param_b24_16, __xlx_apatb_param_b24_17, __xlx_apatb_param_b24_18, __xlx_apatb_param_b24_19, __xlx_apatb_param_b24_20, __xlx_apatb_param_b24_21, __xlx_apatb_param_b24_22, __xlx_apatb_param_b24_23, __xlx_apatb_param_b24_24, __xlx_apatb_param_b24_25, __xlx_apatb_param_b24_26, __xlx_apatb_param_b24_27, __xlx_apatb_param_b24_28, __xlx_apatb_param_b24_29, __xlx_apatb_param_b24_30, __xlx_apatb_param_b24_31, __xlx_apatb_param_b24_32, __xlx_apatb_param_b24_33, __xlx_apatb_param_b24_34, __xlx_apatb_param_b24_35, __xlx_apatb_param_b24_36, __xlx_apatb_param_b24_37, __xlx_apatb_param_b24_38, __xlx_apatb_param_b24_39, __xlx_apatb_param_b24_40, __xlx_apatb_param_b24_41, __xlx_apatb_param_b24_42, __xlx_apatb_param_b24_43, __xlx_apatb_param_b24_44, __xlx_apatb_param_b24_45, __xlx_apatb_param_b24_46, __xlx_apatb_param_b24_47, __xlx_apatb_param_b24_48, __xlx_apatb_param_b24_49, __xlx_apatb_param_b24_50, __xlx_apatb_param_b24_51, __xlx_apatb_param_b24_52, __xlx_apatb_param_b24_53, __xlx_apatb_param_b24_54, __xlx_apatb_param_b24_55, __xlx_apatb_param_b24_56, __xlx_apatb_param_b24_57, __xlx_apatb_param_w27, __xlx_apatb_param_b27);
    CodeState = DUMP_OUTPUTS;
    dump(port4, port4.owriter, tcl.AESL_transaction);
    tcl.AESL_transaction++;
#endif
  } catch (const hls::sim::SimException &e) {
    hls::sim::errExit(e.line, e.msg);
  }
}