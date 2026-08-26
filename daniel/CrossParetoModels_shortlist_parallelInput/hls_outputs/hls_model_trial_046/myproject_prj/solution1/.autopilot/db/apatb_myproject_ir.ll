; ModuleID = '/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_shortlist_parallelInput/hls_outputs/hls_model_trial_046/myproject_prj/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>" = type { %"struct.ap_fixed_base<16, 6, true, AP_TRN, AP_WRAP, 0>" }
%"struct.ap_fixed_base<16, 6, true, AP_TRN, AP_WRAP, 0>" = type { %"struct.ssdm_int<16, true>" }
%"struct.ssdm_int<16, true>" = type { i16 }
%"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>" = type { %"struct.ap_fixed_base<8, 0, false, AP_RND_CONV, AP_SAT, 0>" }
%"struct.ap_fixed_base<8, 0, false, AP_RND_CONV, AP_SAT, 0>" = type { %"class.std::ios_base::Init" }
%"class.std::ios_base::Init" = type { i8 }
%"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>" = type { %"struct.ap_fixed_base<10, 1, true, AP_TRN, AP_WRAP, 0>" }
%"struct.ap_fixed_base<10, 1, true, AP_TRN, AP_WRAP, 0>" = type { %"struct.ssdm_int<10, true>" }
%"struct.ssdm_int<10, true>" = type { i10 }

; Function Attrs: noinline
define void @apatb_myproject_ir(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="273" %cluster, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %nModule, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %x_local, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %y_local, %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"* noalias nocapture nonnull "fpga.decayed.dim.hint"="1" "partition" %layer29_out, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="18" %w9, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="2" %b9, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="48" %w16, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="16" %b16, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="9792" %w21, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="72" %b21, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="4176" %w24, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="58" %b24, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="58" %w27, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %b27) local_unnamed_addr #0 {
entry:
  %cluster_copy21 = alloca i4368, align 512
  %nModule_copy22 = alloca i16, align 512
  %x_local_copy23 = alloca i16, align 512
  %y_local_copy24 = alloca i16, align 512
  %layer29_out_copy19 = alloca i8, align 512
  %w9_copy = alloca [18 x i10], align 512
  %b9_copy_0 = alloca i10, align 512
  %b9_copy_1 = alloca i10, align 512
  %w16_copy = alloca [48 x i10], align 512
  %b16_copy_0 = alloca i10, align 512
  %b16_copy_1 = alloca i10, align 512
  %b16_copy_2 = alloca i10, align 512
  %b16_copy_3 = alloca i10, align 512
  %b16_copy_4 = alloca i10, align 512
  %b16_copy_5 = alloca i10, align 512
  %b16_copy_6 = alloca i10, align 512
  %b16_copy_7 = alloca i10, align 512
  %b16_copy_8 = alloca i10, align 512
  %b16_copy_9 = alloca i10, align 512
  %b16_copy_10 = alloca i10, align 512
  %b16_copy_11 = alloca i10, align 512
  %b16_copy_12 = alloca i10, align 512
  %b16_copy_13 = alloca i10, align 512
  %b16_copy_14 = alloca i10, align 512
  %b16_copy_15 = alloca i10, align 512
  %malloccall = call i8* @malloc(i64 19584)
  %w21_copy = bitcast i8* %malloccall to [9792 x i10]*
  %b21_copy_0 = alloca i10, align 512
  %b21_copy_1 = alloca i10, align 512
  %b21_copy_2 = alloca i10, align 512
  %b21_copy_3 = alloca i10, align 512
  %b21_copy_4 = alloca i10, align 512
  %b21_copy_5 = alloca i10, align 512
  %b21_copy_6 = alloca i10, align 512
  %b21_copy_7 = alloca i10, align 512
  %b21_copy_8 = alloca i10, align 512
  %b21_copy_9 = alloca i10, align 512
  %b21_copy_10 = alloca i10, align 512
  %b21_copy_11 = alloca i10, align 512
  %b21_copy_12 = alloca i10, align 512
  %b21_copy_13 = alloca i10, align 512
  %b21_copy_14 = alloca i10, align 512
  %b21_copy_15 = alloca i10, align 512
  %b21_copy_16 = alloca i10, align 512
  %b21_copy_17 = alloca i10, align 512
  %b21_copy_18 = alloca i10, align 512
  %b21_copy_19 = alloca i10, align 512
  %b21_copy_20 = alloca i10, align 512
  %b21_copy_21 = alloca i10, align 512
  %b21_copy_22 = alloca i10, align 512
  %b21_copy_23 = alloca i10, align 512
  %b21_copy_24 = alloca i10, align 512
  %b21_copy_25 = alloca i10, align 512
  %b21_copy_26 = alloca i10, align 512
  %b21_copy_27 = alloca i10, align 512
  %b21_copy_28 = alloca i10, align 512
  %b21_copy_29 = alloca i10, align 512
  %b21_copy_30 = alloca i10, align 512
  %b21_copy_31 = alloca i10, align 512
  %b21_copy_32 = alloca i10, align 512
  %b21_copy_33 = alloca i10, align 512
  %b21_copy_34 = alloca i10, align 512
  %b21_copy_35 = alloca i10, align 512
  %b21_copy_36 = alloca i10, align 512
  %b21_copy_37 = alloca i10, align 512
  %b21_copy_38 = alloca i10, align 512
  %b21_copy_39 = alloca i10, align 512
  %b21_copy_40 = alloca i10, align 512
  %b21_copy_41 = alloca i10, align 512
  %b21_copy_42 = alloca i10, align 512
  %b21_copy_43 = alloca i10, align 512
  %b21_copy_44 = alloca i10, align 512
  %b21_copy_45 = alloca i10, align 512
  %b21_copy_46 = alloca i10, align 512
  %b21_copy_47 = alloca i10, align 512
  %b21_copy_48 = alloca i10, align 512
  %b21_copy_49 = alloca i10, align 512
  %b21_copy_50 = alloca i10, align 512
  %b21_copy_51 = alloca i10, align 512
  %b21_copy_52 = alloca i10, align 512
  %b21_copy_53 = alloca i10, align 512
  %b21_copy_54 = alloca i10, align 512
  %b21_copy_55 = alloca i10, align 512
  %b21_copy_56 = alloca i10, align 512
  %b21_copy_57 = alloca i10, align 512
  %b21_copy_58 = alloca i10, align 512
  %b21_copy_59 = alloca i10, align 512
  %b21_copy_60 = alloca i10, align 512
  %b21_copy_61 = alloca i10, align 512
  %b21_copy_62 = alloca i10, align 512
  %b21_copy_63 = alloca i10, align 512
  %b21_copy_64 = alloca i10, align 512
  %b21_copy_65 = alloca i10, align 512
  %b21_copy_66 = alloca i10, align 512
  %b21_copy_67 = alloca i10, align 512
  %b21_copy_68 = alloca i10, align 512
  %b21_copy_69 = alloca i10, align 512
  %b21_copy_70 = alloca i10, align 512
  %b21_copy_71 = alloca i10, align 512
  %malloccall1 = call i8* @malloc(i64 8352)
  %w24_copy = bitcast i8* %malloccall1 to [4176 x i10]*
  %b24_copy_0 = alloca i10, align 512
  %b24_copy_1 = alloca i10, align 512
  %b24_copy_2 = alloca i10, align 512
  %b24_copy_3 = alloca i10, align 512
  %b24_copy_4 = alloca i10, align 512
  %b24_copy_5 = alloca i10, align 512
  %b24_copy_6 = alloca i10, align 512
  %b24_copy_7 = alloca i10, align 512
  %b24_copy_8 = alloca i10, align 512
  %b24_copy_9 = alloca i10, align 512
  %b24_copy_10 = alloca i10, align 512
  %b24_copy_11 = alloca i10, align 512
  %b24_copy_12 = alloca i10, align 512
  %b24_copy_13 = alloca i10, align 512
  %b24_copy_14 = alloca i10, align 512
  %b24_copy_15 = alloca i10, align 512
  %b24_copy_16 = alloca i10, align 512
  %b24_copy_17 = alloca i10, align 512
  %b24_copy_18 = alloca i10, align 512
  %b24_copy_19 = alloca i10, align 512
  %b24_copy_20 = alloca i10, align 512
  %b24_copy_21 = alloca i10, align 512
  %b24_copy_22 = alloca i10, align 512
  %b24_copy_23 = alloca i10, align 512
  %b24_copy_24 = alloca i10, align 512
  %b24_copy_25 = alloca i10, align 512
  %b24_copy_26 = alloca i10, align 512
  %b24_copy_27 = alloca i10, align 512
  %b24_copy_28 = alloca i10, align 512
  %b24_copy_29 = alloca i10, align 512
  %b24_copy_30 = alloca i10, align 512
  %b24_copy_31 = alloca i10, align 512
  %b24_copy_32 = alloca i10, align 512
  %b24_copy_33 = alloca i10, align 512
  %b24_copy_34 = alloca i10, align 512
  %b24_copy_35 = alloca i10, align 512
  %b24_copy_36 = alloca i10, align 512
  %b24_copy_37 = alloca i10, align 512
  %b24_copy_38 = alloca i10, align 512
  %b24_copy_39 = alloca i10, align 512
  %b24_copy_40 = alloca i10, align 512
  %b24_copy_41 = alloca i10, align 512
  %b24_copy_42 = alloca i10, align 512
  %b24_copy_43 = alloca i10, align 512
  %b24_copy_44 = alloca i10, align 512
  %b24_copy_45 = alloca i10, align 512
  %b24_copy_46 = alloca i10, align 512
  %b24_copy_47 = alloca i10, align 512
  %b24_copy_48 = alloca i10, align 512
  %b24_copy_49 = alloca i10, align 512
  %b24_copy_50 = alloca i10, align 512
  %b24_copy_51 = alloca i10, align 512
  %b24_copy_52 = alloca i10, align 512
  %b24_copy_53 = alloca i10, align 512
  %b24_copy_54 = alloca i10, align 512
  %b24_copy_55 = alloca i10, align 512
  %b24_copy_56 = alloca i10, align 512
  %b24_copy_57 = alloca i10, align 512
  %w27_copy = alloca [58 x i10], align 512
  %b27_copy20 = alloca i10, align 512
  %0 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %cluster to [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %1 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %nModule to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %2 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %x_local to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %3 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %y_local to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %4 = bitcast %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"* %layer29_out to [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]*
  %5 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %w9 to [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %6 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %b9 to [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %7 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %w16 to [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %8 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %b16 to [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %9 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %w21 to [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %10 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %b21 to [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %11 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %w24 to [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %12 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %b24 to [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %13 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %w27 to [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %14 = bitcast %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %b27 to [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  call void @copy_in([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %0, i4368* nonnull align 512 %cluster_copy21, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %1, i16* nonnull align 512 %nModule_copy22, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %2, i16* nonnull align 512 %x_local_copy23, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %3, i16* nonnull align 512 %y_local_copy24, [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* nonnull %4, i8* nonnull align 512 %layer29_out_copy19, [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %5, [18 x i10]* nonnull align 512 %w9_copy, [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %6, i10* nonnull align 512 %b9_copy_0, i10* nonnull align 512 %b9_copy_1, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %7, [48 x i10]* nonnull align 512 %w16_copy, [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %8, i10* nonnull align 512 %b16_copy_0, i10* nonnull align 512 %b16_copy_1, i10* nonnull align 512 %b16_copy_2, i10* nonnull align 512 %b16_copy_3, i10* nonnull align 512 %b16_copy_4, i10* nonnull align 512 %b16_copy_5, i10* nonnull align 512 %b16_copy_6, i10* nonnull align 512 %b16_copy_7, i10* nonnull align 512 %b16_copy_8, i10* nonnull align 512 %b16_copy_9, i10* nonnull align 512 %b16_copy_10, i10* nonnull align 512 %b16_copy_11, i10* nonnull align 512 %b16_copy_12, i10* nonnull align 512 %b16_copy_13, i10* nonnull align 512 %b16_copy_14, i10* nonnull align 512 %b16_copy_15, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %9, [9792 x i10]* %w21_copy, [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %10, i10* nonnull align 512 %b21_copy_0, i10* nonnull align 512 %b21_copy_1, i10* nonnull align 512 %b21_copy_2, i10* nonnull align 512 %b21_copy_3, i10* nonnull align 512 %b21_copy_4, i10* nonnull align 512 %b21_copy_5, i10* nonnull align 512 %b21_copy_6, i10* nonnull align 512 %b21_copy_7, i10* nonnull align 512 %b21_copy_8, i10* nonnull align 512 %b21_copy_9, i10* nonnull align 512 %b21_copy_10, i10* nonnull align 512 %b21_copy_11, i10* nonnull align 512 %b21_copy_12, i10* nonnull align 512 %b21_copy_13, i10* nonnull align 512 %b21_copy_14, i10* nonnull align 512 %b21_copy_15, i10* nonnull align 512 %b21_copy_16, i10* nonnull align 512 %b21_copy_17, i10* nonnull align 512 %b21_copy_18, i10* nonnull align 512 %b21_copy_19, i10* nonnull align 512 %b21_copy_20, i10* nonnull align 512 %b21_copy_21, i10* nonnull align 512 %b21_copy_22, i10* nonnull align 512 %b21_copy_23, i10* nonnull align 512 %b21_copy_24, i10* nonnull align 512 %b21_copy_25, i10* nonnull align 512 %b21_copy_26, i10* nonnull align 512 %b21_copy_27, i10* nonnull align 512 %b21_copy_28, i10* nonnull align 512 %b21_copy_29, i10* nonnull align 512 %b21_copy_30, i10* nonnull align 512 %b21_copy_31, i10* nonnull align 512 %b21_copy_32, i10* nonnull align 512 %b21_copy_33, i10* nonnull align 512 %b21_copy_34, i10* nonnull align 512 %b21_copy_35, i10* nonnull align 512 %b21_copy_36, i10* nonnull align 512 %b21_copy_37, i10* nonnull align 512 %b21_copy_38, i10* nonnull align 512 %b21_copy_39, i10* nonnull align 512 %b21_copy_40, i10* nonnull align 512 %b21_copy_41, i10* nonnull align 512 %b21_copy_42, i10* nonnull align 512 %b21_copy_43, i10* nonnull align 512 %b21_copy_44, i10* nonnull align 512 %b21_copy_45, i10* nonnull align 512 %b21_copy_46, i10* nonnull align 512 %b21_copy_47, i10* nonnull align 512 %b21_copy_48, i10* nonnull align 512 %b21_copy_49, i10* nonnull align 512 %b21_copy_50, i10* nonnull align 512 %b21_copy_51, i10* nonnull align 512 %b21_copy_52, i10* nonnull align 512 %b21_copy_53, i10* nonnull align 512 %b21_copy_54, i10* nonnull align 512 %b21_copy_55, i10* nonnull align 512 %b21_copy_56, i10* nonnull align 512 %b21_copy_57, i10* nonnull align 512 %b21_copy_58, i10* nonnull align 512 %b21_copy_59, i10* nonnull align 512 %b21_copy_60, i10* nonnull align 512 %b21_copy_61, i10* nonnull align 512 %b21_copy_62, i10* nonnull align 512 %b21_copy_63, i10* nonnull align 512 %b21_copy_64, i10* nonnull align 512 %b21_copy_65, i10* nonnull align 512 %b21_copy_66, i10* nonnull align 512 %b21_copy_67, i10* nonnull align 512 %b21_copy_68, i10* nonnull align 512 %b21_copy_69, i10* nonnull align 512 %b21_copy_70, i10* nonnull align 512 %b21_copy_71, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %11, [4176 x i10]* %w24_copy, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %12, i10* nonnull align 512 %b24_copy_0, i10* nonnull align 512 %b24_copy_1, i10* nonnull align 512 %b24_copy_2, i10* nonnull align 512 %b24_copy_3, i10* nonnull align 512 %b24_copy_4, i10* nonnull align 512 %b24_copy_5, i10* nonnull align 512 %b24_copy_6, i10* nonnull align 512 %b24_copy_7, i10* nonnull align 512 %b24_copy_8, i10* nonnull align 512 %b24_copy_9, i10* nonnull align 512 %b24_copy_10, i10* nonnull align 512 %b24_copy_11, i10* nonnull align 512 %b24_copy_12, i10* nonnull align 512 %b24_copy_13, i10* nonnull align 512 %b24_copy_14, i10* nonnull align 512 %b24_copy_15, i10* nonnull align 512 %b24_copy_16, i10* nonnull align 512 %b24_copy_17, i10* nonnull align 512 %b24_copy_18, i10* nonnull align 512 %b24_copy_19, i10* nonnull align 512 %b24_copy_20, i10* nonnull align 512 %b24_copy_21, i10* nonnull align 512 %b24_copy_22, i10* nonnull align 512 %b24_copy_23, i10* nonnull align 512 %b24_copy_24, i10* nonnull align 512 %b24_copy_25, i10* nonnull align 512 %b24_copy_26, i10* nonnull align 512 %b24_copy_27, i10* nonnull align 512 %b24_copy_28, i10* nonnull align 512 %b24_copy_29, i10* nonnull align 512 %b24_copy_30, i10* nonnull align 512 %b24_copy_31, i10* nonnull align 512 %b24_copy_32, i10* nonnull align 512 %b24_copy_33, i10* nonnull align 512 %b24_copy_34, i10* nonnull align 512 %b24_copy_35, i10* nonnull align 512 %b24_copy_36, i10* nonnull align 512 %b24_copy_37, i10* nonnull align 512 %b24_copy_38, i10* nonnull align 512 %b24_copy_39, i10* nonnull align 512 %b24_copy_40, i10* nonnull align 512 %b24_copy_41, i10* nonnull align 512 %b24_copy_42, i10* nonnull align 512 %b24_copy_43, i10* nonnull align 512 %b24_copy_44, i10* nonnull align 512 %b24_copy_45, i10* nonnull align 512 %b24_copy_46, i10* nonnull align 512 %b24_copy_47, i10* nonnull align 512 %b24_copy_48, i10* nonnull align 512 %b24_copy_49, i10* nonnull align 512 %b24_copy_50, i10* nonnull align 512 %b24_copy_51, i10* nonnull align 512 %b24_copy_52, i10* nonnull align 512 %b24_copy_53, i10* nonnull align 512 %b24_copy_54, i10* nonnull align 512 %b24_copy_55, i10* nonnull align 512 %b24_copy_56, i10* nonnull align 512 %b24_copy_57, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %13, [58 x i10]* nonnull align 512 %w27_copy, [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %14, i10* nonnull align 512 %b27_copy20)
  call void @apatb_myproject_hw(i4368* %cluster_copy21, i16* %nModule_copy22, i16* %x_local_copy23, i16* %y_local_copy24, i8* %layer29_out_copy19, [18 x i10]* %w9_copy, i10* %b9_copy_0, i10* %b9_copy_1, [48 x i10]* %w16_copy, i10* %b16_copy_0, i10* %b16_copy_1, i10* %b16_copy_2, i10* %b16_copy_3, i10* %b16_copy_4, i10* %b16_copy_5, i10* %b16_copy_6, i10* %b16_copy_7, i10* %b16_copy_8, i10* %b16_copy_9, i10* %b16_copy_10, i10* %b16_copy_11, i10* %b16_copy_12, i10* %b16_copy_13, i10* %b16_copy_14, i10* %b16_copy_15, [9792 x i10]* %w21_copy, i10* %b21_copy_0, i10* %b21_copy_1, i10* %b21_copy_2, i10* %b21_copy_3, i10* %b21_copy_4, i10* %b21_copy_5, i10* %b21_copy_6, i10* %b21_copy_7, i10* %b21_copy_8, i10* %b21_copy_9, i10* %b21_copy_10, i10* %b21_copy_11, i10* %b21_copy_12, i10* %b21_copy_13, i10* %b21_copy_14, i10* %b21_copy_15, i10* %b21_copy_16, i10* %b21_copy_17, i10* %b21_copy_18, i10* %b21_copy_19, i10* %b21_copy_20, i10* %b21_copy_21, i10* %b21_copy_22, i10* %b21_copy_23, i10* %b21_copy_24, i10* %b21_copy_25, i10* %b21_copy_26, i10* %b21_copy_27, i10* %b21_copy_28, i10* %b21_copy_29, i10* %b21_copy_30, i10* %b21_copy_31, i10* %b21_copy_32, i10* %b21_copy_33, i10* %b21_copy_34, i10* %b21_copy_35, i10* %b21_copy_36, i10* %b21_copy_37, i10* %b21_copy_38, i10* %b21_copy_39, i10* %b21_copy_40, i10* %b21_copy_41, i10* %b21_copy_42, i10* %b21_copy_43, i10* %b21_copy_44, i10* %b21_copy_45, i10* %b21_copy_46, i10* %b21_copy_47, i10* %b21_copy_48, i10* %b21_copy_49, i10* %b21_copy_50, i10* %b21_copy_51, i10* %b21_copy_52, i10* %b21_copy_53, i10* %b21_copy_54, i10* %b21_copy_55, i10* %b21_copy_56, i10* %b21_copy_57, i10* %b21_copy_58, i10* %b21_copy_59, i10* %b21_copy_60, i10* %b21_copy_61, i10* %b21_copy_62, i10* %b21_copy_63, i10* %b21_copy_64, i10* %b21_copy_65, i10* %b21_copy_66, i10* %b21_copy_67, i10* %b21_copy_68, i10* %b21_copy_69, i10* %b21_copy_70, i10* %b21_copy_71, [4176 x i10]* %w24_copy, i10* %b24_copy_0, i10* %b24_copy_1, i10* %b24_copy_2, i10* %b24_copy_3, i10* %b24_copy_4, i10* %b24_copy_5, i10* %b24_copy_6, i10* %b24_copy_7, i10* %b24_copy_8, i10* %b24_copy_9, i10* %b24_copy_10, i10* %b24_copy_11, i10* %b24_copy_12, i10* %b24_copy_13, i10* %b24_copy_14, i10* %b24_copy_15, i10* %b24_copy_16, i10* %b24_copy_17, i10* %b24_copy_18, i10* %b24_copy_19, i10* %b24_copy_20, i10* %b24_copy_21, i10* %b24_copy_22, i10* %b24_copy_23, i10* %b24_copy_24, i10* %b24_copy_25, i10* %b24_copy_26, i10* %b24_copy_27, i10* %b24_copy_28, i10* %b24_copy_29, i10* %b24_copy_30, i10* %b24_copy_31, i10* %b24_copy_32, i10* %b24_copy_33, i10* %b24_copy_34, i10* %b24_copy_35, i10* %b24_copy_36, i10* %b24_copy_37, i10* %b24_copy_38, i10* %b24_copy_39, i10* %b24_copy_40, i10* %b24_copy_41, i10* %b24_copy_42, i10* %b24_copy_43, i10* %b24_copy_44, i10* %b24_copy_45, i10* %b24_copy_46, i10* %b24_copy_47, i10* %b24_copy_48, i10* %b24_copy_49, i10* %b24_copy_50, i10* %b24_copy_51, i10* %b24_copy_52, i10* %b24_copy_53, i10* %b24_copy_54, i10* %b24_copy_55, i10* %b24_copy_56, i10* %b24_copy_57, [58 x i10]* %w27_copy, i10* %b27_copy20)
  call void @copy_back([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i4368* %cluster_copy21, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %1, i16* %nModule_copy22, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2, i16* %x_local_copy23, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %3, i16* %y_local_copy24, [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %4, i8* %layer29_out_copy19, [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %5, [18 x i10]* %w9_copy, [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %6, i10* %b9_copy_0, i10* %b9_copy_1, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %7, [48 x i10]* %w16_copy, [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %8, i10* %b16_copy_0, i10* %b16_copy_1, i10* %b16_copy_2, i10* %b16_copy_3, i10* %b16_copy_4, i10* %b16_copy_5, i10* %b16_copy_6, i10* %b16_copy_7, i10* %b16_copy_8, i10* %b16_copy_9, i10* %b16_copy_10, i10* %b16_copy_11, i10* %b16_copy_12, i10* %b16_copy_13, i10* %b16_copy_14, i10* %b16_copy_15, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %9, [9792 x i10]* %w21_copy, [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %10, i10* %b21_copy_0, i10* %b21_copy_1, i10* %b21_copy_2, i10* %b21_copy_3, i10* %b21_copy_4, i10* %b21_copy_5, i10* %b21_copy_6, i10* %b21_copy_7, i10* %b21_copy_8, i10* %b21_copy_9, i10* %b21_copy_10, i10* %b21_copy_11, i10* %b21_copy_12, i10* %b21_copy_13, i10* %b21_copy_14, i10* %b21_copy_15, i10* %b21_copy_16, i10* %b21_copy_17, i10* %b21_copy_18, i10* %b21_copy_19, i10* %b21_copy_20, i10* %b21_copy_21, i10* %b21_copy_22, i10* %b21_copy_23, i10* %b21_copy_24, i10* %b21_copy_25, i10* %b21_copy_26, i10* %b21_copy_27, i10* %b21_copy_28, i10* %b21_copy_29, i10* %b21_copy_30, i10* %b21_copy_31, i10* %b21_copy_32, i10* %b21_copy_33, i10* %b21_copy_34, i10* %b21_copy_35, i10* %b21_copy_36, i10* %b21_copy_37, i10* %b21_copy_38, i10* %b21_copy_39, i10* %b21_copy_40, i10* %b21_copy_41, i10* %b21_copy_42, i10* %b21_copy_43, i10* %b21_copy_44, i10* %b21_copy_45, i10* %b21_copy_46, i10* %b21_copy_47, i10* %b21_copy_48, i10* %b21_copy_49, i10* %b21_copy_50, i10* %b21_copy_51, i10* %b21_copy_52, i10* %b21_copy_53, i10* %b21_copy_54, i10* %b21_copy_55, i10* %b21_copy_56, i10* %b21_copy_57, i10* %b21_copy_58, i10* %b21_copy_59, i10* %b21_copy_60, i10* %b21_copy_61, i10* %b21_copy_62, i10* %b21_copy_63, i10* %b21_copy_64, i10* %b21_copy_65, i10* %b21_copy_66, i10* %b21_copy_67, i10* %b21_copy_68, i10* %b21_copy_69, i10* %b21_copy_70, i10* %b21_copy_71, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %11, [4176 x i10]* %w24_copy, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %12, i10* %b24_copy_0, i10* %b24_copy_1, i10* %b24_copy_2, i10* %b24_copy_3, i10* %b24_copy_4, i10* %b24_copy_5, i10* %b24_copy_6, i10* %b24_copy_7, i10* %b24_copy_8, i10* %b24_copy_9, i10* %b24_copy_10, i10* %b24_copy_11, i10* %b24_copy_12, i10* %b24_copy_13, i10* %b24_copy_14, i10* %b24_copy_15, i10* %b24_copy_16, i10* %b24_copy_17, i10* %b24_copy_18, i10* %b24_copy_19, i10* %b24_copy_20, i10* %b24_copy_21, i10* %b24_copy_22, i10* %b24_copy_23, i10* %b24_copy_24, i10* %b24_copy_25, i10* %b24_copy_26, i10* %b24_copy_27, i10* %b24_copy_28, i10* %b24_copy_29, i10* %b24_copy_30, i10* %b24_copy_31, i10* %b24_copy_32, i10* %b24_copy_33, i10* %b24_copy_34, i10* %b24_copy_35, i10* %b24_copy_36, i10* %b24_copy_37, i10* %b24_copy_38, i10* %b24_copy_39, i10* %b24_copy_40, i10* %b24_copy_41, i10* %b24_copy_42, i10* %b24_copy_43, i10* %b24_copy_44, i10* %b24_copy_45, i10* %b24_copy_46, i10* %b24_copy_47, i10* %b24_copy_48, i10* %b24_copy_49, i10* %b24_copy_50, i10* %b24_copy_51, i10* %b24_copy_52, i10* %b24_copy_53, i10* %b24_copy_54, i10* %b24_copy_55, i10* %b24_copy_56, i10* %b24_copy_57, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %13, [58 x i10]* %w27_copy, [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %14, i10* %b27_copy20)
  call void @free(i8* %malloccall)
  call void @free(i8* %malloccall1)
  ret void
}

declare noalias i8* @malloc(i64) local_unnamed_addr

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a18struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "unpacked"="0" %dst, [18 x i10]* noalias nocapture readonly align 512 "unpacked"="1.0" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a18struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, [18 x i10]* %src, i64 18)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a18struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "unpacked"="0" %dst, [18 x i10]* nocapture readonly "unpacked"="1.0" %src, i64 "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [18 x i10], [18 x i10]* %src, i64 0, i64 %for.loop.idx2
  %dst.addr.0.0.06 = getelementptr [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a48struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "unpacked"="0" %dst, [48 x i10]* noalias nocapture readonly align 512 "unpacked"="1.0" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a48struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, [48 x i10]* %src, i64 48)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a48struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "unpacked"="0" %dst, [48 x i10]* nocapture readonly "unpacked"="1.0" %src, i64 "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [48 x i10], [48 x i10]* %src, i64 0, i64 %for.loop.idx2
  %dst.addr.0.0.06 = getelementptr [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a9792struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "unpacked"="0" %dst, [9792 x i10]* noalias nocapture readonly "unpacked"="1.0" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a9792struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, [9792 x i10]* %src, i64 9792)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a9792struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "unpacked"="0" %dst, [9792 x i10]* nocapture readonly "unpacked"="1.0" %src, i64 "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [9792 x i10], [9792 x i10]* %src, i64 0, i64 %for.loop.idx2
  %dst.addr.0.0.06 = getelementptr [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a4176struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "unpacked"="0" %dst, [4176 x i10]* noalias nocapture readonly "unpacked"="1.0" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a4176struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, [4176 x i10]* %src, i64 4176)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a4176struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "unpacked"="0" %dst, [4176 x i10]* nocapture readonly "unpacked"="1.0" %src, i64 "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [4176 x i10], [4176 x i10]* %src, i64 0, i64 %for.loop.idx2
  %dst.addr.0.0.06 = getelementptr [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

declare void @free(i8*) local_unnamed_addr

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([58 x i10]* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.156"([58 x i10]* %dst, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 58)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.156"([58 x i10]* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %dst.addr.0.0.06 = getelementptr [58 x i10], [58 x i10]* %dst, i64 0, i64 %for.loop.idx2
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.160"([58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, [58 x i10]* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.163"([58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, [58 x i10]* %src, i64 58)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.163"([58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, [58 x i10]* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [58 x i10], [58 x i10]* %src, i64 0, i64 %for.loop.idx2
  %dst.addr.0.0.06 = getelementptr [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a4176struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.174"([4176 x i10]* noalias nocapture "unpacked"="0.0" %dst, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "unpacked"="1" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a4176struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.177"([4176 x i10]* %dst, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 4176)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a4176struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.177"([4176 x i10]* nocapture "unpacked"="0.0" %dst, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "unpacked"="1" %src, i64 "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %dst.addr.0.0.06 = getelementptr [4176 x i10], [4176 x i10]* %dst, i64 0, i64 %for.loop.idx2
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a9792struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.198"([9792 x i10]* noalias nocapture "unpacked"="0.0" %dst, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "unpacked"="1" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a9792struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.201"([9792 x i10]* %dst, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 9792)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a9792struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.201"([9792 x i10]* nocapture "unpacked"="0.0" %dst, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "unpacked"="1" %src, i64 "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %dst.addr.0.0.06 = getelementptr [9792 x i10], [9792 x i10]* %dst, i64 0, i64 %for.loop.idx2
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a48struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.222"([48 x i10]* noalias nocapture align 512 "unpacked"="0.0" %dst, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "unpacked"="1" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a48struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.225"([48 x i10]* %dst, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 48)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a48struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.225"([48 x i10]* nocapture "unpacked"="0.0" %dst, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "unpacked"="1" %src, i64 "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %dst.addr.0.0.06 = getelementptr [48 x i10], [48 x i10]* %dst, i64 0, i64 %for.loop.idx2
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a18struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.247"([18 x i10]* noalias nocapture align 512 "unpacked"="0.0" %dst, [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "unpacked"="1" %src) unnamed_addr #1 {
entry:
  %0 = icmp eq [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a18struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.250"([18 x i10]* %dst, [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 18)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a18struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.250"([18 x i10]* nocapture "unpacked"="0.0" %dst, [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "unpacked"="1" %src, i64 "unpacked"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %dst.addr.0.0.06 = getelementptr [18 x i10], [18 x i10]* %dst, i64 0, i64 %for.loop.idx2
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1) #3

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"(i8* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"], [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = load i8, i8* %src.addr.0.0.05, align 1
  store i8 %1, i8* %dst, align 1
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"(i8* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"(i8* %dst, [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* nonnull %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a2struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %dst.addr.0.0.06.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %dst.addr.0.0.06.exit ]
  %src.addr.0.0.05 = getelementptr [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  %cond = icmp eq i64 %for.loop.idx2, 0
  br i1 %cond, label %dst.addr.0.0.06.case.0, label %dst.addr.0.0.06.case.1

dst.addr.0.0.06.case.0:                           ; preds = %for.loop
  store i10 %3, i10* %dst_0, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.1:                           ; preds = %for.loop
  %4 = icmp eq i64 %for.loop.idx2, 1
  call void @llvm.assume(i1 %4)
  store i10 %3, i10* %dst_1, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.exit:                             ; preds = %dst.addr.0.0.06.case.1, %dst.addr.0.0.06.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %dst.addr.0.0.06.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a2struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #1 {
entry:
  %0 = icmp eq [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a2struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* %dst_0, i10* %dst_1, [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 2)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a16struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.2" %dst_2, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.3" %dst_3, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.4" %dst_4, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.5" %dst_5, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.6" %dst_6, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.7" %dst_7, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.8" %dst_8, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.9" %dst_9, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.10" %dst_10, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.11" %dst_11, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.12" %dst_12, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.13" %dst_13, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.14" %dst_14, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.15" %dst_15, [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %dst.addr.0.0.06.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %dst.addr.0.0.06.exit ]
  %src.addr.0.0.05 = getelementptr [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  switch i64 %for.loop.idx2, label %dst.addr.0.0.06.case.15 [
    i64 0, label %dst.addr.0.0.06.case.0
    i64 1, label %dst.addr.0.0.06.case.1
    i64 2, label %dst.addr.0.0.06.case.2
    i64 3, label %dst.addr.0.0.06.case.3
    i64 4, label %dst.addr.0.0.06.case.4
    i64 5, label %dst.addr.0.0.06.case.5
    i64 6, label %dst.addr.0.0.06.case.6
    i64 7, label %dst.addr.0.0.06.case.7
    i64 8, label %dst.addr.0.0.06.case.8
    i64 9, label %dst.addr.0.0.06.case.9
    i64 10, label %dst.addr.0.0.06.case.10
    i64 11, label %dst.addr.0.0.06.case.11
    i64 12, label %dst.addr.0.0.06.case.12
    i64 13, label %dst.addr.0.0.06.case.13
    i64 14, label %dst.addr.0.0.06.case.14
  ]

dst.addr.0.0.06.case.0:                           ; preds = %for.loop
  store i10 %3, i10* %dst_0, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.1:                           ; preds = %for.loop
  store i10 %3, i10* %dst_1, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.2:                           ; preds = %for.loop
  store i10 %3, i10* %dst_2, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.3:                           ; preds = %for.loop
  store i10 %3, i10* %dst_3, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.4:                           ; preds = %for.loop
  store i10 %3, i10* %dst_4, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.5:                           ; preds = %for.loop
  store i10 %3, i10* %dst_5, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.6:                           ; preds = %for.loop
  store i10 %3, i10* %dst_6, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.7:                           ; preds = %for.loop
  store i10 %3, i10* %dst_7, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.8:                           ; preds = %for.loop
  store i10 %3, i10* %dst_8, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.9:                           ; preds = %for.loop
  store i10 %3, i10* %dst_9, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.10:                          ; preds = %for.loop
  store i10 %3, i10* %dst_10, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.11:                          ; preds = %for.loop
  store i10 %3, i10* %dst_11, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.12:                          ; preds = %for.loop
  store i10 %3, i10* %dst_12, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.13:                          ; preds = %for.loop
  store i10 %3, i10* %dst_13, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.14:                          ; preds = %for.loop
  store i10 %3, i10* %dst_14, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.15:                          ; preds = %for.loop
  %4 = icmp eq i64 %for.loop.idx2, 15
  call void @llvm.assume(i1 %4)
  store i10 %3, i10* %dst_15, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.exit:                             ; preds = %dst.addr.0.0.06.case.15, %dst.addr.0.0.06.case.14, %dst.addr.0.0.06.case.13, %dst.addr.0.0.06.case.12, %dst.addr.0.0.06.case.11, %dst.addr.0.0.06.case.10, %dst.addr.0.0.06.case.9, %dst.addr.0.0.06.case.8, %dst.addr.0.0.06.case.7, %dst.addr.0.0.06.case.6, %dst.addr.0.0.06.case.5, %dst.addr.0.0.06.case.4, %dst.addr.0.0.06.case.3, %dst.addr.0.0.06.case.2, %dst.addr.0.0.06.case.1, %dst.addr.0.0.06.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %dst.addr.0.0.06.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a16struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.2" %dst_2, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.3" %dst_3, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.4" %dst_4, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.5" %dst_5, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.6" %dst_6, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.7" %dst_7, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.8" %dst_8, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.9" %dst_9, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.10" %dst_10, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.11" %dst_11, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.12" %dst_12, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.13" %dst_13, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.14" %dst_14, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.15" %dst_15, [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #1 {
entry:
  %0 = icmp eq [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a16struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* %dst_0, i10* %dst_1, i10* %dst_2, i10* %dst_3, i10* %dst_4, i10* %dst_5, i10* %dst_6, i10* %dst_7, i10* %dst_8, i10* %dst_9, i10* %dst_10, i10* %dst_11, i10* %dst_12, i10* %dst_13, i10* %dst_14, i10* %dst_15, [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 16)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a72struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.2" %dst_2, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.3" %dst_3, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.4" %dst_4, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.5" %dst_5, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.6" %dst_6, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.7" %dst_7, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.8" %dst_8, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.9" %dst_9, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.10" %dst_10, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.11" %dst_11, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.12" %dst_12, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.13" %dst_13, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.14" %dst_14, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.15" %dst_15, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.16" %dst_16, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.17" %dst_17, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.18" %dst_18, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.19" %dst_19, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.20" %dst_20, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.21" %dst_21, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.22" %dst_22, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.23" %dst_23, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.24" %dst_24, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.25" %dst_25, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.26" %dst_26, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.27" %dst_27, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.28" %dst_28, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.29" %dst_29, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.30" %dst_30, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.31" %dst_31, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.32" %dst_32, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.33" %dst_33, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.34" %dst_34, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.35" %dst_35, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.36" %dst_36, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.37" %dst_37, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.38" %dst_38, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.39" %dst_39, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.40" %dst_40, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.41" %dst_41, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.42" %dst_42, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.43" %dst_43, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.44" %dst_44, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.45" %dst_45, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.46" %dst_46, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.47" %dst_47, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.48" %dst_48, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.49" %dst_49, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.50" %dst_50, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.51" %dst_51, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.52" %dst_52, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.53" %dst_53, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.54" %dst_54, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.55" %dst_55, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.56" %dst_56, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.57" %dst_57, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.58" %dst_58, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.59" %dst_59, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.60" %dst_60, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.61" %dst_61, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.62" %dst_62, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.63" %dst_63, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.64" %dst_64, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.65" %dst_65, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.66" %dst_66, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.67" %dst_67, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.68" %dst_68, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.69" %dst_69, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.70" %dst_70, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.71" %dst_71, [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %dst.addr.0.0.06.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %dst.addr.0.0.06.exit ]
  %src.addr.0.0.05 = getelementptr [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  switch i64 %for.loop.idx2, label %dst.addr.0.0.06.case.71 [
    i64 0, label %dst.addr.0.0.06.case.0
    i64 1, label %dst.addr.0.0.06.case.1
    i64 2, label %dst.addr.0.0.06.case.2
    i64 3, label %dst.addr.0.0.06.case.3
    i64 4, label %dst.addr.0.0.06.case.4
    i64 5, label %dst.addr.0.0.06.case.5
    i64 6, label %dst.addr.0.0.06.case.6
    i64 7, label %dst.addr.0.0.06.case.7
    i64 8, label %dst.addr.0.0.06.case.8
    i64 9, label %dst.addr.0.0.06.case.9
    i64 10, label %dst.addr.0.0.06.case.10
    i64 11, label %dst.addr.0.0.06.case.11
    i64 12, label %dst.addr.0.0.06.case.12
    i64 13, label %dst.addr.0.0.06.case.13
    i64 14, label %dst.addr.0.0.06.case.14
    i64 15, label %dst.addr.0.0.06.case.15
    i64 16, label %dst.addr.0.0.06.case.16
    i64 17, label %dst.addr.0.0.06.case.17
    i64 18, label %dst.addr.0.0.06.case.18
    i64 19, label %dst.addr.0.0.06.case.19
    i64 20, label %dst.addr.0.0.06.case.20
    i64 21, label %dst.addr.0.0.06.case.21
    i64 22, label %dst.addr.0.0.06.case.22
    i64 23, label %dst.addr.0.0.06.case.23
    i64 24, label %dst.addr.0.0.06.case.24
    i64 25, label %dst.addr.0.0.06.case.25
    i64 26, label %dst.addr.0.0.06.case.26
    i64 27, label %dst.addr.0.0.06.case.27
    i64 28, label %dst.addr.0.0.06.case.28
    i64 29, label %dst.addr.0.0.06.case.29
    i64 30, label %dst.addr.0.0.06.case.30
    i64 31, label %dst.addr.0.0.06.case.31
    i64 32, label %dst.addr.0.0.06.case.32
    i64 33, label %dst.addr.0.0.06.case.33
    i64 34, label %dst.addr.0.0.06.case.34
    i64 35, label %dst.addr.0.0.06.case.35
    i64 36, label %dst.addr.0.0.06.case.36
    i64 37, label %dst.addr.0.0.06.case.37
    i64 38, label %dst.addr.0.0.06.case.38
    i64 39, label %dst.addr.0.0.06.case.39
    i64 40, label %dst.addr.0.0.06.case.40
    i64 41, label %dst.addr.0.0.06.case.41
    i64 42, label %dst.addr.0.0.06.case.42
    i64 43, label %dst.addr.0.0.06.case.43
    i64 44, label %dst.addr.0.0.06.case.44
    i64 45, label %dst.addr.0.0.06.case.45
    i64 46, label %dst.addr.0.0.06.case.46
    i64 47, label %dst.addr.0.0.06.case.47
    i64 48, label %dst.addr.0.0.06.case.48
    i64 49, label %dst.addr.0.0.06.case.49
    i64 50, label %dst.addr.0.0.06.case.50
    i64 51, label %dst.addr.0.0.06.case.51
    i64 52, label %dst.addr.0.0.06.case.52
    i64 53, label %dst.addr.0.0.06.case.53
    i64 54, label %dst.addr.0.0.06.case.54
    i64 55, label %dst.addr.0.0.06.case.55
    i64 56, label %dst.addr.0.0.06.case.56
    i64 57, label %dst.addr.0.0.06.case.57
    i64 58, label %dst.addr.0.0.06.case.58
    i64 59, label %dst.addr.0.0.06.case.59
    i64 60, label %dst.addr.0.0.06.case.60
    i64 61, label %dst.addr.0.0.06.case.61
    i64 62, label %dst.addr.0.0.06.case.62
    i64 63, label %dst.addr.0.0.06.case.63
    i64 64, label %dst.addr.0.0.06.case.64
    i64 65, label %dst.addr.0.0.06.case.65
    i64 66, label %dst.addr.0.0.06.case.66
    i64 67, label %dst.addr.0.0.06.case.67
    i64 68, label %dst.addr.0.0.06.case.68
    i64 69, label %dst.addr.0.0.06.case.69
    i64 70, label %dst.addr.0.0.06.case.70
  ]

dst.addr.0.0.06.case.0:                           ; preds = %for.loop
  store i10 %3, i10* %dst_0, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.1:                           ; preds = %for.loop
  store i10 %3, i10* %dst_1, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.2:                           ; preds = %for.loop
  store i10 %3, i10* %dst_2, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.3:                           ; preds = %for.loop
  store i10 %3, i10* %dst_3, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.4:                           ; preds = %for.loop
  store i10 %3, i10* %dst_4, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.5:                           ; preds = %for.loop
  store i10 %3, i10* %dst_5, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.6:                           ; preds = %for.loop
  store i10 %3, i10* %dst_6, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.7:                           ; preds = %for.loop
  store i10 %3, i10* %dst_7, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.8:                           ; preds = %for.loop
  store i10 %3, i10* %dst_8, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.9:                           ; preds = %for.loop
  store i10 %3, i10* %dst_9, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.10:                          ; preds = %for.loop
  store i10 %3, i10* %dst_10, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.11:                          ; preds = %for.loop
  store i10 %3, i10* %dst_11, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.12:                          ; preds = %for.loop
  store i10 %3, i10* %dst_12, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.13:                          ; preds = %for.loop
  store i10 %3, i10* %dst_13, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.14:                          ; preds = %for.loop
  store i10 %3, i10* %dst_14, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.15:                          ; preds = %for.loop
  store i10 %3, i10* %dst_15, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.16:                          ; preds = %for.loop
  store i10 %3, i10* %dst_16, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.17:                          ; preds = %for.loop
  store i10 %3, i10* %dst_17, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.18:                          ; preds = %for.loop
  store i10 %3, i10* %dst_18, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.19:                          ; preds = %for.loop
  store i10 %3, i10* %dst_19, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.20:                          ; preds = %for.loop
  store i10 %3, i10* %dst_20, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.21:                          ; preds = %for.loop
  store i10 %3, i10* %dst_21, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.22:                          ; preds = %for.loop
  store i10 %3, i10* %dst_22, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.23:                          ; preds = %for.loop
  store i10 %3, i10* %dst_23, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.24:                          ; preds = %for.loop
  store i10 %3, i10* %dst_24, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.25:                          ; preds = %for.loop
  store i10 %3, i10* %dst_25, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.26:                          ; preds = %for.loop
  store i10 %3, i10* %dst_26, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.27:                          ; preds = %for.loop
  store i10 %3, i10* %dst_27, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.28:                          ; preds = %for.loop
  store i10 %3, i10* %dst_28, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.29:                          ; preds = %for.loop
  store i10 %3, i10* %dst_29, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.30:                          ; preds = %for.loop
  store i10 %3, i10* %dst_30, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.31:                          ; preds = %for.loop
  store i10 %3, i10* %dst_31, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.32:                          ; preds = %for.loop
  store i10 %3, i10* %dst_32, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.33:                          ; preds = %for.loop
  store i10 %3, i10* %dst_33, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.34:                          ; preds = %for.loop
  store i10 %3, i10* %dst_34, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.35:                          ; preds = %for.loop
  store i10 %3, i10* %dst_35, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.36:                          ; preds = %for.loop
  store i10 %3, i10* %dst_36, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.37:                          ; preds = %for.loop
  store i10 %3, i10* %dst_37, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.38:                          ; preds = %for.loop
  store i10 %3, i10* %dst_38, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.39:                          ; preds = %for.loop
  store i10 %3, i10* %dst_39, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.40:                          ; preds = %for.loop
  store i10 %3, i10* %dst_40, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.41:                          ; preds = %for.loop
  store i10 %3, i10* %dst_41, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.42:                          ; preds = %for.loop
  store i10 %3, i10* %dst_42, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.43:                          ; preds = %for.loop
  store i10 %3, i10* %dst_43, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.44:                          ; preds = %for.loop
  store i10 %3, i10* %dst_44, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.45:                          ; preds = %for.loop
  store i10 %3, i10* %dst_45, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.46:                          ; preds = %for.loop
  store i10 %3, i10* %dst_46, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.47:                          ; preds = %for.loop
  store i10 %3, i10* %dst_47, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.48:                          ; preds = %for.loop
  store i10 %3, i10* %dst_48, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.49:                          ; preds = %for.loop
  store i10 %3, i10* %dst_49, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.50:                          ; preds = %for.loop
  store i10 %3, i10* %dst_50, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.51:                          ; preds = %for.loop
  store i10 %3, i10* %dst_51, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.52:                          ; preds = %for.loop
  store i10 %3, i10* %dst_52, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.53:                          ; preds = %for.loop
  store i10 %3, i10* %dst_53, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.54:                          ; preds = %for.loop
  store i10 %3, i10* %dst_54, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.55:                          ; preds = %for.loop
  store i10 %3, i10* %dst_55, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.56:                          ; preds = %for.loop
  store i10 %3, i10* %dst_56, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.57:                          ; preds = %for.loop
  store i10 %3, i10* %dst_57, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.58:                          ; preds = %for.loop
  store i10 %3, i10* %dst_58, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.59:                          ; preds = %for.loop
  store i10 %3, i10* %dst_59, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.60:                          ; preds = %for.loop
  store i10 %3, i10* %dst_60, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.61:                          ; preds = %for.loop
  store i10 %3, i10* %dst_61, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.62:                          ; preds = %for.loop
  store i10 %3, i10* %dst_62, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.63:                          ; preds = %for.loop
  store i10 %3, i10* %dst_63, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.64:                          ; preds = %for.loop
  store i10 %3, i10* %dst_64, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.65:                          ; preds = %for.loop
  store i10 %3, i10* %dst_65, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.66:                          ; preds = %for.loop
  store i10 %3, i10* %dst_66, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.67:                          ; preds = %for.loop
  store i10 %3, i10* %dst_67, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.68:                          ; preds = %for.loop
  store i10 %3, i10* %dst_68, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.69:                          ; preds = %for.loop
  store i10 %3, i10* %dst_69, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.70:                          ; preds = %for.loop
  store i10 %3, i10* %dst_70, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.71:                          ; preds = %for.loop
  %4 = icmp eq i64 %for.loop.idx2, 71
  call void @llvm.assume(i1 %4)
  store i10 %3, i10* %dst_71, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.exit:                             ; preds = %dst.addr.0.0.06.case.71, %dst.addr.0.0.06.case.70, %dst.addr.0.0.06.case.69, %dst.addr.0.0.06.case.68, %dst.addr.0.0.06.case.67, %dst.addr.0.0.06.case.66, %dst.addr.0.0.06.case.65, %dst.addr.0.0.06.case.64, %dst.addr.0.0.06.case.63, %dst.addr.0.0.06.case.62, %dst.addr.0.0.06.case.61, %dst.addr.0.0.06.case.60, %dst.addr.0.0.06.case.59, %dst.addr.0.0.06.case.58, %dst.addr.0.0.06.case.57, %dst.addr.0.0.06.case.56, %dst.addr.0.0.06.case.55, %dst.addr.0.0.06.case.54, %dst.addr.0.0.06.case.53, %dst.addr.0.0.06.case.52, %dst.addr.0.0.06.case.51, %dst.addr.0.0.06.case.50, %dst.addr.0.0.06.case.49, %dst.addr.0.0.06.case.48, %dst.addr.0.0.06.case.47, %dst.addr.0.0.06.case.46, %dst.addr.0.0.06.case.45, %dst.addr.0.0.06.case.44, %dst.addr.0.0.06.case.43, %dst.addr.0.0.06.case.42, %dst.addr.0.0.06.case.41, %dst.addr.0.0.06.case.40, %dst.addr.0.0.06.case.39, %dst.addr.0.0.06.case.38, %dst.addr.0.0.06.case.37, %dst.addr.0.0.06.case.36, %dst.addr.0.0.06.case.35, %dst.addr.0.0.06.case.34, %dst.addr.0.0.06.case.33, %dst.addr.0.0.06.case.32, %dst.addr.0.0.06.case.31, %dst.addr.0.0.06.case.30, %dst.addr.0.0.06.case.29, %dst.addr.0.0.06.case.28, %dst.addr.0.0.06.case.27, %dst.addr.0.0.06.case.26, %dst.addr.0.0.06.case.25, %dst.addr.0.0.06.case.24, %dst.addr.0.0.06.case.23, %dst.addr.0.0.06.case.22, %dst.addr.0.0.06.case.21, %dst.addr.0.0.06.case.20, %dst.addr.0.0.06.case.19, %dst.addr.0.0.06.case.18, %dst.addr.0.0.06.case.17, %dst.addr.0.0.06.case.16, %dst.addr.0.0.06.case.15, %dst.addr.0.0.06.case.14, %dst.addr.0.0.06.case.13, %dst.addr.0.0.06.case.12, %dst.addr.0.0.06.case.11, %dst.addr.0.0.06.case.10, %dst.addr.0.0.06.case.9, %dst.addr.0.0.06.case.8, %dst.addr.0.0.06.case.7, %dst.addr.0.0.06.case.6, %dst.addr.0.0.06.case.5, %dst.addr.0.0.06.case.4, %dst.addr.0.0.06.case.3, %dst.addr.0.0.06.case.2, %dst.addr.0.0.06.case.1, %dst.addr.0.0.06.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %dst.addr.0.0.06.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a72struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.2" %dst_2, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.3" %dst_3, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.4" %dst_4, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.5" %dst_5, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.6" %dst_6, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.7" %dst_7, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.8" %dst_8, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.9" %dst_9, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.10" %dst_10, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.11" %dst_11, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.12" %dst_12, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.13" %dst_13, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.14" %dst_14, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.15" %dst_15, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.16" %dst_16, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.17" %dst_17, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.18" %dst_18, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.19" %dst_19, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.20" %dst_20, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.21" %dst_21, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.22" %dst_22, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.23" %dst_23, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.24" %dst_24, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.25" %dst_25, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.26" %dst_26, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.27" %dst_27, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.28" %dst_28, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.29" %dst_29, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.30" %dst_30, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.31" %dst_31, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.32" %dst_32, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.33" %dst_33, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.34" %dst_34, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.35" %dst_35, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.36" %dst_36, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.37" %dst_37, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.38" %dst_38, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.39" %dst_39, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.40" %dst_40, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.41" %dst_41, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.42" %dst_42, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.43" %dst_43, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.44" %dst_44, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.45" %dst_45, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.46" %dst_46, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.47" %dst_47, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.48" %dst_48, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.49" %dst_49, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.50" %dst_50, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.51" %dst_51, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.52" %dst_52, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.53" %dst_53, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.54" %dst_54, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.55" %dst_55, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.56" %dst_56, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.57" %dst_57, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.58" %dst_58, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.59" %dst_59, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.60" %dst_60, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.61" %dst_61, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.62" %dst_62, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.63" %dst_63, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.64" %dst_64, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.65" %dst_65, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.66" %dst_66, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.67" %dst_67, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.68" %dst_68, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.69" %dst_69, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.70" %dst_70, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.71" %dst_71, [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #1 {
entry:
  %0 = icmp eq [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a72struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* %dst_0, i10* %dst_1, i10* %dst_2, i10* %dst_3, i10* %dst_4, i10* %dst_5, i10* %dst_6, i10* %dst_7, i10* %dst_8, i10* %dst_9, i10* %dst_10, i10* %dst_11, i10* %dst_12, i10* %dst_13, i10* %dst_14, i10* %dst_15, i10* %dst_16, i10* %dst_17, i10* %dst_18, i10* %dst_19, i10* %dst_20, i10* %dst_21, i10* %dst_22, i10* %dst_23, i10* %dst_24, i10* %dst_25, i10* %dst_26, i10* %dst_27, i10* %dst_28, i10* %dst_29, i10* %dst_30, i10* %dst_31, i10* %dst_32, i10* %dst_33, i10* %dst_34, i10* %dst_35, i10* %dst_36, i10* %dst_37, i10* %dst_38, i10* %dst_39, i10* %dst_40, i10* %dst_41, i10* %dst_42, i10* %dst_43, i10* %dst_44, i10* %dst_45, i10* %dst_46, i10* %dst_47, i10* %dst_48, i10* %dst_49, i10* %dst_50, i10* %dst_51, i10* %dst_52, i10* %dst_53, i10* %dst_54, i10* %dst_55, i10* %dst_56, i10* %dst_57, i10* %dst_58, i10* %dst_59, i10* %dst_60, i10* %dst_61, i10* %dst_62, i10* %dst_63, i10* %dst_64, i10* %dst_65, i10* %dst_66, i10* %dst_67, i10* %dst_68, i10* %dst_69, i10* %dst_70, i10* %dst_71, [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 72)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.156.384.385"(i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.2" %dst_2, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.3" %dst_3, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.4" %dst_4, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.5" %dst_5, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.6" %dst_6, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.7" %dst_7, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.8" %dst_8, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.9" %dst_9, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.10" %dst_10, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.11" %dst_11, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.12" %dst_12, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.13" %dst_13, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.14" %dst_14, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.15" %dst_15, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.16" %dst_16, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.17" %dst_17, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.18" %dst_18, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.19" %dst_19, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.20" %dst_20, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.21" %dst_21, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.22" %dst_22, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.23" %dst_23, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.24" %dst_24, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.25" %dst_25, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.26" %dst_26, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.27" %dst_27, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.28" %dst_28, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.29" %dst_29, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.30" %dst_30, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.31" %dst_31, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.32" %dst_32, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.33" %dst_33, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.34" %dst_34, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.35" %dst_35, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.36" %dst_36, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.37" %dst_37, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.38" %dst_38, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.39" %dst_39, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.40" %dst_40, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.41" %dst_41, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.42" %dst_42, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.43" %dst_43, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.44" %dst_44, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.45" %dst_45, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.46" %dst_46, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.47" %dst_47, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.48" %dst_48, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.49" %dst_49, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.50" %dst_50, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.51" %dst_51, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.52" %dst_52, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.53" %dst_53, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.54" %dst_54, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.55" %dst_55, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.56" %dst_56, i10* nocapture "orig.arg.no"="0" "unpacked"="0.0.57" %dst_57, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %dst.addr.0.0.06.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %dst.addr.0.0.06.exit ]
  %src.addr.0.0.05 = getelementptr [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  switch i64 %for.loop.idx2, label %dst.addr.0.0.06.case.57 [
    i64 0, label %dst.addr.0.0.06.case.0
    i64 1, label %dst.addr.0.0.06.case.1
    i64 2, label %dst.addr.0.0.06.case.2
    i64 3, label %dst.addr.0.0.06.case.3
    i64 4, label %dst.addr.0.0.06.case.4
    i64 5, label %dst.addr.0.0.06.case.5
    i64 6, label %dst.addr.0.0.06.case.6
    i64 7, label %dst.addr.0.0.06.case.7
    i64 8, label %dst.addr.0.0.06.case.8
    i64 9, label %dst.addr.0.0.06.case.9
    i64 10, label %dst.addr.0.0.06.case.10
    i64 11, label %dst.addr.0.0.06.case.11
    i64 12, label %dst.addr.0.0.06.case.12
    i64 13, label %dst.addr.0.0.06.case.13
    i64 14, label %dst.addr.0.0.06.case.14
    i64 15, label %dst.addr.0.0.06.case.15
    i64 16, label %dst.addr.0.0.06.case.16
    i64 17, label %dst.addr.0.0.06.case.17
    i64 18, label %dst.addr.0.0.06.case.18
    i64 19, label %dst.addr.0.0.06.case.19
    i64 20, label %dst.addr.0.0.06.case.20
    i64 21, label %dst.addr.0.0.06.case.21
    i64 22, label %dst.addr.0.0.06.case.22
    i64 23, label %dst.addr.0.0.06.case.23
    i64 24, label %dst.addr.0.0.06.case.24
    i64 25, label %dst.addr.0.0.06.case.25
    i64 26, label %dst.addr.0.0.06.case.26
    i64 27, label %dst.addr.0.0.06.case.27
    i64 28, label %dst.addr.0.0.06.case.28
    i64 29, label %dst.addr.0.0.06.case.29
    i64 30, label %dst.addr.0.0.06.case.30
    i64 31, label %dst.addr.0.0.06.case.31
    i64 32, label %dst.addr.0.0.06.case.32
    i64 33, label %dst.addr.0.0.06.case.33
    i64 34, label %dst.addr.0.0.06.case.34
    i64 35, label %dst.addr.0.0.06.case.35
    i64 36, label %dst.addr.0.0.06.case.36
    i64 37, label %dst.addr.0.0.06.case.37
    i64 38, label %dst.addr.0.0.06.case.38
    i64 39, label %dst.addr.0.0.06.case.39
    i64 40, label %dst.addr.0.0.06.case.40
    i64 41, label %dst.addr.0.0.06.case.41
    i64 42, label %dst.addr.0.0.06.case.42
    i64 43, label %dst.addr.0.0.06.case.43
    i64 44, label %dst.addr.0.0.06.case.44
    i64 45, label %dst.addr.0.0.06.case.45
    i64 46, label %dst.addr.0.0.06.case.46
    i64 47, label %dst.addr.0.0.06.case.47
    i64 48, label %dst.addr.0.0.06.case.48
    i64 49, label %dst.addr.0.0.06.case.49
    i64 50, label %dst.addr.0.0.06.case.50
    i64 51, label %dst.addr.0.0.06.case.51
    i64 52, label %dst.addr.0.0.06.case.52
    i64 53, label %dst.addr.0.0.06.case.53
    i64 54, label %dst.addr.0.0.06.case.54
    i64 55, label %dst.addr.0.0.06.case.55
    i64 56, label %dst.addr.0.0.06.case.56
  ]

dst.addr.0.0.06.case.0:                           ; preds = %for.loop
  store i10 %3, i10* %dst_0, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.1:                           ; preds = %for.loop
  store i10 %3, i10* %dst_1, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.2:                           ; preds = %for.loop
  store i10 %3, i10* %dst_2, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.3:                           ; preds = %for.loop
  store i10 %3, i10* %dst_3, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.4:                           ; preds = %for.loop
  store i10 %3, i10* %dst_4, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.5:                           ; preds = %for.loop
  store i10 %3, i10* %dst_5, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.6:                           ; preds = %for.loop
  store i10 %3, i10* %dst_6, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.7:                           ; preds = %for.loop
  store i10 %3, i10* %dst_7, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.8:                           ; preds = %for.loop
  store i10 %3, i10* %dst_8, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.9:                           ; preds = %for.loop
  store i10 %3, i10* %dst_9, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.10:                          ; preds = %for.loop
  store i10 %3, i10* %dst_10, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.11:                          ; preds = %for.loop
  store i10 %3, i10* %dst_11, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.12:                          ; preds = %for.loop
  store i10 %3, i10* %dst_12, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.13:                          ; preds = %for.loop
  store i10 %3, i10* %dst_13, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.14:                          ; preds = %for.loop
  store i10 %3, i10* %dst_14, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.15:                          ; preds = %for.loop
  store i10 %3, i10* %dst_15, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.16:                          ; preds = %for.loop
  store i10 %3, i10* %dst_16, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.17:                          ; preds = %for.loop
  store i10 %3, i10* %dst_17, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.18:                          ; preds = %for.loop
  store i10 %3, i10* %dst_18, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.19:                          ; preds = %for.loop
  store i10 %3, i10* %dst_19, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.20:                          ; preds = %for.loop
  store i10 %3, i10* %dst_20, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.21:                          ; preds = %for.loop
  store i10 %3, i10* %dst_21, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.22:                          ; preds = %for.loop
  store i10 %3, i10* %dst_22, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.23:                          ; preds = %for.loop
  store i10 %3, i10* %dst_23, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.24:                          ; preds = %for.loop
  store i10 %3, i10* %dst_24, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.25:                          ; preds = %for.loop
  store i10 %3, i10* %dst_25, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.26:                          ; preds = %for.loop
  store i10 %3, i10* %dst_26, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.27:                          ; preds = %for.loop
  store i10 %3, i10* %dst_27, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.28:                          ; preds = %for.loop
  store i10 %3, i10* %dst_28, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.29:                          ; preds = %for.loop
  store i10 %3, i10* %dst_29, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.30:                          ; preds = %for.loop
  store i10 %3, i10* %dst_30, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.31:                          ; preds = %for.loop
  store i10 %3, i10* %dst_31, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.32:                          ; preds = %for.loop
  store i10 %3, i10* %dst_32, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.33:                          ; preds = %for.loop
  store i10 %3, i10* %dst_33, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.34:                          ; preds = %for.loop
  store i10 %3, i10* %dst_34, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.35:                          ; preds = %for.loop
  store i10 %3, i10* %dst_35, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.36:                          ; preds = %for.loop
  store i10 %3, i10* %dst_36, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.37:                          ; preds = %for.loop
  store i10 %3, i10* %dst_37, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.38:                          ; preds = %for.loop
  store i10 %3, i10* %dst_38, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.39:                          ; preds = %for.loop
  store i10 %3, i10* %dst_39, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.40:                          ; preds = %for.loop
  store i10 %3, i10* %dst_40, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.41:                          ; preds = %for.loop
  store i10 %3, i10* %dst_41, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.42:                          ; preds = %for.loop
  store i10 %3, i10* %dst_42, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.43:                          ; preds = %for.loop
  store i10 %3, i10* %dst_43, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.44:                          ; preds = %for.loop
  store i10 %3, i10* %dst_44, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.45:                          ; preds = %for.loop
  store i10 %3, i10* %dst_45, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.46:                          ; preds = %for.loop
  store i10 %3, i10* %dst_46, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.47:                          ; preds = %for.loop
  store i10 %3, i10* %dst_47, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.48:                          ; preds = %for.loop
  store i10 %3, i10* %dst_48, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.49:                          ; preds = %for.loop
  store i10 %3, i10* %dst_49, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.50:                          ; preds = %for.loop
  store i10 %3, i10* %dst_50, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.51:                          ; preds = %for.loop
  store i10 %3, i10* %dst_51, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.52:                          ; preds = %for.loop
  store i10 %3, i10* %dst_52, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.53:                          ; preds = %for.loop
  store i10 %3, i10* %dst_53, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.54:                          ; preds = %for.loop
  store i10 %3, i10* %dst_54, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.55:                          ; preds = %for.loop
  store i10 %3, i10* %dst_55, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.56:                          ; preds = %for.loop
  store i10 %3, i10* %dst_56, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.57:                          ; preds = %for.loop
  %4 = icmp eq i64 %for.loop.idx2, 57
  call void @llvm.assume(i1 %4)
  store i10 %3, i10* %dst_57, align 2
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.exit:                             ; preds = %dst.addr.0.0.06.case.57, %dst.addr.0.0.06.case.56, %dst.addr.0.0.06.case.55, %dst.addr.0.0.06.case.54, %dst.addr.0.0.06.case.53, %dst.addr.0.0.06.case.52, %dst.addr.0.0.06.case.51, %dst.addr.0.0.06.case.50, %dst.addr.0.0.06.case.49, %dst.addr.0.0.06.case.48, %dst.addr.0.0.06.case.47, %dst.addr.0.0.06.case.46, %dst.addr.0.0.06.case.45, %dst.addr.0.0.06.case.44, %dst.addr.0.0.06.case.43, %dst.addr.0.0.06.case.42, %dst.addr.0.0.06.case.41, %dst.addr.0.0.06.case.40, %dst.addr.0.0.06.case.39, %dst.addr.0.0.06.case.38, %dst.addr.0.0.06.case.37, %dst.addr.0.0.06.case.36, %dst.addr.0.0.06.case.35, %dst.addr.0.0.06.case.34, %dst.addr.0.0.06.case.33, %dst.addr.0.0.06.case.32, %dst.addr.0.0.06.case.31, %dst.addr.0.0.06.case.30, %dst.addr.0.0.06.case.29, %dst.addr.0.0.06.case.28, %dst.addr.0.0.06.case.27, %dst.addr.0.0.06.case.26, %dst.addr.0.0.06.case.25, %dst.addr.0.0.06.case.24, %dst.addr.0.0.06.case.23, %dst.addr.0.0.06.case.22, %dst.addr.0.0.06.case.21, %dst.addr.0.0.06.case.20, %dst.addr.0.0.06.case.19, %dst.addr.0.0.06.case.18, %dst.addr.0.0.06.case.17, %dst.addr.0.0.06.case.16, %dst.addr.0.0.06.case.15, %dst.addr.0.0.06.case.14, %dst.addr.0.0.06.case.13, %dst.addr.0.0.06.case.12, %dst.addr.0.0.06.case.11, %dst.addr.0.0.06.case.10, %dst.addr.0.0.06.case.9, %dst.addr.0.0.06.case.8, %dst.addr.0.0.06.case.7, %dst.addr.0.0.06.case.6, %dst.addr.0.0.06.case.5, %dst.addr.0.0.06.case.4, %dst.addr.0.0.06.case.3, %dst.addr.0.0.06.case.2, %dst.addr.0.0.06.case.1, %dst.addr.0.0.06.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %dst.addr.0.0.06.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.383.386"(i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.2" %dst_2, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.3" %dst_3, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.4" %dst_4, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.5" %dst_5, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.6" %dst_6, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.7" %dst_7, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.8" %dst_8, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.9" %dst_9, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.10" %dst_10, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.11" %dst_11, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.12" %dst_12, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.13" %dst_13, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.14" %dst_14, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.15" %dst_15, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.16" %dst_16, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.17" %dst_17, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.18" %dst_18, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.19" %dst_19, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.20" %dst_20, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.21" %dst_21, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.22" %dst_22, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.23" %dst_23, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.24" %dst_24, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.25" %dst_25, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.26" %dst_26, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.27" %dst_27, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.28" %dst_28, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.29" %dst_29, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.30" %dst_30, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.31" %dst_31, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.32" %dst_32, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.33" %dst_33, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.34" %dst_34, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.35" %dst_35, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.36" %dst_36, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.37" %dst_37, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.38" %dst_38, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.39" %dst_39, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.40" %dst_40, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.41" %dst_41, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.42" %dst_42, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.43" %dst_43, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.44" %dst_44, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.45" %dst_45, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.46" %dst_46, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.47" %dst_47, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.48" %dst_48, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.49" %dst_49, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.50" %dst_50, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.51" %dst_51, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.52" %dst_52, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.53" %dst_53, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.54" %dst_54, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.55" %dst_55, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.56" %dst_56, i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.57" %dst_57, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #1 {
entry:
  %0 = icmp eq [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.156.384.385"(i10* %dst_0, i10* %dst_1, i10* %dst_2, i10* %dst_3, i10* %dst_4, i10* %dst_5, i10* %dst_6, i10* %dst_7, i10* %dst_8, i10* %dst_9, i10* %dst_10, i10* %dst_11, i10* %dst_12, i10* %dst_13, i10* %dst_14, i10* %dst_15, i10* %dst_16, i10* %dst_17, i10* %dst_18, i10* %dst_19, i10* %dst_20, i10* %dst_21, i10* %dst_22, i10* %dst_23, i10* %dst_24, i10* %dst_25, i10* %dst_26, i10* %dst_27, i10* %dst_28, i10* %dst_29, i10* %dst_30, i10* %dst_31, i10* %dst_32, i10* %dst_33, i10* %dst_34, i10* %dst_35, i10* %dst_36, i10* %dst_37, i10* %dst_38, i10* %dst_39, i10* %dst_40, i10* %dst_41, i10* %dst_42, i10* %dst_43, i10* %dst_44, i10* %dst_45, i10* %dst_46, i10* %dst_47, i10* %dst_48, i10* %dst_49, i10* %dst_50, i10* %dst_51, i10* %dst_52, i10* %dst_53, i10* %dst_54, i10* %dst_55, i10* %dst_56, i10* %dst_57, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 58)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src.addr.0.0.05 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* %dst, [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>.263"([1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i8* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr.0.0.06 = getelementptr [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"], [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = load i8, i8* %src, align 1
  store i8 %1, i8* %dst.addr.0.0.06, align 1
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>.260"([1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i8* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>.263"([1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* nonnull %dst, i8* %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a2struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.237"([2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.1" %src_1, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %src.addr.0.0.05.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %src.addr.0.0.05.exit ]
  %dst.addr.0.0.06 = getelementptr [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %cond = icmp eq i64 %for.loop.idx2, 0
  br i1 %cond, label %src.addr.0.0.05.case.0, label %src.addr.0.0.05.case.1

src.addr.0.0.05.case.0:                           ; preds = %for.loop
  %1 = bitcast i10* %src_0 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.1:                           ; preds = %for.loop
  %4 = icmp eq i64 %for.loop.idx2, 1
  call void @llvm.assume(i1 %4)
  %5 = bitcast i10* %src_1 to i16*
  %6 = load i16, i16* %5
  %7 = trunc i16 %6 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.exit:                             ; preds = %src.addr.0.0.05.case.1, %src.addr.0.0.05.case.0
  %8 = phi i10 [ %3, %src.addr.0.0.05.case.0 ], [ %7, %src.addr.0.0.05.case.1 ]
  store i10 %8, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %src.addr.0.0.05.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a2struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.234"([2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.1" %src_1) #1 {
entry:
  %0 = icmp eq [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a2struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.237"([2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i10* %src_0, i10* %src_1, i64 2)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a16struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.213"([16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.1" %src_1, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.2" %src_2, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.3" %src_3, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.4" %src_4, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.5" %src_5, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.6" %src_6, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.7" %src_7, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.8" %src_8, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.9" %src_9, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.10" %src_10, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.11" %src_11, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.12" %src_12, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.13" %src_13, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.14" %src_14, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.15" %src_15, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %src.addr.0.0.05.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %src.addr.0.0.05.exit ]
  %dst.addr.0.0.06 = getelementptr [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  switch i64 %for.loop.idx2, label %src.addr.0.0.05.case.15 [
    i64 0, label %src.addr.0.0.05.case.0
    i64 1, label %src.addr.0.0.05.case.1
    i64 2, label %src.addr.0.0.05.case.2
    i64 3, label %src.addr.0.0.05.case.3
    i64 4, label %src.addr.0.0.05.case.4
    i64 5, label %src.addr.0.0.05.case.5
    i64 6, label %src.addr.0.0.05.case.6
    i64 7, label %src.addr.0.0.05.case.7
    i64 8, label %src.addr.0.0.05.case.8
    i64 9, label %src.addr.0.0.05.case.9
    i64 10, label %src.addr.0.0.05.case.10
    i64 11, label %src.addr.0.0.05.case.11
    i64 12, label %src.addr.0.0.05.case.12
    i64 13, label %src.addr.0.0.05.case.13
    i64 14, label %src.addr.0.0.05.case.14
  ]

src.addr.0.0.05.case.0:                           ; preds = %for.loop
  %1 = bitcast i10* %src_0 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.1:                           ; preds = %for.loop
  %4 = bitcast i10* %src_1 to i16*
  %5 = load i16, i16* %4
  %6 = trunc i16 %5 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.2:                           ; preds = %for.loop
  %7 = bitcast i10* %src_2 to i16*
  %8 = load i16, i16* %7
  %9 = trunc i16 %8 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.3:                           ; preds = %for.loop
  %10 = bitcast i10* %src_3 to i16*
  %11 = load i16, i16* %10
  %12 = trunc i16 %11 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.4:                           ; preds = %for.loop
  %13 = bitcast i10* %src_4 to i16*
  %14 = load i16, i16* %13
  %15 = trunc i16 %14 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.5:                           ; preds = %for.loop
  %16 = bitcast i10* %src_5 to i16*
  %17 = load i16, i16* %16
  %18 = trunc i16 %17 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.6:                           ; preds = %for.loop
  %19 = bitcast i10* %src_6 to i16*
  %20 = load i16, i16* %19
  %21 = trunc i16 %20 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.7:                           ; preds = %for.loop
  %22 = bitcast i10* %src_7 to i16*
  %23 = load i16, i16* %22
  %24 = trunc i16 %23 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.8:                           ; preds = %for.loop
  %25 = bitcast i10* %src_8 to i16*
  %26 = load i16, i16* %25
  %27 = trunc i16 %26 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.9:                           ; preds = %for.loop
  %28 = bitcast i10* %src_9 to i16*
  %29 = load i16, i16* %28
  %30 = trunc i16 %29 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.10:                          ; preds = %for.loop
  %31 = bitcast i10* %src_10 to i16*
  %32 = load i16, i16* %31
  %33 = trunc i16 %32 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.11:                          ; preds = %for.loop
  %34 = bitcast i10* %src_11 to i16*
  %35 = load i16, i16* %34
  %36 = trunc i16 %35 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.12:                          ; preds = %for.loop
  %37 = bitcast i10* %src_12 to i16*
  %38 = load i16, i16* %37
  %39 = trunc i16 %38 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.13:                          ; preds = %for.loop
  %40 = bitcast i10* %src_13 to i16*
  %41 = load i16, i16* %40
  %42 = trunc i16 %41 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.14:                          ; preds = %for.loop
  %43 = bitcast i10* %src_14 to i16*
  %44 = load i16, i16* %43
  %45 = trunc i16 %44 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.15:                          ; preds = %for.loop
  %46 = icmp eq i64 %for.loop.idx2, 15
  call void @llvm.assume(i1 %46)
  %47 = bitcast i10* %src_15 to i16*
  %48 = load i16, i16* %47
  %49 = trunc i16 %48 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.exit:                             ; preds = %src.addr.0.0.05.case.15, %src.addr.0.0.05.case.14, %src.addr.0.0.05.case.13, %src.addr.0.0.05.case.12, %src.addr.0.0.05.case.11, %src.addr.0.0.05.case.10, %src.addr.0.0.05.case.9, %src.addr.0.0.05.case.8, %src.addr.0.0.05.case.7, %src.addr.0.0.05.case.6, %src.addr.0.0.05.case.5, %src.addr.0.0.05.case.4, %src.addr.0.0.05.case.3, %src.addr.0.0.05.case.2, %src.addr.0.0.05.case.1, %src.addr.0.0.05.case.0
  %50 = phi i10 [ %3, %src.addr.0.0.05.case.0 ], [ %6, %src.addr.0.0.05.case.1 ], [ %9, %src.addr.0.0.05.case.2 ], [ %12, %src.addr.0.0.05.case.3 ], [ %15, %src.addr.0.0.05.case.4 ], [ %18, %src.addr.0.0.05.case.5 ], [ %21, %src.addr.0.0.05.case.6 ], [ %24, %src.addr.0.0.05.case.7 ], [ %27, %src.addr.0.0.05.case.8 ], [ %30, %src.addr.0.0.05.case.9 ], [ %33, %src.addr.0.0.05.case.10 ], [ %36, %src.addr.0.0.05.case.11 ], [ %39, %src.addr.0.0.05.case.12 ], [ %42, %src.addr.0.0.05.case.13 ], [ %45, %src.addr.0.0.05.case.14 ], [ %49, %src.addr.0.0.05.case.15 ]
  store i10 %50, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %src.addr.0.0.05.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a16struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.210"([16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.1" %src_1, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.2" %src_2, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.3" %src_3, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.4" %src_4, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.5" %src_5, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.6" %src_6, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.7" %src_7, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.8" %src_8, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.9" %src_9, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.10" %src_10, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.11" %src_11, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.12" %src_12, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.13" %src_13, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.14" %src_14, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.15" %src_15) #1 {
entry:
  %0 = icmp eq [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a16struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.213"([16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i10* %src_0, i10* %src_1, i10* %src_2, i10* %src_3, i10* %src_4, i10* %src_5, i10* %src_6, i10* %src_7, i10* %src_8, i10* %src_9, i10* %src_10, i10* %src_11, i10* %src_12, i10* %src_13, i10* %src_14, i10* %src_15, i64 16)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a72struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.189"([72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.1" %src_1, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.2" %src_2, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.3" %src_3, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.4" %src_4, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.5" %src_5, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.6" %src_6, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.7" %src_7, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.8" %src_8, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.9" %src_9, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.10" %src_10, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.11" %src_11, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.12" %src_12, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.13" %src_13, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.14" %src_14, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.15" %src_15, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.16" %src_16, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.17" %src_17, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.18" %src_18, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.19" %src_19, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.20" %src_20, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.21" %src_21, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.22" %src_22, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.23" %src_23, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.24" %src_24, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.25" %src_25, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.26" %src_26, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.27" %src_27, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.28" %src_28, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.29" %src_29, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.30" %src_30, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.31" %src_31, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.32" %src_32, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.33" %src_33, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.34" %src_34, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.35" %src_35, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.36" %src_36, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.37" %src_37, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.38" %src_38, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.39" %src_39, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.40" %src_40, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.41" %src_41, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.42" %src_42, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.43" %src_43, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.44" %src_44, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.45" %src_45, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.46" %src_46, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.47" %src_47, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.48" %src_48, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.49" %src_49, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.50" %src_50, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.51" %src_51, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.52" %src_52, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.53" %src_53, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.54" %src_54, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.55" %src_55, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.56" %src_56, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.57" %src_57, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.58" %src_58, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.59" %src_59, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.60" %src_60, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.61" %src_61, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.62" %src_62, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.63" %src_63, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.64" %src_64, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.65" %src_65, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.66" %src_66, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.67" %src_67, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.68" %src_68, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.69" %src_69, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.70" %src_70, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.71" %src_71, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %src.addr.0.0.05.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %src.addr.0.0.05.exit ]
  %dst.addr.0.0.06 = getelementptr [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  switch i64 %for.loop.idx2, label %src.addr.0.0.05.case.71 [
    i64 0, label %src.addr.0.0.05.case.0
    i64 1, label %src.addr.0.0.05.case.1
    i64 2, label %src.addr.0.0.05.case.2
    i64 3, label %src.addr.0.0.05.case.3
    i64 4, label %src.addr.0.0.05.case.4
    i64 5, label %src.addr.0.0.05.case.5
    i64 6, label %src.addr.0.0.05.case.6
    i64 7, label %src.addr.0.0.05.case.7
    i64 8, label %src.addr.0.0.05.case.8
    i64 9, label %src.addr.0.0.05.case.9
    i64 10, label %src.addr.0.0.05.case.10
    i64 11, label %src.addr.0.0.05.case.11
    i64 12, label %src.addr.0.0.05.case.12
    i64 13, label %src.addr.0.0.05.case.13
    i64 14, label %src.addr.0.0.05.case.14
    i64 15, label %src.addr.0.0.05.case.15
    i64 16, label %src.addr.0.0.05.case.16
    i64 17, label %src.addr.0.0.05.case.17
    i64 18, label %src.addr.0.0.05.case.18
    i64 19, label %src.addr.0.0.05.case.19
    i64 20, label %src.addr.0.0.05.case.20
    i64 21, label %src.addr.0.0.05.case.21
    i64 22, label %src.addr.0.0.05.case.22
    i64 23, label %src.addr.0.0.05.case.23
    i64 24, label %src.addr.0.0.05.case.24
    i64 25, label %src.addr.0.0.05.case.25
    i64 26, label %src.addr.0.0.05.case.26
    i64 27, label %src.addr.0.0.05.case.27
    i64 28, label %src.addr.0.0.05.case.28
    i64 29, label %src.addr.0.0.05.case.29
    i64 30, label %src.addr.0.0.05.case.30
    i64 31, label %src.addr.0.0.05.case.31
    i64 32, label %src.addr.0.0.05.case.32
    i64 33, label %src.addr.0.0.05.case.33
    i64 34, label %src.addr.0.0.05.case.34
    i64 35, label %src.addr.0.0.05.case.35
    i64 36, label %src.addr.0.0.05.case.36
    i64 37, label %src.addr.0.0.05.case.37
    i64 38, label %src.addr.0.0.05.case.38
    i64 39, label %src.addr.0.0.05.case.39
    i64 40, label %src.addr.0.0.05.case.40
    i64 41, label %src.addr.0.0.05.case.41
    i64 42, label %src.addr.0.0.05.case.42
    i64 43, label %src.addr.0.0.05.case.43
    i64 44, label %src.addr.0.0.05.case.44
    i64 45, label %src.addr.0.0.05.case.45
    i64 46, label %src.addr.0.0.05.case.46
    i64 47, label %src.addr.0.0.05.case.47
    i64 48, label %src.addr.0.0.05.case.48
    i64 49, label %src.addr.0.0.05.case.49
    i64 50, label %src.addr.0.0.05.case.50
    i64 51, label %src.addr.0.0.05.case.51
    i64 52, label %src.addr.0.0.05.case.52
    i64 53, label %src.addr.0.0.05.case.53
    i64 54, label %src.addr.0.0.05.case.54
    i64 55, label %src.addr.0.0.05.case.55
    i64 56, label %src.addr.0.0.05.case.56
    i64 57, label %src.addr.0.0.05.case.57
    i64 58, label %src.addr.0.0.05.case.58
    i64 59, label %src.addr.0.0.05.case.59
    i64 60, label %src.addr.0.0.05.case.60
    i64 61, label %src.addr.0.0.05.case.61
    i64 62, label %src.addr.0.0.05.case.62
    i64 63, label %src.addr.0.0.05.case.63
    i64 64, label %src.addr.0.0.05.case.64
    i64 65, label %src.addr.0.0.05.case.65
    i64 66, label %src.addr.0.0.05.case.66
    i64 67, label %src.addr.0.0.05.case.67
    i64 68, label %src.addr.0.0.05.case.68
    i64 69, label %src.addr.0.0.05.case.69
    i64 70, label %src.addr.0.0.05.case.70
  ]

src.addr.0.0.05.case.0:                           ; preds = %for.loop
  %1 = bitcast i10* %src_0 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.1:                           ; preds = %for.loop
  %4 = bitcast i10* %src_1 to i16*
  %5 = load i16, i16* %4
  %6 = trunc i16 %5 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.2:                           ; preds = %for.loop
  %7 = bitcast i10* %src_2 to i16*
  %8 = load i16, i16* %7
  %9 = trunc i16 %8 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.3:                           ; preds = %for.loop
  %10 = bitcast i10* %src_3 to i16*
  %11 = load i16, i16* %10
  %12 = trunc i16 %11 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.4:                           ; preds = %for.loop
  %13 = bitcast i10* %src_4 to i16*
  %14 = load i16, i16* %13
  %15 = trunc i16 %14 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.5:                           ; preds = %for.loop
  %16 = bitcast i10* %src_5 to i16*
  %17 = load i16, i16* %16
  %18 = trunc i16 %17 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.6:                           ; preds = %for.loop
  %19 = bitcast i10* %src_6 to i16*
  %20 = load i16, i16* %19
  %21 = trunc i16 %20 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.7:                           ; preds = %for.loop
  %22 = bitcast i10* %src_7 to i16*
  %23 = load i16, i16* %22
  %24 = trunc i16 %23 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.8:                           ; preds = %for.loop
  %25 = bitcast i10* %src_8 to i16*
  %26 = load i16, i16* %25
  %27 = trunc i16 %26 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.9:                           ; preds = %for.loop
  %28 = bitcast i10* %src_9 to i16*
  %29 = load i16, i16* %28
  %30 = trunc i16 %29 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.10:                          ; preds = %for.loop
  %31 = bitcast i10* %src_10 to i16*
  %32 = load i16, i16* %31
  %33 = trunc i16 %32 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.11:                          ; preds = %for.loop
  %34 = bitcast i10* %src_11 to i16*
  %35 = load i16, i16* %34
  %36 = trunc i16 %35 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.12:                          ; preds = %for.loop
  %37 = bitcast i10* %src_12 to i16*
  %38 = load i16, i16* %37
  %39 = trunc i16 %38 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.13:                          ; preds = %for.loop
  %40 = bitcast i10* %src_13 to i16*
  %41 = load i16, i16* %40
  %42 = trunc i16 %41 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.14:                          ; preds = %for.loop
  %43 = bitcast i10* %src_14 to i16*
  %44 = load i16, i16* %43
  %45 = trunc i16 %44 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.15:                          ; preds = %for.loop
  %46 = bitcast i10* %src_15 to i16*
  %47 = load i16, i16* %46
  %48 = trunc i16 %47 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.16:                          ; preds = %for.loop
  %49 = bitcast i10* %src_16 to i16*
  %50 = load i16, i16* %49
  %51 = trunc i16 %50 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.17:                          ; preds = %for.loop
  %52 = bitcast i10* %src_17 to i16*
  %53 = load i16, i16* %52
  %54 = trunc i16 %53 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.18:                          ; preds = %for.loop
  %55 = bitcast i10* %src_18 to i16*
  %56 = load i16, i16* %55
  %57 = trunc i16 %56 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.19:                          ; preds = %for.loop
  %58 = bitcast i10* %src_19 to i16*
  %59 = load i16, i16* %58
  %60 = trunc i16 %59 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.20:                          ; preds = %for.loop
  %61 = bitcast i10* %src_20 to i16*
  %62 = load i16, i16* %61
  %63 = trunc i16 %62 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.21:                          ; preds = %for.loop
  %64 = bitcast i10* %src_21 to i16*
  %65 = load i16, i16* %64
  %66 = trunc i16 %65 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.22:                          ; preds = %for.loop
  %67 = bitcast i10* %src_22 to i16*
  %68 = load i16, i16* %67
  %69 = trunc i16 %68 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.23:                          ; preds = %for.loop
  %70 = bitcast i10* %src_23 to i16*
  %71 = load i16, i16* %70
  %72 = trunc i16 %71 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.24:                          ; preds = %for.loop
  %73 = bitcast i10* %src_24 to i16*
  %74 = load i16, i16* %73
  %75 = trunc i16 %74 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.25:                          ; preds = %for.loop
  %76 = bitcast i10* %src_25 to i16*
  %77 = load i16, i16* %76
  %78 = trunc i16 %77 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.26:                          ; preds = %for.loop
  %79 = bitcast i10* %src_26 to i16*
  %80 = load i16, i16* %79
  %81 = trunc i16 %80 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.27:                          ; preds = %for.loop
  %82 = bitcast i10* %src_27 to i16*
  %83 = load i16, i16* %82
  %84 = trunc i16 %83 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.28:                          ; preds = %for.loop
  %85 = bitcast i10* %src_28 to i16*
  %86 = load i16, i16* %85
  %87 = trunc i16 %86 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.29:                          ; preds = %for.loop
  %88 = bitcast i10* %src_29 to i16*
  %89 = load i16, i16* %88
  %90 = trunc i16 %89 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.30:                          ; preds = %for.loop
  %91 = bitcast i10* %src_30 to i16*
  %92 = load i16, i16* %91
  %93 = trunc i16 %92 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.31:                          ; preds = %for.loop
  %94 = bitcast i10* %src_31 to i16*
  %95 = load i16, i16* %94
  %96 = trunc i16 %95 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.32:                          ; preds = %for.loop
  %97 = bitcast i10* %src_32 to i16*
  %98 = load i16, i16* %97
  %99 = trunc i16 %98 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.33:                          ; preds = %for.loop
  %100 = bitcast i10* %src_33 to i16*
  %101 = load i16, i16* %100
  %102 = trunc i16 %101 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.34:                          ; preds = %for.loop
  %103 = bitcast i10* %src_34 to i16*
  %104 = load i16, i16* %103
  %105 = trunc i16 %104 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.35:                          ; preds = %for.loop
  %106 = bitcast i10* %src_35 to i16*
  %107 = load i16, i16* %106
  %108 = trunc i16 %107 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.36:                          ; preds = %for.loop
  %109 = bitcast i10* %src_36 to i16*
  %110 = load i16, i16* %109
  %111 = trunc i16 %110 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.37:                          ; preds = %for.loop
  %112 = bitcast i10* %src_37 to i16*
  %113 = load i16, i16* %112
  %114 = trunc i16 %113 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.38:                          ; preds = %for.loop
  %115 = bitcast i10* %src_38 to i16*
  %116 = load i16, i16* %115
  %117 = trunc i16 %116 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.39:                          ; preds = %for.loop
  %118 = bitcast i10* %src_39 to i16*
  %119 = load i16, i16* %118
  %120 = trunc i16 %119 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.40:                          ; preds = %for.loop
  %121 = bitcast i10* %src_40 to i16*
  %122 = load i16, i16* %121
  %123 = trunc i16 %122 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.41:                          ; preds = %for.loop
  %124 = bitcast i10* %src_41 to i16*
  %125 = load i16, i16* %124
  %126 = trunc i16 %125 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.42:                          ; preds = %for.loop
  %127 = bitcast i10* %src_42 to i16*
  %128 = load i16, i16* %127
  %129 = trunc i16 %128 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.43:                          ; preds = %for.loop
  %130 = bitcast i10* %src_43 to i16*
  %131 = load i16, i16* %130
  %132 = trunc i16 %131 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.44:                          ; preds = %for.loop
  %133 = bitcast i10* %src_44 to i16*
  %134 = load i16, i16* %133
  %135 = trunc i16 %134 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.45:                          ; preds = %for.loop
  %136 = bitcast i10* %src_45 to i16*
  %137 = load i16, i16* %136
  %138 = trunc i16 %137 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.46:                          ; preds = %for.loop
  %139 = bitcast i10* %src_46 to i16*
  %140 = load i16, i16* %139
  %141 = trunc i16 %140 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.47:                          ; preds = %for.loop
  %142 = bitcast i10* %src_47 to i16*
  %143 = load i16, i16* %142
  %144 = trunc i16 %143 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.48:                          ; preds = %for.loop
  %145 = bitcast i10* %src_48 to i16*
  %146 = load i16, i16* %145
  %147 = trunc i16 %146 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.49:                          ; preds = %for.loop
  %148 = bitcast i10* %src_49 to i16*
  %149 = load i16, i16* %148
  %150 = trunc i16 %149 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.50:                          ; preds = %for.loop
  %151 = bitcast i10* %src_50 to i16*
  %152 = load i16, i16* %151
  %153 = trunc i16 %152 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.51:                          ; preds = %for.loop
  %154 = bitcast i10* %src_51 to i16*
  %155 = load i16, i16* %154
  %156 = trunc i16 %155 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.52:                          ; preds = %for.loop
  %157 = bitcast i10* %src_52 to i16*
  %158 = load i16, i16* %157
  %159 = trunc i16 %158 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.53:                          ; preds = %for.loop
  %160 = bitcast i10* %src_53 to i16*
  %161 = load i16, i16* %160
  %162 = trunc i16 %161 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.54:                          ; preds = %for.loop
  %163 = bitcast i10* %src_54 to i16*
  %164 = load i16, i16* %163
  %165 = trunc i16 %164 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.55:                          ; preds = %for.loop
  %166 = bitcast i10* %src_55 to i16*
  %167 = load i16, i16* %166
  %168 = trunc i16 %167 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.56:                          ; preds = %for.loop
  %169 = bitcast i10* %src_56 to i16*
  %170 = load i16, i16* %169
  %171 = trunc i16 %170 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.57:                          ; preds = %for.loop
  %172 = bitcast i10* %src_57 to i16*
  %173 = load i16, i16* %172
  %174 = trunc i16 %173 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.58:                          ; preds = %for.loop
  %175 = bitcast i10* %src_58 to i16*
  %176 = load i16, i16* %175
  %177 = trunc i16 %176 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.59:                          ; preds = %for.loop
  %178 = bitcast i10* %src_59 to i16*
  %179 = load i16, i16* %178
  %180 = trunc i16 %179 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.60:                          ; preds = %for.loop
  %181 = bitcast i10* %src_60 to i16*
  %182 = load i16, i16* %181
  %183 = trunc i16 %182 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.61:                          ; preds = %for.loop
  %184 = bitcast i10* %src_61 to i16*
  %185 = load i16, i16* %184
  %186 = trunc i16 %185 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.62:                          ; preds = %for.loop
  %187 = bitcast i10* %src_62 to i16*
  %188 = load i16, i16* %187
  %189 = trunc i16 %188 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.63:                          ; preds = %for.loop
  %190 = bitcast i10* %src_63 to i16*
  %191 = load i16, i16* %190
  %192 = trunc i16 %191 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.64:                          ; preds = %for.loop
  %193 = bitcast i10* %src_64 to i16*
  %194 = load i16, i16* %193
  %195 = trunc i16 %194 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.65:                          ; preds = %for.loop
  %196 = bitcast i10* %src_65 to i16*
  %197 = load i16, i16* %196
  %198 = trunc i16 %197 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.66:                          ; preds = %for.loop
  %199 = bitcast i10* %src_66 to i16*
  %200 = load i16, i16* %199
  %201 = trunc i16 %200 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.67:                          ; preds = %for.loop
  %202 = bitcast i10* %src_67 to i16*
  %203 = load i16, i16* %202
  %204 = trunc i16 %203 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.68:                          ; preds = %for.loop
  %205 = bitcast i10* %src_68 to i16*
  %206 = load i16, i16* %205
  %207 = trunc i16 %206 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.69:                          ; preds = %for.loop
  %208 = bitcast i10* %src_69 to i16*
  %209 = load i16, i16* %208
  %210 = trunc i16 %209 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.70:                          ; preds = %for.loop
  %211 = bitcast i10* %src_70 to i16*
  %212 = load i16, i16* %211
  %213 = trunc i16 %212 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.71:                          ; preds = %for.loop
  %214 = icmp eq i64 %for.loop.idx2, 71
  call void @llvm.assume(i1 %214)
  %215 = bitcast i10* %src_71 to i16*
  %216 = load i16, i16* %215
  %217 = trunc i16 %216 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.exit:                             ; preds = %src.addr.0.0.05.case.71, %src.addr.0.0.05.case.70, %src.addr.0.0.05.case.69, %src.addr.0.0.05.case.68, %src.addr.0.0.05.case.67, %src.addr.0.0.05.case.66, %src.addr.0.0.05.case.65, %src.addr.0.0.05.case.64, %src.addr.0.0.05.case.63, %src.addr.0.0.05.case.62, %src.addr.0.0.05.case.61, %src.addr.0.0.05.case.60, %src.addr.0.0.05.case.59, %src.addr.0.0.05.case.58, %src.addr.0.0.05.case.57, %src.addr.0.0.05.case.56, %src.addr.0.0.05.case.55, %src.addr.0.0.05.case.54, %src.addr.0.0.05.case.53, %src.addr.0.0.05.case.52, %src.addr.0.0.05.case.51, %src.addr.0.0.05.case.50, %src.addr.0.0.05.case.49, %src.addr.0.0.05.case.48, %src.addr.0.0.05.case.47, %src.addr.0.0.05.case.46, %src.addr.0.0.05.case.45, %src.addr.0.0.05.case.44, %src.addr.0.0.05.case.43, %src.addr.0.0.05.case.42, %src.addr.0.0.05.case.41, %src.addr.0.0.05.case.40, %src.addr.0.0.05.case.39, %src.addr.0.0.05.case.38, %src.addr.0.0.05.case.37, %src.addr.0.0.05.case.36, %src.addr.0.0.05.case.35, %src.addr.0.0.05.case.34, %src.addr.0.0.05.case.33, %src.addr.0.0.05.case.32, %src.addr.0.0.05.case.31, %src.addr.0.0.05.case.30, %src.addr.0.0.05.case.29, %src.addr.0.0.05.case.28, %src.addr.0.0.05.case.27, %src.addr.0.0.05.case.26, %src.addr.0.0.05.case.25, %src.addr.0.0.05.case.24, %src.addr.0.0.05.case.23, %src.addr.0.0.05.case.22, %src.addr.0.0.05.case.21, %src.addr.0.0.05.case.20, %src.addr.0.0.05.case.19, %src.addr.0.0.05.case.18, %src.addr.0.0.05.case.17, %src.addr.0.0.05.case.16, %src.addr.0.0.05.case.15, %src.addr.0.0.05.case.14, %src.addr.0.0.05.case.13, %src.addr.0.0.05.case.12, %src.addr.0.0.05.case.11, %src.addr.0.0.05.case.10, %src.addr.0.0.05.case.9, %src.addr.0.0.05.case.8, %src.addr.0.0.05.case.7, %src.addr.0.0.05.case.6, %src.addr.0.0.05.case.5, %src.addr.0.0.05.case.4, %src.addr.0.0.05.case.3, %src.addr.0.0.05.case.2, %src.addr.0.0.05.case.1, %src.addr.0.0.05.case.0
  %218 = phi i10 [ %3, %src.addr.0.0.05.case.0 ], [ %6, %src.addr.0.0.05.case.1 ], [ %9, %src.addr.0.0.05.case.2 ], [ %12, %src.addr.0.0.05.case.3 ], [ %15, %src.addr.0.0.05.case.4 ], [ %18, %src.addr.0.0.05.case.5 ], [ %21, %src.addr.0.0.05.case.6 ], [ %24, %src.addr.0.0.05.case.7 ], [ %27, %src.addr.0.0.05.case.8 ], [ %30, %src.addr.0.0.05.case.9 ], [ %33, %src.addr.0.0.05.case.10 ], [ %36, %src.addr.0.0.05.case.11 ], [ %39, %src.addr.0.0.05.case.12 ], [ %42, %src.addr.0.0.05.case.13 ], [ %45, %src.addr.0.0.05.case.14 ], [ %48, %src.addr.0.0.05.case.15 ], [ %51, %src.addr.0.0.05.case.16 ], [ %54, %src.addr.0.0.05.case.17 ], [ %57, %src.addr.0.0.05.case.18 ], [ %60, %src.addr.0.0.05.case.19 ], [ %63, %src.addr.0.0.05.case.20 ], [ %66, %src.addr.0.0.05.case.21 ], [ %69, %src.addr.0.0.05.case.22 ], [ %72, %src.addr.0.0.05.case.23 ], [ %75, %src.addr.0.0.05.case.24 ], [ %78, %src.addr.0.0.05.case.25 ], [ %81, %src.addr.0.0.05.case.26 ], [ %84, %src.addr.0.0.05.case.27 ], [ %87, %src.addr.0.0.05.case.28 ], [ %90, %src.addr.0.0.05.case.29 ], [ %93, %src.addr.0.0.05.case.30 ], [ %96, %src.addr.0.0.05.case.31 ], [ %99, %src.addr.0.0.05.case.32 ], [ %102, %src.addr.0.0.05.case.33 ], [ %105, %src.addr.0.0.05.case.34 ], [ %108, %src.addr.0.0.05.case.35 ], [ %111, %src.addr.0.0.05.case.36 ], [ %114, %src.addr.0.0.05.case.37 ], [ %117, %src.addr.0.0.05.case.38 ], [ %120, %src.addr.0.0.05.case.39 ], [ %123, %src.addr.0.0.05.case.40 ], [ %126, %src.addr.0.0.05.case.41 ], [ %129, %src.addr.0.0.05.case.42 ], [ %132, %src.addr.0.0.05.case.43 ], [ %135, %src.addr.0.0.05.case.44 ], [ %138, %src.addr.0.0.05.case.45 ], [ %141, %src.addr.0.0.05.case.46 ], [ %144, %src.addr.0.0.05.case.47 ], [ %147, %src.addr.0.0.05.case.48 ], [ %150, %src.addr.0.0.05.case.49 ], [ %153, %src.addr.0.0.05.case.50 ], [ %156, %src.addr.0.0.05.case.51 ], [ %159, %src.addr.0.0.05.case.52 ], [ %162, %src.addr.0.0.05.case.53 ], [ %165, %src.addr.0.0.05.case.54 ], [ %168, %src.addr.0.0.05.case.55 ], [ %171, %src.addr.0.0.05.case.56 ], [ %174, %src.addr.0.0.05.case.57 ], [ %177, %src.addr.0.0.05.case.58 ], [ %180, %src.addr.0.0.05.case.59 ], [ %183, %src.addr.0.0.05.case.60 ], [ %186, %src.addr.0.0.05.case.61 ], [ %189, %src.addr.0.0.05.case.62 ], [ %192, %src.addr.0.0.05.case.63 ], [ %195, %src.addr.0.0.05.case.64 ], [ %198, %src.addr.0.0.05.case.65 ], [ %201, %src.addr.0.0.05.case.66 ], [ %204, %src.addr.0.0.05.case.67 ], [ %207, %src.addr.0.0.05.case.68 ], [ %210, %src.addr.0.0.05.case.69 ], [ %213, %src.addr.0.0.05.case.70 ], [ %217, %src.addr.0.0.05.case.71 ]
  store i10 %218, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %src.addr.0.0.05.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a72struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.186"([72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.1" %src_1, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.2" %src_2, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.3" %src_3, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.4" %src_4, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.5" %src_5, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.6" %src_6, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.7" %src_7, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.8" %src_8, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.9" %src_9, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.10" %src_10, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.11" %src_11, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.12" %src_12, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.13" %src_13, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.14" %src_14, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.15" %src_15, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.16" %src_16, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.17" %src_17, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.18" %src_18, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.19" %src_19, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.20" %src_20, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.21" %src_21, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.22" %src_22, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.23" %src_23, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.24" %src_24, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.25" %src_25, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.26" %src_26, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.27" %src_27, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.28" %src_28, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.29" %src_29, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.30" %src_30, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.31" %src_31, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.32" %src_32, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.33" %src_33, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.34" %src_34, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.35" %src_35, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.36" %src_36, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.37" %src_37, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.38" %src_38, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.39" %src_39, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.40" %src_40, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.41" %src_41, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.42" %src_42, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.43" %src_43, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.44" %src_44, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.45" %src_45, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.46" %src_46, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.47" %src_47, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.48" %src_48, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.49" %src_49, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.50" %src_50, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.51" %src_51, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.52" %src_52, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.53" %src_53, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.54" %src_54, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.55" %src_55, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.56" %src_56, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.57" %src_57, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.58" %src_58, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.59" %src_59, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.60" %src_60, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.61" %src_61, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.62" %src_62, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.63" %src_63, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.64" %src_64, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.65" %src_65, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.66" %src_66, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.67" %src_67, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.68" %src_68, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.69" %src_69, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.70" %src_70, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.71" %src_71) #1 {
entry:
  %0 = icmp eq [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a72struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.189"([72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i10* %src_0, i10* %src_1, i10* %src_2, i10* %src_3, i10* %src_4, i10* %src_5, i10* %src_6, i10* %src_7, i10* %src_8, i10* %src_9, i10* %src_10, i10* %src_11, i10* %src_12, i10* %src_13, i10* %src_14, i10* %src_15, i10* %src_16, i10* %src_17, i10* %src_18, i10* %src_19, i10* %src_20, i10* %src_21, i10* %src_22, i10* %src_23, i10* %src_24, i10* %src_25, i10* %src_26, i10* %src_27, i10* %src_28, i10* %src_29, i10* %src_30, i10* %src_31, i10* %src_32, i10* %src_33, i10* %src_34, i10* %src_35, i10* %src_36, i10* %src_37, i10* %src_38, i10* %src_39, i10* %src_40, i10* %src_41, i10* %src_42, i10* %src_43, i10* %src_44, i10* %src_45, i10* %src_46, i10* %src_47, i10* %src_48, i10* %src_49, i10* %src_50, i10* %src_51, i10* %src_52, i10* %src_53, i10* %src_54, i10* %src_55, i10* %src_56, i10* %src_57, i10* %src_58, i10* %src_59, i10* %src_60, i10* %src_61, i10* %src_62, i10* %src_63, i10* %src_64, i10* %src_65, i10* %src_66, i10* %src_67, i10* %src_68, i10* %src_69, i10* %src_70, i10* %src_71, i64 72)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.163.424.425"([58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.1" %src_1, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.2" %src_2, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.3" %src_3, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.4" %src_4, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.5" %src_5, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.6" %src_6, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.7" %src_7, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.8" %src_8, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.9" %src_9, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.10" %src_10, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.11" %src_11, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.12" %src_12, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.13" %src_13, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.14" %src_14, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.15" %src_15, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.16" %src_16, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.17" %src_17, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.18" %src_18, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.19" %src_19, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.20" %src_20, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.21" %src_21, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.22" %src_22, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.23" %src_23, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.24" %src_24, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.25" %src_25, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.26" %src_26, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.27" %src_27, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.28" %src_28, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.29" %src_29, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.30" %src_30, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.31" %src_31, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.32" %src_32, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.33" %src_33, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.34" %src_34, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.35" %src_35, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.36" %src_36, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.37" %src_37, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.38" %src_38, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.39" %src_39, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.40" %src_40, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.41" %src_41, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.42" %src_42, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.43" %src_43, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.44" %src_44, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.45" %src_45, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.46" %src_46, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.47" %src_47, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.48" %src_48, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.49" %src_49, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.50" %src_50, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.51" %src_51, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.52" %src_52, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.53" %src_53, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.54" %src_54, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.55" %src_55, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.56" %src_56, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.57" %src_57, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %src.addr.0.0.05.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %src.addr.0.0.05.exit ]
  %dst.addr.0.0.06 = getelementptr [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  switch i64 %for.loop.idx2, label %src.addr.0.0.05.case.57 [
    i64 0, label %src.addr.0.0.05.case.0
    i64 1, label %src.addr.0.0.05.case.1
    i64 2, label %src.addr.0.0.05.case.2
    i64 3, label %src.addr.0.0.05.case.3
    i64 4, label %src.addr.0.0.05.case.4
    i64 5, label %src.addr.0.0.05.case.5
    i64 6, label %src.addr.0.0.05.case.6
    i64 7, label %src.addr.0.0.05.case.7
    i64 8, label %src.addr.0.0.05.case.8
    i64 9, label %src.addr.0.0.05.case.9
    i64 10, label %src.addr.0.0.05.case.10
    i64 11, label %src.addr.0.0.05.case.11
    i64 12, label %src.addr.0.0.05.case.12
    i64 13, label %src.addr.0.0.05.case.13
    i64 14, label %src.addr.0.0.05.case.14
    i64 15, label %src.addr.0.0.05.case.15
    i64 16, label %src.addr.0.0.05.case.16
    i64 17, label %src.addr.0.0.05.case.17
    i64 18, label %src.addr.0.0.05.case.18
    i64 19, label %src.addr.0.0.05.case.19
    i64 20, label %src.addr.0.0.05.case.20
    i64 21, label %src.addr.0.0.05.case.21
    i64 22, label %src.addr.0.0.05.case.22
    i64 23, label %src.addr.0.0.05.case.23
    i64 24, label %src.addr.0.0.05.case.24
    i64 25, label %src.addr.0.0.05.case.25
    i64 26, label %src.addr.0.0.05.case.26
    i64 27, label %src.addr.0.0.05.case.27
    i64 28, label %src.addr.0.0.05.case.28
    i64 29, label %src.addr.0.0.05.case.29
    i64 30, label %src.addr.0.0.05.case.30
    i64 31, label %src.addr.0.0.05.case.31
    i64 32, label %src.addr.0.0.05.case.32
    i64 33, label %src.addr.0.0.05.case.33
    i64 34, label %src.addr.0.0.05.case.34
    i64 35, label %src.addr.0.0.05.case.35
    i64 36, label %src.addr.0.0.05.case.36
    i64 37, label %src.addr.0.0.05.case.37
    i64 38, label %src.addr.0.0.05.case.38
    i64 39, label %src.addr.0.0.05.case.39
    i64 40, label %src.addr.0.0.05.case.40
    i64 41, label %src.addr.0.0.05.case.41
    i64 42, label %src.addr.0.0.05.case.42
    i64 43, label %src.addr.0.0.05.case.43
    i64 44, label %src.addr.0.0.05.case.44
    i64 45, label %src.addr.0.0.05.case.45
    i64 46, label %src.addr.0.0.05.case.46
    i64 47, label %src.addr.0.0.05.case.47
    i64 48, label %src.addr.0.0.05.case.48
    i64 49, label %src.addr.0.0.05.case.49
    i64 50, label %src.addr.0.0.05.case.50
    i64 51, label %src.addr.0.0.05.case.51
    i64 52, label %src.addr.0.0.05.case.52
    i64 53, label %src.addr.0.0.05.case.53
    i64 54, label %src.addr.0.0.05.case.54
    i64 55, label %src.addr.0.0.05.case.55
    i64 56, label %src.addr.0.0.05.case.56
  ]

src.addr.0.0.05.case.0:                           ; preds = %for.loop
  %1 = bitcast i10* %src_0 to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.1:                           ; preds = %for.loop
  %4 = bitcast i10* %src_1 to i16*
  %5 = load i16, i16* %4
  %6 = trunc i16 %5 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.2:                           ; preds = %for.loop
  %7 = bitcast i10* %src_2 to i16*
  %8 = load i16, i16* %7
  %9 = trunc i16 %8 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.3:                           ; preds = %for.loop
  %10 = bitcast i10* %src_3 to i16*
  %11 = load i16, i16* %10
  %12 = trunc i16 %11 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.4:                           ; preds = %for.loop
  %13 = bitcast i10* %src_4 to i16*
  %14 = load i16, i16* %13
  %15 = trunc i16 %14 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.5:                           ; preds = %for.loop
  %16 = bitcast i10* %src_5 to i16*
  %17 = load i16, i16* %16
  %18 = trunc i16 %17 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.6:                           ; preds = %for.loop
  %19 = bitcast i10* %src_6 to i16*
  %20 = load i16, i16* %19
  %21 = trunc i16 %20 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.7:                           ; preds = %for.loop
  %22 = bitcast i10* %src_7 to i16*
  %23 = load i16, i16* %22
  %24 = trunc i16 %23 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.8:                           ; preds = %for.loop
  %25 = bitcast i10* %src_8 to i16*
  %26 = load i16, i16* %25
  %27 = trunc i16 %26 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.9:                           ; preds = %for.loop
  %28 = bitcast i10* %src_9 to i16*
  %29 = load i16, i16* %28
  %30 = trunc i16 %29 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.10:                          ; preds = %for.loop
  %31 = bitcast i10* %src_10 to i16*
  %32 = load i16, i16* %31
  %33 = trunc i16 %32 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.11:                          ; preds = %for.loop
  %34 = bitcast i10* %src_11 to i16*
  %35 = load i16, i16* %34
  %36 = trunc i16 %35 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.12:                          ; preds = %for.loop
  %37 = bitcast i10* %src_12 to i16*
  %38 = load i16, i16* %37
  %39 = trunc i16 %38 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.13:                          ; preds = %for.loop
  %40 = bitcast i10* %src_13 to i16*
  %41 = load i16, i16* %40
  %42 = trunc i16 %41 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.14:                          ; preds = %for.loop
  %43 = bitcast i10* %src_14 to i16*
  %44 = load i16, i16* %43
  %45 = trunc i16 %44 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.15:                          ; preds = %for.loop
  %46 = bitcast i10* %src_15 to i16*
  %47 = load i16, i16* %46
  %48 = trunc i16 %47 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.16:                          ; preds = %for.loop
  %49 = bitcast i10* %src_16 to i16*
  %50 = load i16, i16* %49
  %51 = trunc i16 %50 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.17:                          ; preds = %for.loop
  %52 = bitcast i10* %src_17 to i16*
  %53 = load i16, i16* %52
  %54 = trunc i16 %53 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.18:                          ; preds = %for.loop
  %55 = bitcast i10* %src_18 to i16*
  %56 = load i16, i16* %55
  %57 = trunc i16 %56 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.19:                          ; preds = %for.loop
  %58 = bitcast i10* %src_19 to i16*
  %59 = load i16, i16* %58
  %60 = trunc i16 %59 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.20:                          ; preds = %for.loop
  %61 = bitcast i10* %src_20 to i16*
  %62 = load i16, i16* %61
  %63 = trunc i16 %62 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.21:                          ; preds = %for.loop
  %64 = bitcast i10* %src_21 to i16*
  %65 = load i16, i16* %64
  %66 = trunc i16 %65 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.22:                          ; preds = %for.loop
  %67 = bitcast i10* %src_22 to i16*
  %68 = load i16, i16* %67
  %69 = trunc i16 %68 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.23:                          ; preds = %for.loop
  %70 = bitcast i10* %src_23 to i16*
  %71 = load i16, i16* %70
  %72 = trunc i16 %71 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.24:                          ; preds = %for.loop
  %73 = bitcast i10* %src_24 to i16*
  %74 = load i16, i16* %73
  %75 = trunc i16 %74 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.25:                          ; preds = %for.loop
  %76 = bitcast i10* %src_25 to i16*
  %77 = load i16, i16* %76
  %78 = trunc i16 %77 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.26:                          ; preds = %for.loop
  %79 = bitcast i10* %src_26 to i16*
  %80 = load i16, i16* %79
  %81 = trunc i16 %80 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.27:                          ; preds = %for.loop
  %82 = bitcast i10* %src_27 to i16*
  %83 = load i16, i16* %82
  %84 = trunc i16 %83 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.28:                          ; preds = %for.loop
  %85 = bitcast i10* %src_28 to i16*
  %86 = load i16, i16* %85
  %87 = trunc i16 %86 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.29:                          ; preds = %for.loop
  %88 = bitcast i10* %src_29 to i16*
  %89 = load i16, i16* %88
  %90 = trunc i16 %89 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.30:                          ; preds = %for.loop
  %91 = bitcast i10* %src_30 to i16*
  %92 = load i16, i16* %91
  %93 = trunc i16 %92 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.31:                          ; preds = %for.loop
  %94 = bitcast i10* %src_31 to i16*
  %95 = load i16, i16* %94
  %96 = trunc i16 %95 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.32:                          ; preds = %for.loop
  %97 = bitcast i10* %src_32 to i16*
  %98 = load i16, i16* %97
  %99 = trunc i16 %98 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.33:                          ; preds = %for.loop
  %100 = bitcast i10* %src_33 to i16*
  %101 = load i16, i16* %100
  %102 = trunc i16 %101 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.34:                          ; preds = %for.loop
  %103 = bitcast i10* %src_34 to i16*
  %104 = load i16, i16* %103
  %105 = trunc i16 %104 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.35:                          ; preds = %for.loop
  %106 = bitcast i10* %src_35 to i16*
  %107 = load i16, i16* %106
  %108 = trunc i16 %107 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.36:                          ; preds = %for.loop
  %109 = bitcast i10* %src_36 to i16*
  %110 = load i16, i16* %109
  %111 = trunc i16 %110 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.37:                          ; preds = %for.loop
  %112 = bitcast i10* %src_37 to i16*
  %113 = load i16, i16* %112
  %114 = trunc i16 %113 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.38:                          ; preds = %for.loop
  %115 = bitcast i10* %src_38 to i16*
  %116 = load i16, i16* %115
  %117 = trunc i16 %116 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.39:                          ; preds = %for.loop
  %118 = bitcast i10* %src_39 to i16*
  %119 = load i16, i16* %118
  %120 = trunc i16 %119 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.40:                          ; preds = %for.loop
  %121 = bitcast i10* %src_40 to i16*
  %122 = load i16, i16* %121
  %123 = trunc i16 %122 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.41:                          ; preds = %for.loop
  %124 = bitcast i10* %src_41 to i16*
  %125 = load i16, i16* %124
  %126 = trunc i16 %125 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.42:                          ; preds = %for.loop
  %127 = bitcast i10* %src_42 to i16*
  %128 = load i16, i16* %127
  %129 = trunc i16 %128 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.43:                          ; preds = %for.loop
  %130 = bitcast i10* %src_43 to i16*
  %131 = load i16, i16* %130
  %132 = trunc i16 %131 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.44:                          ; preds = %for.loop
  %133 = bitcast i10* %src_44 to i16*
  %134 = load i16, i16* %133
  %135 = trunc i16 %134 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.45:                          ; preds = %for.loop
  %136 = bitcast i10* %src_45 to i16*
  %137 = load i16, i16* %136
  %138 = trunc i16 %137 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.46:                          ; preds = %for.loop
  %139 = bitcast i10* %src_46 to i16*
  %140 = load i16, i16* %139
  %141 = trunc i16 %140 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.47:                          ; preds = %for.loop
  %142 = bitcast i10* %src_47 to i16*
  %143 = load i16, i16* %142
  %144 = trunc i16 %143 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.48:                          ; preds = %for.loop
  %145 = bitcast i10* %src_48 to i16*
  %146 = load i16, i16* %145
  %147 = trunc i16 %146 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.49:                          ; preds = %for.loop
  %148 = bitcast i10* %src_49 to i16*
  %149 = load i16, i16* %148
  %150 = trunc i16 %149 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.50:                          ; preds = %for.loop
  %151 = bitcast i10* %src_50 to i16*
  %152 = load i16, i16* %151
  %153 = trunc i16 %152 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.51:                          ; preds = %for.loop
  %154 = bitcast i10* %src_51 to i16*
  %155 = load i16, i16* %154
  %156 = trunc i16 %155 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.52:                          ; preds = %for.loop
  %157 = bitcast i10* %src_52 to i16*
  %158 = load i16, i16* %157
  %159 = trunc i16 %158 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.53:                          ; preds = %for.loop
  %160 = bitcast i10* %src_53 to i16*
  %161 = load i16, i16* %160
  %162 = trunc i16 %161 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.54:                          ; preds = %for.loop
  %163 = bitcast i10* %src_54 to i16*
  %164 = load i16, i16* %163
  %165 = trunc i16 %164 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.55:                          ; preds = %for.loop
  %166 = bitcast i10* %src_55 to i16*
  %167 = load i16, i16* %166
  %168 = trunc i16 %167 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.56:                          ; preds = %for.loop
  %169 = bitcast i10* %src_56 to i16*
  %170 = load i16, i16* %169
  %171 = trunc i16 %170 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.57:                          ; preds = %for.loop
  %172 = icmp eq i64 %for.loop.idx2, 57
  call void @llvm.assume(i1 %172)
  %173 = bitcast i10* %src_57 to i16*
  %174 = load i16, i16* %173
  %175 = trunc i16 %174 to i10
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.exit:                             ; preds = %src.addr.0.0.05.case.57, %src.addr.0.0.05.case.56, %src.addr.0.0.05.case.55, %src.addr.0.0.05.case.54, %src.addr.0.0.05.case.53, %src.addr.0.0.05.case.52, %src.addr.0.0.05.case.51, %src.addr.0.0.05.case.50, %src.addr.0.0.05.case.49, %src.addr.0.0.05.case.48, %src.addr.0.0.05.case.47, %src.addr.0.0.05.case.46, %src.addr.0.0.05.case.45, %src.addr.0.0.05.case.44, %src.addr.0.0.05.case.43, %src.addr.0.0.05.case.42, %src.addr.0.0.05.case.41, %src.addr.0.0.05.case.40, %src.addr.0.0.05.case.39, %src.addr.0.0.05.case.38, %src.addr.0.0.05.case.37, %src.addr.0.0.05.case.36, %src.addr.0.0.05.case.35, %src.addr.0.0.05.case.34, %src.addr.0.0.05.case.33, %src.addr.0.0.05.case.32, %src.addr.0.0.05.case.31, %src.addr.0.0.05.case.30, %src.addr.0.0.05.case.29, %src.addr.0.0.05.case.28, %src.addr.0.0.05.case.27, %src.addr.0.0.05.case.26, %src.addr.0.0.05.case.25, %src.addr.0.0.05.case.24, %src.addr.0.0.05.case.23, %src.addr.0.0.05.case.22, %src.addr.0.0.05.case.21, %src.addr.0.0.05.case.20, %src.addr.0.0.05.case.19, %src.addr.0.0.05.case.18, %src.addr.0.0.05.case.17, %src.addr.0.0.05.case.16, %src.addr.0.0.05.case.15, %src.addr.0.0.05.case.14, %src.addr.0.0.05.case.13, %src.addr.0.0.05.case.12, %src.addr.0.0.05.case.11, %src.addr.0.0.05.case.10, %src.addr.0.0.05.case.9, %src.addr.0.0.05.case.8, %src.addr.0.0.05.case.7, %src.addr.0.0.05.case.6, %src.addr.0.0.05.case.5, %src.addr.0.0.05.case.4, %src.addr.0.0.05.case.3, %src.addr.0.0.05.case.2, %src.addr.0.0.05.case.1, %src.addr.0.0.05.case.0
  %176 = phi i10 [ %3, %src.addr.0.0.05.case.0 ], [ %6, %src.addr.0.0.05.case.1 ], [ %9, %src.addr.0.0.05.case.2 ], [ %12, %src.addr.0.0.05.case.3 ], [ %15, %src.addr.0.0.05.case.4 ], [ %18, %src.addr.0.0.05.case.5 ], [ %21, %src.addr.0.0.05.case.6 ], [ %24, %src.addr.0.0.05.case.7 ], [ %27, %src.addr.0.0.05.case.8 ], [ %30, %src.addr.0.0.05.case.9 ], [ %33, %src.addr.0.0.05.case.10 ], [ %36, %src.addr.0.0.05.case.11 ], [ %39, %src.addr.0.0.05.case.12 ], [ %42, %src.addr.0.0.05.case.13 ], [ %45, %src.addr.0.0.05.case.14 ], [ %48, %src.addr.0.0.05.case.15 ], [ %51, %src.addr.0.0.05.case.16 ], [ %54, %src.addr.0.0.05.case.17 ], [ %57, %src.addr.0.0.05.case.18 ], [ %60, %src.addr.0.0.05.case.19 ], [ %63, %src.addr.0.0.05.case.20 ], [ %66, %src.addr.0.0.05.case.21 ], [ %69, %src.addr.0.0.05.case.22 ], [ %72, %src.addr.0.0.05.case.23 ], [ %75, %src.addr.0.0.05.case.24 ], [ %78, %src.addr.0.0.05.case.25 ], [ %81, %src.addr.0.0.05.case.26 ], [ %84, %src.addr.0.0.05.case.27 ], [ %87, %src.addr.0.0.05.case.28 ], [ %90, %src.addr.0.0.05.case.29 ], [ %93, %src.addr.0.0.05.case.30 ], [ %96, %src.addr.0.0.05.case.31 ], [ %99, %src.addr.0.0.05.case.32 ], [ %102, %src.addr.0.0.05.case.33 ], [ %105, %src.addr.0.0.05.case.34 ], [ %108, %src.addr.0.0.05.case.35 ], [ %111, %src.addr.0.0.05.case.36 ], [ %114, %src.addr.0.0.05.case.37 ], [ %117, %src.addr.0.0.05.case.38 ], [ %120, %src.addr.0.0.05.case.39 ], [ %123, %src.addr.0.0.05.case.40 ], [ %126, %src.addr.0.0.05.case.41 ], [ %129, %src.addr.0.0.05.case.42 ], [ %132, %src.addr.0.0.05.case.43 ], [ %135, %src.addr.0.0.05.case.44 ], [ %138, %src.addr.0.0.05.case.45 ], [ %141, %src.addr.0.0.05.case.46 ], [ %144, %src.addr.0.0.05.case.47 ], [ %147, %src.addr.0.0.05.case.48 ], [ %150, %src.addr.0.0.05.case.49 ], [ %153, %src.addr.0.0.05.case.50 ], [ %156, %src.addr.0.0.05.case.51 ], [ %159, %src.addr.0.0.05.case.52 ], [ %162, %src.addr.0.0.05.case.53 ], [ %165, %src.addr.0.0.05.case.54 ], [ %168, %src.addr.0.0.05.case.55 ], [ %171, %src.addr.0.0.05.case.56 ], [ %175, %src.addr.0.0.05.case.57 ]
  store i10 %176, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %src.addr.0.0.05.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.160.423.426"([58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.1" %src_1, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.2" %src_2, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.3" %src_3, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.4" %src_4, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.5" %src_5, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.6" %src_6, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.7" %src_7, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.8" %src_8, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.9" %src_9, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.10" %src_10, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.11" %src_11, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.12" %src_12, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.13" %src_13, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.14" %src_14, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.15" %src_15, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.16" %src_16, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.17" %src_17, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.18" %src_18, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.19" %src_19, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.20" %src_20, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.21" %src_21, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.22" %src_22, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.23" %src_23, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.24" %src_24, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.25" %src_25, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.26" %src_26, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.27" %src_27, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.28" %src_28, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.29" %src_29, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.30" %src_30, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.31" %src_31, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.32" %src_32, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.33" %src_33, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.34" %src_34, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.35" %src_35, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.36" %src_36, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.37" %src_37, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.38" %src_38, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.39" %src_39, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.40" %src_40, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.41" %src_41, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.42" %src_42, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.43" %src_43, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.44" %src_44, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.45" %src_45, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.46" %src_46, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.47" %src_47, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.48" %src_48, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.49" %src_49, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.50" %src_50, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.51" %src_51, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.52" %src_52, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.53" %src_53, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.54" %src_54, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.55" %src_55, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.56" %src_56, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.57" %src_57) #1 {
entry:
  %0 = icmp eq [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.163.424.425"([58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i10* %src_0, i10* %src_1, i10* %src_2, i10* %src_3, i10* %src_4, i10* %src_5, i10* %src_6, i10* %src_7, i10* %src_8, i10* %src_9, i10* %src_10, i10* %src_11, i10* %src_12, i10* %src_13, i10* %src_14, i10* %src_15, i10* %src_16, i10* %src_17, i10* %src_18, i10* %src_19, i10* %src_20, i10* %src_21, i10* %src_22, i10* %src_23, i10* %src_24, i10* %src_25, i10* %src_26, i10* %src_27, i10* %src_28, i10* %src_29, i10* %src_30, i10* %src_31, i10* %src_32, i10* %src_33, i10* %src_34, i10* %src_35, i10* %src_36, i10* %src_37, i10* %src_38, i10* %src_39, i10* %src_40, i10* %src_41, i10* %src_42, i10* %src_43, i10* %src_44, i10* %src_45, i10* %src_46, i10* %src_47, i10* %src_48, i10* %src_49, i10* %src_50, i10* %src_51, i10* %src_52, i10* %src_53, i10* %src_54, i10* %src_55, i10* %src_56, i10* %src_57, i64 58)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.146"([1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i10* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr.0.0.06 = getelementptr [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i10* %src to i16*
  %2 = load i16, i16* %1
  %3 = trunc i16 %2 to i10
  store i10 %3, i10* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.143"([1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i10* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.146"([1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i10* %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a273struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i4368* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, i64 %dst_shift, [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = mul i64 16, %for.loop.idx2
  %2 = add i64 %dst_shift, %1
  %3 = load i16, i16* %src.addr.0.0.05, align 2
  %4 = load i4368, i4368* %dst, align 512
  %5 = zext i64 %2 to i4368
  %6 = shl i4368 65535, %5
  %7 = zext i16 %3 to i4368
  %8 = shl i4368 %7, %5
  %thr.xor1 = xor i4368 %6, -1
  %thr.and2 = and i4368 %4, %thr.xor1
  %thr.or3 = or i4368 %thr.and2, %8
  store i4368 %thr.or3, i4368* %dst, align 512
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a273struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i4368* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #1 {
entry:
  %0 = icmp eq [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a273struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i4368* %dst, i64 0, [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 273)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.275"(i16* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, i64 %dst_shift, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  %1 = trunc i64 %dst_shift to i16
  %2 = shl i16 -1, %1
  %3 = xor i16 %2, -1
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %4 = load i16, i16* %src.addr.0.0.05, align 2
  %5 = load i16, i16* %dst, align 2
  %6 = shl i16 %4, %1
  %7 = and i16 %5, %3
  %8 = or i16 %7, %6
  store i16 %8, i16* %dst, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.272"(i16* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.275"(i16* %dst, i64 0, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_in([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="0" "unpacked"="0", i4368* noalias nocapture align 512 "orig.arg.no"="1" "unpacked"="1.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture align 512 "orig.arg.no"="3" "unpacked"="3.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="4" "unpacked"="4", i16* noalias nocapture align 512 "orig.arg.no"="5" "unpacked"="5.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="6" "unpacked"="6", i16* noalias nocapture align 512 "orig.arg.no"="7" "unpacked"="7.0", [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* noalias readonly "orig.arg.no"="8" "unpacked"="8", i8* noalias nocapture align 512 "orig.arg.no"="9" "unpacked"="9.0", [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="10" "unpacked"="10", [18 x i10]* noalias nocapture align 512 "orig.arg.no"="11" "unpacked"="11.0", [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="12" "unpacked"="12", i10* noalias nocapture align 512 "orig.arg.no"="13" "unpacked"="13.0.0" %_0, i10* noalias nocapture align 512 "orig.arg.no"="13" "unpacked"="13.0.1" %_1, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="14" "unpacked"="14", [48 x i10]* noalias nocapture align 512 "orig.arg.no"="15" "unpacked"="15.0", [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="16" "unpacked"="16", i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.0" %_01, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.1" %_12, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.2" %_2, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.3" %_3, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.4" %_4, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.5" %_5, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.6" %_6, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.7" %_7, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.8" %_8, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.9" %_9, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.10" %_10, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.11" %_11, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.12" %_123, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.13" %_13, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.14" %_14, i10* noalias nocapture align 512 "orig.arg.no"="17" "unpacked"="17.0.15" %_15, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="18" "unpacked"="18", [9792 x i10]* noalias nocapture "orig.arg.no"="19" "unpacked"="19.0", [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="20" "unpacked"="20", i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.0" %_04, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.1" %_16, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.2" %_27, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.3" %_38, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.4" %_49, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.5" %_510, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.6" %_611, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.7" %_712, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.8" %_813, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.9" %_914, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.10" %_1015, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.11" %_1116, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.12" %_1217, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.13" %_1318, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.14" %_1419, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.15" %_1520, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.16" %_1621, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.17" %_17, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.18" %_18, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.19" %_19, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.20" %_20, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.21" %_21, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.22" %_22, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.23" %_23, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.24" %_24, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.25" %_25, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.26" %_26, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.27" %_2722, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.28" %_28, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.29" %_29, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.30" %_30, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.31" %_31, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.32" %_32, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.33" %_33, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.34" %_34, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.35" %_35, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.36" %_36, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.37" %_37, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.38" %_3823, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.39" %_39, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.40" %_40, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.41" %_41, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.42" %_42, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.43" %_43, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.44" %_44, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.45" %_45, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.46" %_46, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.47" %_47, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.48" %_48, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.49" %_4924, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.50" %_50, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.51" %_51, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.52" %_52, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.53" %_53, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.54" %_54, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.55" %_55, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.56" %_56, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.57" %_57, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.58" %_58, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.59" %_59, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.60" %_60, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.61" %_61, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.62" %_62, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.63" %_63, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.64" %_64, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.65" %_65, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.66" %_66, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.67" %_67, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.68" %_68, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.69" %_69, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.70" %_70, i10* noalias nocapture align 512 "orig.arg.no"="21" "unpacked"="21.0.71" %_71, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="22" "unpacked"="22", [4176 x i10]* noalias nocapture "orig.arg.no"="23" "unpacked"="23.0", [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="24" "unpacked"="24", i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.0" %_025, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.1" %_126, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.2" %_227, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.3" %_328, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.4" %_429, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.5" %_530, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.6" %_631, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.7" %_732, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.8" %_833, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.9" %_934, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.10" %_1035, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.11" %_1136, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.12" %_1237, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.13" %_1338, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.14" %_1439, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.15" %_1540, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.16" %_1641, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.17" %_1742, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.18" %_1843, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.19" %_1944, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.20" %_2045, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.21" %_2146, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.22" %_2247, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.23" %_2348, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.24" %_2449, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.25" %_2550, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.26" %_2651, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.27" %_2752, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.28" %_2853, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.29" %_2954, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.30" %_3055, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.31" %_3156, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.32" %_3257, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.33" %_3358, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.34" %_3459, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.35" %_3560, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.36" %_3661, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.37" %_3762, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.38" %_3863, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.39" %_3964, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.40" %_4065, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.41" %_4166, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.42" %_4267, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.43" %_4368, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.44" %_4469, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.45" %_4570, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.46" %_4671, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.47" %_4772, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.48" %_4873, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.49" %_4974, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.50" %_5075, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.51" %_5176, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.52" %_5277, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.53" %_5378, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.54" %_5479, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.55" %_5580, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.56" %_5681, i10* noalias nocapture align 512 "orig.arg.no"="25" "unpacked"="25.0.57" %_5782, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="26" "unpacked"="26", [58 x i10]* noalias nocapture align 512 "orig.arg.no"="27" "unpacked"="27.0", [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="28" "unpacked"="28", i10* noalias nocapture align 512 "orig.arg.no"="29" "unpacked"="29.0") #4 {
entry:
  call void @"onebyonecpy_hls.p0a273struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i4368* align 512 %1, [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.272"(i16* align 512 %3, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.272"(i16* align 512 %5, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %4)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.272"(i16* align 512 %7, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6)
  call void @"onebyonecpy_hls.p0a1struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"(i8* align 512 %9, [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %8)
  call fastcc void @"onebyonecpy_hls.p0a18struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.247"([18 x i10]* align 512 %11, [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %10)
  call void @"onebyonecpy_hls.p0a2struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* align 512 %_0, i10* align 512 %_1, [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %12)
  call fastcc void @"onebyonecpy_hls.p0a48struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.222"([48 x i10]* align 512 %14, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %13)
  call void @"onebyonecpy_hls.p0a16struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* align 512 %_01, i10* align 512 %_12, i10* align 512 %_2, i10* align 512 %_3, i10* align 512 %_4, i10* align 512 %_5, i10* align 512 %_6, i10* align 512 %_7, i10* align 512 %_8, i10* align 512 %_9, i10* align 512 %_10, i10* align 512 %_11, i10* align 512 %_123, i10* align 512 %_13, i10* align 512 %_14, i10* align 512 %_15, [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %15)
  call fastcc void @"onebyonecpy_hls.p0a9792struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.198"([9792 x i10]* %17, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %16)
  call void @"onebyonecpy_hls.p0a72struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* align 512 %_04, i10* align 512 %_16, i10* align 512 %_27, i10* align 512 %_38, i10* align 512 %_49, i10* align 512 %_510, i10* align 512 %_611, i10* align 512 %_712, i10* align 512 %_813, i10* align 512 %_914, i10* align 512 %_1015, i10* align 512 %_1116, i10* align 512 %_1217, i10* align 512 %_1318, i10* align 512 %_1419, i10* align 512 %_1520, i10* align 512 %_1621, i10* align 512 %_17, i10* align 512 %_18, i10* align 512 %_19, i10* align 512 %_20, i10* align 512 %_21, i10* align 512 %_22, i10* align 512 %_23, i10* align 512 %_24, i10* align 512 %_25, i10* align 512 %_26, i10* align 512 %_2722, i10* align 512 %_28, i10* align 512 %_29, i10* align 512 %_30, i10* align 512 %_31, i10* align 512 %_32, i10* align 512 %_33, i10* align 512 %_34, i10* align 512 %_35, i10* align 512 %_36, i10* align 512 %_37, i10* align 512 %_3823, i10* align 512 %_39, i10* align 512 %_40, i10* align 512 %_41, i10* align 512 %_42, i10* align 512 %_43, i10* align 512 %_44, i10* align 512 %_45, i10* align 512 %_46, i10* align 512 %_47, i10* align 512 %_48, i10* align 512 %_4924, i10* align 512 %_50, i10* align 512 %_51, i10* align 512 %_52, i10* align 512 %_53, i10* align 512 %_54, i10* align 512 %_55, i10* align 512 %_56, i10* align 512 %_57, i10* align 512 %_58, i10* align 512 %_59, i10* align 512 %_60, i10* align 512 %_61, i10* align 512 %_62, i10* align 512 %_63, i10* align 512 %_64, i10* align 512 %_65, i10* align 512 %_66, i10* align 512 %_67, i10* align 512 %_68, i10* align 512 %_69, i10* align 512 %_70, i10* align 512 %_71, [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %18)
  call fastcc void @"onebyonecpy_hls.p0a4176struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.174"([4176 x i10]* %20, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %19)
  call void @"onebyonecpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.383.386"(i10* align 512 %_025, i10* align 512 %_126, i10* align 512 %_227, i10* align 512 %_328, i10* align 512 %_429, i10* align 512 %_530, i10* align 512 %_631, i10* align 512 %_732, i10* align 512 %_833, i10* align 512 %_934, i10* align 512 %_1035, i10* align 512 %_1136, i10* align 512 %_1237, i10* align 512 %_1338, i10* align 512 %_1439, i10* align 512 %_1540, i10* align 512 %_1641, i10* align 512 %_1742, i10* align 512 %_1843, i10* align 512 %_1944, i10* align 512 %_2045, i10* align 512 %_2146, i10* align 512 %_2247, i10* align 512 %_2348, i10* align 512 %_2449, i10* align 512 %_2550, i10* align 512 %_2651, i10* align 512 %_2752, i10* align 512 %_2853, i10* align 512 %_2954, i10* align 512 %_3055, i10* align 512 %_3156, i10* align 512 %_3257, i10* align 512 %_3358, i10* align 512 %_3459, i10* align 512 %_3560, i10* align 512 %_3661, i10* align 512 %_3762, i10* align 512 %_3863, i10* align 512 %_3964, i10* align 512 %_4065, i10* align 512 %_4166, i10* align 512 %_4267, i10* align 512 %_4368, i10* align 512 %_4469, i10* align 512 %_4570, i10* align 512 %_4671, i10* align 512 %_4772, i10* align 512 %_4873, i10* align 512 %_4974, i10* align 512 %_5075, i10* align 512 %_5176, i10* align 512 %_5277, i10* align 512 %_5378, i10* align 512 %_5479, i10* align 512 %_5580, i10* align 512 %_5681, i10* align 512 %_5782, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %21)
  call fastcc void @"onebyonecpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([58 x i10]* align 512 %23, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %22)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"(i10* align 512 %25, [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %24)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a273struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.300"([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i4368* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 %src_shift, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %1 = mul i64 16, %for.loop.idx2
  %2 = add i64 %src_shift, %1
  %dst.addr.0.0.06 = getelementptr [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %3 = load i4368, i4368* %src, align 512
  %4 = zext i64 %2 to i4368
  %5 = lshr i4368 %3, %4
  %6 = trunc i4368 %5 to i16
  store i16 %6, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a273struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.297"([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i4368* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #1 {
entry:
  %0 = icmp eq [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a273struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.300"([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i4368* %src, i64 0, i64 273)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.282"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i16* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 %src_shift, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  %1 = trunc i64 %src_shift to i16
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr.0.0.06 = getelementptr [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %2 = load i16, i16* %src, align 2
  %3 = lshr i16 %2, %1
  store i16 %3, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i16* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.282"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i16* %src, i64 0, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_out([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i4368* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="4" "unpacked"="4", i16* noalias nocapture readonly align 512 "orig.arg.no"="5" "unpacked"="5.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="6" "unpacked"="6", i16* noalias nocapture readonly align 512 "orig.arg.no"="7" "unpacked"="7.0", [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* noalias "orig.arg.no"="8" "unpacked"="8", i8* noalias nocapture readonly align 512 "orig.arg.no"="9" "unpacked"="9.0", [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="10" "unpacked"="10", [18 x i10]* noalias nocapture readonly align 512 "orig.arg.no"="11" "unpacked"="11.0", [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="12" "unpacked"="12", i10* noalias nocapture readonly align 512 "orig.arg.no"="13" "unpacked"="13.0.0" %_0, i10* noalias nocapture readonly align 512 "orig.arg.no"="13" "unpacked"="13.0.1" %_1, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="14" "unpacked"="14", [48 x i10]* noalias nocapture readonly align 512 "orig.arg.no"="15" "unpacked"="15.0", [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="16" "unpacked"="16", i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.0" %_01, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.1" %_12, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.2" %_2, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.3" %_3, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.4" %_4, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.5" %_5, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.6" %_6, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.7" %_7, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.8" %_8, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.9" %_9, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.10" %_10, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.11" %_11, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.12" %_123, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.13" %_13, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.14" %_14, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.15" %_15, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="18" "unpacked"="18", [9792 x i10]* noalias nocapture readonly "orig.arg.no"="19" "unpacked"="19.0", [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="20" "unpacked"="20", i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.0" %_04, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.1" %_16, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.2" %_27, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.3" %_38, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.4" %_49, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.5" %_510, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.6" %_611, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.7" %_712, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.8" %_813, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.9" %_914, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.10" %_1015, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.11" %_1116, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.12" %_1217, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.13" %_1318, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.14" %_1419, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.15" %_1520, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.16" %_1621, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.17" %_17, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.18" %_18, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.19" %_19, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.20" %_20, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.21" %_21, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.22" %_22, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.23" %_23, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.24" %_24, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.25" %_25, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.26" %_26, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.27" %_2722, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.28" %_28, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.29" %_29, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.30" %_30, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.31" %_31, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.32" %_32, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.33" %_33, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.34" %_34, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.35" %_35, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.36" %_36, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.37" %_37, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.38" %_3823, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.39" %_39, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.40" %_40, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.41" %_41, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.42" %_42, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.43" %_43, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.44" %_44, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.45" %_45, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.46" %_46, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.47" %_47, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.48" %_48, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.49" %_4924, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.50" %_50, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.51" %_51, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.52" %_52, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.53" %_53, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.54" %_54, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.55" %_55, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.56" %_56, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.57" %_57, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.58" %_58, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.59" %_59, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.60" %_60, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.61" %_61, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.62" %_62, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.63" %_63, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.64" %_64, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.65" %_65, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.66" %_66, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.67" %_67, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.68" %_68, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.69" %_69, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.70" %_70, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.71" %_71, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="22" "unpacked"="22", [4176 x i10]* noalias nocapture readonly "orig.arg.no"="23" "unpacked"="23.0", [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="24" "unpacked"="24", i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.0" %_025, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.1" %_126, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.2" %_227, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.3" %_328, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.4" %_429, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.5" %_530, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.6" %_631, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.7" %_732, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.8" %_833, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.9" %_934, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.10" %_1035, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.11" %_1136, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.12" %_1237, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.13" %_1338, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.14" %_1439, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.15" %_1540, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.16" %_1641, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.17" %_1742, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.18" %_1843, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.19" %_1944, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.20" %_2045, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.21" %_2146, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.22" %_2247, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.23" %_2348, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.24" %_2449, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.25" %_2550, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.26" %_2651, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.27" %_2752, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.28" %_2853, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.29" %_2954, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.30" %_3055, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.31" %_3156, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.32" %_3257, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.33" %_3358, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.34" %_3459, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.35" %_3560, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.36" %_3661, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.37" %_3762, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.38" %_3863, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.39" %_3964, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.40" %_4065, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.41" %_4166, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.42" %_4267, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.43" %_4368, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.44" %_4469, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.45" %_4570, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.46" %_4671, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.47" %_4772, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.48" %_4873, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.49" %_4974, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.50" %_5075, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.51" %_5176, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.52" %_5277, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.53" %_5378, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.54" %_5479, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.55" %_5580, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.56" %_5681, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.57" %_5782, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="26" "unpacked"="26", [58 x i10]* noalias nocapture readonly align 512 "orig.arg.no"="27" "unpacked"="27.0", [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="28" "unpacked"="28", i10* noalias nocapture readonly align 512 "orig.arg.no"="29" "unpacked"="29.0") #5 {
entry:
  call void @"onebyonecpy_hls.p0a273struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.297"([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i4368* align 512 %1)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2, i16* align 512 %3)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %4, i16* align 512 %5)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6, i16* align 512 %7)
  call void @"onebyonecpy_hls.p0a1struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>.260"([1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %8, i8* align 512 %9)
  call fastcc void @"onebyonecpy_hls.p0a18struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %10, [18 x i10]* align 512 %11)
  call void @"onebyonecpy_hls.p0a2struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.234"([2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %12, i10* align 512 %_0, i10* align 512 %_1)
  call fastcc void @"onebyonecpy_hls.p0a48struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %13, [48 x i10]* align 512 %14)
  call void @"onebyonecpy_hls.p0a16struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.210"([16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %15, i10* align 512 %_01, i10* align 512 %_12, i10* align 512 %_2, i10* align 512 %_3, i10* align 512 %_4, i10* align 512 %_5, i10* align 512 %_6, i10* align 512 %_7, i10* align 512 %_8, i10* align 512 %_9, i10* align 512 %_10, i10* align 512 %_11, i10* align 512 %_123, i10* align 512 %_13, i10* align 512 %_14, i10* align 512 %_15)
  call fastcc void @"onebyonecpy_hls.p0a9792struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %16, [9792 x i10]* %17)
  call void @"onebyonecpy_hls.p0a72struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.186"([72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %18, i10* align 512 %_04, i10* align 512 %_16, i10* align 512 %_27, i10* align 512 %_38, i10* align 512 %_49, i10* align 512 %_510, i10* align 512 %_611, i10* align 512 %_712, i10* align 512 %_813, i10* align 512 %_914, i10* align 512 %_1015, i10* align 512 %_1116, i10* align 512 %_1217, i10* align 512 %_1318, i10* align 512 %_1419, i10* align 512 %_1520, i10* align 512 %_1621, i10* align 512 %_17, i10* align 512 %_18, i10* align 512 %_19, i10* align 512 %_20, i10* align 512 %_21, i10* align 512 %_22, i10* align 512 %_23, i10* align 512 %_24, i10* align 512 %_25, i10* align 512 %_26, i10* align 512 %_2722, i10* align 512 %_28, i10* align 512 %_29, i10* align 512 %_30, i10* align 512 %_31, i10* align 512 %_32, i10* align 512 %_33, i10* align 512 %_34, i10* align 512 %_35, i10* align 512 %_36, i10* align 512 %_37, i10* align 512 %_3823, i10* align 512 %_39, i10* align 512 %_40, i10* align 512 %_41, i10* align 512 %_42, i10* align 512 %_43, i10* align 512 %_44, i10* align 512 %_45, i10* align 512 %_46, i10* align 512 %_47, i10* align 512 %_48, i10* align 512 %_4924, i10* align 512 %_50, i10* align 512 %_51, i10* align 512 %_52, i10* align 512 %_53, i10* align 512 %_54, i10* align 512 %_55, i10* align 512 %_56, i10* align 512 %_57, i10* align 512 %_58, i10* align 512 %_59, i10* align 512 %_60, i10* align 512 %_61, i10* align 512 %_62, i10* align 512 %_63, i10* align 512 %_64, i10* align 512 %_65, i10* align 512 %_66, i10* align 512 %_67, i10* align 512 %_68, i10* align 512 %_69, i10* align 512 %_70, i10* align 512 %_71)
  call fastcc void @"onebyonecpy_hls.p0a4176struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"([4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %19, [4176 x i10]* %20)
  call void @"onebyonecpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.160.423.426"([58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %21, i10* align 512 %_025, i10* align 512 %_126, i10* align 512 %_227, i10* align 512 %_328, i10* align 512 %_429, i10* align 512 %_530, i10* align 512 %_631, i10* align 512 %_732, i10* align 512 %_833, i10* align 512 %_934, i10* align 512 %_1035, i10* align 512 %_1136, i10* align 512 %_1237, i10* align 512 %_1338, i10* align 512 %_1439, i10* align 512 %_1540, i10* align 512 %_1641, i10* align 512 %_1742, i10* align 512 %_1843, i10* align 512 %_1944, i10* align 512 %_2045, i10* align 512 %_2146, i10* align 512 %_2247, i10* align 512 %_2348, i10* align 512 %_2449, i10* align 512 %_2550, i10* align 512 %_2651, i10* align 512 %_2752, i10* align 512 %_2853, i10* align 512 %_2954, i10* align 512 %_3055, i10* align 512 %_3156, i10* align 512 %_3257, i10* align 512 %_3358, i10* align 512 %_3459, i10* align 512 %_3560, i10* align 512 %_3661, i10* align 512 %_3762, i10* align 512 %_3863, i10* align 512 %_3964, i10* align 512 %_4065, i10* align 512 %_4166, i10* align 512 %_4267, i10* align 512 %_4368, i10* align 512 %_4469, i10* align 512 %_4570, i10* align 512 %_4671, i10* align 512 %_4772, i10* align 512 %_4873, i10* align 512 %_4974, i10* align 512 %_5075, i10* align 512 %_5176, i10* align 512 %_5277, i10* align 512 %_5378, i10* align 512 %_5479, i10* align 512 %_5580, i10* align 512 %_5681, i10* align 512 %_5782)
  call fastcc void @"onebyonecpy_hls.p0a58struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.160"([58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %22, [58 x i10]* align 512 %23)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>.143"([1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %24, i10* align 512 %25)
  ret void
}

declare void @apatb_myproject_hw(i4368*, i16*, i16*, i16*, i8*, [18 x i10]*, i10*, i10*, [48 x i10]*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, [9792 x i10]*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, [4176 x i10]*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, [58 x i10]*, i10*)

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_back([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i4368* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="4" "unpacked"="4", i16* noalias nocapture readonly align 512 "orig.arg.no"="5" "unpacked"="5.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="6" "unpacked"="6", i16* noalias nocapture readonly align 512 "orig.arg.no"="7" "unpacked"="7.0", [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* noalias "orig.arg.no"="8" "unpacked"="8", i8* noalias nocapture readonly align 512 "orig.arg.no"="9" "unpacked"="9.0", [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="10" "unpacked"="10", [18 x i10]* noalias nocapture readonly align 512 "orig.arg.no"="11" "unpacked"="11.0", [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="12" "unpacked"="12", i10* noalias nocapture readonly align 512 "orig.arg.no"="13" "unpacked"="13.0.0" %_0, i10* noalias nocapture readonly align 512 "orig.arg.no"="13" "unpacked"="13.0.1" %_1, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="14" "unpacked"="14", [48 x i10]* noalias nocapture readonly align 512 "orig.arg.no"="15" "unpacked"="15.0", [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="16" "unpacked"="16", i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.0" %_01, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.1" %_12, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.2" %_2, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.3" %_3, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.4" %_4, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.5" %_5, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.6" %_6, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.7" %_7, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.8" %_8, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.9" %_9, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.10" %_10, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.11" %_11, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.12" %_123, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.13" %_13, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.14" %_14, i10* noalias nocapture readonly align 512 "orig.arg.no"="17" "unpacked"="17.0.15" %_15, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="18" "unpacked"="18", [9792 x i10]* noalias nocapture readonly "orig.arg.no"="19" "unpacked"="19.0", [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="20" "unpacked"="20", i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.0" %_04, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.1" %_16, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.2" %_27, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.3" %_38, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.4" %_49, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.5" %_510, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.6" %_611, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.7" %_712, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.8" %_813, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.9" %_914, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.10" %_1015, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.11" %_1116, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.12" %_1217, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.13" %_1318, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.14" %_1419, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.15" %_1520, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.16" %_1621, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.17" %_17, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.18" %_18, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.19" %_19, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.20" %_20, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.21" %_21, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.22" %_22, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.23" %_23, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.24" %_24, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.25" %_25, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.26" %_26, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.27" %_2722, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.28" %_28, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.29" %_29, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.30" %_30, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.31" %_31, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.32" %_32, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.33" %_33, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.34" %_34, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.35" %_35, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.36" %_36, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.37" %_37, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.38" %_3823, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.39" %_39, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.40" %_40, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.41" %_41, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.42" %_42, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.43" %_43, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.44" %_44, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.45" %_45, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.46" %_46, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.47" %_47, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.48" %_48, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.49" %_4924, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.50" %_50, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.51" %_51, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.52" %_52, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.53" %_53, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.54" %_54, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.55" %_55, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.56" %_56, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.57" %_57, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.58" %_58, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.59" %_59, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.60" %_60, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.61" %_61, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.62" %_62, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.63" %_63, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.64" %_64, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.65" %_65, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.66" %_66, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.67" %_67, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.68" %_68, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.69" %_69, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.70" %_70, i10* noalias nocapture readonly align 512 "orig.arg.no"="21" "unpacked"="21.0.71" %_71, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="22" "unpacked"="22", [4176 x i10]* noalias nocapture readonly "orig.arg.no"="23" "unpacked"="23.0", [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="24" "unpacked"="24", i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.0" %_025, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.1" %_126, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.2" %_227, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.3" %_328, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.4" %_429, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.5" %_530, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.6" %_631, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.7" %_732, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.8" %_833, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.9" %_934, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.10" %_1035, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.11" %_1136, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.12" %_1237, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.13" %_1338, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.14" %_1439, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.15" %_1540, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.16" %_1641, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.17" %_1742, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.18" %_1843, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.19" %_1944, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.20" %_2045, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.21" %_2146, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.22" %_2247, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.23" %_2348, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.24" %_2449, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.25" %_2550, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.26" %_2651, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.27" %_2752, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.28" %_2853, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.29" %_2954, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.30" %_3055, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.31" %_3156, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.32" %_3257, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.33" %_3358, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.34" %_3459, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.35" %_3560, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.36" %_3661, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.37" %_3762, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.38" %_3863, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.39" %_3964, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.40" %_4065, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.41" %_4166, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.42" %_4267, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.43" %_4368, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.44" %_4469, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.45" %_4570, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.46" %_4671, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.47" %_4772, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.48" %_4873, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.49" %_4974, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.50" %_5075, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.51" %_5176, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.52" %_5277, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.53" %_5378, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.54" %_5479, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.55" %_5580, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.56" %_5681, i10* noalias nocapture readonly align 512 "orig.arg.no"="25" "unpacked"="25.0.57" %_5782, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="26" "unpacked"="26", [58 x i10]* noalias nocapture readonly align 512 "orig.arg.no"="27" "unpacked"="27.0", [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="28" "unpacked"="28", i10* noalias nocapture readonly align 512 "orig.arg.no"="29" "unpacked"="29.0") #5 {
entry:
  call void @"onebyonecpy_hls.p0a1struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>.260"([1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %8, i8* align 512 %9)
  ret void
}

define void @myproject_hw_stub_wrapper(i4368*, i16*, i16*, i16*, i8*, [18 x i10]*, i10*, i10*, [48 x i10]*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, [9792 x i10]*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, [4176 x i10]*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, i10*, [58 x i10]*, i10*) #6 {
entry:
  %159 = alloca [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %160 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %161 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %162 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %163 = alloca [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]
  %164 = alloca [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]
  %165 = alloca [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]
  %166 = alloca [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]
  %167 = alloca [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]
  %malloccall = tail call i8* @malloc(i64 19584)
  %168 = bitcast i8* %malloccall to [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %169 = alloca [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]
  %malloccall1 = tail call i8* @malloc(i64 8352)
  %170 = bitcast i8* %malloccall1 to [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]*
  %171 = alloca [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]
  %172 = alloca [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]
  %173 = alloca [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]
  call void @copy_out([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %159, i4368* %0, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %160, i16* %1, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %161, i16* %2, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %162, i16* %3, [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %163, i8* %4, [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %164, [18 x i10]* %5, [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %165, i10* %6, i10* %7, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %166, [48 x i10]* %8, [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %167, i10* %9, i10* %10, i10* %11, i10* %12, i10* %13, i10* %14, i10* %15, i10* %16, i10* %17, i10* %18, i10* %19, i10* %20, i10* %21, i10* %22, i10* %23, i10* %24, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %168, [9792 x i10]* %25, [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %169, i10* %26, i10* %27, i10* %28, i10* %29, i10* %30, i10* %31, i10* %32, i10* %33, i10* %34, i10* %35, i10* %36, i10* %37, i10* %38, i10* %39, i10* %40, i10* %41, i10* %42, i10* %43, i10* %44, i10* %45, i10* %46, i10* %47, i10* %48, i10* %49, i10* %50, i10* %51, i10* %52, i10* %53, i10* %54, i10* %55, i10* %56, i10* %57, i10* %58, i10* %59, i10* %60, i10* %61, i10* %62, i10* %63, i10* %64, i10* %65, i10* %66, i10* %67, i10* %68, i10* %69, i10* %70, i10* %71, i10* %72, i10* %73, i10* %74, i10* %75, i10* %76, i10* %77, i10* %78, i10* %79, i10* %80, i10* %81, i10* %82, i10* %83, i10* %84, i10* %85, i10* %86, i10* %87, i10* %88, i10* %89, i10* %90, i10* %91, i10* %92, i10* %93, i10* %94, i10* %95, i10* %96, i10* %97, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %170, [4176 x i10]* %98, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %171, i10* %99, i10* %100, i10* %101, i10* %102, i10* %103, i10* %104, i10* %105, i10* %106, i10* %107, i10* %108, i10* %109, i10* %110, i10* %111, i10* %112, i10* %113, i10* %114, i10* %115, i10* %116, i10* %117, i10* %118, i10* %119, i10* %120, i10* %121, i10* %122, i10* %123, i10* %124, i10* %125, i10* %126, i10* %127, i10* %128, i10* %129, i10* %130, i10* %131, i10* %132, i10* %133, i10* %134, i10* %135, i10* %136, i10* %137, i10* %138, i10* %139, i10* %140, i10* %141, i10* %142, i10* %143, i10* %144, i10* %145, i10* %146, i10* %147, i10* %148, i10* %149, i10* %150, i10* %151, i10* %152, i10* %153, i10* %154, i10* %155, i10* %156, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %172, [58 x i10]* %157, [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %173, i10* %158)
  %174 = bitcast [273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %159 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %175 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %160 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %176 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %161 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %177 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %162 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %178 = bitcast [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %163 to %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"*
  %179 = bitcast [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %164 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  %180 = bitcast [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %165 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  %181 = bitcast [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %166 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  %182 = bitcast [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %167 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  %183 = bitcast [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %168 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  %184 = bitcast [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %169 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  %185 = bitcast [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %170 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  %186 = bitcast [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %171 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  %187 = bitcast [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %172 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  %188 = bitcast [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %173 to %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"*
  call void @myproject_hw_stub(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %174, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %175, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %176, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %177, %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"* %178, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %179, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %180, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %181, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %182, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %183, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %184, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %185, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %186, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %187, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* %188)
  call void @copy_in([273 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %159, i4368* %0, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %160, i16* %1, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %161, i16* %2, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %162, i16* %3, [1 x %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"]* %163, i8* %4, [18 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %164, [18 x i10]* %5, [2 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %165, i10* %6, i10* %7, [48 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %166, [48 x i10]* %8, [16 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %167, i10* %9, i10* %10, i10* %11, i10* %12, i10* %13, i10* %14, i10* %15, i10* %16, i10* %17, i10* %18, i10* %19, i10* %20, i10* %21, i10* %22, i10* %23, i10* %24, [9792 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %168, [9792 x i10]* %25, [72 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %169, i10* %26, i10* %27, i10* %28, i10* %29, i10* %30, i10* %31, i10* %32, i10* %33, i10* %34, i10* %35, i10* %36, i10* %37, i10* %38, i10* %39, i10* %40, i10* %41, i10* %42, i10* %43, i10* %44, i10* %45, i10* %46, i10* %47, i10* %48, i10* %49, i10* %50, i10* %51, i10* %52, i10* %53, i10* %54, i10* %55, i10* %56, i10* %57, i10* %58, i10* %59, i10* %60, i10* %61, i10* %62, i10* %63, i10* %64, i10* %65, i10* %66, i10* %67, i10* %68, i10* %69, i10* %70, i10* %71, i10* %72, i10* %73, i10* %74, i10* %75, i10* %76, i10* %77, i10* %78, i10* %79, i10* %80, i10* %81, i10* %82, i10* %83, i10* %84, i10* %85, i10* %86, i10* %87, i10* %88, i10* %89, i10* %90, i10* %91, i10* %92, i10* %93, i10* %94, i10* %95, i10* %96, i10* %97, [4176 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %170, [4176 x i10]* %98, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %171, i10* %99, i10* %100, i10* %101, i10* %102, i10* %103, i10* %104, i10* %105, i10* %106, i10* %107, i10* %108, i10* %109, i10* %110, i10* %111, i10* %112, i10* %113, i10* %114, i10* %115, i10* %116, i10* %117, i10* %118, i10* %119, i10* %120, i10* %121, i10* %122, i10* %123, i10* %124, i10* %125, i10* %126, i10* %127, i10* %128, i10* %129, i10* %130, i10* %131, i10* %132, i10* %133, i10* %134, i10* %135, i10* %136, i10* %137, i10* %138, i10* %139, i10* %140, i10* %141, i10* %142, i10* %143, i10* %144, i10* %145, i10* %146, i10* %147, i10* %148, i10* %149, i10* %150, i10* %151, i10* %152, i10* %153, i10* %154, i10* %155, i10* %156, [58 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %172, [58 x i10]* %157, [1 x %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"]* %173, i10* %158)
  ret void
}

declare void @myproject_hw_stub(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_ufixed<8, 0, AP_RND_CONV, AP_SAT, 0>"* noalias nocapture nonnull, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<10, 1, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly)

attributes #0 = { noinline "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #2 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="arraycpy_hls" }
attributes #3 = { nounwind willreturn }
attributes #4 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyin" }
attributes #5 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyout" }
attributes #6 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}
!datalayout.transforms.on.top = !{!5, !12, !18, !38, !114, !176}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
!5 = !{!6, !8, !10}
!6 = !{!7}
!7 = !{!"4.0", [1 x i8]* null}
!8 = !{!9}
!9 = !{!"array_partition", !"type=Complete", !"dim=1"}
!10 = !{!11}
!11 = !{!"4.0", i8* null}
!12 = !{!13, !8, !15}
!13 = !{!14}
!14 = !{!"6.0", [2 x i10]* null}
!15 = !{!16, !17}
!16 = !{!"6.0.0", i10* null}
!17 = !{!"6.0.1", i10* null}
!18 = !{!19, !8, !21}
!19 = !{!20}
!20 = !{!"8.0", [16 x i10]* null}
!21 = !{!22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37}
!22 = !{!"8.0.0", i10* null}
!23 = !{!"8.0.1", i10* null}
!24 = !{!"8.0.2", i10* null}
!25 = !{!"8.0.3", i10* null}
!26 = !{!"8.0.4", i10* null}
!27 = !{!"8.0.5", i10* null}
!28 = !{!"8.0.6", i10* null}
!29 = !{!"8.0.7", i10* null}
!30 = !{!"8.0.8", i10* null}
!31 = !{!"8.0.9", i10* null}
!32 = !{!"8.0.10", i10* null}
!33 = !{!"8.0.11", i10* null}
!34 = !{!"8.0.12", i10* null}
!35 = !{!"8.0.13", i10* null}
!36 = !{!"8.0.14", i10* null}
!37 = !{!"8.0.15", i10* null}
!38 = !{!39, !8, !41}
!39 = !{!40}
!40 = !{!"10.0", [72 x i10]* null}
!41 = !{!42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113}
!42 = !{!"10.0.0", i10* null}
!43 = !{!"10.0.1", i10* null}
!44 = !{!"10.0.2", i10* null}
!45 = !{!"10.0.3", i10* null}
!46 = !{!"10.0.4", i10* null}
!47 = !{!"10.0.5", i10* null}
!48 = !{!"10.0.6", i10* null}
!49 = !{!"10.0.7", i10* null}
!50 = !{!"10.0.8", i10* null}
!51 = !{!"10.0.9", i10* null}
!52 = !{!"10.0.10", i10* null}
!53 = !{!"10.0.11", i10* null}
!54 = !{!"10.0.12", i10* null}
!55 = !{!"10.0.13", i10* null}
!56 = !{!"10.0.14", i10* null}
!57 = !{!"10.0.15", i10* null}
!58 = !{!"10.0.16", i10* null}
!59 = !{!"10.0.17", i10* null}
!60 = !{!"10.0.18", i10* null}
!61 = !{!"10.0.19", i10* null}
!62 = !{!"10.0.20", i10* null}
!63 = !{!"10.0.21", i10* null}
!64 = !{!"10.0.22", i10* null}
!65 = !{!"10.0.23", i10* null}
!66 = !{!"10.0.24", i10* null}
!67 = !{!"10.0.25", i10* null}
!68 = !{!"10.0.26", i10* null}
!69 = !{!"10.0.27", i10* null}
!70 = !{!"10.0.28", i10* null}
!71 = !{!"10.0.29", i10* null}
!72 = !{!"10.0.30", i10* null}
!73 = !{!"10.0.31", i10* null}
!74 = !{!"10.0.32", i10* null}
!75 = !{!"10.0.33", i10* null}
!76 = !{!"10.0.34", i10* null}
!77 = !{!"10.0.35", i10* null}
!78 = !{!"10.0.36", i10* null}
!79 = !{!"10.0.37", i10* null}
!80 = !{!"10.0.38", i10* null}
!81 = !{!"10.0.39", i10* null}
!82 = !{!"10.0.40", i10* null}
!83 = !{!"10.0.41", i10* null}
!84 = !{!"10.0.42", i10* null}
!85 = !{!"10.0.43", i10* null}
!86 = !{!"10.0.44", i10* null}
!87 = !{!"10.0.45", i10* null}
!88 = !{!"10.0.46", i10* null}
!89 = !{!"10.0.47", i10* null}
!90 = !{!"10.0.48", i10* null}
!91 = !{!"10.0.49", i10* null}
!92 = !{!"10.0.50", i10* null}
!93 = !{!"10.0.51", i10* null}
!94 = !{!"10.0.52", i10* null}
!95 = !{!"10.0.53", i10* null}
!96 = !{!"10.0.54", i10* null}
!97 = !{!"10.0.55", i10* null}
!98 = !{!"10.0.56", i10* null}
!99 = !{!"10.0.57", i10* null}
!100 = !{!"10.0.58", i10* null}
!101 = !{!"10.0.59", i10* null}
!102 = !{!"10.0.60", i10* null}
!103 = !{!"10.0.61", i10* null}
!104 = !{!"10.0.62", i10* null}
!105 = !{!"10.0.63", i10* null}
!106 = !{!"10.0.64", i10* null}
!107 = !{!"10.0.65", i10* null}
!108 = !{!"10.0.66", i10* null}
!109 = !{!"10.0.67", i10* null}
!110 = !{!"10.0.68", i10* null}
!111 = !{!"10.0.69", i10* null}
!112 = !{!"10.0.70", i10* null}
!113 = !{!"10.0.71", i10* null}
!114 = !{!115, !8, !117}
!115 = !{!116}
!116 = !{!"12.0", [58 x i10]* null}
!117 = !{!118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140, !141, !142, !143, !144, !145, !146, !147, !148, !149, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174, !175}
!118 = !{!"12.0.0", i10* null}
!119 = !{!"12.0.1", i10* null}
!120 = !{!"12.0.2", i10* null}
!121 = !{!"12.0.3", i10* null}
!122 = !{!"12.0.4", i10* null}
!123 = !{!"12.0.5", i10* null}
!124 = !{!"12.0.6", i10* null}
!125 = !{!"12.0.7", i10* null}
!126 = !{!"12.0.8", i10* null}
!127 = !{!"12.0.9", i10* null}
!128 = !{!"12.0.10", i10* null}
!129 = !{!"12.0.11", i10* null}
!130 = !{!"12.0.12", i10* null}
!131 = !{!"12.0.13", i10* null}
!132 = !{!"12.0.14", i10* null}
!133 = !{!"12.0.15", i10* null}
!134 = !{!"12.0.16", i10* null}
!135 = !{!"12.0.17", i10* null}
!136 = !{!"12.0.18", i10* null}
!137 = !{!"12.0.19", i10* null}
!138 = !{!"12.0.20", i10* null}
!139 = !{!"12.0.21", i10* null}
!140 = !{!"12.0.22", i10* null}
!141 = !{!"12.0.23", i10* null}
!142 = !{!"12.0.24", i10* null}
!143 = !{!"12.0.25", i10* null}
!144 = !{!"12.0.26", i10* null}
!145 = !{!"12.0.27", i10* null}
!146 = !{!"12.0.28", i10* null}
!147 = !{!"12.0.29", i10* null}
!148 = !{!"12.0.30", i10* null}
!149 = !{!"12.0.31", i10* null}
!150 = !{!"12.0.32", i10* null}
!151 = !{!"12.0.33", i10* null}
!152 = !{!"12.0.34", i10* null}
!153 = !{!"12.0.35", i10* null}
!154 = !{!"12.0.36", i10* null}
!155 = !{!"12.0.37", i10* null}
!156 = !{!"12.0.38", i10* null}
!157 = !{!"12.0.39", i10* null}
!158 = !{!"12.0.40", i10* null}
!159 = !{!"12.0.41", i10* null}
!160 = !{!"12.0.42", i10* null}
!161 = !{!"12.0.43", i10* null}
!162 = !{!"12.0.44", i10* null}
!163 = !{!"12.0.45", i10* null}
!164 = !{!"12.0.46", i10* null}
!165 = !{!"12.0.47", i10* null}
!166 = !{!"12.0.48", i10* null}
!167 = !{!"12.0.49", i10* null}
!168 = !{!"12.0.50", i10* null}
!169 = !{!"12.0.51", i10* null}
!170 = !{!"12.0.52", i10* null}
!171 = !{!"12.0.53", i10* null}
!172 = !{!"12.0.54", i10* null}
!173 = !{!"12.0.55", i10* null}
!174 = !{!"12.0.56", i10* null}
!175 = !{!"12.0.57", i10* null}
!176 = !{!177, !8, !179}
!177 = !{!178}
!178 = !{!"14.0", [1 x i10]* null}
!179 = !{!180}
!180 = !{!"14.0", i10* null}
