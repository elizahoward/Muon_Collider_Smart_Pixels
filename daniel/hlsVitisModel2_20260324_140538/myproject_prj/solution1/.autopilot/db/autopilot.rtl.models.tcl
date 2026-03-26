set SynModuleInfo {
  {SRCNAME concatenate1d<ap_fixed,ap_fixed,ap_fixed<16,6,5,3,0>,config3> MODELNAME concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config3_s RTLNAME myproject_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config3_s}
  {SRCNAME concatenate1d<ap_fixed,ap_fixed,ap_fixed<16,6,5,3,0>,config7> MODELNAME concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_s RTLNAME myproject_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_s}
  {SRCNAME concatenate1d<ap_fixed,ap_fixed,ap_fixed<16,6,5,3,0>,config8> MODELNAME concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config8_s RTLNAME myproject_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config8_s}
  {SRCNAME {dense_latency<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, config9>} MODELNAME dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s RTLNAME myproject_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s
    SUBMODULES {
      {MODELNAME myproject_mul_16s_8s_23_2_0 RTLNAME myproject_mul_16s_8s_23_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_7s_23_2_0 RTLNAME myproject_mul_16s_7s_23_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_6s_22_2_0 RTLNAME myproject_mul_16s_6s_22_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_5ns_21_2_0 RTLNAME myproject_mul_16s_5ns_21_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_7ns_23_2_0 RTLNAME myproject_mul_16s_7ns_23_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_6ns_22_2_0 RTLNAME myproject_mul_16s_6ns_22_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_5s_21_2_0 RTLNAME myproject_mul_16s_5s_21_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_8ns_23_2_0 RTLNAME myproject_mul_16s_8ns_23_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {dense_latency<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, config11>} MODELNAME dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config11_s RTLNAME myproject_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config11_s
    SUBMODULES {
      {MODELNAME myproject_mul_16s_6ns_21_2_1 RTLNAME myproject_mul_16s_6ns_21_2_1 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {relu<ap_fixed<16, 6, 5, 3, 0>, ap_ufixed<8, 0, 4, 0, 0>, relu_config13>} MODELNAME relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config13_s RTLNAME myproject_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config13_s}
  {SRCNAME {relu<ap_fixed<16, 6, 5, 3, 0>, ap_ufixed<8, 0, 4, 0, 0>, relu_config14>} MODELNAME relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config14_s RTLNAME myproject_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config14_s}
  {SRCNAME concatenate1d<ap_ufixed,ap_ufixed,ap_fixed<16,6,5,3,0>,config15> MODELNAME concatenate1d_ap_ufixed_ap_ufixed_ap_fixed_16_6_5_3_0_config15_s RTLNAME myproject_concatenate1d_ap_ufixed_ap_ufixed_ap_fixed_16_6_5_3_0_config15_s}
  {SRCNAME {dense_latency<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, config16>} MODELNAME dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s RTLNAME myproject_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s}
  {SRCNAME {relu<ap_fixed<16, 6, 5, 3, 0>, ap_ufixed<8, 0, 4, 0, 0>, relu_config18>} MODELNAME relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config18_s RTLNAME myproject_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config18_s}
  {SRCNAME {dense_latency<ap_ufixed<8, 0, 4, 0, 0>, ap_fixed<16, 6, 5, 3, 0>, config19>} MODELNAME dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s RTLNAME myproject_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s
    SUBMODULES {
      {MODELNAME myproject_mul_8ns_7s_15_1_1 RTLNAME myproject_mul_8ns_7s_15_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_8ns_8ns_15_1_1 RTLNAME myproject_mul_8ns_8ns_15_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_8ns_8s_16_1_1 RTLNAME myproject_mul_8ns_8s_16_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_8ns_7ns_14_1_1 RTLNAME myproject_mul_8ns_7ns_14_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {relu<ap_fixed<16, 6, 5, 3, 0>, ap_ufixed<8, 0, 4, 0, 0>, relu_config21>} MODELNAME relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config21_s RTLNAME myproject_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config21_s}
  {SRCNAME {dense_latency<ap_ufixed<8, 0, 4, 0, 0>, ap_fixed<16, 6, 5, 3, 0>, config22>} MODELNAME dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config22_s RTLNAME myproject_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config22_s}
  {SRCNAME myproject MODELNAME myproject RTLNAME myproject IS_TOP 1}
}
