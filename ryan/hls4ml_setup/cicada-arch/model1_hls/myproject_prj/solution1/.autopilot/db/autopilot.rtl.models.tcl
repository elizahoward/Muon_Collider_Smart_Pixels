set SynModuleInfo {
  {SRCNAME concatenate1d<ap_fixed,ap_fixed,ap_fixed<16,6,5,3,0>,config3> MODELNAME concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config3_s RTLNAME myproject_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config3_s}
  {SRCNAME concatenate1d<ap_fixed,ap_fixed,ap_fixed<16,6,5,3,0>,config5> MODELNAME concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config5_s RTLNAME myproject_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config5_s}
  {SRCNAME concatenate1d<ap_fixed,ap_fixed,ap_fixed<16,6,5,3,0>,config7> MODELNAME concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_s RTLNAME myproject_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_s}
  {SRCNAME {dense_latency<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, config8>} MODELNAME dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s RTLNAME myproject_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s
    SUBMODULES {
      {MODELNAME myproject_mul_16s_10s_26_2_0 RTLNAME myproject_mul_16s_10s_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_9s_25_2_0 RTLNAME myproject_mul_16s_9s_25_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_13ns_26_2_0 RTLNAME myproject_mul_16s_13ns_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_8s_24_2_0 RTLNAME myproject_mul_16s_8s_24_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_11s_26_2_0 RTLNAME myproject_mul_16s_11s_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_13s_26_2_0 RTLNAME myproject_mul_16s_13s_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_10ns_26_2_0 RTLNAME myproject_mul_16s_10ns_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_11ns_26_2_0 RTLNAME myproject_mul_16s_11ns_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_8ns_24_2_0 RTLNAME myproject_mul_16s_8ns_24_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_7ns_23_2_0 RTLNAME myproject_mul_16s_7ns_23_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {relu<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, relu_config9>} MODELNAME relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config9_s RTLNAME myproject_relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config9_s}
  {SRCNAME {dense_latency<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, config10>} MODELNAME dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s RTLNAME myproject_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s
    SUBMODULES {
      {MODELNAME myproject_mul_16s_14ns_26_2_0 RTLNAME myproject_mul_16s_14ns_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_12s_26_2_0 RTLNAME myproject_mul_16s_12s_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_12ns_26_2_0 RTLNAME myproject_mul_16s_12ns_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {relu<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, relu_config11>} MODELNAME relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config11_s RTLNAME myproject_relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config11_s}
  {SRCNAME {dense_latency<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, config12>} MODELNAME dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s RTLNAME myproject_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s
    SUBMODULES {
      {MODELNAME myproject_mul_16s_14s_26_2_0 RTLNAME myproject_mul_16s_14s_26_2_0 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {sigmoid<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, sigmoid_config13>} MODELNAME sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s RTLNAME myproject_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s
    SUBMODULES {
      {MODELNAME myproject_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s_sigmoid_tabkb RTLNAME myproject_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s_sigmoid_tabkb BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME myproject MODELNAME myproject RTLNAME myproject IS_TOP 1}
}
