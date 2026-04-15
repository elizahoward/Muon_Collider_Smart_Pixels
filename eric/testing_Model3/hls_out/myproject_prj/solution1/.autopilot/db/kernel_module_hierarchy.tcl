set ModuleHierarchy {[{
"Name" : "myproject","ID" : "0","Type" : "dataflow",
"SubInsts" : [
	{"Name" : "entry_proc_U0","ID" : "1","Type" : "sequential"},
	{"Name" : "conv_2d_cl_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config5_U0","ID" : "2","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "PartitionLoop","ID" : "3","Type" : "no",
		"SubInsts" : [
		{"Name" : "grp_fill_buffer_fu_3567","ID" : "4","Type" : "sequential"},],
		"SubLoops" : [
		{"Name" : "ReuseLoop","ID" : "5","Type" : "pipeline"},]},]},
	{"Name" : "concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_U0","ID" : "6","Type" : "pipeline"},
	{"Name" : "concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config10_U0","ID" : "7","Type" : "pipeline"},
	{"Name" : "dense_resource_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_U0","ID" : "8","Type" : "pipeline",
		"SubLoops" : [
		{"Name" : "ReuseLoop","ID" : "9","Type" : "pipeline"},]},
	{"Name" : "relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config9_U0","ID" : "10","Type" : "pipeline"},
	{"Name" : "relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config15_U0","ID" : "11","Type" : "pipeline"},
	{"Name" : "pooling2d_cl_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config11_U0","ID" : "12","Type" : "pipeline",
		"SubInsts" : [
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5816","ID" : "13","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5796","ID" : "14","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5804","ID" : "15","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5798","ID" : "16","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5794","ID" : "17","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5815","ID" : "18","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5792","ID" : "19","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5807","ID" : "20","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5801","ID" : "21","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5810","ID" : "22","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5790","ID" : "23","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5808","ID" : "24","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5809","ID" : "25","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5799","ID" : "26","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5814","ID" : "27","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5817","ID" : "28","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5797","ID" : "29","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5805","ID" : "30","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5803","ID" : "31","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5795","ID" : "32","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5813","ID" : "33","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5793","ID" : "34","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5812","ID" : "35","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5802","ID" : "36","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5811","ID" : "37","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5791","ID" : "38","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5806","ID" : "39","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5800","ID" : "40","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5819","ID" : "41","Type" : "pipeline"},
		{"Name" : "grp_pool_op_ap_ufixed_8_0_4_0_0_4_0_s_fu_5818","ID" : "42","Type" : "pipeline"},]},
	{"Name" : "concatenate1d_ap_fixed_ap_ufixed_ap_fixed_16_6_5_3_0_config16_U0","ID" : "43","Type" : "pipeline"},
	{"Name" : "dense_resource_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config17_U0","ID" : "44","Type" : "pipeline",
		"SubLoops" : [
		{"Name" : "ReuseLoop","ID" : "45","Type" : "pipeline"},]},
	{"Name" : "relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config19_U0","ID" : "46","Type" : "pipeline"},
	{"Name" : "dense_resource_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config20_U0","ID" : "47","Type" : "pipeline",
		"SubLoops" : [
		{"Name" : "ReuseLoop","ID" : "48","Type" : "pipeline"},]},
	{"Name" : "relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config22_U0","ID" : "49","Type" : "pipeline"},
	{"Name" : "layer23_out_U","ID" : "50","Type" : "pipeline",
		"SubLoops" : [
		{"Name" : "ReuseLoop","ID" : "51","Type" : "pipeline"},]},
	{"Name" : "hard_tanh_ap_fixed_16_6_5_3_0_ap_fixed_8_1_4_0_0_hard_tanh_config25_U0","ID" : "52","Type" : "sequential"},]
}]}