set moduleName myproject
set isTopModule 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type function
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {myproject}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ x_profile int 336 regular {pointer 0}  }
	{ nModule int 16 regular {pointer 0}  }
	{ x_local int 16 regular {pointer 0}  }
	{ y_profile int 208 regular {pointer 0}  }
	{ y_local int 16 regular {pointer 0}  }
	{ layer24_out int 8 regular {pointer 1}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "x_profile", "interface" : "wire", "bitwidth" : 336, "direction" : "READONLY"} , 
 	{ "Name" : "nModule", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "x_local", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "y_profile", "interface" : "wire", "bitwidth" : 208, "direction" : "READONLY"} , 
 	{ "Name" : "y_local", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "layer24_out", "interface" : "wire", "bitwidth" : 8, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 18
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ x_profile_ap_vld sc_in sc_logic 1 invld 0 } 
	{ y_profile_ap_vld sc_in sc_logic 1 invld 3 } 
	{ y_local_ap_vld sc_in sc_logic 1 invld 4 } 
	{ nModule_ap_vld sc_in sc_logic 1 invld 1 } 
	{ x_local_ap_vld sc_in sc_logic 1 invld 2 } 
	{ x_profile sc_in sc_lv 336 signal 0 } 
	{ nModule sc_in sc_lv 16 signal 1 } 
	{ x_local sc_in sc_lv 16 signal 2 } 
	{ y_profile sc_in sc_lv 208 signal 3 } 
	{ y_local sc_in sc_lv 16 signal 4 } 
	{ layer24_out sc_out sc_lv 8 signal 5 } 
	{ layer24_out_ap_vld sc_out sc_logic 1 outvld 5 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "x_profile_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "x_profile", "role": "ap_vld" }} , 
 	{ "name": "y_profile_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "y_profile", "role": "ap_vld" }} , 
 	{ "name": "y_local_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "y_local", "role": "ap_vld" }} , 
 	{ "name": "nModule_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "nModule", "role": "ap_vld" }} , 
 	{ "name": "x_local_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "x_local", "role": "ap_vld" }} , 
 	{ "name": "x_profile", "direction": "in", "datatype": "sc_lv", "bitwidth":336, "type": "signal", "bundle":{"name": "x_profile", "role": "default" }} , 
 	{ "name": "nModule", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "nModule", "role": "default" }} , 
 	{ "name": "x_local", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "x_local", "role": "default" }} , 
 	{ "name": "y_profile", "direction": "in", "datatype": "sc_lv", "bitwidth":208, "type": "signal", "bundle":{"name": "y_profile", "role": "default" }} , 
 	{ "name": "y_local", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "y_local", "role": "default" }} , 
 	{ "name": "layer24_out", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "layer24_out", "role": "default" }} , 
 	{ "name": "layer24_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "layer24_out", "role": "ap_vld" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "105", "106", "108", "109", "110", "111", "148", "149", "156", "157"],
		"CDFG" : "myproject",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "29", "EstimateLatencyMin" : "29", "EstimateLatencyMax" : "29",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "x_profile", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "x_profile_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "nModule", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "nModule_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "x_local", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "x_local_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "y_profile", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "y_profile_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "y_local", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "y_local_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer24_out", "Type" : "Vld", "Direction" : "O"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret1_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config3_s_fu_143", "Parent" : "0",
		"CDFG" : "concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config3_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "1", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "1",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret2_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_s_fu_151", "Parent" : "0",
		"CDFG" : "concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "1", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "1",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data1_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_6_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_7_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_8_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_9_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_10_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_11_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_12_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_13_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_14_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_15_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_16_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_17_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_18_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_19_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_20_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_21_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_22_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_23_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_24_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_25_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_26_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_27_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_28_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_29_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_30_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_31_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_32_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_33_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191", "Parent" : "0", "Child" : ["4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104"],
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "6", "EstimateLatencyMin" : "6", "EstimateLatencyMax" : "6",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_6_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_7_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_8_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_9_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_10_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_11_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_12_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_13_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_14_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_15_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_16_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_17_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_18_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_19_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_20_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_21_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_22_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_23_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_24_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_25_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_26_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_27_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_28_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_29_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_30_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_31_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_32_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_33_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_34_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U40", "Parent" : "3"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U41", "Parent" : "3"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U42", "Parent" : "3"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5ns_21_2_0_U43", "Parent" : "3"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U44", "Parent" : "3"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U45", "Parent" : "3"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U46", "Parent" : "3"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U47", "Parent" : "3"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U48", "Parent" : "3"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U49", "Parent" : "3"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U50", "Parent" : "3"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U51", "Parent" : "3"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U52", "Parent" : "3"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U53", "Parent" : "3"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U54", "Parent" : "3"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U55", "Parent" : "3"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U56", "Parent" : "3"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U57", "Parent" : "3"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U58", "Parent" : "3"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5s_21_2_0_U59", "Parent" : "3"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5s_21_2_0_U60", "Parent" : "3"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U61", "Parent" : "3"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5s_21_2_0_U62", "Parent" : "3"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U63", "Parent" : "3"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U64", "Parent" : "3"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U65", "Parent" : "3"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5ns_21_2_0_U66", "Parent" : "3"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U67", "Parent" : "3"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U68", "Parent" : "3"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5s_21_2_0_U69", "Parent" : "3"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U70", "Parent" : "3"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5s_21_2_0_U71", "Parent" : "3"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U72", "Parent" : "3"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U73", "Parent" : "3"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U74", "Parent" : "3"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U75", "Parent" : "3"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5ns_21_2_0_U76", "Parent" : "3"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U77", "Parent" : "3"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U78", "Parent" : "3"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U79", "Parent" : "3"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U80", "Parent" : "3"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8ns_23_2_0_U81", "Parent" : "3"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U82", "Parent" : "3"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U83", "Parent" : "3"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8ns_23_2_0_U84", "Parent" : "3"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5s_21_2_0_U85", "Parent" : "3"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U86", "Parent" : "3"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U87", "Parent" : "3"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U88", "Parent" : "3"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U89", "Parent" : "3"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U90", "Parent" : "3"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8ns_23_2_0_U91", "Parent" : "3"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U92", "Parent" : "3"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U93", "Parent" : "3"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U94", "Parent" : "3"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U95", "Parent" : "3"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U96", "Parent" : "3"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U97", "Parent" : "3"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U98", "Parent" : "3"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8ns_23_2_0_U99", "Parent" : "3"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U100", "Parent" : "3"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U101", "Parent" : "3"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U102", "Parent" : "3"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8ns_23_2_0_U103", "Parent" : "3"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U104", "Parent" : "3"},
	{"ID" : "69", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U105", "Parent" : "3"},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U106", "Parent" : "3"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U107", "Parent" : "3"},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U108", "Parent" : "3"},
	{"ID" : "73", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5ns_21_2_0_U109", "Parent" : "3"},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U110", "Parent" : "3"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U111", "Parent" : "3"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U112", "Parent" : "3"},
	{"ID" : "77", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U113", "Parent" : "3"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U114", "Parent" : "3"},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U115", "Parent" : "3"},
	{"ID" : "80", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U116", "Parent" : "3"},
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U117", "Parent" : "3"},
	{"ID" : "82", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U118", "Parent" : "3"},
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U119", "Parent" : "3"},
	{"ID" : "84", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5s_21_2_0_U120", "Parent" : "3"},
	{"ID" : "85", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U121", "Parent" : "3"},
	{"ID" : "86", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U122", "Parent" : "3"},
	{"ID" : "87", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5ns_21_2_0_U123", "Parent" : "3"},
	{"ID" : "88", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U124", "Parent" : "3"},
	{"ID" : "89", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U125", "Parent" : "3"},
	{"ID" : "90", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_5ns_21_2_0_U126", "Parent" : "3"},
	{"ID" : "91", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U127", "Parent" : "3"},
	{"ID" : "92", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U128", "Parent" : "3"},
	{"ID" : "93", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U129", "Parent" : "3"},
	{"ID" : "94", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U130", "Parent" : "3"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U131", "Parent" : "3"},
	{"ID" : "96", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U132", "Parent" : "3"},
	{"ID" : "97", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U133", "Parent" : "3"},
	{"ID" : "98", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U134", "Parent" : "3"},
	{"ID" : "99", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7ns_23_2_0_U135", "Parent" : "3"},
	{"ID" : "100", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_8s_23_2_0_U136", "Parent" : "3"},
	{"ID" : "101", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_7s_23_2_0_U137", "Parent" : "3"},
	{"ID" : "102", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6s_22_2_0_U138", "Parent" : "3"},
	{"ID" : "103", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U139", "Parent" : "3"},
	{"ID" : "104", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_191.mul_16s_6ns_22_2_0_U140", "Parent" : "3"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret3_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config8_s_fu_230", "Parent" : "0",
		"CDFG" : "concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config8_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "1", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "1",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config11_s_fu_236", "Parent" : "0", "Child" : ["107"],
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config11_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "2", "EstimateLatencyMin" : "2", "EstimateLatencyMax" : "2",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config11_s_fu_236.mul_16s_6ns_21_2_1_U184", "Parent" : "106"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config13_s_fu_242", "Parent" : "0",
		"CDFG" : "relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config13_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_6_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_7_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config14_s_fu_254", "Parent" : "0",
		"CDFG" : "relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config14_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret8_concatenate1d_ap_ufixed_ap_ufixed_ap_fixed_16_6_5_3_0_config15_s_fu_264", "Parent" : "0",
		"CDFG" : "concatenate1d_ap_ufixed_ap_ufixed_ap_fixed_16_6_5_3_0_config15_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "1", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "1",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data1_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_6_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_7_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_5_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "111", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282", "Parent" : "0", "Child" : ["112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147"],
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "5", "EstimateLatencyMin" : "5", "EstimateLatencyMax" : "5",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_6_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_7_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_8_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_9_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_10_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_11_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_12_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_13_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "112", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7s_23_2_0_U216", "Parent" : "111"},
	{"ID" : "113", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6ns_22_2_0_U217", "Parent" : "111"},
	{"ID" : "114", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6ns_22_2_0_U218", "Parent" : "111"},
	{"ID" : "115", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7ns_23_2_0_U219", "Parent" : "111"},
	{"ID" : "116", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7ns_23_2_0_U220", "Parent" : "111"},
	{"ID" : "117", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7s_23_2_0_U221", "Parent" : "111"},
	{"ID" : "118", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7ns_23_2_0_U222", "Parent" : "111"},
	{"ID" : "119", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8ns_23_2_0_U223", "Parent" : "111"},
	{"ID" : "120", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7s_23_2_0_U224", "Parent" : "111"},
	{"ID" : "121", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6ns_22_2_0_U225", "Parent" : "111"},
	{"ID" : "122", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7ns_23_2_0_U226", "Parent" : "111"},
	{"ID" : "123", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6s_22_2_0_U227", "Parent" : "111"},
	{"ID" : "124", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8ns_23_2_0_U228", "Parent" : "111"},
	{"ID" : "125", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6s_22_2_0_U229", "Parent" : "111"},
	{"ID" : "126", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8s_23_2_0_U230", "Parent" : "111"},
	{"ID" : "127", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8ns_23_2_0_U231", "Parent" : "111"},
	{"ID" : "128", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6s_22_2_0_U232", "Parent" : "111"},
	{"ID" : "129", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8s_23_2_0_U233", "Parent" : "111"},
	{"ID" : "130", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7s_23_2_0_U234", "Parent" : "111"},
	{"ID" : "131", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6s_22_2_0_U235", "Parent" : "111"},
	{"ID" : "132", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6ns_22_2_0_U236", "Parent" : "111"},
	{"ID" : "133", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7ns_23_2_0_U237", "Parent" : "111"},
	{"ID" : "134", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8ns_23_2_0_U238", "Parent" : "111"},
	{"ID" : "135", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6s_22_2_0_U239", "Parent" : "111"},
	{"ID" : "136", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8ns_23_2_0_U240", "Parent" : "111"},
	{"ID" : "137", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7s_23_2_0_U241", "Parent" : "111"},
	{"ID" : "138", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7ns_23_2_0_U242", "Parent" : "111"},
	{"ID" : "139", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6ns_22_2_0_U243", "Parent" : "111"},
	{"ID" : "140", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8ns_23_2_0_U244", "Parent" : "111"},
	{"ID" : "141", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8s_23_2_0_U245", "Parent" : "111"},
	{"ID" : "142", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6ns_22_2_0_U246", "Parent" : "111"},
	{"ID" : "143", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7s_23_2_0_U247", "Parent" : "111"},
	{"ID" : "144", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_6ns_22_2_0_U248", "Parent" : "111"},
	{"ID" : "145", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8s_23_2_0_U249", "Parent" : "111"},
	{"ID" : "146", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_7ns_23_2_0_U250", "Parent" : "111"},
	{"ID" : "147", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_282.mul_16s_8ns_23_2_0_U251", "Parent" : "111"},
	{"ID" : "148", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config18_s_fu_300", "Parent" : "0",
		"CDFG" : "relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config18_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "149", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s_fu_308", "Parent" : "0", "Child" : ["150", "151", "152", "153", "154", "155"],
		"CDFG" : "dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "2", "EstimateLatencyMin" : "2", "EstimateLatencyMax" : "2",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "150", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s_fu_308.mul_8ns_7s_15_1_1_U270", "Parent" : "149"},
	{"ID" : "151", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s_fu_308.mul_8ns_8ns_15_1_1_U271", "Parent" : "149"},
	{"ID" : "152", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s_fu_308.mul_8ns_8ns_15_1_1_U272", "Parent" : "149"},
	{"ID" : "153", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s_fu_308.mul_8ns_8s_16_1_1_U273", "Parent" : "149"},
	{"ID" : "154", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s_fu_308.mul_8ns_7ns_14_1_1_U274", "Parent" : "149"},
	{"ID" : "155", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s_fu_308.mul_8ns_8ns_15_1_1_U275", "Parent" : "149"},
	{"ID" : "156", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config21_s_fu_316", "Parent" : "0",
		"CDFG" : "relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config21_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "157", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config22_s_fu_324", "Parent" : "0", "Child" : ["158", "159"],
		"CDFG" : "dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config22_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "2", "EstimateLatencyMin" : "2", "EstimateLatencyMax" : "2",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "158", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config22_s_fu_324.mul_8ns_7ns_14_1_1_U288", "Parent" : "157"},
	{"ID" : "159", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config22_s_fu_324.mul_8ns_8ns_15_1_1_U289", "Parent" : "157"}]}


set ArgLastReadFirstWriteLatency {
	myproject {
		x_profile {Type I LastRead 0 FirstWrite -1}
		nModule {Type I LastRead 0 FirstWrite -1}
		x_local {Type I LastRead 0 FirstWrite -1}
		y_profile {Type I LastRead 0 FirstWrite -1}
		y_local {Type I LastRead 0 FirstWrite -1}
		layer24_out {Type O LastRead -1 FirstWrite 29}}
	concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config3_s {
		data1_val {Type I LastRead 0 FirstWrite -1}
		data2_val {Type I LastRead 0 FirstWrite -1}}
	concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_s {
		data1_0_val {Type I LastRead 0 FirstWrite -1}
		data1_1_val {Type I LastRead 0 FirstWrite -1}
		data1_2_val {Type I LastRead 0 FirstWrite -1}
		data1_3_val {Type I LastRead 0 FirstWrite -1}
		data1_4_val {Type I LastRead 0 FirstWrite -1}
		data1_5_val {Type I LastRead 0 FirstWrite -1}
		data1_6_val {Type I LastRead 0 FirstWrite -1}
		data1_7_val {Type I LastRead 0 FirstWrite -1}
		data1_8_val {Type I LastRead 0 FirstWrite -1}
		data1_9_val {Type I LastRead 0 FirstWrite -1}
		data1_10_val {Type I LastRead 0 FirstWrite -1}
		data1_11_val {Type I LastRead 0 FirstWrite -1}
		data1_12_val {Type I LastRead 0 FirstWrite -1}
		data1_13_val {Type I LastRead 0 FirstWrite -1}
		data1_14_val {Type I LastRead 0 FirstWrite -1}
		data1_15_val {Type I LastRead 0 FirstWrite -1}
		data1_16_val {Type I LastRead 0 FirstWrite -1}
		data1_17_val {Type I LastRead 0 FirstWrite -1}
		data1_18_val {Type I LastRead 0 FirstWrite -1}
		data1_19_val {Type I LastRead 0 FirstWrite -1}
		data1_20_val {Type I LastRead 0 FirstWrite -1}
		data1_21_val {Type I LastRead 0 FirstWrite -1}
		data1_22_val {Type I LastRead 0 FirstWrite -1}
		data1_23_val {Type I LastRead 0 FirstWrite -1}
		data1_24_val {Type I LastRead 0 FirstWrite -1}
		data1_25_val {Type I LastRead 0 FirstWrite -1}
		data1_26_val {Type I LastRead 0 FirstWrite -1}
		data1_27_val {Type I LastRead 0 FirstWrite -1}
		data1_28_val {Type I LastRead 0 FirstWrite -1}
		data1_29_val {Type I LastRead 0 FirstWrite -1}
		data1_30_val {Type I LastRead 0 FirstWrite -1}
		data1_31_val {Type I LastRead 0 FirstWrite -1}
		data1_32_val {Type I LastRead 0 FirstWrite -1}
		data1_33_val {Type I LastRead 0 FirstWrite -1}
		data2_val {Type I LastRead 0 FirstWrite -1}}
	dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}
		data_4_val {Type I LastRead 0 FirstWrite -1}
		data_5_val {Type I LastRead 0 FirstWrite -1}
		data_6_val {Type I LastRead 0 FirstWrite -1}
		data_7_val {Type I LastRead 0 FirstWrite -1}
		data_8_val {Type I LastRead 0 FirstWrite -1}
		data_9_val {Type I LastRead 0 FirstWrite -1}
		data_10_val {Type I LastRead 0 FirstWrite -1}
		data_11_val {Type I LastRead 0 FirstWrite -1}
		data_12_val {Type I LastRead 0 FirstWrite -1}
		data_13_val {Type I LastRead 0 FirstWrite -1}
		data_14_val {Type I LastRead 0 FirstWrite -1}
		data_15_val {Type I LastRead 0 FirstWrite -1}
		data_16_val {Type I LastRead 0 FirstWrite -1}
		data_17_val {Type I LastRead 0 FirstWrite -1}
		data_18_val {Type I LastRead 0 FirstWrite -1}
		data_19_val {Type I LastRead 0 FirstWrite -1}
		data_20_val {Type I LastRead 0 FirstWrite -1}
		data_21_val {Type I LastRead 0 FirstWrite -1}
		data_22_val {Type I LastRead 0 FirstWrite -1}
		data_23_val {Type I LastRead 0 FirstWrite -1}
		data_24_val {Type I LastRead 0 FirstWrite -1}
		data_25_val {Type I LastRead 0 FirstWrite -1}
		data_26_val {Type I LastRead 0 FirstWrite -1}
		data_27_val {Type I LastRead 0 FirstWrite -1}
		data_28_val {Type I LastRead 0 FirstWrite -1}
		data_29_val {Type I LastRead 0 FirstWrite -1}
		data_30_val {Type I LastRead 0 FirstWrite -1}
		data_31_val {Type I LastRead 0 FirstWrite -1}
		data_32_val {Type I LastRead 0 FirstWrite -1}
		data_33_val {Type I LastRead 0 FirstWrite -1}
		data_34_val {Type I LastRead 0 FirstWrite -1}}
	concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config8_s {
		data1_val {Type I LastRead 0 FirstWrite -1}
		data2_val {Type I LastRead 0 FirstWrite -1}}
	dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config11_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}}
	relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config13_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}
		data_4_val {Type I LastRead 0 FirstWrite -1}
		data_5_val {Type I LastRead 0 FirstWrite -1}
		data_6_val {Type I LastRead 0 FirstWrite -1}
		data_7_val {Type I LastRead 0 FirstWrite -1}}
	relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config14_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}
		data_4_val {Type I LastRead 0 FirstWrite -1}
		data_5_val {Type I LastRead 0 FirstWrite -1}}
	concatenate1d_ap_ufixed_ap_ufixed_ap_fixed_16_6_5_3_0_config15_s {
		data1_0_val {Type I LastRead 0 FirstWrite -1}
		data1_1_val {Type I LastRead 0 FirstWrite -1}
		data1_2_val {Type I LastRead 0 FirstWrite -1}
		data1_3_val {Type I LastRead 0 FirstWrite -1}
		data1_4_val {Type I LastRead 0 FirstWrite -1}
		data1_5_val {Type I LastRead 0 FirstWrite -1}
		data1_6_val {Type I LastRead 0 FirstWrite -1}
		data1_7_val {Type I LastRead 0 FirstWrite -1}
		data2_0_val {Type I LastRead 0 FirstWrite -1}
		data2_1_val {Type I LastRead 0 FirstWrite -1}
		data2_2_val {Type I LastRead 0 FirstWrite -1}
		data2_3_val {Type I LastRead 0 FirstWrite -1}
		data2_4_val {Type I LastRead 0 FirstWrite -1}
		data2_5_val {Type I LastRead 0 FirstWrite -1}}
	dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}
		data_4_val {Type I LastRead 0 FirstWrite -1}
		data_5_val {Type I LastRead 0 FirstWrite -1}
		data_6_val {Type I LastRead 0 FirstWrite -1}
		data_7_val {Type I LastRead 0 FirstWrite -1}
		data_8_val {Type I LastRead 0 FirstWrite -1}
		data_9_val {Type I LastRead 0 FirstWrite -1}
		data_10_val {Type I LastRead 0 FirstWrite -1}
		data_11_val {Type I LastRead 0 FirstWrite -1}
		data_12_val {Type I LastRead 0 FirstWrite -1}
		data_13_val {Type I LastRead 0 FirstWrite -1}}
	relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config18_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}}
	dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}}
	relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config21_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}}
	dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config22_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "29", "Max" : "29"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	x_profile { ap_vld {  { x_profile_ap_vld in_vld 0 1 }  { x_profile in_data 0 336 } } }
	nModule { ap_vld {  { nModule_ap_vld in_vld 0 1 }  { nModule in_data 0 16 } } }
	x_local { ap_vld {  { x_local_ap_vld in_vld 0 1 }  { x_local in_data 0 16 } } }
	y_profile { ap_vld {  { y_profile_ap_vld in_vld 0 1 }  { y_profile in_data 0 208 } } }
	y_local { ap_vld {  { y_local_ap_vld in_vld 0 1 }  { y_local in_data 0 16 } } }
	layer24_out { ap_vld {  { layer24_out out_data 1 8 }  { layer24_out_ap_vld out_vld 1 1 } } }
}

set maxi_interface_dict [dict create]

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
