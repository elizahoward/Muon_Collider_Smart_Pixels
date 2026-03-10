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
	{ z_global int 16 regular {pointer 0}  }
	{ x_size int 16 regular {pointer 0}  }
	{ y_size int 16 regular {pointer 0}  }
	{ y_local int 16 regular {pointer 0}  }
	{ layer13_out int 16 regular {pointer 1}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "z_global", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "x_size", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "y_size", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "y_local", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "layer13_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 16
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ z_global_ap_vld sc_in sc_logic 1 invld 0 } 
	{ x_size_ap_vld sc_in sc_logic 1 invld 1 } 
	{ y_size_ap_vld sc_in sc_logic 1 invld 2 } 
	{ y_local_ap_vld sc_in sc_logic 1 invld 3 } 
	{ z_global sc_in sc_lv 16 signal 0 } 
	{ x_size sc_in sc_lv 16 signal 1 } 
	{ y_size sc_in sc_lv 16 signal 2 } 
	{ y_local sc_in sc_lv 16 signal 3 } 
	{ layer13_out sc_out sc_lv 16 signal 4 } 
	{ layer13_out_ap_vld sc_out sc_logic 1 outvld 4 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "z_global_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "z_global", "role": "ap_vld" }} , 
 	{ "name": "x_size_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "x_size", "role": "ap_vld" }} , 
 	{ "name": "y_size_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "y_size", "role": "ap_vld" }} , 
 	{ "name": "y_local_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "y_local", "role": "ap_vld" }} , 
 	{ "name": "z_global", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z_global", "role": "default" }} , 
 	{ "name": "x_size", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "x_size", "role": "default" }} , 
 	{ "name": "y_size", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "y_size", "role": "default" }} , 
 	{ "name": "y_local", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "y_local", "role": "default" }} , 
 	{ "name": "layer13_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "layer13_out", "role": "default" }} , 
 	{ "name": "layer13_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "layer13_out", "role": "ap_vld" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "24", "25", "71", "72", "80"],
		"CDFG" : "myproject",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "23", "EstimateLatencyMin" : "23", "EstimateLatencyMax" : "23",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "z_global", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "z_global_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "x_size", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "x_size_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "y_size", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "y_size_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "y_local", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "y_local_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "layer13_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "sigmoid_table", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "80", "SubInstance" : "grp_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s_fu_160", "Port" : "sigmoid_table"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret1_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config3_s_fu_83", "Parent" : "0",
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
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret2_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config5_s_fu_91", "Parent" : "0",
		"CDFG" : "concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config5_s",
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
			{"Name" : "data2_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret3_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_s_fu_99", "Parent" : "0",
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
			{"Name" : "data2_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108", "Parent" : "0", "Child" : ["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"],
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "3", "EstimateLatencyMin" : "3", "EstimateLatencyMax" : "3",
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
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_10s_26_2_0_U10", "Parent" : "4"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_9s_25_2_0_U11", "Parent" : "4"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_13ns_26_2_0_U12", "Parent" : "4"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_8s_24_2_0_U13", "Parent" : "4"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_13ns_26_2_0_U14", "Parent" : "4"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_11s_26_2_0_U15", "Parent" : "4"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_13s_26_2_0_U16", "Parent" : "4"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_10ns_26_2_0_U17", "Parent" : "4"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_11ns_26_2_0_U18", "Parent" : "4"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_10s_26_2_0_U19", "Parent" : "4"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_8ns_24_2_0_U20", "Parent" : "4"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_8s_24_2_0_U21", "Parent" : "4"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_11s_26_2_0_U22", "Parent" : "4"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_7ns_23_2_0_U23", "Parent" : "4"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_11ns_26_2_0_U24", "Parent" : "4"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_13ns_26_2_0_U25", "Parent" : "4"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_13s_26_2_0_U26", "Parent" : "4"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_13ns_26_2_0_U27", "Parent" : "4"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s_fu_108.mul_16s_11ns_26_2_0_U28", "Parent" : "4"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret5_relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config9_s_fu_116", "Parent" : "0",
		"CDFG" : "relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config9_s",
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
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_6_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127", "Parent" : "0", "Child" : ["26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70"],
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "4", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
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
			{"Name" : "data_6_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_11ns_26_2_0_U50", "Parent" : "25"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_14ns_26_2_0_U51", "Parent" : "25"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13ns_26_2_0_U52", "Parent" : "25"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_8ns_24_2_0_U53", "Parent" : "25"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_8ns_24_2_0_U54", "Parent" : "25"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12s_26_2_0_U55", "Parent" : "25"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12ns_26_2_0_U56", "Parent" : "25"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12ns_26_2_0_U57", "Parent" : "25"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12s_26_2_0_U58", "Parent" : "25"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_9s_25_2_0_U59", "Parent" : "25"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12s_26_2_0_U60", "Parent" : "25"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12s_26_2_0_U61", "Parent" : "25"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13ns_26_2_0_U62", "Parent" : "25"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_10s_26_2_0_U63", "Parent" : "25"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13s_26_2_0_U64", "Parent" : "25"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_11ns_26_2_0_U65", "Parent" : "25"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13ns_26_2_0_U66", "Parent" : "25"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_11ns_26_2_0_U67", "Parent" : "25"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_11s_26_2_0_U68", "Parent" : "25"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_11ns_26_2_0_U69", "Parent" : "25"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13s_26_2_0_U70", "Parent" : "25"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12ns_26_2_0_U71", "Parent" : "25"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_10s_26_2_0_U72", "Parent" : "25"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_9s_25_2_0_U73", "Parent" : "25"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12s_26_2_0_U74", "Parent" : "25"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13s_26_2_0_U75", "Parent" : "25"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_10s_26_2_0_U76", "Parent" : "25"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_14ns_26_2_0_U77", "Parent" : "25"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_11s_26_2_0_U78", "Parent" : "25"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13s_26_2_0_U79", "Parent" : "25"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12ns_26_2_0_U80", "Parent" : "25"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12ns_26_2_0_U81", "Parent" : "25"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12ns_26_2_0_U82", "Parent" : "25"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_11s_26_2_0_U83", "Parent" : "25"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13s_26_2_0_U84", "Parent" : "25"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12s_26_2_0_U85", "Parent" : "25"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_8ns_24_2_0_U86", "Parent" : "25"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_10s_26_2_0_U87", "Parent" : "25"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13s_26_2_0_U88", "Parent" : "25"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12ns_26_2_0_U89", "Parent" : "25"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12ns_26_2_0_U90", "Parent" : "25"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_14ns_26_2_0_U91", "Parent" : "25"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_12s_26_2_0_U92", "Parent" : "25"},
	{"ID" : "69", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13s_26_2_0_U93", "Parent" : "25"},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s_fu_127.mul_16s_13ns_26_2_0_U94", "Parent" : "25"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret7_relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config11_s_fu_138", "Parent" : "0",
		"CDFG" : "relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config11_s",
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
			{"Name" : "data_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_6_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s_fu_149", "Parent" : "0", "Child" : ["73", "74", "75", "76", "77", "78", "79"],
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "4", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
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
			{"Name" : "data_6_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "73", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s_fu_149.mul_16s_13ns_26_2_0_U112", "Parent" : "72"},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s_fu_149.mul_16s_13ns_26_2_0_U113", "Parent" : "72"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s_fu_149.mul_16s_14ns_26_2_0_U114", "Parent" : "72"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s_fu_149.mul_16s_10ns_26_2_0_U115", "Parent" : "72"},
	{"ID" : "77", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s_fu_149.mul_16s_12s_26_2_0_U116", "Parent" : "72"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s_fu_149.mul_16s_13s_26_2_0_U117", "Parent" : "72"},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s_fu_149.mul_16s_14s_26_2_0_U118", "Parent" : "72"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s_fu_160", "Parent" : "0", "Child" : ["81"],
		"CDFG" : "sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "3", "EstimateLatencyMin" : "3", "EstimateLatencyMax" : "3",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "sigmoid_table", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s_fu_160.sigmoid_table_U", "Parent" : "80"}]}


set ArgLastReadFirstWriteLatency {
	myproject {
		z_global {Type I LastRead 0 FirstWrite -1}
		x_size {Type I LastRead 0 FirstWrite -1}
		y_size {Type I LastRead 0 FirstWrite -1}
		y_local {Type I LastRead 0 FirstWrite -1}
		layer13_out {Type O LastRead -1 FirstWrite 23}
		sigmoid_table {Type I LastRead -1 FirstWrite -1}}
	concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config3_s {
		data1_val {Type I LastRead 0 FirstWrite -1}
		data2_val {Type I LastRead 0 FirstWrite -1}}
	concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config5_s {
		data1_0_val {Type I LastRead 0 FirstWrite -1}
		data1_1_val {Type I LastRead 0 FirstWrite -1}
		data2_val {Type I LastRead 0 FirstWrite -1}}
	concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config7_s {
		data1_0_val {Type I LastRead 0 FirstWrite -1}
		data1_1_val {Type I LastRead 0 FirstWrite -1}
		data1_2_val {Type I LastRead 0 FirstWrite -1}
		data2_val {Type I LastRead 0 FirstWrite -1}}
	dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config8_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}}
	relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config9_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}
		data_4_val {Type I LastRead 0 FirstWrite -1}
		data_5_val {Type I LastRead 0 FirstWrite -1}
		data_6_val {Type I LastRead 0 FirstWrite -1}}
	dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config10_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}
		data_4_val {Type I LastRead 0 FirstWrite -1}
		data_5_val {Type I LastRead 0 FirstWrite -1}
		data_6_val {Type I LastRead 0 FirstWrite -1}}
	relu_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_relu_config11_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}
		data_4_val {Type I LastRead 0 FirstWrite -1}
		data_5_val {Type I LastRead 0 FirstWrite -1}
		data_6_val {Type I LastRead 0 FirstWrite -1}}
	dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config12_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}
		data_2_val {Type I LastRead 0 FirstWrite -1}
		data_3_val {Type I LastRead 0 FirstWrite -1}
		data_4_val {Type I LastRead 0 FirstWrite -1}
		data_5_val {Type I LastRead 0 FirstWrite -1}
		data_6_val {Type I LastRead 0 FirstWrite -1}}
	sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s {
		data_val {Type I LastRead 0 FirstWrite -1}
		sigmoid_table {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "23", "Max" : "23"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	z_global { ap_vld {  { z_global_ap_vld in_vld 0 1 }  { z_global in_data 0 16 } } }
	x_size { ap_vld {  { x_size_ap_vld in_vld 0 1 }  { x_size in_data 0 16 } } }
	y_size { ap_vld {  { y_size_ap_vld in_vld 0 1 }  { y_size in_data 0 16 } } }
	y_local { ap_vld {  { y_local_ap_vld in_vld 0 1 }  { y_local in_data 0 16 } } }
	layer13_out { ap_vld {  { layer13_out out_data 1 16 }  { layer13_out_ap_vld out_vld 1 1 } } }
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
