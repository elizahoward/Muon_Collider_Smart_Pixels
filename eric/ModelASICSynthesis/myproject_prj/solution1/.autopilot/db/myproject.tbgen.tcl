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
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"],
		"CDFG" : "myproject",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "24", "EstimateLatencyMin" : "24", "EstimateLatencyMax" : "24",
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
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret3_concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config8_s_fu_191", "Parent" : "0",
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
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s_fu_199", "Parent" : "0",
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config9_s",
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
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config11_s_fu_238", "Parent" : "0",
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config11_s",
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
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config14_s_fu_244", "Parent" : "0",
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
			{"Name" : "data_1_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config13_s_fu_250", "Parent" : "0",
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
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret8_concatenate1d_ap_ufixed_ap_ufixed_ap_fixed_16_6_5_3_0_config15_s_fu_262", "Parent" : "0",
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
			{"Name" : "data2_1_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s_fu_276", "Parent" : "0",
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config16_s",
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
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_6_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_7_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_8_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_9_val", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config18_s_fu_290", "Parent" : "0",
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
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config19_s_fu_298", "Parent" : "0",
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
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config21_s_fu_306", "Parent" : "0",
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
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config22_s_fu_314", "Parent" : "0",
		"CDFG" : "dense_latency_ap_ufixed_8_0_4_0_0_ap_fixed_16_6_5_3_0_config22_s",
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
			{"Name" : "data_3_val", "Type" : "None", "Direction" : "I"}]}]}


set ArgLastReadFirstWriteLatency {
	myproject {
		x_profile {Type I LastRead 0 FirstWrite -1}
		nModule {Type I LastRead 0 FirstWrite -1}
		x_local {Type I LastRead 0 FirstWrite -1}
		y_profile {Type I LastRead 0 FirstWrite -1}
		y_local {Type I LastRead 0 FirstWrite -1}
		layer24_out {Type O LastRead -1 FirstWrite 24}}
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
	concatenate1d_ap_fixed_ap_fixed_ap_fixed_16_6_5_3_0_config8_s {
		data1_val {Type I LastRead 0 FirstWrite -1}
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
	dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config11_s {
		data_0_val {Type I LastRead 0 FirstWrite -1}
		data_1_val {Type I LastRead 0 FirstWrite -1}}
	relu_ap_fixed_16_6_5_3_0_ap_ufixed_8_0_4_0_0_relu_config14_s {
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
		data2_1_val {Type I LastRead 0 FirstWrite -1}}
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
		data_9_val {Type I LastRead 0 FirstWrite -1}}
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
	{"Name" : "Latency", "Min" : "24", "Max" : "24"}
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
