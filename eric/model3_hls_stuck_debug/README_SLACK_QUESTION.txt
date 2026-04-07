================================================================================
  MODEL3 HLS SYNTHESIS STUCK — HELP NEEDED
  Prepared: April 7, 2026
  Context: Muon Collider Smart Pixels project, FPGA synthesis via hls4ml / Vitis HLS
================================================================================

--------------------------------------------------------------------------------
BACKGROUND
--------------------------------------------------------------------------------

We are running HLS (High-Level Synthesis) on a Keras neural network (Model3) using
hls4ml with the Vitis backend, targeting the xc7z020clg400-1 FPGA part (Zynq-7020).

The same pipeline (same script, same FPGA part, same hls4ml version) works
perfectly on our smaller model (Model2_5), which completes full synthesis
including C-simulation, RTL co-simulation, export, and Vivado synthesis.

For Model3 (trial_008), two separate synthesis attempts were made (hls_model_trial_008
and hls_model_trial_008_trail1). Both get permanently stuck at the same point and
never finish. Neither run produced a crashed error — the process just stalled
indefinitely.

--------------------------------------------------------------------------------
WHAT THE MODEL IS
--------------------------------------------------------------------------------

Model3 is a mixed-input neural network with the following architecture:
  - Input 1: 2D pixel cluster image (13x21x1)
  - Input 2: scalar features (nModule, x_local, y_local)
  - Conv2D layer: 3x3 kernel, 24 filters, stride=1, same padding → output 13x21x24
  - ReLU activation
  - 2D Max-Pooling: 2x2 stride=2 → output 6x10x24 = 1,440 elements
  - Parallel scalar branch: Dense(3 → 64) + ReLU
  - Concatenate: pool output (1,440) + scalar dense output (64) = 1,504 inputs
  - Dense merged_dense1: 1,504 → 110 (4-bit weights)   ← KEY BOTTLENECK
  - ReLU activation
  - Dense merged_dense2: 110 → 88 (4-bit weights)
  - ReLU activation
  - Dense output_dense: 88 → 1 (4-bit weights)
  - hard_tanh output activation

The model was chosen from a Pareto front based on AUC and background rejection at 95%.
Trial 008 stats: 175,903 parameters, AUC = 0.9748, bkg_rej@95% = 0.9164

hls4ml configuration:
  - Backend: Vitis
  - IOType: io_parallel
  - Strategy: Latency
  - ReuseFactor: 1
  - ClockPeriod: 5 ns (200 MHz)
  - All weights quantized to fixed<4,1> (4-bit)
  - Activation outputs at ufixed<8,0> or fixed<16,6>

--------------------------------------------------------------------------------
WHAT HAPPENED — STEP BY STEP
--------------------------------------------------------------------------------

The synthesis was launched via:
    hls_model.build(csim=False, synth=True, cosim=True, export=True, vsynth=True, reset=True)

which internally calls Vitis HLS via build_prj.tcl.

Step 1 — Setup (works, ~1 sec):
  config_array_partition -maximum_size 4096   → ERROR (deprecated flag, caught by
                                                 "catch" in TCL, synthesis continues)
  config_compile, set_part, create_clock, etc. → all OK

Step 2 — C/RTL Synthesis starts:
  Analyzing source files, clang preprocessing → OK
  Compile/Link phase reports: 32,767 instructions
    (vs 8,079 for Model2_5 — already 4x larger before any unrolling)

  The csynth_design_size.rpt shows:
    - conv_2d/fill_buffer alone: 24,105 instructions
    - The Unroll/Inline phase shows "pending" — it never starts

Step 3 — LTO / Unroll/Inline phase (where it gets STUCK):
  The autopilot.flow.log shows the last action taken was:
    "Doing LTO" → launching background clang process for reflow/unrolling

  That background process (a.g.ld.0.bc.clang.reflow.err.log) ran for ~16 hours
  (from Mar 26 16:26 to Mar 27 08:11) before dying, with no output to vitis_hls.log.

  The last thousands of lines of the reflow error log are all identical:
    remark: ... Unrolling loop 'Product2' in function
    'nnet::dense_latency<..., config17>' completely with a factor of 110
    (repeated thousands of times for the merged_dense1 layer)

  The vitis_hls.log ends abruptly after:
    INFO: [HLS 200-1995] There were 32,767 instructions in the design after the
    'Compile/Link' phase ...

  No error, no exit code — the process just silently stalled/died during LTO.

--------------------------------------------------------------------------------
ROOT CAUSE (OUR HYPOTHESIS)
--------------------------------------------------------------------------------

The problem appears to be that the merged_dense1 layer has:
  - 1,504 inputs (from conv2d pool output flattened + scalar branch)
  - 110 outputs
  - = 165,440 multiplications

With Strategy: Latency + ReuseFactor: 1 + IOType: io_parallel, all 165,440
multiplications must be fully unrolled into parallel hardware during the
Unroll/Inline compilation phase.

For comparison, Model2_5 (which works):
  - Compile/Link: 8,079 instructions
  - After Unroll/Inline: peaks at ~542,764 instructions (~67x blowup)
  - Then simplifies down to ~47,000 for the final HW transforms
  - Synthesis completes in a few minutes

For Model3:
  - Compile/Link: 32,767 instructions
  - Projected Unroll/Inline blowup at the same ratio: ~2.2 million instructions
  - The LTO/reflow clang process cannot handle this — it ran for 16 hours,
    consumed massive memory, and died silently

This is also why it fails identically on two separate attempts (trial_008 and
trial_008_trail1) — it is not a fluke crash but a deterministic design-size problem.

Additionally, note that the conv2d fill_buffer alone generates 24,105 instructions
at Compile/Link (vs 0 for Model2_5 which has no conv layers), consuming 74% of the
instruction budget before the dense layers are even counted.

--------------------------------------------------------------------------------
QUESTION FOR SLACK
--------------------------------------------------------------------------------

Has anyone dealt with hls4ml synthesis getting permanently stuck in the
Unroll/Inline / LTO phase for models with large dense layers after a Conv2D?

Specifically:
  1. Is there a known threshold for how many instructions/multiplications Vitis HLS
     (2024.1) can handle in Latency + io_parallel + ReuseFactor=1 mode?

  2. Would switching to Strategy: Resource or increasing ReuseFactor (e.g., 32 or 64)
     specifically for merged_dense1 (1504 → 110) fix the LTO timeout/OOM, while
     keeping the rest Latency? Or does changing Strategy for one layer require
     changing the whole model?

  3. Is IOType: io_stream a better fit for this Conv+Dense mixed architecture?
     We previously used io_parallel for all Model2_5 work and it worked fine there.

  4. Has anyone successfully synthesized a conv2d model of this scale (~165k weights
     in the first merged dense layer) with this FPGA part? What config worked?

--------------------------------------------------------------------------------
FILES IN THIS FOLDER
--------------------------------------------------------------------------------

  model_trial_008.h5              — The Keras model (175,903 params, 4-bit weights)
  hls4ml_config.yml               — The hls4ml configuration used
  build_prj.tcl                   — The Vitis HLS TCL build script (generated by hls4ml)
  project.tcl                     — Project TCL (part number, clock, etc.)
  firmware/myproject.cpp          — Top-level HLS C++ file
  firmware/defines.h              — Type definitions and array size #defines
  firmware/parameters.h           — Layer config structs (reuse factors, sizes, etc.)
  vitis_hls_trial_008.log         — Vitis HLS log from first attempt (ends at line 92)
  vitis_hls_trial_008_trail1.log  — Vitis HLS log from second attempt (identical stuck point)
  reports_trial_008/
    csynth_design_size.rpt        — Design size after Compile/Link (synthesis incomplete)
  reports_trial_008_trail1/
    csynth_design_size.rpt        — Same, from second attempt
  autopilot_db/
    autopilot.flow.log            — Internal HLS flow log, shows last step taken
    a.g.ld.0.bc.clang.reflow.err.log — The reflow/LTO log (961 KB, 3224 lines of
                                        repeated unroll remarks, ends mid-unroll)
  pareto_optimal_models_roc_primary.csv  — Pareto front (trial_008 is best by AUC)
  roc_based_analysis_detailed.csv        — Full set of trained trials + metrics

--------------------------------------------------------------------------------
TOOL VERSIONS
--------------------------------------------------------------------------------

  hls4ml: (see hls4ml_config.yml — Vitis backend, 2024.1 build)
  Vitis HLS: 2024.1 (SW Build 5069499, May 21 2024)
  Vivado: 2024.1
  TensorFlow / Keras: used for model training
  QKeras: used for 4-bit quantized weights
  FPGA Target: xc7z020clg400-1 (Zynq-7020, 53,200 LUTs, 220 DSPs, 280 BRAMs)
  Host OS: Linux x86_64 (RHEL9), host: kdplab01

================================================================================
