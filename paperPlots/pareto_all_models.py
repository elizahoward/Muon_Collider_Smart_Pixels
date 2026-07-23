#!/usr/bin/env python3
"""
Combined Pareto plot: Model 1 vs Model 2.5 vs Model 3 (all bit widths)

HLS resource extraction priority:
  - Model 1:   vsynth (Vivado LUT/FF from vitis_hls.log) preferred;
               falls back to csynth estimate when vsynth log is absent.
  - Model 2.5: csynth estimates only (no vsynth available).
  - Model 3:   csynth estimates only, searched under pareto_primary/hls_outputs/.

Outputs (same variants as pareto_model1_vs_model2_5.py):
  combined_all_models_pareto/
    pareto_all_models_full[_nolabels].png
    pareto_all_models_zoomed[_nolabels].png
    pareto_all_models_subfronts_xlinear_ylinear[_nolabels].png
    pareto_all_models_subfronts_xlog_ylinear[_nolabels].png
    pareto_all_models_subfronts_xlinear_ycomplog[_nolabels].png
    pareto_all_models_subfronts_xlog_ycomplog[_nolabels].png
    combined_all_detailed.csv
    pareto_primary.csv
    summary.json
"""

import os
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

ERIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"eric")
newFolderStructure = True; #for the consistent folder structure when you have pareto_primary and pareto_secondary for everyone
if newFolderStructure:
    ERICF = os.path.join(ERIC, "Results_June2026_99SigEff") 
else:
    print("WARNING!!!!!!!!! WHY ARE YOU USING THE OLD FOLDER STRUCTURE FOR PAPER PLOTS?????")
    raise ValueError("Are you sure you want to do that? If so, comment out this line")
    ERICF = os.path.join(ERIC, "Final_Results")

MODEL1_DIR  = os.path.join(ERICF, "model1_fin_results")
MODEL25_DIR = os.path.join(ERICF, "model2.5_fin_results")
MODEL3_DIR  = os.path.join(ERICF, "model3_fin_results")
# MODEL3_DIR  = os.path.join(os.path.join(ERIC, "Final_Results"), "model3_fin_results")
if newFolderStructure:
    # OUTPUT_DIR  = os.path.join(ERIC, "combined_all_models_pareto_newJune2026")
    OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combined_all_models_pareto_newJune2026")
else:
    OUTPUT_DIR  = os.path.join(ERIC, "combined_all_models_pareto")

PRIMARY_METRIC = "primary_metric"

PRIMARY_METRIC = "bkg_rej_@99%"

if PRIMARY_METRIC == "primary_metric":
    METRIC_NAME = "Weighted Bkg Rejection"
elif PRIMARY_METRIC == "bkg_rej_@99%":
    METRIC_NAME = "Bkg Rejection @ 99% Sig. Eff."
else:
    raise ValueError("invalid PRIMARY_METRIC")

# ── Bit-width folder configs: (folder_name, run_name, label, color) ────────────
if newFolderStructure:
    MODEL1_CONFIGS = [
        ("model1_3bit_normalised_selected",   "model1_3w5i",   "Model 1 (3-bit)",   "#ff6b6b"),
        ("model1_4bit_normalised_selected",   "model1_4w6i",   "Model 1 (4-bit)",   "#ee5a24"),
        ("model1_6bit_normalised_selected",   "model1_6w8i",   "Model 1 (6-bit)",   "#c0392b"),
        ("model1_8bit_normalised_selected",  "model1_8w10i",  "Model 1 (8-bit)",  "#922b21"),
        ("model1_10bit_normalised_selected", "model1_10w12i", "Model 1 (10-bit)", "#641e16"),
    ]
else:
    MODEL1_CONFIGS = [
        ("3w0i_i5_sigmoid",   "model1_3w5i",   "Model 1 (3-bit)",   "#ff6b6b"),
        ("4w0i_i6_sigmoid",   "model1_4w6i",   "Model 1 (4-bit)",   "#ee5a24"),
        ("6w0i_i8_sigmoid",   "model1_6w8i",   "Model 1 (6-bit)",   "#c0392b"),
        ("8w0i_i10_sigmoid",  "model1_8w10i",  "Model 1 (8-bit)",  "#922b21"),
        ("10w0i_i12_sigmoid", "model1_10w12i", "Model 1 (10-bit)", "#641e16"),
    ]

if newFolderStructure:
    MODEL25_CONFIGS = [
        ("model2_5_3bit_normalised_selected",  "model25_3bit",  "Model 2 (3-bit)",  "#74b9ff"),
        ("model2_5_4bit_normalised_selected",  "model25_4bit",  "Model 2 (4-bit)",  "#0984e3"),
        ("model2_5_6bit_normalised_selected",  "model25_6bit",  "Model 2 (6-bit)",  "#2980b9"),
        ("model2_5_8bit_normalised_selected",  "model25_8bit",  "Model 2 (8-bit)",  "#1a5276"),
        ("model2_5_10bit_normalised_selected", "model25_10bit", "Model 2 (10-bit)", "#154360"),
    ]
else:
    MODEL25_CONFIGS = [
        ("model2_5_3bit_normalised_selected",  "model25_3bit",  "Model 2 (3-bit)",  "#74b9ff"),
        ("model2.5_4bit_normalised_selected",  "model25_4bit",  "Model 2 (4-bit)",  "#0984e3"),
        ("model2.5_6bit_normalised_selected",  "model25_6bit",  "Model 2 (6-bit)",  "#2980b9"),
        ("model2.5_8bit_normalised_selected",  "model25_8bit",  "Model 2 (8-bit)",  "#1a5276"),
        ("model2.5_10bit_normalised_selected", "model25_10bit", "Model 2 (10-bit)", "#154360"),
    ]

MODEL3_CONFIGS = [
    ("model3_3bit_normalised_selected",  "model3_3bit",  "Model 3 (3-bit)",  "#9bfea5"),
    ("model3_4bit_normalised_selected",  "model3_4bit",  "Model 3 (4-bit)",  "#2ecc31"),
    ("model3_6bit_normalised_selected",  "model3_6bit",  "Model 3 (6-bit)",  "#3bad58"),
    ("model3_8bit_normalised_selected",  "model3_8bit",  "Model 3 (8-bit)",  "#00b8ac"),
    ("model3_10bit_normalised_selected", "model3_10bit", "Model 3 (10-bit)", "#00745D"),
]

_ALL_CONFIGS = MODEL1_CONFIGS + MODEL25_CONFIGS + MODEL3_CONFIGS
RUN_COLORS   = {rn: col for _, rn, _, col in _ALL_CONFIGS}
RUN_LABELS   = {rn: lbl for _, rn, lbl, _ in _ALL_CONFIGS}

FAMILY_LINE = {
    "Model 1":   dict(color="#c0392b", lw=1.4, ls="--", label="Model 1 Pareto front"),
    "Model 2.5": dict(color="#1a5276", lw=1.4, ls="-.", label="Model 2 Pareto front"),
    "Model 3":   dict(color="#27ae60", lw=1.4, ls=":",  label="Model 3 Pareto front"),
    "Combined":  dict(color="black",   lw=1.8, ls="-",  label="Combined Pareto front"),
}

COMP_LOG_TICKS = [0.9, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.30, 0.10, 0.0]


#References from synthesizing some different models
#of format (ff_plus_lut, label,color,labelYHeight,textOffset) in HARDWARE_REFS
#for filtering model, reference DOI 10.1088/2632-2153/ad6a00

HARDWARE_REFS = [
    # (34568,"Smartpixel Filtering Model (csynth) DOI 10.1088/2632-2153/ad6a00","teal",0.05,0),#for top alignment, 0.86 #qmodel_file = "/local/d1/smartpixLab/fermiModels/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
    (34568,"Smartpixel Filtering Model, arxiv:2310.02474","teal",0.05,0),#for top alignment, 0.86 #qmodel_file = "/local/d1/smartpixLab/fermiModels/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
    # (26376,"Smartpixel Filtering Model (vsynth)","teal",0.05,0), #for top alignment 0.5 #qmodel_file = "/local/d1/smartpixLab/fermiModels/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
    (14289+57398,"Smartpixel Regression Model, arxiv:2312.11676v1","purple",0.05,0), #for top alignment 0.5 #qmodel_file = "/local/d1/smartpixLab/fermiModels/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
    # (35216,"Smartpixel Filtering Model (csynth) but add an input quantization layer","teal",0.95), #singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/ASIC Model_results_20260610_055759/models/ASIC Model_quantized_4bit.h5"
    # (24853,"Smartpixel Filtering Model (vsynth) but add an input quantization layer","teal",0.95), #singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/ASIC Model_results_20260610_055759/models/ASIC Model_quantized_4bit.h5"
    (106400+53200,"FPGA: Xilinx Zynq (xc7z020clg400-1), featured on PYNQ-Z2","fuchsia",0.05,0),
    # (20736+15552,"FPGA: Tang Nano 20k","pink",0.05,0.1),
    # (14400+28800,"FPGA: Xilinx Zynq 7007S (xc7z007z)","purple",0.05,0.1),#https://www.mouser.com/datasheet/2/903/ds190-Zynq-7000-Overview-1595492.pdf
    # (10000+20000,'Small FPGA: "Pink Board" Tang Nano 20k', ), #accroding to https://deepwiki.com/sipeed/sipeed_wiki/4.1-tang-nano-20k
    (3456000,"FPGA: Xilinx Alveo U250 (xcu250-figd2104-2L-e)","goldenrod",0.05,0) #https://pcbsync.com/xilinx-alveo-u200/ says 1341000+2682000, but https://docs.amd.com/r/en-US/ds962-u200-u250/Summary says 3456000
]

SELECTED_MODELS = [
# ("model3_10bit","110"),
("model3_10bit","046"),
# ("model25_8bit","063"),
("model25_10bit","087"),
("model25_10bit","057"),
# ("model25_8bit","051"),
("model1_8w10i","1046"),
# ("model1_6w8i","836"),
# ("model1_10w12i","1224"),
# ("model1_8w10i","1028"),
]
styleSheet = "seaborn-v0_8-colorblind"
plt.style.use(styleSheet)
# ── HLS resource extraction ────────────────────────────────────────────────────
def _lut_ff_from_vsynth_log(log_path):
    """Parse Vivado synthesis cell-usage table from vitis_hls.log. Returns (lut, ff) or None."""
    try:
        with open(log_path) as f:
            text = f.read()
        m = re.search(r'Report Cell Usage:.*?(?=Report Instance Areas:)', text, re.DOTALL)
        if not m:
            return None
        lut = ff = 0
        for line in m.group(0).splitlines():
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 4:
                continue
            cell    = parts[2] if len(parts) > 2 else ""
            count_s = parts[3] if len(parts) > 3 else ""
            if not count_s.isdigit():
                continue
            count = int(count_s)
            if re.match(r'LUT[1-6]$', cell):
                lut += count
            elif cell in ('FDRE', 'FDSE', 'FDCE', 'FDPE'):
                ff += count
        return (lut, ff) if (lut or ff) else None
    except Exception:
        return None


def _lut_ff_from_csynth_rpt(rpt_path):
    """Parse HLS csynth report Total row. Returns (lut, ff, dsp, bram) or None.
    Column order in report: BRAM_18K | DSP | FF | LUT | URAM
    """
    try:
        with open(rpt_path) as f:
            for line in f:
                if line.startswith('|Total'):
                    parts = [p.strip() for p in line.split('|')]
                    def _int(s):
                        return int(s) if s.lstrip('-').isdigit() else 0
                    return _int(parts[5]), _int(parts[4]), _int(parts[3]), _int(parts[2])
        return None
    except Exception:
        return None


def _hls_resources_model1(bit_dir, trial_id):
    """
    trial_id e.g. "hp4q_trial_006" → folder "hls_hp4q_model_trial_006"
    Tries vsynth first, falls back to csynth.
    Returns (lut, ff, dsp, bram, source_str) or all-None tuple.
    """
    hls_name = "hls_" + trial_id.replace("_trial_", "_model_trial_")
    csynth_rel = os.path.join(hls_name, "myproject_prj", "solution1",
                              "syn", "report", "myproject_csynth.rpt")

    vlog = os.path.join(bit_dir, "hls_outputs_vsynth", hls_name, "vitis_hls.log")
    if os.path.isfile(vlog):
        res = _lut_ff_from_vsynth_log(vlog)
        if res:
            return res[0], res[1], None, None, "vsynth"

    rpt = os.path.join(bit_dir, "hls_outputs", csynth_rel)
    if os.path.isfile(rpt):
        res = _lut_ff_from_csynth_rpt(rpt)
        if res:
            return res[0], res[1], res[2], res[3], "csynth"

    return None, None, None, None, None


def _hls_resources_csynth_only(bit_dir, trial_id, search_prefixes):
    """
    trial_id e.g. "063" → folder "hls_model_trial_063"
    Searches each prefix in order for the csynth report.
    Returns (lut, ff, dsp, bram, "csynth") or all-None tuple.
    """
    hls_name = f"hls_model_trial_{trial_id}"
    csynth_rel = os.path.join(hls_name, "myproject_prj", "solution1",
                              "syn", "report", "myproject_csynth.rpt")
    for prefix in search_prefixes:
        rpt = os.path.join(bit_dir, prefix, csynth_rel)
        if os.path.isfile(rpt):
            res = _lut_ff_from_csynth_rpt(rpt)
            if res:
                return res[0], res[1], res[2], res[3], "csynth"
    return None, None, None, None, None

def _hls_resources_csynth_or_vsynth(bit_dir, trial_id, search_prefixes):
    """
    trial_id e.g. "063" → folder "hls_model_trial_063"
    Searches each prefix in order for the csynth report.
    Returns (lut, ff, dsp, bram, "csynth") or all-None tuple.
    """
    hls_name = f"hls_model_trial_{trial_id}"
    csynth_rel = os.path.join(hls_name, "myproject_prj", "solution1",
                              "syn", "report", "myproject_csynth.rpt")
    for prefix in search_prefixes:
        vlog = os.path.join(bit_dir, prefix, hls_name, "vitis_hls.log")
        if os.path.isfile(vlog):
            res = _lut_ff_from_vsynth_log(vlog)
            if res:
                return res[0], res[1], None, None, "vsynth"
        rpt = os.path.join(bit_dir, prefix, csynth_rel)
        if os.path.isfile(rpt):
            res = _lut_ff_from_csynth_rpt(rpt)
            if res:
                return res[0], res[1], res[2], res[3], "csynth"
    return None, None, None, None, None

# ── Data loading ───────────────────────────────────────────────────────────────
def _build_row(model_key, run_name, trial_id, csv_row, lut, ff, dsp, bram, src,fullpath):
    return {
        "model":          model_key,
        "run_name":       run_name,
        "trial_id":       trial_id,
        "parameters":     csv_row.get("parameters", np.nan),
        "auc":            csv_row.get("auc", np.nan),
        PRIMARY_METRIC: csv_row.get(PRIMARY_METRIC, np.nan),
        "luts":           lut,
        "registers":      ff,
        "luts_plus_ff":   lut + ff,
        "dsp":            dsp or 0,
        "bram":           bram or 0,
        "hls_source":     src,
        "fullPath": fullpath, #added by Daniel to make it easier to load models
    }


def load_model1(base_dir):
    rows = []
    for folder, run_name, _, _ in MODEL1_CONFIGS:
        bit_dir  = os.path.join(base_dir, folder)
        csv_path = os.path.join(bit_dir, "roc_based_analysis_detailed.csv")
        if not os.path.isfile(csv_path):
            print(f"  [WARN] Missing: {csv_path}")
            continue
        for _, csv_row in pd.read_csv(csv_path).iterrows():
            trial_id = str(csv_row["trial_id"])
            lut, ff, dsp, bram, src = _hls_resources_model1(bit_dir, trial_id)
            if lut is None:
                continue
            fullpath = os.path.join(bit_dir,trial_id[0:4]+"_model"+trial_id[4:]+".h5") #by hand edited it
            rows.append(_build_row("model1", run_name, trial_id, csv_row,
                                   lut, ff, dsp, bram, src,fullpath))
    return pd.DataFrame(rows)


def load_model25(base_dir):
    rows = []
    for folder, run_name, _, _ in MODEL25_CONFIGS:
        bit_dir  = os.path.join(base_dir, folder)
        csv_path = os.path.join(bit_dir, "roc_based_analysis_detailed.csv")
        if not os.path.isfile(csv_path):
            print(f"  [WARN] Missing: {csv_path}")
            continue
        for _, csv_row in pd.read_csv(csv_path).iterrows():
            trial_id = str(int(csv_row["trial_id"])).zfill(3)
            lut, ff, dsp, bram, src = _hls_resources_csynth_only(
                bit_dir, trial_id, ["hls_outputs"])
            if lut is None:
                continue
            fullpath = os.path.join(bit_dir,"model_trial_"+trial_id+".h5")
            rows.append(_build_row("model2_5", run_name, trial_id, csv_row,
                                   lut, ff, dsp, bram, src,fullpath))
    return pd.DataFrame(rows)


def load_model3(base_dir):
    rows = []
    for folder, run_name, _, _ in MODEL3_CONFIGS:
        bit_dir  = os.path.join(base_dir, folder)
        csv_path = os.path.join(bit_dir, "roc_based_analysis_detailed.csv")
        if not os.path.isfile(csv_path):
            print(f"  [WARN] Missing: {csv_path}")
            continue
        for _, csv_row in pd.read_csv(csv_path).iterrows():
            trial_id = str(int(csv_row["trial_id"])).zfill(3)
            lut, ff, dsp, bram, src = _hls_resources_csynth_only(
                bit_dir, trial_id,
                ["pareto_primary/hls_outputs",
                 "pareto_secondary/hls_outputs",
                 "hls_outputs"])
            if lut is None:
                continue
            fullpath1 = os.path.join(bit_dir,"pareto_primary/")
            fullpath2 = os.path.join(bit_dir,"pareto_secondary/")
            fullpath = os.path.join(fullpath1,"model_trial_"+trial_id+'.h5')
            if not os.path.isfile(fullpath):
                fullpath = os.path.join(fullpath2,"model_trial_"+trial_id+'.h5')
            rows.append(_build_row("model3", run_name, trial_id, csv_row,
                                   lut, ff, dsp, bram, src,fullpath))
    return pd.DataFrame(rows)

def load_modelNEW(base_dir,modelNum):
    if modelNum ==1:
        modelName = "model1"
        modelConfig = MODEL1_CONFIGS
    elif modelNum == 2 or modelNum == 2.5:
        modelName = "model2_5"
        modelConfig = MODEL25_CONFIGS
    elif modelNum == 3:
        modelName = "model3"
        modelConfig = MODEL3_CONFIGS
    else:
        raise ValueError("Not supported model number")
    if modelName not in ["model1", "model2_5", "model3"]:
        raise ValueError("Not supported modelName")
    rows = []
    for folder, run_name, _, _ in modelConfig:
        bit_dir  = os.path.join(base_dir, folder)
        csv_path = os.path.join(bit_dir, "roc_based_analysis_detailed.csv") #TODO: replace with paperPlots version of this directory
        print("TODO: want to replace ",csv_path," with version in paperPlots after fix brej nums")
        if not os.path.isfile(csv_path):
            print(f"\n\n  [WARN] Missing: {csv_path}\n\n")
            continue
        for _, csv_row in pd.read_csv(csv_path).iterrows():
            trial_id = str(int(csv_row["trial_id"])).zfill(3)
            lut, ff, dsp, bram, src = _hls_resources_csynth_only(
            # lut, ff, dsp, bram, src = _hls_resources_csynth_or_vsynth(
                bit_dir, trial_id,
                ["pareto_primary/hls_outputs",
                 "pareto_secondary/hls_outputs",
                 "hls_outputs"])
            if lut is None:
                continue
            if dsp != 0:
                print(f"\n\n  [WARN] NONZERO DSPs !!!: {bit_dir}, {trial_id}\n\n")
            fullpath1 = os.path.join(bit_dir,"pareto_primary/")
            fullpath2 = os.path.join(bit_dir,"pareto_secondary/")
            fullpath3 = os.path.join(fullpath1,"finishedTrials") #added for model3, which has moved files once finished
            fullpath = os.path.join(fullpath1,"model_trial_"+trial_id+'.h5')
            if not os.path.isfile(fullpath):
                fullpath = os.path.join(fullpath2,"model_trial_"+trial_id+'.h5')
                if not os.path.isfile(fullpath):
                    # print("using fullpath3\n\n\n")
                    fullpath = os.path.join(fullpath3,"model_trial_"+trial_id+'.h5')
            rows.append(_build_row(modelName, run_name, trial_id, csv_row,
                                   lut, ff, dsp, bram, src,fullpath))
    return pd.DataFrame(rows)


# ── Pareto helpers ─────────────────────────────────────────────────────────────
def _is_dominated(point, others):
    for _, other in others.iterrows():
        if (other[PRIMARY_METRIC] >= point[PRIMARY_METRIC] and
                other["luts_plus_ff"]   <= point["luts_plus_ff"] and
                (other[PRIMARY_METRIC] > point[PRIMARY_METRIC] or
                 other["luts_plus_ff"]  < point["luts_plus_ff"])):
            return True
    return False


def find_pareto_front(df):
    idx = [i for i, row in df.iterrows()
           if not _is_dominated(row, df.drop(index=i))]
    return (df.loc[idx].copy()
              .sort_values(by=[PRIMARY_METRIC, "luts_plus_ff"],
                           ascending=[False, True]))

# Added to make vertical lines for reference points for FF+LUT
#textOffset is a ratio
def add_threshold_line(ax, x_val, label, color='gray', linestyle='--', linewidth=1.5,labelYHeight = 0.5,textOffset = 0):

    """Adds a vertical line with a vertical text label at the top of the axis."""
    if textOffset ==0:
        textOffset = 0.05
    if x_val < 2000000:
        ax.axvline(x=x_val, color=color, linestyle=linestyle, linewidth=linewidth, zorder=2,
                label='_nolegend_',  # Keeps the script's legend logic from breaking
                )
        #doesn't work for too big x vals
        # # 1. Grab current y-limits so the line spans the full height of the plot
        # ymin, ymax = ax.get_ylim()
        
        # from matplotlib.lines import Line2D
        # line = Line2D(
        #     [x_val, x_val], [ymin, ymax],
        #     color=color, 
        #     linestyle=linestyle, 
        #     linewidth=linewidth, 
        #     zorder=2
        # )
        # ax.add_line(line) # Forces line onto plot without registering it to the legend subsystem


        # Transform: x is data coordinates, y is axis fraction (0 to 1)
        transform = ax.get_xaxis_transform()
        ax.text(
            x=x_val+x_val*textOffset, 
            y=labelYHeight,               
            s=label,
            rotation=90,          # Vertical text
            transform=transform, 
            color=color, 
            va='bottom',         
            ha='left',        
            fontsize=12,
            zorder=3
        )
    else:
        print(f"Not plotting vertical line for {label} because {x_val} is too big")

# ── Y-axis transform helpers ───────────────────────────────────────────────────
def _y(v, complement):
    if complement:
        return np.clip(1.0 - np.asarray(v, dtype=float), 1e-4, None)
    return v


def _setup_complement_log_y(ax, df):
    ax.set_yscale("log")
    ax.invert_yaxis()
    ticks_actual = [t for t in COMP_LOG_TICKS
                    if t < df[PRIMARY_METRIC].max() + 0.05]
    ticks_comp   = np.clip(1.0 - np.array(ticks_actual, dtype=float), 1e-4, None)
    ax.set_yticks(ticks_comp)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{1 - v:.2f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    y_min = max(1e-4, float(np.min(_y(df[PRIMARY_METRIC], True))) * 0.5)
    y_max = float(np.max(_y(df[PRIMARY_METRIC], True))) * 2.0
    ax.set_ylim(y_max, y_min)


# ── Core drawing helpers ───────────────────────────────────────────────────────
def _draw_scatter(ax, df, pareto_df, x_col, complement=False, annotate=True):
    for run_name in df["run_name"].unique():
        color  = RUN_COLORS.get(run_name, "gray")
        label  = RUN_LABELS.get(run_name, run_name)
        subset = df[df["run_name"] == run_name]
        ax.scatter(subset[x_col], _y(subset[PRIMARY_METRIC], complement),
                   alpha=0.30, s=25, c=color, edgecolors="none",
                   label=label, zorder=1)

    for run_name in pareto_df["run_name"].unique():
        color = RUN_COLORS.get(run_name, "gray")
        p_sub = pareto_df[pareto_df["run_name"] == run_name]
        ax.scatter(p_sub[x_col], _y(p_sub[PRIMARY_METRIC], complement),
                   alpha=0.90, s=60, c=color, edgecolors="black",
                   linewidth=1.2, marker="D", zorder=3)
    for _, row in pareto_df.iterrows():
        if (row['run_name'],row['trial_id']) in SELECTED_MODELS:
            color = RUN_COLORS.get(row['run_name'], "gray")
            ax.scatter(row[x_col], _y(row[PRIMARY_METRIC], complement),
                   alpha=0.90, s=700, c=color, edgecolors="black",
                   linewidth=1.2, marker="*", zorder=3)


    ps    = pareto_df.sort_values(x_col)
    style = {k: v for k, v in FAMILY_LINE["Combined"].items()
             if k not in ("label", "lw")}
    ax.plot(ps[x_col], _y(ps[PRIMARY_METRIC], complement),
            **style, alpha=0.25, zorder=2, lw=1.2)

    if annotate:
        for _, row in pareto_df.iterrows():
            short = (RUN_LABELS.get(row["run_name"], row["run_name"])
                     .replace("Model 1 ", "M1 ")
                     .replace("Model 2.5 ", "M2.5 ")
                     .replace("Model 3 ", "M3 "))
            yval = float(_y(row[PRIMARY_METRIC], complement))
            ax.annotate(f"{short}\n{row['trial_id']}",
                        xy=(row[x_col], yval),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=4, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow",
                                  alpha=0.8, edgecolor="black", linewidth=0.8),
                        zorder=4)


def _draw_subfronts(ax, pareto_m1, pareto_m25, pareto_m3, pareto_all,
                    x_col, complement=False):
    for pareto, key, alpha in [
            (pareto_m1,  "Model 1",   0.85),
            (pareto_m25, "Model 2.5", 0.85),
            (pareto_m3,  "Model 3",   0.85),
            (pareto_all, "Combined",  0.25)]:
        style = FAMILY_LINE[key]
        ps = pareto.sort_values(x_col)
        ax.plot(ps[x_col], _y(ps[PRIMARY_METRIC], complement),
                color=style["color"], lw=style["lw"], ls=style["ls"],
                alpha=alpha, zorder=5, label=style["label"])

#modified from https://www.statology.org/matplotlib-legend-order/
#to swap order of items in legend
def reorderLegend(handles, labels, legendOrder=[]):
    #get handles and labels
    assert len(handles) == len(labels)

    #specify order of items in legend
    if len(legendOrder) != len(handles):
        print(f"Legend reordering is invalid since length {len(legendOrder), len(handles)} mismatch, so doing nothing")
        return handles, labels
    return [handles[idx] for idx in legendOrder],[labels[idx] for idx in legendOrder]
  
def _add_legend_and_stats(ax, df, pareto_df, complement=False, extra_handles=None):
    handles, labels = ax.get_legend_handles_labels()
    if extra_handles:
        handles += extra_handles
        labels  += [h.get_label() for h in extra_handles]
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    # loc = "upper left" if complement else "lower right"
    loc = "lower right"
    legendOrder = [0,1,2,3,4,15,18,5,6,7,8,9,16,10,11,12,13,14,17]
    h2, l2 = reorderLegend(h2, l2, legendOrder=legendOrder)
    ax.legend(h2, l2, loc=loc, fontsize=11, framealpha=0.9, ncol=3, columnspacing=0.6)

    stats = (f"Total models plotted: {len(df)}\n"
             f"Pareto optimal: {len(pareto_df)} ({100*len(pareto_df)/len(df):.1f}%)\n"
             f"Metric range: {df[PRIMARY_METRIC].min():.4f} – {df[PRIMARY_METRIC].max():.4f}")
    ax.text(0.02, 0.02 if complement else 0.98, stats,
            transform=ax.transAxes, fontsize=9,
            va="bottom" if complement else "top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))


def _finalize(ax, title, xscale="linear", complement=False):
    tags = []
    if xscale != "linear": tags.append("x-log")
    if complement:          tags.append("y: 1−metric log")
    tag = f" [{', '.join(tags)}]" if tags else ""
    ax.set_title(title + tag, fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("LUTs + FFs (registers) (csynth)", fontsize=16, fontweight="bold")
    ax.set_ylabel(
        f"{METRIC_NAME}  (1−metric, log scale)" if complement
        else METRIC_NAME,
        fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.28, linestyle="--", which="both")
    for (ff_plus_lut, label,color,labelYHeight,textOffset) in HARDWARE_REFS:
        add_threshold_line(ax,ff_plus_lut,label,color=color,labelYHeight=labelYHeight,textOffset = textOffset)
    # add_threshold_line(ax,34568,"Smartpixel Filtering Model (csynth) DOI 10.1088/2632-2153/ad6a00",color="teal",labelYHeight=0.9)
    # add_threshold_line(ax,26376,"Smartpixel Filtering Model (vsynth) DOI 10.1088/2632-2153/ad6a00",color="teal",labelYHeight=0.9)
    # add_threshold_line(ax,35216,"Smartpixel Filtering Model (csynth) but add an input quantization layer",color="teal",labelYHeight=0.95)
    # add_threshold_line(ax,24853,"Smartpixel Filtering Model (vsynth) but add an input quantization layer",color="teal",labelYHeight=0.95)
    if xscale == "log":
        ax.set_xscale("log")


# ── Public plot functions ──────────────────────────────────────────────────────
def _save(fig, output_dir, fname):
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  ✓ {path}")
    plt.close(fig)


def make_plot_simple(df, pareto_df, output_dir, zoomed=False, annotate=True):
    x_col = "luts_plus_ff"
    fig, ax = plt.subplots(figsize=(16, 9))
    _draw_scatter(ax, df, pareto_df, x_col, annotate=annotate)
    if zoomed:
        ax.set_xlim(0, 1.5e5); ax.set_ylim(0.6, 1.0)
        suffix = "_zoomed"
        title  = "Model 1 vs Model 2 vs Model 3 — Pareto Front (Zoomed)"
    else:
        suffix = "_full"
        title  = f"Model 1 vs Model 2 vs Model 3 — Combined Pareto Front\n{METRIC_NAME} vs LUTs + FF"
        title  = f"Model 1 vs Model 2 vs Model 3 — Combined Pareto Front"
    _finalize(ax, title)
    _add_legend_and_stats(ax, df, pareto_df)
    plt.tight_layout()
    tag = "" if annotate else "_nolabels"
    _save(fig, output_dir, f"pareto_all_models{suffix}{tag}.png")


def make_plot_subfronts(df, pareto_df, pareto_m1, pareto_m25, pareto_m3,
                        output_dir, xscale="linear", complement=False, annotate=True):
    x_col = "luts_plus_ff"
    fig, ax = plt.subplots(figsize=(16, 9))
    fig, ax = plt.subplots(figsize=(14, 9*14/16))
    _draw_scatter(ax, df, pareto_df, x_col, complement=complement, annotate=annotate)
    _draw_subfronts(ax, pareto_m1, pareto_m25, pareto_m3, pareto_df,
                    x_col, complement=complement)
    if complement:
        _setup_complement_log_y(ax, df)

    title = ("Model 1 vs Model 2 vs Model 3 — Sub-fronts + Combined Pareto")
            #  f"\n{METRIC_NAME} vs LUTs + FF")
    _finalize(ax, title, xscale=xscale, complement=complement)

    extra = [mlines.Line2D([], [], color=FAMILY_LINE[k]["color"],
                           lw=FAMILY_LINE[k]["lw"], ls=FAMILY_LINE[k]["ls"],
                           label=FAMILY_LINE[k]["label"])
             for k in ("Model 1", "Model 2.5", "Model 3", "Combined")]
    _add_legend_and_stats(ax, df, pareto_df, complement=complement, extra_handles=extra)
    plt.tight_layout()

    parts = ["xlog" if xscale == "log" else "xlinear",
             "ycomplog" if complement else "ylinear"]
    tag   = "" if annotate else "_nolabels"
    _save(fig, output_dir,
          "pareto_all_models_subfronts_" + "_".join(parts) + tag + ".png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading Model 1...")
    if newFolderStructure:
        df1 = load_modelNEW(MODEL1_DIR,1)
    else:
        df1 = load_model1(MODEL1_DIR)
    print(f"  {len(df1)} trials with HLS resources "
          f"({(df1['hls_source'] == 'vsynth').sum()} vsynth, "
          f"{(df1['hls_source'] == 'csynth').sum()} csynth)")
    # print(df1["dsp"])
    # print(df1["bram"])
    # print([df1[key].head() for key in df1.keys()])
    # print(df1.keys())

    print("Loading Model 2.5...")
    if newFolderStructure:
        df25 = load_modelNEW(MODEL25_DIR,2)
    else:
        df25 = load_model25(MODEL25_DIR)
    print(f"  {len(df25)} trials with HLS resources (all csynth)")

    print("Loading Model 3...")
    if newFolderStructure:
        df3 = load_modelNEW(MODEL3_DIR,3)
    else:
        df3 = load_model3(MODEL3_DIR)
    print(f"  {len(df3)} trials with HLS resources (all csynth)")
    # print([df3[key].head() for key in df1.keys()])

    df = pd.concat([df1, df25, df3], ignore_index=True)
    
    print("dsps",df["dsp"])
    print("bram", df["bram"])

    print(f"\nCombined total: {len(df)} trials")
    print(f"Primary metric range: {df[PRIMARY_METRIC].min():.4f} – {df[PRIMARY_METRIC].max():.4f}")
    print(f"LUTs+FF range:        {df['luts_plus_ff'].min()} – {df['luts_plus_ff'].max()}")

    # print(df)
    # raise ValueError("Stop now")
    print("\nComputing Pareto fronts...")
    pareto_all = find_pareto_front(df)
    pareto_m1  = find_pareto_front(df1)
    pareto_m25 = find_pareto_front(df25)
    pareto_m3  = find_pareto_front(df3)
    print(f"  Joint Pareto:     {len(pareto_all)}")
    print(f"  Model 1 Pareto:   {len(pareto_m1)}")
    print(f"  Model 2.5 Pareto: {len(pareto_m25)}")
    print(f"  Model 3 Pareto:   {len(pareto_m3)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(os.path.join(OUTPUT_DIR, "combined_all_detailed.csv"), index=False)
    cols = ["model", "run_name", "trial_id", "parameters", "auc", PRIMARY_METRIC,
            "luts", "registers", "luts_plus_ff", "dsp", "bram", "hls_source","fullPath"]
    pareto_all[[c for c in cols if c in pareto_all.columns]].to_csv(
        os.path.join(OUTPUT_DIR, "pareto_primary.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "timestamp":          datetime.now().isoformat(),
            "total_models":       len(df),
            "joint_pareto":       len(pareto_all),
            "model1_pareto":      len(pareto_m1),
            "model25_pareto":     len(pareto_m25),
            "model3_pareto":      len(pareto_m3),
            "model1_vsynth":      int((df1["hls_source"] == "vsynth").sum()),
            "model1_csynth":      int((df1["hls_source"] == "csynth").sum()),
            "metric_range":       {"min": float(df[PRIMARY_METRIC].min()),
                                   "max": float(df[PRIMARY_METRIC].max())},
            "luts_plus_ff_range": {"min": int(df["luts_plus_ff"].min()),
                                   "max": int(df["luts_plus_ff"].max())},
        }, f, indent=2)

    print("\nGenerating plots...")
    for annotate in (True, False):
        make_plot_simple(df, pareto_all, OUTPUT_DIR, zoomed=False, annotate=annotate)
        make_plot_simple(df, pareto_all, OUTPUT_DIR, zoomed=True,  annotate=annotate)
        make_plot_subfronts(df, pareto_all, pareto_m1, pareto_m25, pareto_m3,
                            OUTPUT_DIR, xscale="linear", complement=False, annotate=annotate)
        make_plot_subfronts(df, pareto_all, pareto_m1, pareto_m25, pareto_m3,
                            OUTPUT_DIR, xscale="log",    complement=False, annotate=annotate)
        make_plot_subfronts(df, pareto_all, pareto_m1, pareto_m25, pareto_m3,
                            OUTPUT_DIR, xscale="linear", complement=True,  annotate=annotate)
        make_plot_subfronts(df, pareto_all, pareto_m1, pareto_m25, pareto_m3,
                            OUTPUT_DIR, xscale="log",    complement=True,  annotate=annotate)

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
