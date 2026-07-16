#Author: Daniel Abadjiev
#Date: April 22, 2026
#Description: Runner for hlsVerification.py, that runs the class, taken from the original testCatapultNtbk.ipynb

enviroModelAsic = True

import os
if enviroModelAsic:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import hlsVerification
import re
import shutil

print("Current MGC_HOME:", os.environ.get("MGC_HOME"))
print("Resolved Catapult Binary:", shutil.which("catapult"))

runParetoVerification = False
runSingleVerification = True

tfRecordFolder = "" #The default, which will go to tfLoaderUtils defaults, which are not normalized actually, 
# that default appropriate for some older models or for model1, but not great for the newest (as of May/June2026) models 2/3
tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V4_June/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized/"

singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/model2.5_quantizedinputs_quantized_6w0i_qi2_hyperparameter_results_20260424_112349/model_trial_0.h5",
singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/ASIC Model_results_20260610_055759/models/ASIC Model_quantized_4bit.h5"
qmodel_file = "/local/d1/smartpixLab/fermiModels/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
singleFilepath = qmodel_file
modelType = "ASIC"
singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Results_June2026_99SigEff/model2.5_fin_results/model2_5_8bit_normalised_selected/pareto_primary/model_trial_077.h5"
singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Results_June2026_99SigEff/model2.5_fin_results/model2_5_10bit_normalised_selected/pareto_primary/model_trial_057.h5"
modelType = 2.5
singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_June2026/model3_10bit_normalised_selected_pareto_primary__model_trial_046.h5"
singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_June2026/model3_8bit_normalised_selected_pareto_primary__model_trial_100.h5"
singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_June2026/model3_10bit_normalised_selected_pareto_primary__model_trial_110.h5"#current 110 10,
#need to do 110 8 bit
# need to do 009 8 bit
#need to do 084 10 bit
singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model1_fin_results_model1_8bit_normalised_selected__model_trial_1046.h5"
singleFilepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected/model3_10bit_normalised_selected_pareto_primary__model_trial_046.h5"
modelType = 3
bitNumber = 10
hlsDir = f"./hlsVerification/m{modelType}_b{bitNumber}_{singleFilepath[-20:-1]}"
if runSingleVerification:
    hlsGuy = hlsVerification.hlsVerifier(
        doingCatapult = True, #If using catapult, use the ccs_env python environment
        doingVitis = False, #If using vitis, use the hls4ml "default" environment that works with Vitis      
        loadTestVectors = True,
        saveTestVectors = False,
        buildModel = True,
        # customModel = False,
        modelType = modelType,        
        # filepath = "",
        interactivePlots = False,
        fullRunOnInit = True,
        filepath = singleFilepath,
        baseSaveDir = hlsDir,
        tfRecordFolder=tfRecordFolder,
        doTrace = False
    )
    print("finished with hls verification of a single file")

paretoDir = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Model2_5_tahn/model2.5_quantizedinputs_8w0i_pareto_roc_selected"
paretoDir = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_June2026"
paretoDir = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/CrossParetoModels_selected"
modelTypeFromFile = True
# for e in os.scandir(paretoDir):
#     if "model1" in e.path:
#         # print("type 1", e.path)
#         modelType = 1
#     if "model2" in e.path:
#         # print("type 2", e.path)
#         modelType = 2.5
#     if "model3" in e.path:
#         modelType = 3
#         # print("type 3", e.path)
if runParetoVerification:
    for e in os.scandir(paretoDir):
        if e.is_file():
            if ".h5" in e.path:
                if modelTypeFromFile:
                    if "model1" in e.path:
                        modelType = 1
                    if "model2" in e.path:
                        modelType = 2.5
                        # print("skipping model 2 type")
                        # continue
                    if "model3" in e.path:
                        modelType = 3
                        # print("skipping model 3 type")
                        # continue
                match = re.search(r'(\d+)bit', e.path)
                bitNumber = int(match.group(1)) if match else -1
                hlsDir = f"./hlsVerification/m{modelType}_b{bitNumber}_{e.path[-20:-1]}"
                print(hlsDir)
                hlsGuy = hlsVerification.hlsVerifier(
                    doingCatapult = False, #If using catapult, use the ccs_env python environment
                    doingVitis = True, #If using vitis, use the hls4ml "default" environment that works with Vitis (e.g. miscGithubEnviro)     
                    # loadTestVectors = True,
                    # saveTestVectors = False,
                    buildModel = True,
                    # customModel = False,
                    modelType = modelType,
                    # filepath = "",
                    baseSaveDir = hlsDir,
                    interactivePlots = False,
                    fullRunOnInit = True,
                    filepath = e.path,
                    tfRecordFolder=tfRecordFolder,
                    doTrace = True,
                )