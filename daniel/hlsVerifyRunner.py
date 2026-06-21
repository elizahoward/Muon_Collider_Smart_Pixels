#Author: Daniel Abadjiev
#Date: April 22, 2026
#Description: Runner for hlsVerification.py, that runs the class, taken from the original testCatapultNtbk.ipynb

enviroModelAsic = True

import os
if enviroModelAsic:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import hlsVerification

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
modelType = 2.5
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
        tfRecordFolder=tfRecordFolder,
        doTrace = False
    )
    print("finished with hls verification of a single file")

paretoDir = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Model2_5_tahn/model2.5_quantizedinputs_8w0i_pareto_roc_selected"
if runParetoVerification:
    for e in os.scandir(paretoDir):
        if e.is_file():
            if ".h5" in e.path:
                hlsGuy = hlsVerification.hlsVerifier(
                    doingCatapult = False, #If using catapult, use the ccs_env python environment
                    doingVitis = True, #If using vitis, use the hls4ml "default" environment that works with Vitis      
                    # loadTestVectors = True,
                    # saveTestVectors = False,
                    buildModel = False,
                    # customModel = False,
                    # modelType = 2.5,
                    # filepath = "",
                    interactivePlots = False,
                    fullRunOnInit = True,
                    filepath = e.path,
                    tfRecordFolder=tfRecordFolder,
                )