#Author: Daniel Abadjiev
#Date: April 22, 2026
#Description: Runner for hlsVerification.py, that runs the class, taken from the original testCatapultNtbk.ipynb

import hlsVerification
import os

hlsGuy = hlsVerification.hlsVerifier(
    doingCatapult = False, #If using catapult, use the ccs_env python environment
    doingVitis = True, #If using vitis, use the hls4ml "default" environment that works with Vitis      
    # loadTestVectors = True,
    # saveTestVectors = False,
    buildModel = True,
    # customModel = False,
    # modelType = 2.5, #so far using 1 and 2.5, but in future will use the specification in hlsUtils
    # filepath = "",
    interactivePlots = False,
    fullRunOnInit = False,
    filepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/model2.5_quantizedinputs_quantized_6w0i_qi2_hyperparameter_results_20260424_112349/model_trial_0.h5",
)

paretoDir = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Model2_5_tahn/model2.5_quantizedinputs_8w0i_pareto_roc_selected"

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
                # modelType = 2.5, #so far using 1 and 2.5, but in future will use the specification in hlsUtils
                # filepath = "",
                interactivePlots = False,
                fullRunOnInit = True,
                filepath = e.path
            )