#Author: Daniel Abadjiev
#Date: April 22, 2026
#Description: Runner for hlsVerification.py, that runs the class, taken from the original testCatapultNtbk.ipynb

import hlsVerification

hlsGuy = hlsVerification.hlsVerifier(
    doingCatapult = True, #If using catapult, use the ccs_env python environment
    doingVitis = False, #If using vitis, use the hls4ml "default" environment that works with Vitis      
    # loadTestVectors = True,
    # saveTestVectors = False,
    buildModel = True,
    # customModel = False,
    # modelType = 2.5, #so far using 1 and 2.5, but in future will use the specification in hlsUtils
    # filepath = "",
    interactivePlots = False,
    filepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/model2.5_quantizedinputs_quantized_6w0i_qi2_hyperparameter_results_20260424_112349/model_trial_0.h5",
)