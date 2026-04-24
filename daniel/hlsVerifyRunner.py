#Author: Daniel Abadjiev
#Date: April 22, 2026
#Description: Runner for hlsVerification.py, that runs the class, taken from the original testCatapultNtbk.ipynb

import hlsVerification

hlsGuy = hlsVerification.hlsVerifier(
    doingCatapult = False, #If using catapult, use the ccs_env python environment
    doingVitis = True, #If using vitis, use the hls4ml "default" environment that works with Vitis      
    # loadTestVectors = True,
    # saveTestVectors = False,
    buildModel = True,
    # customModel = False,
    # modelType = 2.5, #so far using 1 and 2.5, but in future will use the specification in hlsUtils
    # filepath = "",
)