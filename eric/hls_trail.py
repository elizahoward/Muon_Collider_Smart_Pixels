from OptimizedDataGenerator4 import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from qkeras import QDense, QActivation, QDenseBatchnorm
# from qkeras.quantizers import quantized_bits, quantized_relu
import hls4ml
import os

noGPU=False
if noGPU:
    tf.config.set_visible_devices([], 'GPU')

print(tf.config.experimental.list_physical_devices())
print(tf.test.is_built_with_cuda())
print(tf.test.is_built_with_gpu_support())
print(tf.test.is_gpu_available())

import os
import numpy as np
import tensorflow as tf
import csv
import pandas as pd
# import model as md
# import utils as ut

# from qkeras import QDenseBatchnorm
import qkeras
print("\n\nhls4ml version: ",hls4ml.__version__)

os.system("source ~/.bashrc")
os.system("echo $PATH")
filepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/erics_git/quantized_model1_results_20250922_162405/quantized_8w0i_8a0i/trial_1/quantized_mlp_quantized_8w0i_8a0i_trial1.keras"#"./DanielModels/model1.keras"
# qmodel_file = "/local/d1/smartpixLab/fermiModels/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
filepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/model1Test4hls.h5"
filepath = "/home/youeric/PixelML/smart_pixels_ml/eric_training_models/Model3/combined_results_20250730_021907/quantized_model_3bit_0int/best_model_3bit_0int.h5"
filepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/best_model_3bit_0int.h5"
filepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/model2_results_20251021_213803/models/model2_quantized_8bit.h5"
filepathModel2 = "/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/eric/model2_quantized_6w0i_hyperparameter_results_20251104_145942/model_trial_021.h5"
filepath=filepathModel2
# filepath = "./quantized_mlp_quantized_2w0i_2a0i_trial1.h5"
# filepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/daniel/quantized_mlp_quantized_2w0i_2a0i_trial1.h5"
# filepath = ""
# filepath = ""
co = {}       
qkeras.utils._add_supported_quantized_objects(co)
quantizedModel = tf.keras.models.load_model(filepath,custom_objects=co,compile=True)
output_dir = "./hlsTmpModel2"

config = hls4ml.utils.config_from_keras_model(quantizedModel, granularity='name')
# Convert to an hls model

hls_model = hls4ml.converters.convert_from_keras_model(quantizedModel, hls_config=config, part = 'xc7z020clg400-1', output_dir=output_dir,backend="Vitis")
##Part number input in above line as part='xcu
# part='xcu250-figd2104-2L-e',
hls_model.write()

hls_model.build(csim=False, synth=True, cosim=True, validation=False, export=True, vsynth=True, reset=True )