"""
Train a single quantized Model2.5 for 2 epochs on CPU
and save the H5 file for HLS synthesis testing.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import sys
sys.path.append("/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/")
sys.path.append("../MuC_Smartpix_ML/")

from model2_5 import Model2_5

DATA_FOLDER = (
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/"
    "Data_Files/Data_Set_2026V2_Apr/TF_Records/"
    "filtering_records16384_data_shuffled_single_bigData"
)

WEIGHT_BITS = 4
INT_BITS    = 0
OUTPUT_H5   = "model2_5_4w0i_hls_test.h5"

model = Model2_5(
    tfRecordFolder=DATA_FOLDER,
    bit_configs=[(WEIGHT_BITS, INT_BITS)],
    nmodule_xlocal_weight_bits=WEIGHT_BITS,
    nmodule_xlocal_int_bits=INT_BITS,
)

model.makeQuantizedModel()
model.loadTfRecords()

config_name = f"quantized_{WEIGHT_BITS}w{INT_BITS}i"
keras_model = model.models[config_name]

keras_model.fit(
    model.training_generator,
    validation_data=model.validation_generator,
    epochs=2,
    verbose=1,
)

keras_model.save(OUTPUT_H5)
print(f"\nSaved: {OUTPUT_H5}")
