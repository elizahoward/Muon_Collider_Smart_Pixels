"""
Author: Daniel Abadjiev
Date: Setpember 25, 2026
Description: Runner for distillers.py / studentTeacherTesting.ipynb

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import qkeras
import sys
sys.path.append("../MuC_Smartpix_ML")
sys.path.append("../eric")
sys.path.append("../ryan")
import Model_Classes
from model1 import Model1
from model2 import Model2
from model3 import Model3
from model2_5 import Model2_5
from ASICModel import ModelASIC
import tfLoaderUtils
import distillers
tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V4_June/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized/"


sys.path.append("../MuC_Smartpix_Data_Production/tfRecords")
import OptimizedDataGenerator4_data_shuffled_bigData_NewFormat as ODG2



tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V4_June/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized/"
filepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Results_June2026_99SigEff/model3_fin_results/model3_10bit_normalised_selected/pareto_primary/FailedTrialsRetryIfCoureageous/model_trial_095.h5"
teacher = Model_Classes.loadQuantizedModel(filepath)
def numOutNodes(model):
    outputLayerNodes = model.layers[-2].input_spec.axes[-1]
    return outputLayerNodes;



def numOutNodes(model):
    outputLayerNodes = model.layers[-2].input_spec.axes[-1]
    return outputLayerNodes;
def constructModel (
            input_bits = 12,
            w_bits = 10,
            rownodes = 10,
            hintNodes = numOutNodes(teacher),):

    input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
    input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
    input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
    input4 = tf.keras.layers.Input(shape=(1,), name="y_local")
    # input5 = tf.keras.layers.Input(shape=(1,), name="nModule")
    # input6 = tf.keras.layers.Input(shape=(1,), name="x_local")

    # inputList = [input2, input3, input4, input5, input6]
    inputList = [input1, input2, input3, input4]

    q_input1 = qkeras.QActivation(activation=qkeras.quantized_bits(input_bits, 0), name="q_input_1")(input1)
    q_input2 = qkeras.QActivation(activation=qkeras.quantized_bits(input_bits, 0), name="q_input_2")(input2)
    q_input3 = qkeras.QActivation(activation=qkeras.quantized_bits(input_bits, 0), name="q_input_3")(input3)
    q_input4 = qkeras.QActivation(activation=qkeras.quantized_bits(input_bits, 0), name="q_input_4")(input4)
    # q_input5 = QActivation(activation=qkeras.quantized_bits(input_bits, 0), name="q_input_5")(input5)
    # q_input6 = QActivation(activation=qkeras.quantized_bits(input_bits, 0), name="q_input_6")(input6)

    x_concat1 = tf.keras.layers.Concatenate()([q_input2, q_input3])
    x_concat2 = tf.keras.layers.Concatenate()([x_concat1, q_input4])
    x_concat3 = tf.keras.layers.Concatenate()([x_concat2, q_input1])
    # x_concat3 = tf.keras.layers.Concatenate()([x_concat2, q_input5])
    # x_concat4 = tf.keras.layers.Concatenate()([x_concat3, q_input6])
    x=x_concat3

    # layer 1
    x = qkeras.QDense(
    rownodes,
    kernel_quantizer=qkeras.quantized_bits(w_bits, 0, alpha=1),
    bias_quantizer=qkeras.quantized_bits(w_bits, 0, alpha=1),
    )(x)
    x = qkeras.QActivation(
    activation=qkeras.quantized_relu(8, 0),
    name="q_relu1"
    )(x)
    ## layer 2
    x = qkeras.QDense(
    hintNodes,
    kernel_quantizer=qkeras.quantized_bits(w_bits, 0, alpha=1),
    bias_quantizer=qkeras.quantized_bits(w_bits, 0, alpha=1),
    )(x)
    x = qkeras.QActivation(
    activation=qkeras.quantized_relu(8, 0),
    name="q_relu2"
    )(x)
    ## output layer
    x = qkeras.QDense(
    1,
    kernel_quantizer=qkeras.quantized_bits(w_bits, 0, alpha=1),
    bias_quantizer=qkeras.quantized_bits(w_bits, 0, alpha=1),
    )(x)


    ## output later
    output = qkeras.QActivation("quantized_sigmoid(8,0)", name="output_activation")(x)

    model = tf.keras.Model(inputs=inputList, outputs=output)
    return model
# --- Step 0: build student (quantized, matching teacher's input features) ---
student = constructModel()




d = distillers.OfflineDistiller(teacher, student,)
                            #  temperature=3.0, alpha=0, beta=0)

model1Dummy = Model1(tfRecordFolder = tfRecordFolder)     
model3Dummy = Model3(tfRecordFolder = tfRecordFolder)    
model1Dummy.x_feature_description = model1Dummy.x_feature_description + ["z_global"] + model3Dummy.x_feature_description + ["nPix"]
model1Dummy.loadTfRecords()
odgTrain = model1Dummy.training_generator
odgTest = model1Dummy.validation_generator

augTfRecordDir = "./augRecords"
augTrainDir = f"{augTfRecordDir}/tfrecords_train/"
augValDir = f"{augTfRecordDir}/tfrecords_validation/"



regenerateRecords=False
if regenerateRecords:
    d.extract_and_save_teacher_outputs(
        training_generator=odgTrain,
        output_dir=augTrainDir,
        validation_generator=odgTest,
        val_output_dir=augValDir,
    )




aug_train_gen = ODG2.OptimizedDataGeneratorDataShuffledBigData(
    load_records=True,
    tf_records_dir=augTrainDir,
    x_feature_description=model1Dummy.x_feature_description + ["teacher_logits", "teacher_feat"],
    batch_size=16384,
)
aug_val_gen = ODG2.OptimizedDataGeneratorDataShuffledBigData(
    load_records=True,
    tf_records_dir=augValDir,
    x_feature_description=model1Dummy.x_feature_description + ["teacher_logits", "teacher_feat"],
    batch_size=16384,
)


model = d.build_student_model()
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     student_loss_fn=tf.keras.losses.BinaryCrossentropy(),
#     metrics=[tf.keras.metrics.BinaryAccuracy()],
#     run_eagerly=True,  # required for QKeras models
#     loss = None,
# )
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    student_loss_fn=tf.keras.losses.BinaryCrossentropy(),
    alpha=0,
    beta=0,
    temperature=1.0,
    metrics=[tf.keras.metrics.BinaryAccuracy()],
    run_eagerly=True,
)
# # Check what student_input_keys resolved to
# print("student input keys:", model.student_input_keys)

# # Check one batch manually
# x_batch, y_batch = aug_train_gen[0]
# print("batch keys:", list(x_batch.keys()))
# student_x = {k: v for k, v in x_batch.items() if k in model.student_input_keys}
# print("filtered student keys:", list(student_x.keys()))
# y_pred = model.student(student_x, training=False)
# print("y_pred sample:", y_pred[:5])
model.fit(aug_train_gen,
        validation_data=aug_val_gen,
        epochs=2)
