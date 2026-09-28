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
# tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V4_June/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized/"
# from collections.abc import Callable
from typing import Callable

sys.path.append("../MuC_Smartpix_Data_Production/tfRecords")
import OptimizedDataGenerator4_data_shuffled_bigData_NewFormat as ODG2
import pathlib
from datetime import datetime
import os
import json



def numOutNodes(model):
    outputLayerNodes = model.layers[-2].input_spec.axes[-1]
    return outputLayerNodes;
def constructStudentModel (
            input_bits = 12,
            w_bits = 10,
            rownodes = 10,
            hintNodes = None,
            teacher = None,):
    if hintNodes is None:
        if teacher is None:
            raise ValueError("Can't guess hint nodes without a teacher")
        else:
            hintNodes = numOutNodes(teacher)

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

class DistillationRunner():
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        beta: float = 0.1,
        tfRecordFolder:str="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V4_June/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized/",
        teacherFilepath:str = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Results_June2026_99SigEff/model3_fin_results/model3_10bit_normalised_selected/pareto_primary/FailedTrialsRetryIfCoureageous/model_trial_095.h5",
        studentConstructor: Callable[...,tf.keras.Model] = constructStudentModel,
        runAllOnInit: bool = True,
        nEpochs: int = 3,
        regenerateRecords:bool=False,
        augTfRecordDir:str = "./augRecords",
        learningRate = 1e-3,#if None then should do a scheduler
        saveDir:str = None,
    ):
        self.temperature = temperature
        self.alpha       = alpha
        self.beta        = beta
        self.tfRecordFolder = tfRecordFolder
        self.teacherFilepath = teacherFilepath
        self.studentConstructor = studentConstructor
        self.runAllOnInit = runAllOnInit
        self.teacher = None;
        self.student = None
        self.distiller = None
        self.odgTrain = None
        self.odgTest = None
        self.nEpochs = nEpochs
        self.regenerateRecords = regenerateRecords
        self.augTfRecordDir = augTfRecordDir
        self.learningRate = learningRate
        if self.learningRate is None:
            raise NotImplementedError("Need to add learning rate scheduler")
        self.saveDir = saveDir

        if self.runAllOnInit:
            self.runAll();

    def makeAugGens(self):
        model1Dummy = Model1(tfRecordFolder = self.tfRecordFolder)     
        model3Dummy = Model3(tfRecordFolder = self.tfRecordFolder)    
        model1Dummy.x_feature_description = model1Dummy.x_feature_description + ["z_global"] + model3Dummy.x_feature_description + ["nPix"]
        
        self.augTrainDir = f"{self.augTfRecordDir}/tfrecords_train/"
        self.augValDir = f"{self.augTfRecordDir}/tfrecords_validation/"
        
        if self.regenerateRecords:
            model1Dummy.loadTfRecords()
            odgTrain = model1Dummy.training_generator
            odgTest = model1Dummy.validation_generator
            self.distiller.extract_and_save_teacher_outputs(
                training_generator=odgTrain,
                output_dir=self.augTrainDir,
                validation_generator=odgTest,
                val_output_dir=self.augValDir,
            )
        model1Dummy.x_feature_description = model1Dummy.x_feature_description + ["teacher_logits", "teacher_feat"]

        self.aug_train_gen = ODG2.OptimizedDataGeneratorDataShuffledBigData(
            load_records=True,
            tf_records_dir=self.augTrainDir,
            x_feature_description=model1Dummy.x_feature_description,
            batch_size=16384,
        )
        self.aug_val_gen = ODG2.OptimizedDataGeneratorDataShuffledBigData(
            load_records=True,
            tf_records_dir=self.augValDir,
            x_feature_description=model1Dummy.x_feature_description,
            batch_size=16384,
        )
        return self.aug_train_gen, self.aug_val_gen
    def makeOffStuModel(self):
        self.model = self.distiller.build_student_model()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            student_loss_fn=tf.keras.losses.BinaryCrossentropy(),
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
            # metrics=[tf.keras.metrics.BinaryAccuracy()],
            metrics = "binary_accuracy",
            run_eagerly=True,
        )
        return self.model
    def trainModel(self,nEpochs):
        self.history = self.model.fit(self.aug_train_gen,
                validation_data=self.aug_val_gen,
                epochs=nEpochs)
    def saveModel(self):
        if self.saveDir is None:
            raise ValueError("Cannot save to no save directory")

        pathlib.Path(self.saveDir).mkdir(parents=True,exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.saveDir,f"studentModel_{timestamp}_a{self.alpha}_b{self.beta}_t{self.temperature}_nE{self.nEpochs}")
        pathlib.Path(self.output_dir).mkdir(parents=True,exist_ok=True)
        self.model.save(os.path.join(self.output_dir,"model.keras"))
        with open(os.path.join(self.output_dir,"history.json"),"w") as f:
            f.write(json.dumps(self.history.history,indent=4))
        
    def runAll(self,nEpochs:int = None):
        if nEpochs is None:
            nEpochs = self.nEpochs

        self.teacher = Model_Classes.loadQuantizedModel(self.teacherFilepath)

        self.student = constructStudentModel(teacher=self.teacher)

        self.distiller = distillers.OfflineDistiller(self.teacher, self.student,)

        self.makeAugGens()

        self.makeOffStuModel()
        self.trainModel(nEpochs)
        if self.saveDir is not None:
            self.saveModel()
        

def main():
    runner = DistillationRunner(nEpochs=3,saveDir = "./distillRuns")
if __name__=="__main__":
    main()