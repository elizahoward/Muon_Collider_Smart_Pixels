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
import glob
from matplotlib import colors
import pandas as pd


def plotHistory(history,accKey="binary_accuracy",yscale="log", savePlotName=None,extraValPlot=False,title="",figsize=(10,10),typeMDMM=False):
    plt.figure(figsize=figsize)
    if accKey not in history.keys():
        raise ValueError("wrong accuracy key")
    plt.subplot(211)
    plt.plot(history[accKey],label=accKey)
    plt.plot(history[f"val_{accKey}"],label=f"val_{accKey}")
    plt.yscale(yscale)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(accKey+" training student")
    plt.title(title)
    plt.subplot(212)
    for lossKey in history.keys():
        if ("loss" in lossKey) or ("constraint" in lossKey):
            plt.plot(history[lossKey],label=lossKey,alpha=0.7)    
    if extraValPlot:
        plt.plot(history['val_loss'],"o",label="val_loss")
    plt.ylabel("loss training student")
    plt.xlabel("epoch")
    plt.yscale(yscale)
    plt.legend()
    if savePlotName is None:
        plt.show()
    else:
        plt.savefig(savePlotName)
        plt.close()
def evaluateModelFromPath(modelFolderPath,tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V4_June/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized/"):
    configName="justThisOne"
    smodel = Model1(tfRecordFolder = tfRecordFolder) 
    smodel.models[configName] = Model_Classes.loadQuantizedModel(modelFolderPath+"/model.h5")
    smodel.models[configName].compile(metrics=[tf.keras.metrics.BinaryAccuracy()])
    evalResults = smodel.evaluate(config_name=configName,predictionPlots=False,signal_efficiencies=[0.95, 0.98, 0.99])
    return evalResults,evalResults['bkg_rej_at_99pct']
def paramsFromStudentPath(studentPath):
    parts = studentPath.split("_")
    epochs = parts[-1][2:]
    temp = parts[-2][1:]
    beta = parts[-3][1:]
    alpha = parts[-4][1:]
    timestamp = parts[-6] + "_" + parts[-5]
    return epochs,temp,beta,alpha,timestamp
def paramsFromMDMMStudentPath(studentPath):
    parts = studentPath.split("_")
    epochs = parts[-1][2:]
    hintDamping = parts[-2][1:]
    distilDamping = parts[-3][1:]
    hintMax = parts[-4][1:]
    distillMax = parts[-5][1:]
    timestamp = parts[-7] + "_" + parts[-6]
    return epochs,hintDamping,distilDamping,hintMax,distillMax,timestamp
def showAllStudentResults(modelFolderPath,makePlots=False,typeMDMM=False):
    with open(modelFolderPath+"/history.json","r") as f:
        history = json.load(f)
    # print(history)
    print(history["val_binary_accuracy"][-1])
    if typeMDMM:
        epochs,hintDamping,distilDamping,hintMax,distillMax,timestamp = paramsFromMDMMStudentPath(modelFolderPath)
    else:
        epochs,temp,beta,alpha,timestamp = paramsFromStudentPath(modelFolderPath)
    evalResults,brej99se = evaluateModelFromPath(modelFolderPath)
    print(brej99se)
    if makePlots:
        if typeMDMM:
            plotHistory(history,title=f"time_{timestamp} trainFor{epochs}Epochs brejAt99SE:{brej99se} \n hintDamping:{hintDamping} distilDamping:{distilDamping} hintMaxLoss:{hintMax} distilMaxLoss:{distillMax}",figsize=(8,6))
        else:
            plotHistory(history,title=f"time_{timestamp} trainFor{epochs}Epochs brejAt99SE:{brej99se} \n temp:{temp} beta:{beta} alpha:{alpha}",figsize=(8,6))
    if typeMDMM:
        return brej99se,hintDamping,distilDamping,hintMax,distillMax,epochs,timestamp,evalResults,history
    else:
        return brej99se,temp,beta,alpha,epochs,timestamp,evalResults,history

def iterateShowingStudRes(modelResGlob,makePlots=False,typeMDMM=False):
    allPathRes = []
    for path in glob.glob(modelResGlob):
        print(path)
        pathRes = showAllStudentResults(path,makePlots=makePlots,typeMDMM=typeMDMM)
        if typeMDMM:
            allPathRes.append({"brej99se":pathRes[0],"hintDamping":pathRes[1],"distilDamping":pathRes[2],"hintMax":pathRes[3],"distillMax":pathRes[4],"epochs":pathRes[5],})
        else:
            allPathRes.append({"brej99se":pathRes[0],"temp":pathRes[1],"beta":pathRes[2],"alpha":pathRes[3],"epochs":pathRes[4],})
    return allPathRes
def plotAllStuRes(allPathRes,sizeScale=17):

    studResOrig = pd.DataFrame(allPathRes)
    # print(studResOrig)
    studRes = studResOrig.query("brej99se>0.5")
    # # print(studRes)
    # # print(studRes["temp"].to_numpy(dtype="int"))
    # for idx,row in enumerate(allPathRes):
    #     print(np.log(row["brej99se"])*100+30)
    #     plt.plot(float(row["beta"]),float(row["alpha"])+(idx/20),"o",markersize=np.log(row["brej99se"])*100+30)
    #     plt.text(float(row["beta"]),float(row["alpha"])+(idx/20),f"t{row['temp']} a{row['alpha']} b{row['beta']} b99: {row['brej99se']:0.4}")
    # # plt.plot(studRes["beta"],studRes["alpha"],"o",markersize=[1, 3, 5, 1, 3, 5, 1])
    # # print(studRes["brej99se"])


    fig, ax = plt.subplots(figsize=(8, 6))

    temps = sorted(studRes["temp"].unique())
    temp_offsets = {t: (i - len(temps)/2) * 0.05 for i, t in enumerate(temps)}
    sc = ax.scatter(
        studRes["beta"].astype(float),
        studRes["alpha"].astype(float) + studRes["temp"].map(temp_offsets).astype(float),
        c=studRes["brej99se"].astype(float),
        s=studRes["temp"].astype(float)*sizeScale+1,
        cmap="viridis",
        # vmin=studRes["brej99se"].astype(float).min(),
        # vmax=studRes["brej99se"].astype(float).max(),
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
        # norm=colors.LogNorm(vmin=studRes["brej99se"].astype(float).min(), vmax=studRes["brej99se"].astype(float).max())
    )
    plt.colorbar(sc, label="brej99se")
    for row in allPathRes:
        if row["brej99se"] < 0.5:
            continue
        plt.text(float(row["beta"]),float(row["alpha"])+temp_offsets[row["temp"]],f"t{row['temp']} a{row['alpha']} b{row['beta']} b99: {row['brej99se']:0.4}")


    # Legend for temperature offsets
    for t, offset in temp_offsets.items():
        ax.plot([], [], 'o', color='gray',markersize=t, label=f"t={t} (offset {offset:+.2f})")
    ax.legend(title="temperature", loc="center")

    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_xlim([-0.01,0.15])
    ax.set_yticks([0, 0.3, 0.5])
    ax.set_xticks([0,0.1])
    ax.set_title("Training distilled models with different losses — color: brej99, jitter: temperature")

    return studRes,studResOrig



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
        typeMDMM: bool = True,
        distil_max_value:float=0.3,
        hint_max_value:float =0.6,
        dampingHint:float = 1,
        dampingDistil:float = 1,
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
        self.aug_train_gen = None
        self.aug_val_gen = None
        self.nEpochs = nEpochs
        self.regenerateRecords = regenerateRecords
        self.augTfRecordDir = augTfRecordDir
        self.learningRate = learningRate
        self.typeMDMM = typeMDMM
        self.distil_max_value= distil_max_value
        self.hint_max_value = hint_max_value
        self.dampingHint = dampingHint
        self.dampingDistil = dampingDistil
        if self.learningRate is None:
            raise NotImplementedError("Need to add learning rate scheduler")
        else:
            self.optimizer = tf.keras.optimizers.Adam(self.learningRate)
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
        if self.typeMDMM:
            self.model = self.distiller.build_mdmm_model(distil_max_value=self.distil_max_value,
                                    hint_max_value=self.hint_max_value, dampingHint= self.dampingHint,dampingDistil=self.dampingDistil)
            self.model.compile(
                optimizer=self.optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()],
            )
        else:
            self.model = self.distiller.build_student_model()

            self.model.compile(
                optimizer=self.optimizer,
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
        if self.typeMDMM:
            self.output_dir = os.path.join(self.saveDir,f"stuMDMMModel_{timestamp}_dM{self.distil_max_value}_hM{self.hint_max_value}_dD{self.dampingDistil}_hD{self.dampingHint}_nE{self.nEpochs}")
        else:
            self.output_dir = os.path.join(self.saveDir,f"studentModel_{timestamp}_a{self.alpha}_b{self.beta}_t{self.temperature}_nE{self.nEpochs}")
            
        # self.output_dir = os.path.join(self.saveDir,f"studentModel_{timestamp}_a{self.alpha}_b{self.beta}_t{self.temperature}_nE{self.nEpochs}")
        pathlib.Path(self.output_dir).mkdir(parents=True,exist_ok=True)
        if self.typeMDMM:
            self.model.model.save(os.path.join(self.output_dir,"model.h5"))
        else:
            self.model.student.save(os.path.join(self.output_dir,"model.h5"))
        with open(os.path.join(self.output_dir,"history.json"),"w") as f:
            f.write(json.dumps(self.history.history,indent=4))
        plotHistory(self.history.history,savePlotName=os.path.join(self.output_dir,"learning.png"))
        
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
        

def main(doMDMM = True):
    alphas = [0, 0.3, 0.5, 0.7, 1] #suggested by claude
    betas = [0, 0.1, 0.2]#suggested by claude
    temperatures = [1, 2, 5, 10]#suggested by claude
    
    alphas = [0, 0.3, 0.5]
    betas = [0,0.1]
    temperatures = [1, 3]

    alphas = [0,0.3,0.5]
    betas = [0,0.1]
    temperatures = [1,3]
    nEpochs = 100
    nEpochs = 50
    distillationMaxes = [0.27,0.29]
    hintMaxes = [0.6]
    dampings = [1,3,5]
    hintDampingExtras = [0,3]
    if doMDMM:
        for hintDampingExtra in hintDampingExtras:
            for damping in dampings:
                for distillationMax in distillationMaxes:
                    for hintMax in hintMaxes:
                        runner = DistillationRunner(nEpochs=nEpochs,saveDir = "./distillRuns",typeMDMM=True,distil_max_value=distillationMax,hint_max_value=hintMax,dampingHint=damping+hintDampingExtra, dampingDistil=damping)
    else:
        for temperature in temperatures:
            for beta in betas:
                for alpha in alphas:
                    runner = DistillationRunner(nEpochs=nEpochs,saveDir = "./distillRuns",alpha=alpha,beta=beta,temperature=temperature,typeMDMM=False) 

if __name__=="__main__":
    main()