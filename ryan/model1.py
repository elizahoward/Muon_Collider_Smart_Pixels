from abc import ABC, abstractmethod
# import all the necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay, ExponentialDecay, CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import OptimizedDataGenerator4 as ODG

class Model1(SmartPixModel):
    def __init__(self,
            tfRecordFolder: str = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
            nBits: list = None, # just for fractional bits, integer bits
                                ## number of bits is the number of bits for each quantized model and then
                                ## run training should make one model for each bit size
            loadModel: bool = False,
            modelPath: str = None, # Only include if you are loading a model
            ): 
        self.tfRecordFolder = tfRecordFolder
        self.modelName = "Model 1" # for other models, e.g., Model 1, Model 2, etc.
        self.model = None
        self.quantized_model = None
        self.hyperparameterModel = None
        return
    
    def runAllStuff(self,):
        return
    
    def loadTfRecords(self):
        # Load the TFRecords using the OptimizedDataGenerator4
        trainDir = f"{self.tfRecordFolder}/tfrecords_train/"
        valDir   = f"{self.tfRecordFolder}/tfrecords_validation/"

        self.training_generator = ODG.OptimizedDataGenerator(load_records=True, tf_records_dir=training_dir, x_feature_description=x_feature_description)
        self.validation_generator = ODG.OptimizedDataGenerator(load_records=True, tf_records_dir=val_dir, x_feature_description=x_feature_description)
        return 
    
    
    def makeUnquantizedModel(self):
        ## here i will be making a 4-layer neural network 
        ## Model 1: z-global, x size, y size, y local


        ## define the inputs
        input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
        input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
        input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
        input4 = tf.keras.layers.Input(shape=(1,), name="y_local")

        ## concatenate the inputs into one layer
        inputList = [input1, input2, input3, input4]
        inputs = tf.keras.layers.Concatenate()(inputList)


        ## here i will add the layers 

        stack1 = tf.keras.layers.Dense(17,activation='relu')(inputs)
        stack2 = tf.keras.layers.Dense(20, activation='relu')(stack1)
        stack3 = tf.keras.layers.Dense(9, activation='relu')(stack2)
        stack4 = tf.keras.layers.Dense(16, activation='relu')(stack3)
        stack5 = tf.keras.layers.Dense(18, activation='relu')(stack4)
        output = tf.keras.layers.Dense(1,activation='sigmoid')(stack5)

        self.main_model = tf.keras.Model(inputs=inputList, outputs=output)

        return main_model



    def makeUnquatizedModelHyperParameterTuning(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def makeQuantizedModel(self, list: [int]):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def runHyperparameterTuning(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def buildModel(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def loadModel(self, file_path: str):
        self.model=tf.keras.models.load_model(file_path, compile=False)

    def saveModel(self, overwrite=False):
        file_path = Path(f'./{self.modelName}.keras').resolve()
        if not overwrite and os.path.exists(file_path):
            raise Exception("Model exists. To overwrite existing saved model, set overwrite to True.")
        self.model.save(file_path)
    
    #dataset
    def trainModel(self): # in the input, specify the learning rate scheduler, etc.
        raise NotImplementedError("We need to write this code")

    #plot based on history
    def plotModel(self):
        raise NotImplementedError("We need to write this code")

    # Evaluate the model
    def evaluate(self):
        raise NotImplementedError("We need to write this code")
