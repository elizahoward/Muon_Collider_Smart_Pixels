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

class SmartPixModel(ABC):
    def __init__(self,
            tfRecordFolder: str = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
            nBits: list = None, # just for fractional bits, integer bits 
            loadModel: bool = False,
            modelPath: str = None, # Only include if you are loading a model
            ): 
        # Do we want to specify model, modelType, bitSize, etc.
        # Decide here if we want to load a pre-trained model or create a new one from scratch
        self.tfRecordFolder = tfRecordFolder
        self.modelName = "Base Model" # for other models, e.g., Model 1, Model 2, etc.
        self.models = {"Unquantized": None, "Quantized": None} # Maybe have a dictionary to store different versions of the model
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
    
    @abstractmethod
    def makeUnquantizedModel():
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def makeUnquatizedModelHyperParameterTuning(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def makeQuantizedModel():
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
