from abc import ABC, abstractmethod
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
import sys
sys.path.append('/home/ryanmichaud/common_repo/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
import OptimizedDataGenerator4 as ODG
from Model_Classes import SmartPixModel

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

        self.model = tf.keras.Model(inputs=inputList, outputs=output)


    def makeUnquatizedModelHyperParameterTuning(self):
        raise NotImplementedError("Subclasses should implement this method.")


    
    def makeQuantizedModel(self, list: [int]):
        
        def make_model(total_bits, int_bits):
            """
            Build & compile your QKeras model with the given number of integer bits.
            """
            tf.keras.backend.clear_session()
            # inputs
            input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
            input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
            input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
            input4 = tf.keras.layers.Input(shape=(1,), name="y_local")
            x = tf.keras.layers.Concatenate()([input1, input2, input3, input4])

            ## I want to try this with 1 int bit and 7 fractional
            ## I want to try this with 0 int bit and 7 fractional
            
            # layer 1
            x = QDense(
                17,
                kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                bias_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
                ## adds sum of the activations squared to the loss function 
                #activity_regularizer=tf.keras.regularizers.L2(0.0001),
            )(x)
            x = QActivation(
                activation=quantized_relu(total_bits, int_bits),
                name="q_relu1"
            )(x)

            # layer 2 (example—you can tweak per‐layer bits)
            x = QDense(
                20,
                kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                bias_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
                ## adds sum of the activations squared to the loss function 
                #activity_regularizer=tf.keras.regularizers.L2(0.0001),
            )(x)
            x = QActivation(
                activation=quantized_relu(total_bits, int_bits),
                name="q_relu2"
            )(x)

            # layer 3
            x = QDense(
                9,
                kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                bias_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
                ## adds sum of the activations squared to the loss function 
                #activity_regularizer=tf.keras.regularizers.L2(0.0001),
            )(x)
            x = QActivation(
                activation=quantized_relu(total_bits, int_bits),
                name="q_relu3"
            )(x)

            # layer 4
            x = QDense(
                16,
                kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                bias_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
                ## adds sum of the activations squared to the loss function 
                #activity_regularizer=tf.keras.regularizers.L2(0.0001),
            )(x)
            x = QActivation(
                activation=quantized_relu(total_bits, int_bits),
                name="q_relu4"
            )(x)

            # layer 5
            x = QDense(
                8,
                kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                bias_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
                ## adds sum of the activations squared to the loss function 
                #activity_regularizer=tf.keras.regularizers.L2(0.0001),
            )(x)
            x = QActivation(
                activation=quantized_relu(total_bits, int_bits),
                name="q_relu5"
            )(x)

            # output
            x = QDense(
                1,
                kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                bias_quantizer=quantized_bits(total_bits, int_bits, alpha=0.001),
                #kernel_regularizer=tf.keras.regularizers.L2(0.0001),
            )(x)
            out = QActivation("smooth_sigmoid")(x)
            
            m = tf.keras.Model(inputs=[input1, input2, input3, input4], outputs=out)
            m.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["binary_accuracy"],
                run_eagerly=True
            )
            return m

        models = {}
        Quantized_model = None

        # 1) define your sweep
        total_bits = [8]
        int_bits =   [0]

        # 2) containers for final metrics
        train_losses, val_losses = {}, {}
        train_accs,   val_accs   = {}, {}

        for i, val in enumerate(total_bits):
            print(f"\n→ training model with {val} total bits and {int_bits[i]} integer bits")
            model = make_model(val, int_bits[i])
            

            name = f"total: {val}, int: {int_bits[i]}"
            models[name] = model
            Quantized_Model = model

        self.quantized_model = Quantized_model



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
