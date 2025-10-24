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
from Model_Classes import SmartPixModel
from OptimizedDataGenerator4 import *
from qkeras import QDense, QActivation, quantized_bits, quantized_relu



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
        self.unquantized_history = None

        self.quantized_model = None
        self.quantized_history = None
        self.hyperparameterModel = None
        self.trainODG = None
        self.validationODG = None
        return
    
    def runAllStuff(self,):
        return
    
    def loadTfRecords(self):
        # Load the TFRecords using the OptimizedDataGenerator4
        validation_dir = "./tf_records1000Daniel/tfrecords_validation/"
        train_dir = "./tf_records1000Daniel/tfrecords_train/"
        x_feature_description: list = ['x_size','z_global','y_profile','x_profile','cluster', 'y_size', 'x_size', 'y_local']
        self.trainODG = OptimizedDataGenerator(tf_records_dir=train_dir,load_records=True, x_feature_description=x_feature_description)
        self.validationODG = OptimizedDataGenerator(tf_records_dir=validation_dir,load_records=True, x_feature_description=x_feature_description)

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
        def model_builder(hp):
            # ── B) Architecture hyperparams ──────────────────────────────────────────
            # separately tune rows and cols

            row1nodes      = hp.Int("1",   10, 200, step=10)
            row2nodes      = hp.Int("2",   10, 200, step=10)
            row3nodes      = hp.Int("3",   10, 200, step=10)
            row4nodes      = hp.Int("4",   10, 200, step=10)
            row5nodes      = hp.Int("5",   10, 200, step=10)



            input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
            input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
            input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
            input4 = tf.keras.layers.Input(shape=(1,), name="y_local")

            ## concatenate the inputs into one layer
            inputList = [input1, input2, input3, input4]
            inputs = tf.keras.layers.Concatenate()(inputList)


            ## here i will add the layers 

            # layer 1
            x = tf.keras.layers.Dense(row1nodes,activation='relu')(inputs)
            x = tf.keras.layers.Dense(row2nodes, activation='relu')(x)
            x = tf.keras.layers.Dense(row3nodes, activation='relu')(x)
            x = tf.keras.layers.Dense(row4nodes, activation='relu')(x)
            x = tf.keras.layers.Dense(row5nodes, activation='relu')(x)
            output = tf.keras.layers.Dense(1,activation='sigmoid')(x)

            model = tf.keras.Model(inputs=inputList, outputs=output)

            model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly  = True 
            )
            return model

        tuner = kt.RandomSearch(
        model_builder,
        objective           = "val_binary_accuracy",
        max_trials          = 120,
        executions_per_trial = 2,
        project_name        = "new_hyperparam_search"
        )

        tuner.search(
            trainODG,
            validation_data = self.validationODG,
            epochs          = 110,
            callbacks       = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
     


    
    def makeQuantizedModel(self):
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
            kernel_quantizer=quantized_bits(8, 0, alpha=0.001),
            bias_quantizer=quantized_bits(8, 0, alpha=0.001),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(8, 0),
            name="q_relu1"
        )(x)

        # layer 2 (example—you can tweak per‐layer bits)
        x = QDense(
            20,
            kernel_quantizer=quantized_bits(8, 0, alpha=0.001),
            bias_quantizer=quantized_bits(8, 0, alpha=0.001),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(8, 0),
            name="q_relu2"
        )(x)

        # layer 3
        x = QDense(
            9,
            kernel_quantizer=quantized_bits(8, 0, alpha=0.001),
            bias_quantizer=quantized_bits(8, 0, alpha=0.001),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(8, 0),
            name="q_relu3"
        )(x)

        # layer 4
        x = QDense(
            16,
            kernel_quantizer=quantized_bits(8, 0, alpha=0.001),
            bias_quantizer=quantized_bits(8, 0, alpha=0.001),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(8, 0),
            name="q_relu4"
        )(x)

        # layer 5
        x = QDense(
            8,
            kernel_quantizer=quantized_bits(8, 0, alpha=0.001),
            bias_quantizer=quantized_bits(8, 0, alpha=0.001),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(8, 0),
            name="q_relu5"
        )(x)

        # output
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(8, 0, alpha=0.001),
            bias_quantizer=quantized_bits(8, 0, alpha=0.001),
            #kernel_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        out = QActivation("smooth_sigmoid")(x)
        self.quantized_model = tf.keras.Model(inputs=[input1, input2, input3, input4], outputs=out)

    """
    @abstractmethod
    def runHyperparameterTuning(self):
        raise NotImplementedError("Subclasses should implement this method.")
    """
    
    def buildModel(self):
       self.makeUnquantizedModel()

    def loadModel(self, file_path: str):
        self.model=tf.keras.models.load_model(file_path, compile=False)

    def saveModel(self, overwrite=False):
        file_path = Path(f'./{self.modelName}.keras').resolve()
        if not overwrite and os.path.exists(file_path):
            raise Exception("Model exists. To overwrite existing saved model, set overwrite to True.")
        self.model.save(file_path)
    
    #dataset
    def trainUnquantizedModel(self): # in the input, specify the learning rate scheduler, etc.
        self.loadTfRecords()
        callbacks = []
        self.model.compile(optimizer='adam', 
                            loss='binary_crossentropy', 
                            metrics=['binary_accuracy'],
                            run_eagerly=True
                            )
        self.unquantized_history = self.model.fit(x=self.trainODG,validation_data=self.validationODG, callbacks=callbacks,epochs=100)


    def trainQuantizedModel(self): # in the input, specify the learning rate scheduler, etc.
        self.loadTfRecords()
        callbacks = []
        self.quantized_model.compile(optimizer='adam', 
                            loss='binary_crossentropy', 
                            metrics=['binary_accuracy'],
                            run_eagerly=True
                            )
        self.quantized_history = self.quantized_model.fit(x=self.trainODG,validation_data=self.validationODG, callbacks=callbacks,epochs=100)

    #plot based on history
    def plotUnquantizedModel(self):
        def plotModelHistory(history,modelNum = -999):
            plt.subplot(211)
            # Plot training & validation loss values
            plt.plot(history.history['loss'],label="Train")
            plt.plot(history.history['val_loss'],label="Validation")
            plt.title(f'Model {modelNum} loss and accuracy')
            plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            plt.legend()
            plt.subplot(212)
            # Plot training & validation accuracy values
            plt.plot(history.history['binary_accuracy'],label="Train")
            plt.plot(history.history['val_binary_accuracy'],label="Validation")
            # plt.title(f'Model {modelNum} accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(loc='upper left')
            plt.show()
        plotModelHistory(self.unquantized_history, 1)

    def plotQuantizedModel(self):
        def plotModelHistory(history,modelNum = -999):
            plt.subplot(211)
            # Plot training & validation loss values
            plt.plot(history.history['loss'],label="Train")
            plt.plot(history.history['val_loss'],label="Validation")
            plt.title(f'Model {modelNum} loss and accuracy')
            plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            plt.legend()
            plt.subplot(212)
            # Plot training & validation accuracy values
            plt.plot(history.history['binary_accuracy'],label="Train")
            plt.plot(history.history['val_binary_accuracy'],label="Validation")
            # plt.title(f'Model {modelNum} accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(loc='upper left')
            plt.show()
        plotModelHistory(self.quantized_history, 1)


    # Evaluate the model
    def evaluate(self):
        print(self.quantized_model.evaluate(self.validationODG, verbose=0))
