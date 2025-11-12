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
                        # dropout_rate: float = 0.1,
            initial_lr: float = 1e-3,
            end_lr: float = 1e-4,
            power: int = 2,
            bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
            ): 
        self.tfRecordFolder = tfRecordFolder
        self.modelName = "Model1" # for other models, e.g., Model 1, Model 2, etc.
        # self.model = None
        self.histories = {}
        self.models = {"Unquantized": None}
        self.bit_configs = bit_configs
        for total_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{total_bits}w{int_bits}i"
            self.models[config_name] = None
        # self.quantized_model = None
        self.hyperparameterModel = None
        self.training_generator = None
        self.validation_generator = None
        self.x_feature_description: list = ['z_global','x_size', 'y_size', 'y_local']
        # Learning rate parameters
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.power = power
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

        self.models["Unquantized"] = tf.keras.Model(inputs=inputList, outputs=output)


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
            training_generator,
            validation_data = self.validation_generator,
            epochs          = 110,
            callbacks       = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
     


    def makeQuantizedModel(self):
        for total_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{total_bits}w{int_bits}i"
        
        
            print(f"Building {config_name} model...")
            self.makeQuantizedModel_withBits(total_bits=total_bits,int_bits=int_bits)
    def makeQuantizedModel_withBits(self, total_bits = 8,int_bits =0):
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
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
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
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
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
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
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
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
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
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
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
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            #kernel_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        out = QActivation("smooth_sigmoid")(x)
        config_name = f"quantized_{total_bits}w{int_bits}i"
        self.models[config_name] = tf.keras.Model(inputs=[input1, input2, input3, input4], outputs=out)

    """
    @abstractmethod
    def runHyperparameterTuning(self):
        raise NotImplementedError("Subclasses should implement this method.")
    """
    

    # """
    # config_name = Unquantized or 
    # config_name = f"quantized_{total_bits}w{int_bits}i"
    # """
    # def loadModel(self, file_path: str,config_name = "Unquantized"):
    #     self.models[config_name]=tf.keras.models.load_model(file_path, compile=False)

    # def saveModel(self, file_path = None,overwrite=False,config_name = "Unquantized"):
    #     if file_path == None:
    #         file_path = Path(f'./{self.modelName}.keras').resolve()
    #     if not overwrite and os.path.exists(file_path):
    #         raise Exception("Model exists. To overwrite existing saved model, set overwrite to True.")
    #     self.models[config_name].save(file_path)
    


    #THESE SHOULD BE UNNECESSARY NOW, use the abstract version
    def trainModel(self, epochs=100, batch_size=32, learning_rate=None, 
                   save_best=True, early_stopping_patience=20,
                   run_eagerly = False,config_name = "Unquantized"):
        # raise NotImplementedError("Use the abstract class version, trainModel()")
        print("\n\n\nWARNING SHOULD BE USING THE ABSTRACT VERSION INSTEAD\n\n\n")
        print("\n\n\nTHIS IS FOR DEBUGGING WHY YOUR QUANTIZED MODEL IS RANDOMLY GUESSING\n\n\n")
        print("\n\n\nONCE THAT IS FIGURED OUT, GO BACK TO THE ABSTRACT VERSION\n\n\n")
        # self.loadTfRecords()
        callbacks = []
        print(self.models[config_name].summary())
        self.models[config_name].compile(optimizer='adam', 
                            loss='binary_crossentropy', 
                            metrics=['binary_accuracy'],
                            run_eagerly=True
                            )
        self.histories[config_name] = self.models[config_name].fit(x=self.training_generator,validation_data=self.validation_generator, callbacks=callbacks,epochs=epochs)


    def trainUnquantizedModel(self): # in the input, specify the learning rate scheduler, etc.
        raise NotImplementedError("Use the abstract class version, trainModel()")
        self.loadTfRecords()
        callbacks = []
        self.models["Unquantized"].compile(optimizer='adam', 
                            loss='binary_crossentropy', 
                            metrics=['binary_accuracy'],
                            run_eagerly=True
                            )
        self.histories["Unquantized"] = self.models["Unquantized"].fit(x=self.training_generator,validation_data=self.validation_generator, callbacks=callbacks,epochs=100)


    def trainQuantizedModel(self): # in the input, specify the learning rate scheduler, etc.
        raise NotImplementedError("Use the abstract class version, trainModel()")
        self.loadTfRecords()
        callbacks = []
        # config_name = f"quantized_{total_bits}w{int_bits}i"
        #TODO make this based off of total bits, int bits, in bit_configs
        
        self.models["quantized_8w0i"].compile(optimizer='adam', 
                            loss='binary_crossentropy', 
                            metrics=['binary_accuracy'],
                            run_eagerly=True
                            )
        self.histories["quantized_8w0i"] = self.models["quantized_8w0i"].fit(x=self.training_generator,validation_data=self.validation_generator, callbacks=callbacks,epochs=100)

    #plot based on history
    def plotHistories(self):

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
        if self.models["Unquantized"] is not None:
            plotModelHistory(self.histories["Unquantized"], 1)
        #TODO loop through bit_configs
        if self.models["quantized_8w0i"] is not None:
            plotModelHistory(self.histories["quantized_8w0i"], 2)
