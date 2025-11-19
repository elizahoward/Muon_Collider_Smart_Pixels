'''
Author: Daniel Abadjiev
Date: created October 17, 2025
Description: A class that implements the SmartPixModel in Model_Classes.py using the ASIC model from https://github.com/elizahoward/smart-pixels-ml/tree/daniel
'''


##Imports to get the Model_Classes
# import os
import sys
sys.path.append("/local/d1/smartpixML/filtering_models/shuffling_data/") #TODO use the ODG from here
import OptimizedDataGenerator4_data_shuffled_bigData as ODG2
sys.path.append("/home/dabadjiev/smartpixels_ml_dsabadjiev/smart-pixels-ml/")
sys.path.append("../MuC_Smartpix_ML")
import Model_Classes

##now the imports specific to the ASIC model (from ASICModelTest.ipynb)
#TODO get rid of unnecessary imports
from OptimizedDataGenerator4 import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
try:
    from qkeras import QDense, QActivation, QDenseBatchnorm
    from qkeras.quantizers import quantized_bits, quantized_relu
    import hls4ml
except ModuleNotFoundError:
    print("Missing QDenseBatchnorm, hls4ml, or something of that ilk")
except:
    print("weird import error")

noGPU=False
if noGPU:
    tf.config.set_visible_devices([], 'GPU')

print(tf.config.experimental.list_physical_devices())
print(tf.test.is_built_with_cuda())
print(tf.test.is_built_with_gpu_support())
print(tf.test.is_gpu_available())
# os.system("echo $PATH")

class ModelASIC(Model_Classes.SmartPixModel):
    def __init__(self,
            tfRecordFolder: str = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try3_eric/filtering_records16384_data_shuffled_single_bigData/",
            nBits: list = None, # just for fractional bits, integer bits 
            loadModel: bool = False,
            modelPath: str = None, # Only include if you are loading a model
            initial_lr: float = 1e-3,
            end_lr: float = 1e-4,
            power: int = 2,
            ): 
        # Do we want to specify model, modelType, bitSize, etc.
        # Decide here if we want to load a pre-trained model or create a new one from scratch
        self.tfRecordFolder = tfRecordFolder
        self.modelName = "Model ASIC" # for other models, e.g., Model 1, Model 2, etc.
        # self.models = {"Unquantized": None, "Quantized": None} # Maybe have a dictionary to store different versions of the model
        self.hyperparameterModel = None
        self.x_feature_description: list = ['x_size','z_global','y_profile','x_profile','cluster','y_local']
        self.bit_configs = [(4,0)]
        
        self.histories = {}
        self.models = {"Unquantized": None}
        for total_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{total_bits}w{int_bits}i"
            self.models[config_name] = None

        # Learning rate parameters
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.power = power
        return
    
    def makeUnquantizedModel(self):
        #Make a model that has multpile layers
        input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
        input2 = tf.keras.layers.Input(shape=(1,), name="y_local")
        input3 = tf.keras.layers.Input(shape=(1,), name="x_size")
        # input3 = tf.keras.layers.Input(shape=(21,), name="x_profile")
        input4 = tf.keras.layers.Input(shape=(13,), name="y_profile")
        inputList = [input1, input2,input3,input4]
        inputs = tf.keras.layers.Concatenate()(inputList)
        stack = tf.keras.layers.Dense(58,activation='relu')(inputs)
        # stack = tf.keras.layers.Dense(10)(stack)
        output = tf.keras.layers.Dense(1,activation='sigmoid')(stack)

        model1 = tf.keras.Model(inputs=inputList, outputs=output)

        # model.summary()

        model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
        self.models["Unquantized"] = model1
        # callbacks=[]
        # learningRates = [0.1,0.9,0.6,0.3,0.1,0.03,0.01,0.001,0.0001,0.00001,0.000001]
        # callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr : lr if epoch<5 else lr*np.exp(-0.1)))
        callbacks=[]
   
    def makeUnquatizedModelHyperParameterTuning(hp):
        raise NotImplementedError("Subclasses should implement this method.")
    def makeQuantizedModel(self):
        for weight_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            # Define inputs
            input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
            input2 = tf.keras.layers.Input(shape=(1,), name="y_local")
            input3 = tf.keras.layers.Input(shape=(1,), name="x_size")
            input4 = tf.keras.layers.Input(shape=(13,), name="y_profile")
            inputList = [input1, input2, input3, input4]
            inputList = [input2,input4]
            
            # Concatenate all inputs
            # x1 =  tf.keras.layers.Concatenate()([input1, input2])
            # x2 =  tf.keras.layers.Concatenate()([input3, input4])
            # x =  tf.keras.layers.Concatenate()([x1, x2])
            x = tf.keras.layers.Concatenate()(inputList)
            # x = x_in = Input(shape, name="input1")
            x = QDenseBatchnorm(58,
            kernel_quantizer=quantized_bits(4,0,alpha=1),
            bias_quantizer=quantized_bits(4,0,alpha=1),
            name="dense1")(x)
            x = QActivation("quantized_relu(8,0)", name="relu1")(x)
            #Fermilab's output layer, not right for our dataset currently
            # x = QDense(3,
            #     kernel_quantizer=quantized_bits(4,0,alpha=1),
            #     bias_quantizer=quantized_bits(4,0,alpha=1),
            #     name="dense2")(x)
            # x = QActivation("linear", name="linear")(x)

            #Eric's Output layer
            x = QDense(
                1,
                kernel_quantizer=quantized_bits(4,0,alpha=1),
                bias_quantizer=quantized_bits(4,0,alpha=1),
                name="output_dense"
            )(x)
            output = QActivation("quantized_tanh", name="output")(x)
            model = tf.keras.Model(inputs=inputList, outputs=output)

            model.summary()
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
            self.models[config_name] = model
    