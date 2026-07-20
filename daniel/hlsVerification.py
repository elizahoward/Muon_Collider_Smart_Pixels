"""
Author: Daniel Abadjiev
Date: Apr 9, 2026
Description: Turning testCatapultNtbk.ipynb into a script, so these are helper function
"""

#imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print(tf.__version__)

import os
import sys
print(sys.path)


# import hls4ml # if doingCatapult need to add catapult's version of it to path first
import csv
import pandas as pd

import qkeras
import pickle


# if not loadTestVectors:
from tfLoaderUtils import *

from importlib import import_module
import pathlib

#end imports

class hlsVerifier():
    
    def makeCustomModel(self):
        assert self.customModel
        input1 = tf.keras.layers.Input(shape=(1,), name="nModule")
        input2 = tf.keras.layers.Input(shape=(1,), name="y_local")
        input3 = tf.keras.layers.Input(shape=(1,), name="x_local")
        input4 = tf.keras.layers.Input(shape=(13,), name="y_profile")
        inputList = [input1, input2, input3, input4]
        # inputList = [input2,input4]

        # Concatenate all inputs
        # x_concat1 = tf.keras.layers.Concatenate()([input1,input2])
        # x_concat2 = tf.keras.layers.Concatenate()([x_concat1,input3])
        # x_concat3 = tf.keras.layers.Concatenate()([x_concat2,input4])
        # x=x_concat3
        weight_bits = 4;
        int_bits = 0;
        from qkeras import QDense, QActivation#, QDenseBatchnorm
        from qkeras.quantizers import quantized_bits, quantized_relu

        weight_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
        input_q_str = f"quantized_bits({weight_bits},{int_bits})"
        input_quantized  = QActivation(input_q_str, name="q_input_y_profile")(input4)
        x = tf.keras.layers.Concatenate()([input4,input4])
        x = tf.keras.layers.Concatenate()([x,x])
        # x = input4
        x = QDense(
            10,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="hidden_dense"
        )(x)

        x = QDense(
            1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="output_dense"
        )(x)
        output = QActivation("smooth_sigmoid", name="output")(x)
        model = tf.keras.Model(inputs=[input4], outputs=x)

        # model.summary()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
        self.model = model

    def fitCustomModel(self):
        assert self.customModel
        assert self.model is not None
        epochs = 2
        callbacks = []
        self.history = self.model.fit(
                    x=self.xTrain,
                    y=self.yTrain,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                ) 
    
    def plotCustomModel(self):
        assert self.customModel
        assert self.model is not None
        assert self.history is not None
        plt.plot(self.history.history['binary_accuracy'],label="accuracy")
        plt.plot(self.history.history['loss'],label="loss")
        # plt.plot(history.history['val_binary_accuracy'],label="Validation")
        test_loss, test_accuracy = self.model.evaluate(self.xTest, self.yTest)
        print(test_loss,"that was loss, val accuracy  is",test_accuracy)

    # no longer works for some reason
    def fixLayerActivationRegular(self):
        for layer in self.quantizedModel.layers:
            if hasattr(layer, 'activation'):
                act = layer.activation
                
                # 1. Handle QKeras strings like 'quantized_relu(8,0)'
                if isinstance(act, str) and 'quantized_' in act:
                    # This converts the string into a QKeras activation object
                    # layer.activation = qkeras.get_quantized_activation(act)
                    layer.activation = qkeras.get_quantizer(act)
                
                # 2. Handle standard Keras strings like 'relu' or 'linear'
                elif isinstance(act, str):
                    layer.activation = tf.keras.activations.get(act)

                # 3. Final safety check: ensure the object has a __name__ for hls4ml
                if not hasattr(layer.activation, '__name__'):
                    # QKeras objects usually store their name in __name__ or we can use the class name
                    name = getattr(layer.activation, '__name__', layer.activation.__class__.__name__)
                    # If it's still an object we can't edit, we can force a name attribute 
                    # by wrapping it or using setattr if the object allows it
                    try:
                        setattr(layer.activation, '__name__', name)
                    except AttributeError:
                        pass 

    #Rewritten by AI to handle QDenseBatchNorm from ASICModel
    # def fixLayerActivation(self):
    #     for layer in self.quantizedModel.layers:
    #         if hasattr(layer, 'activation'):
    #             act = layer.activation
                
    #             # 1. Skip if it is already a healthy Keras Tuner dictionary
    #             if isinstance(act, dict) and 'class_name' in act and act['class_name'] != 'linear':
    #                 continue

    #             # 2. Extract structural string name from function objects or references
    #             if not isinstance(act, str):
    #                 name = getattr(act, '__name__', act.__class__.__name__)
    #             else:
    #                 name = act

    #             # 3. Clean up serialized parameterized strings like "quantized_relu(8,0)"
    #             if '(' in name:
    #                 name = name.split('(')[0]

    #             # 4. Route accurately to satisfy both to_json() and hls4ml
    #             if 'quantized_' in name or name in ['binary', 'ternary']:
    #                 layer.activation = {
    #                     'class_name': name,
    #                     'config': {
    #                         'bits': 8, 
    #                         'integer': 0
    #                     }
    #                 }
    #             elif name == 'linear':
    #                 # UNIVERSAL FIX: Use the resolved activation function object.
    #                 # This serializes safely to 'linear' string inside Keras json 
    #                 # AND yields a valid .__name__ property to satisfy hls4ml.
    #                 layer.activation = tf.keras.activations.get('linear')
    #             else:
    #                 layer.activation = name
    # def fixLayerActivation(self):
    #     # Define a robust dummy class that acts like a functional QKeras quantizer object 
    #     # to perfectly satisfy hls4ml's internal get_activation_quantizer routine
    #     class Hls4mlCompatibleQuantizer:
    #         def __init__(self, name):
    #             self.__name__ = name
    #             self.class_name = name

    #         def __call__(self, x):
    #             return x

    #         def get_config(self):
    #             # Returns a default configuration to keep downstream dictionaries happy
    #             return {'bits': 8, 'integer': 0, 'negative_slope': 0.0}

    #     # Instantiate custom fake quantizer trackers for hls4ml routing paths
    #     dummy_linear = Hls4mlCompatibleQuantizer('quantized_bits')
    #     dummy_relu = Hls4mlCompatibleQuantizer('quantized_relu')

    #     for layer in self.quantizedModel.layers:
    #         if hasattr(layer, 'activation'):
    #             act = layer.activation
                
    #             # Extract basic text identity identifier
    #             if isinstance(act, dict):
    #                 name = act.get('class_name', str(act))
    #             else:
    #                 name = getattr(act, '__name__', str(act))

    #             if '(' in name:
    #                 name = name.split('(')[0]
    #             if 'function linear' in name or 'builtins' in name:
    #                 name = 'linear'

    #             # Force route the underlying configuration references to our robust dummy tracking objects
    #             if 'quantized_relu' in name:
    #                 layer.activation = dummy_relu
    #             elif name == 'linear' or 'quantized_bits' in name:
    #                 layer.activation = dummy_linear
    #             else:
    #                 # Standard non-quantized activations can safely default to strings
    #                 layer.activation = name

    # #Working version for ASIC Model, up to doing the hls tracing
    # def fixLayerActivation(self):
    #     from qkeras.quantizers import quantized_bits, quantized_relu
    #     # 1. Create true instantiated QKeras Quantizer objects
    #     # These objects possess the 'get_config' attribute hls4ml checks for!
    #     qk_bits_8_0 = quantized_bits(bits=8, integer=0)
    #     qk_relu_8_0 = quantized_relu(bits=8, integer=0)

    #     for layer in self.quantizedModel.layers:
    #         if hasattr(layer, 'activation'):
    #             act = layer.activation
    #             layer_type = layer.__class__.__name__
                
    #             # Skip if it's already an active Keras Tuner structured layout dict
    #             if isinstance(act, dict) and 'class_name' in act and act['class_name'] != 'linear':
    #                 continue

    #             # Extract basic string name 
    #             if not isinstance(act, str):
    #                 name = getattr(act, '__name__', act.__class__.__name__)
    #             else:
    #                 name = act

    #             if '(' in name:
    #                 name = name.split('(')[0]
    #             if 'function linear' in name or 'builtins' in name:
    #                 name = 'linear'

    #             # 2. Route layers directly to fully-instantiated QKeras object classes
    #             if 'quantized_bits' in name:
    #                 layer.activation = qk_bits_8_0
                    
    #             elif 'quantized_relu' in name:
    #                 layer.activation = qk_relu_8_0
                    
    #             elif name == 'linear':
    #                 if layer_type == 'QActivation':
    #                     # Standalone QActivation layers acting as a pass-through 
    #                     # must use an instantiated quantized_bits instance to trigger 'linear'
    #                     layer.activation = qk_bits_8_0
    #                 else:
    #                     # Inline properties on QDense / QDenseBatchnorm can safely accept 
    #                     # standard native Keras functional pointers
    #                     layer.activation = tf.keras.activations.linear
    #             else:
    #                 layer.activation = name


    #Update to resolve tracing error with ASIC Model, this works for ASIC Model
    # def fixLayerActivation(self):
    #     from qkeras.quantizers import quantized_bits, quantized_relu
    #     import tensorflow as tf
        
    #     # 1. Create true instantiated qkeras quantizer objects
    #     qk_bits_8_0 = quantized_bits(bits=8, integer=0)
    #     qk_relu_8_0 = quantized_relu(bits=8, integer=0)
        
    #     # Inject dynamic __name__ strings directly to intercept the hls4ml profiler.
    #     # This prevents the AttributeError without altering native Keras object serialization.
    #     qk_bits_8_0.__name__ = "quantized_bits"
    #     qk_relu_8_0.__name__ = "quantized_relu"
        
    #     for layer in self.quantizedModel.layers:
    #         if hasattr(layer, 'activation'):
    #             act = layer.activation
    #             layer_type = layer.__class__.__name__
                
    #             # Skip if it's already an active keras tuner structured layout dict
    #             if isinstance(act, dict) and 'class_name' in act and act['class_name'] != 'linear':
    #                 continue
                    
    #             # Extract basic string name
    #             if not isinstance(act, str):
    #                 name = getattr(act, '__name__', act.__class__.__name__)
    #             else:
    #                 name = act
                    
    #             # RESTORED: Fixed the split array-index cutoff bug from the previous attempt
    #             if '(' in name:
    #                 name = name.split('(')[0]
                    
    #             if 'function linear' in name or 'builtins' in name or 'linear' in name:
    #                 name = 'linear'
                    
    #             # 2. Route layers directly to fully-instantiated qkeras object classes
    #             if 'quantized_bits' in name:
    #                 layer.activation = qk_bits_8_0

    #             elif 'quantized_relu' in name:
    #                 layer.activation = qk_relu_8_0

    #             elif name == 'linear':
    #                 if layer_type in ['QActivation', 'Activation']:
    #                     # Standalone pass-through layers get the patched quantizer object
    #                     layer.activation = qk_bits_8_0
    #                 else:
    #                     # Inline properties on QDense / QConv2D drop back safely to a Keras linear function
    #                     layer.activation = tf.keras.activations.linear
    #             else:
    #                 # Catch-all fallback for standard activations
    #                 layer.activation = name

    #Update to work for Fermilab model qmodel_file = "/local/d1/smartpixLab/fermiModels/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
    def fixLayerActivationASIC(self):
        from qkeras.quantizers import quantized_bits, quantized_relu
        from qkeras.qlayers import QActivation
        import tensorflow as tf
        
        # 1. Create true instantiated qkeras quantizer objects
        qk_bits_8_0 = quantized_bits(bits=8, integer=0)
        qk_relu_8_0 = quantized_relu(bits=8, integer=0)
        
        # Inject dynamic __name__ strings directly to satisfy the hls4ml profiling trace.
        qk_bits_8_0.__name__ = "quantized_bits"
        qk_relu_8_0.__name__ = "quantized_relu"
        
        for layer in self.quantizedModel.layers:
            if hasattr(layer, 'activation'):
                act = layer.activation
                layer_type = layer.__class__.__name__
                
                # Skip if it's already an active keras tuner structured layout dict
                if isinstance(act, dict) and 'class_name' in act and act['class_name'] != 'linear':
                    continue
                    
                # Extract basic string name
                if not isinstance(act, str):
                    name = getattr(act, '__name__', act.__class__.__name__)
                else:
                    name = act
                    
                if '(' in name:
                    name = name.split('(')
                    
                if 'function linear' in name or 'builtins' in name or 'linear' in name:
                    name = 'linear'
                    
                # TRANSFORM STEP: Convert vanilla Activation layers into native QActivation layers
                if layer_type == 'Activation':
                    layer.__class__ = QActivation
                    layer_type = 'QActivation'
                    
                # 2. Route layers directly to fully-instantiated qkeras object classes
                if 'quantized_bits' in name:
                    layer.activation = qk_bits_8_0
                    if layer_type == 'QActivation':
                        layer.quantizer = qk_bits_8_0  # Fixes execution loop initialization

                elif 'quantized_relu' in name:
                    layer.activation = qk_relu_8_0
                    if layer_type == 'QActivation':
                        layer.quantizer = qk_relu_8_0  # Fixes execution loop initialization

                elif name == 'linear':
                    if layer_type == 'QActivation':
                        layer.activation = qk_bits_8_0
                        layer.quantizer = qk_bits_8_0  # Fixes execution loop initialization
                    else:
                        # Inline properties on QDense / QDenseBatchnorm drop back safely to a Keras linear function
                        layer.activation = tf.keras.activations.linear
                else:
                    # Catch-all fallback for standard activations (relu, sigmoid, etc.)
                    layer.activation = name

    def fixLayerActivationRegularNOTTHIS(self):
        from qkeras.quantizers import quantized_bits, quantized_relu, quantized_sigmoid
        from qkeras.qlayers import QActivation
        import tensorflow as tf
        import types
        
        # 1. Create true instantiated qkeras quantizer objects
        qk_bits_8_0 = quantized_bits(bits=8, integer=0)
        qk_relu_8_0 = quantized_relu(bits=8, integer=0)
        qk_sigmoid_8_0 = quantized_sigmoid(bits=8, integer=0)
        
        # Inject dynamic __name__ strings directly to satisfy the hls4ml profiling trace.
        qk_bits_8_0.__name__ = "quantized_bits"
        qk_relu_8_0.__name__ = "quantized_relu"
        qk_sigmoid_8_0.__name__ = "quantized_sigmoid"
        
        for layer in self.quantizedModel.layers:
            if hasattr(layer, 'activation'):
                act = layer.activation
                layer_type = layer.__class__.__name__
                
                # Skip if it's already an active keras tuner structured layout dict
                if isinstance(act, dict) and 'class_name' in act and act['class_name'] != 'linear':
                    continue
                    
                # Extract basic string name
                if not isinstance(act, str):
                    if isinstance(act, (list, tuple)) or act.__class__.__name__ == 'ListWrapper':
                        name = str(act)
                    else:
                        name = getattr(act, '__name__', act.__class__.__name__)
                else:
                    name = act
                    
                if '(' in name:
                    name = name.split('(')[0]
                    
                if 'function linear' in name or 'builtins' in name or 'linear' in name:
                    name = 'linear'
                    
                # TRANSFORM STEP: Convert vanilla Activation layers into native QActivation layers
                if layer_type == 'Activation':
                    layer.__class__ = QActivation
                    layer_type = 'QActivation'
                    
                # 2. Route layers directly to fully-instantiated qkeras object classes
                target_quantizer = None
                if 'quantized_bits' in name:
                    target_quantizer = qk_bits_8_0
                elif 'quantized_relu' in name:
                    target_quantizer = qk_relu_8_0
                elif 'quantized_sigmoid' in name:
                    target_quantizer = qk_sigmoid_8_0
                elif name == 'linear' and layer_type == 'QActivation':
                    target_quantizer = qk_bits_8_0

                # 3. Apply the routing parameters and force serialization overrides
                if target_quantizer is not None:
                    layer.activation = target_quantizer
                    if layer_type == 'QActivation':
                        layer.quantizer = target_quantizer
                        
                        # PURE ADDITION: Override get_config dynamically to force clean 
                        # dictionary serialization and bypass Keras ListWrapper bugs.
                        def make_forced_get_config(l, q):
                            original_get_config = l.get_config
                            def forced_get_config():
                                config = original_get_config()
                                # Build the native dictionary layout hls4ml expects
                                config['activation'] = {
                                    'class_name': q.__class__.__name__,
                                    'config': q.get_config()
                                }
                                return config
                            return forced_get_config
                            
                        layer.get_config = make_forced_get_config(layer, target_quantizer)
                else:
                    if name == 'linear':
                        # Inline properties on QDense / QConv2D drop back safely to a Keras linear function
                        layer.activation = tf.keras.activations.linear
                    else:
                        # Catch-all fallback for standard activations (relu, sigmoid, etc.)
                        layer.activation = name



    def fixLayerActivation(self):
        if self.modelType in [0, "ASIC"]:
            self.fixLayerActivationASIC()
        else:
            self.fixLayerActivationRegular()


    def assignModel(self):
        if self.customModel:
            self.makeCustomModel()
            self.fitCustomModel()
            self.plotCustomModel()
            self.quantizedModel = self.model
        else:
            from Model_Classes import loadQuantizedModel
            # co = {}       
            # qkeras.utils._add_supported_quantized_objects(co)
            # self.quantizedModel = tf.keras.models.load_model(self.filepath,custom_objects=co,compile=True)
            self.quantizedModel = loadQuantizedModel(self.filepath,compile=True)
            self.fixLayerActivation()

    def printModelSummary(self):
        
        print("Summary of ")
        print(self.quantizedModel.summary())
        for i, layer in enumerate (self.quantizedModel.layers):
            print (i, layer)
            try:
                print ("    ",layer.activation)
            except AttributeError:
                print('   no activation attribute')
    
    def getTestVectors(self):
        if self.loadTestVectors:
            [self.yTest, self.xTestList] = pickle.load(open(f"./testVectors{self.modelType}.pkl",'rb'))
            self.xTest = pickle.load(open(f"./tfTestVectors{self.modelType}.pkl",'rb'))
            
            # [yTest, xTestList] = pickle.load(open(f"./testVectors.pkl",'rb'))
            # xTest = pickle.load(open(f"./tfTestVectors.pkl",'rb'))
        else:
            self.xTest, self.yTest, self.xTestList, self.xTrain, self.yTrain = flattenTfData(self.modelType,tfRecordFolder=self.tfRecordFolder)
            if self.saveTestVectors:
                pickle.dump([self.yTest, self.xTestList],open(f"./testVectors{self.modelType}.pkl","wb"))
                pickle.dump(self.xTest,open(f"./tfTestVectors{self.modelType}.pkl","wb"))
    
    def verifyTestVectors(self):
        if self.modelType in [2.5, "2.5"]:
            print(self.xTest['y_profile'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['x_profile'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['nModule'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['x_local'][(np.random.randint(0,len(self.xTest["y_local"])))])
        if self.modelType in [1,"1"]:
            print(self.xTest['y_size'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['x_size'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['z_global'][(np.random.randint(0,len(self.xTest["y_local"])))])
        if self.modelType in [0,"ASIC"]:
            print(self.xTest['x_size'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['y_profile'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['z_global'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['nModule'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['x_local'][(np.random.randint(0,len(self.xTest["y_local"])))])
            print(self.xTest['inVectAsic'][(np.random.randint(0,len(self.xTest["y_local"])))])

        print(self.xTest['y_local'][(np.random.randint(0,len(self.xTest["y_local"])))])

        print(self.xTest.keys())
        # print(len(xTest["x_profile"]))
        print(len(self.yTest))
        # print(len(yTrain))
        # print(len(yTrain)+len(yTest))
        print(1656656*2)
        print(1600000*2)
        print(1656656*2*(41/(41+164)))
        print(1656656*2*0.2)
        print()
        print(661482/20672)
        print(len(self.xTestList))
        print(len(self.xTestList[0]))
        if self.customModel:
            self.xTestList = self.xTestList[3]
        if self.modelType in [0,"ASIC"]:
            self.xTestList = self.xTestList[-1]   
        # if self.modelType in [1, "1"]:
        #     #Not sure why this hasn't been fixed already?? but need to get rid of nModule and xLocal
        #     if len(self.xTestList) == 6:
        #         print("getting rid of last 2 list elements since assuming they are nModule and xlocal")
        #         self.xTestList = self.xTestList[0:4]     
        print(len(self.xTestList))
        print(len(self.xTestList[0]))

    def __init__(self,
                doingCatapult: bool = True, #If using catapult, use the ccs_env python environment
                doingVitis: bool = False, #If using vitis, use the hls4ml "default" environment that works with Vitis      
                loadTestVectors: bool = True,
                saveTestVectors: bool = False,
                buildModel: bool = False,
                customModel: bool = False,
                fullRunOnInit: bool = True,
                modelType = 2.5, #so far using 1 and 2.5, but in future will use the specification in hlsUtils -> Update June 10, 2026, seems okay now? and follows tfLoaderUtils, there is no hlsUtils
                filepath = "",
                baseSaveDir = "./hlsVerification",
                interactivePlots: bool = True,
                PLOT_DIR = "",
                tfRecordFolder = "", #this is the default used by tfLoaderUtils, which if unchanged will pass to the default directories of tfLoaderUtils
                doTrace: bool = True,
                noDSP: bool = False,
                 ) -> None:
        
        self.doingCatapult = doingCatapult 
        self.doingVitis = doingVitis  
        self.loadTestVectors = loadTestVectors 
        self.saveTestVectors = saveTestVectors 
        self.buildModel = buildModel 
        self.customModel = customModel 
        self.modelType = modelType 
        self.fullRunOnInit = fullRunOnInit
        self.tfRecordFolder = tfRecordFolder
        self.doTrace = doTrace
        self.noDSP = noDSP

        #process input flags
        if self.doingCatapult:
            # sys.path.append("/code/Siemens_EDA/Catapult_Synthesis_2026.1-1267132/Mgc_home/shared/pkgs/ccs_hls4ml/hls4ml/")
            sys.path.append("/code/Siemens_EDA/Catapult_Synthesis_2026.2-1292347/Mgc_home/shared/pkgs/ccs_hls4ml/hls4ml/")
            self.catapult_ai_nn = import_module("catapult_ai_nn")
            if self.doTrace:
                print("SO FAR CATAPULT DOES NOT FULLY SUPPORT THE TRACE STUFF!!!!\n\n")
        self.hls4ml = import_module("hls4ml")
        

        #Verify inputs flags are appropriate
        assert doingCatapult == (not doingVitis)

        if saveTestVectors:
            assert not loadTestVectors

        customModel = False
        if customModel:
            assert not loadTestVectors

        if modelType not in ["1","2","2.5","3",1,2,3,2.5,"ASIC"]:
                raise TypeError("Not supported model type")
        
    
        #process inputs part 2
        if filepath == "":
            filepath="../../Muon_Collider_Smart_Pixels/eric/model2.5_quantized_4w0i_hyperparameter_results_20260222_004048/model_trial_000.h5"
            filepath="../../Muon_Collider_Smart_Pixels/eric/model2.5_quantized_8w0i_hyperparameter_results_20260228_020952/model_trial_0.h5"
            filepath="./model_trial_0.h5"
            if modelType==2:
                filepath="../../Muon_Collider_Smart_Pixels/eric/model2.5_qi_4w0i_pareto_roc_selected/model_trial_25.h5"
            # filepath = "../../smart-pixels-ml/DanielModels/model2_20260325.keras"
            #Now trying an Ryan model
            if modelType==1:
                filepath="/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/ryan/old_quantization_res/model1_quantized_4w0i_pareto/model_trial_034.h5"
        self.filepath = filepath

        pathlib.Path(baseSaveDir).mkdir(parents=True,exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir_catapult = os.path.join(baseSaveDir,"hlsCatapultModel2_"+timestamp)
        self.output_dir_vitis = os.path.join(baseSaveDir,"hlsVitisModel2_"+timestamp)
        self.output_dir = self.output_dir_catapult if doingCatapult else self.output_dir_vitis

        if PLOT_DIR == "":
            self.PLOT_DIR = os.path.join(baseSaveDir,"plots_"+timestamp)
        else:
            self.PLOT_DIR = PLOT_DIR
        self.interactivePlots = interactivePlots
        pathlib.Path(self.PLOT_DIR).mkdir(parents=True,exist_ok=True)


        self.model = None
        self.history = None
        if self.fullRunOnInit:
            self.runAllTheStuff()

    def runAllTheStuff(self):
        self.getTestVectors()
        self.verifyTestVectors()
        self.assignModel()
        self.printModelSummary()
        self.compileHLSModel()
        self.printInputHLSVars()
        self.predictFirstVector()
        print("starting to predict all vectors")
        self.predictAllVectors()
        if self.doTrace:
            print("starting to trace all vectors (this may take a bit)")
            self.traceAllVectors()
            print("Done tracing all vectors")
            self.printHLSTrace()
            self.plotHLSTrace()
            self.writeTrace()
        self.plotHLSVerification()
        if self.buildModel:
            self.finishBuilding()
    
    def compileHLSModel(self):
        if self.doingCatapult:
            config_ccs = self.catapult_ai_nn.config_for_dataflow(model=self.quantizedModel, x_test=None, y_test=None, num_samples=None, 
                                                default_precision='ac_fixed<16,6>', max_precision='ac_fixed<16,6>',
                                                default_reuse_factor=1,
                                                output_dir=self.output_dir_catapult,
                                                tech='asic',
                                                asiclibs='saed32rvt_tt0p78v125c_beh',                                     
                                                asicfifo='hls4ml_lib.mgc_pipe_mem',
                                                clock_period=10,
                                                io_type='io_stream',
                                                csim=0, SCVerify=0, Synth=1)
            # Specific architecture settings
            # Configure input feature precision since it is known
            # print(config_ccs['HLSConfig']['LayerName'].keys())
            # config_ccs['HLSConfig']['LayerName']['x_profile']['Precision'] = 'ac_fixed<8,1,true>' 
            # config_ccs['HLSConfig']['LayerName']['y_profile']['Precision'] = 'ac_fixed<8,1,true>' 
            # config_ccs['HLSConfig']['LayerName']['x_local']['Precision'] = 'ac_fixed<8,1,true>' 
            # config_ccs['HLSConfig']['LayerName']['y_local']['Precision'] = 'ac_fixed<8,1,true>' 
            # config_ccs['HLSConfig']['LayerName']['nModule']['Precision'] = 'ac_fixed<8,1,true>' 
            # config_ccs['HLSConfig']['LayerName']['xy_concat']['Precision'] = 'ac_fixed<8,1,true>' 
            # config_ccs['HLSConfig']['LayerName']['other_features']['Precision'] = 'ac_fixed<8,1,true>' 
            # config_ccs['HLSConfig']['LayerName']['nmodule_xlocal_concat']['Precision'] = 'ac_fixed<8,1,true>' 
            # config_ccs['HLSConfig']['LayerName']['blabla']['Precision'] = 'ac_fixed<8,1,true>' 
            # config_ccs['HLSConfig']['LayerName']['input1']['Precision'] = 'ac_fixed<8,1,true>' ##TODO
            # Performance strategy is set to latency mode
            config_ccs['HLSConfig']['Model']['Strategy'] = 'Latency'
            hls_model_ccs = self.catapult_ai_nn.generate_dataflow(model=self.quantizedModel,config_ccs=config_ccs)
            hls_model_ccs.compile()
            self.hls_model_ccs = hls_model_ccs
        if self.doingVitis:
            #Actually need to import the other hls4ml for this
            config = self.hls4ml.utils.config_from_keras_model(self.quantizedModel, granularity='name',default_precision = "fixed<16,7>",)
            # config = self.hls4ml.utils.config_from_keras_model(quantizedModel, granularity='name',default_precision="ap_fixed<16,6,true>")

            for layer in config['LayerName'].keys():
                print('Enable tracing for layer:', layer)
                config['LayerName'][layer]['Trace'] = True

            #added to get more latency tracing
            # Add this line to force Co-simulation tracking
            config['TraceLevel'] = 'port' # Options: 'all', 'port', or 'none'

            # Convert to an hls model

            hls_model = self.hls4ml.converters.convert_from_keras_model(self.quantizedModel, hls_config=config, part = 'xc7z020clg400-1', output_dir=self.output_dir_vitis,backend="Vitis")
            # hls_model = self.hls4ml.converters.convert_from_keras_model(quantizedModel, hls_config=config, part = 'xc7z020clg400-1', output_dir=output_dir,backend="Catapult")
            ##Part number input in above line as part='xcu
            # part='xcu250-figd2104-2L-e',
            hls_model.compile()
            if self.noDSP:
                print("TURNING OFF DSPS")
                # Force the Vitis HLS 2024.1 backend to map all operations to standard fabric logic
                tcl_path = os.path.join(self.output_dir_vitis, "build_prj.tcl")
                if os.path.exists(tcl_path):
                    with open(tcl_path, "r") as f:
                        tcl_content = f.read()
                    
                    # 1. Clean up the deprecated array partition warning/error line
                    tcl_content = tcl_content.replace(
                        "config_array_partition -maximum_size 4096", 
                        "# config_array_partition -maximum_size 4096"
                    )
                    
                    # 2. Inject valid Vitis HLS 2024.1 global operator fabric overrides
                    if "csynth_design" in tcl_content:
                        patch_directives = (
                            "config_op mul -impl fabric\n"  # Maps all inferred multiplications to LUTs
                            "config_op add -impl fabric\n"  # Maps all inferred additions to LUTs
                            "csynth_design"
                        )
                        tcl_content = tcl_content.replace("csynth_design", patch_directives)
                        
                        with open(tcl_path, "w") as f:
                            f.write(tcl_content)
                        print(f"[{self.output_dir_vitis}] Patched build_prj.tcl to force Vitis 2024.1 operator fabric mapping.")
                    # if "csynth_design" in tcl_content:
                    #     patch_directives = (
                    #         # 1. Globally configure the RTL generation tool to enforce 0 DSP architectures
                    #         "config_rtl -max_dsp 0\n"
                            
                    #         # 2. Prevent memory array and multiplier multi-pumping optimizations
                    #         "config_bind -force_array_dsp off\n"
                            
                    #         # 3. Explicitly tell the global operator framework to bind math functions to LUT fabric
                    #         "config_op mul -impl fabric\n"
                    #         "config_op mac -impl fabric\n"
                            
                    #         # 4. Enforce strict budget limits on standard structural sub-cores
                    #         "alloc_use_core mul_dsp 0\n"
                    #         "alloc_use_core mac_dsp 0\n"
                            
                    #         "csynth_design"
                    #     )
                    #     tcl_content = tcl_content.replace("csynth_design", patch_directives)
                    #     tcl_content = tcl_content.replace("csynth_design", patch_directives)
                        
                    #     with open(tcl_path, "w") as f:
                    #         f.write(tcl_content)
                    #     print(f"[{self.output_dir_vitis}] Applied structural core block to wipe out Instance DSPs.")



            self.hls_model = hls_model
    def printInputHLSVars(self):
        if self.doingCatapult:
    
            for input_var in self.hls_model_ccs.get_input_variables():
                print(input_var.name)
                # hls_model_ccs.config.config['InputShapes'][input_var.name] = list(input_var.shape)
                print(self.hls_model_ccs.config.config['InputShapes'][input_var.name])

            # hls_model_ccs._get_top_function()
            self.hls_model_ccs.config.backend.compile
        else:
            print('not implemented for vitis yet')
    def predictFirstVector(self):
        print([(self.xTestList[i].shape) for i in range(len(self.xTestList))])
        print(self.xTest.keys())
        # print(hls_model_ccs.get_input_variables())
        firstInput = [self.xTestList[i][1:2] for i in range(len(self.xTestList))]
        if self.customModel or (self.modelType in [0,"ASIC"]):
            firstInput = self.xTestList[1:2]
        print(firstInput)
        # firstInputFlat = np.concatenate([firstInput[0][0],firstInput[1],firstInput[2],firstInput[3][0],firstInput[4]])
        # print(*firstInputFlat)
        if self.doingCatapult:
            ccs_hls_model_predictions = [self.hls_model_ccs.predict(firstInput) for i in range(6)]
        if self.doingVitis:
            hls_model_predictions = [self.hls_model.predict(firstInput) for i in range(6)]
        # q_predictions = quantizedModel.predict([xTest[key][1:2] for key in xTest.keys()])
        q_predictions = [self.quantizedModel.predict(firstInput) for i in range(6)]

        # print(self.yTest)
        if self.doingCatapult:
            print(ccs_hls_model_predictions)
        else:
            print(hls_model_predictions)
        print(q_predictions)
    def predictAllVectors(self,smallInput=False):
        if smallInput:
            self.multiInput = [self.xTestList[i][1:20] for i in range(len(self.xTestList))]
        else:
            self.multiInput = self.xTestList
        if self.doingVitis:
            self.hls_model_predictions = self.hls_model.predict(self.multiInput)
        else:    
            self.hls_model_predictions = self.hls_model_ccs.predict(self.multiInput)
        self.q_predictions = self.quantizedModel.predict(self.multiInput)
        if self.q_predictions.shape[1]>1: #In case of ASIC model
            print("applying argmax")
            assert self.modelType in [0,"ASIC"]
            self.q_predictions = np.argmax(self.q_predictions,axis=1)
            self.hls_model_predictions = np.argmax(self.hls_model_predictions,axis=1)
            #reshape to be consistent with other cases
            self.hls_model_predictions = self.hls_model_predictions.reshape(-1, 1)
            self.q_predictions = self.q_predictions.reshape(-1, 1)        
        print(self.hls_model_predictions.shape,self.q_predictions.shape,self.yTest.shape)

    def traceAllVectors(self):
        if self.doingVitis:
            self.hls4ml_pred, self.hls4ml_trace = self.hls_model.trace(self.multiInput)
            self.keras_trace = self.hls4ml.model.profiling.get_ymodel_keras(self.quantizedModel, self.multiInput)
        else:
            self.hls4ml_pred, self.hls4ml_trace = self.hls_model_ccs.trace(self.multiInput)

    def printHLSTrace(self,testVectorIndex = 370,N_ELEMENTS=5):
        if True:#doingVitis:
            # Print the traces on console
            #(array([     3,     18,    217, ..., 661450, 661452, 661477]),)

            # Backup print options
            bkp_threshold = np.get_printoptions()['threshold']
            bkp_linewidth = np.get_printoptions()['linewidth']

            # Set print options
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            if self.modelType in [2.5, "2.5"]:
                print('input xprofile', self.xTestList[0][testVectorIndex])
                print('input nModule', self.xTestList[1][testVectorIndex])
                print('input x_local', self.xTestList[2][testVectorIndex])
                print('input y_profile', self.xTestList[3][testVectorIndex])
                print('input y_local', self.xTestList[4][testVectorIndex])
            if self.modelType in [1,"1"]:
                print('input z_global', self.xTestList[0][testVectorIndex])
                print('input y_local', self.xTestList[3][testVectorIndex])
                print('input xsize', self.xTestList[1][testVectorIndex])
                print('input y_size', self.xTestList[2][testVectorIndex])
            for key in self.hls4ml_trace.keys():
                print('-------')
                print(key, self.hls4ml_trace[key].shape)
                print('[hls4ml]', key, self.hls4ml_trace[key][testVectorIndex].flatten()[:N_ELEMENTS])
                if self.doingVitis:
                    print('[keras] ', key, self.keras_trace[key][testVectorIndex].flatten()[:N_ELEMENTS])
                
                print('[hls4ml]', key, self.hls4ml_trace[key][testVectorIndex].flatten())
                if self.doingVitis:
                    print('[keras] ', key, self.keras_trace[key][testVectorIndex].flatten())
                print('[hls4ml]', key, self.hls4ml_trace[key][testVectorIndex])
                if self.doingVitis:
                    print('[keras] ', key, self.keras_trace[key][testVectorIndex])
                    print('|hls4ml - qkeras| > 0.001 indices', key, np.where(np.abs(self.hls4ml_trace[key][testVectorIndex] - self.keras_trace[key][testVectorIndex])>0.001))
                    print('|1-hls4ml / qkeras| < 0.001 indices', key, np.where(np.abs(1-(self.hls4ml_trace[key][testVectorIndex] / self.keras_trace[key][testVectorIndex]))>0.001))
                
                    # np.set_printoptions(threshold=bkp_threshold, linewidth=bkp_linewidth)
                    print("Are there any values with integer bits (-2 to 2 excluded)?",np.any(np.bitwise_or(self.keras_trace[key][:]>2 ,self.keras_trace[key][:]<-2)) )
                    print("list of indices with integer bits?",np.where(np.bitwise_or(self.keras_trace[key][:]>2 ,self.keras_trace[key][:]<-2))[0][0:20] )
                    print("list of values with integer bits?",self.keras_trace[key][np.where(np.bitwise_or(self.keras_trace[key][:]>2 ,self.keras_trace[key][:]<-2))[0][0:20]] )
                # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            # Restore print options
            np.set_printoptions(threshold=bkp_threshold, linewidth=bkp_linewidth)
        # print(xTestList[0][661452])

    def plotHLSTrace(self): #could make into a function that takes in hls4ml_trace and keras_trace
        numLayers = len(self.hls4ml_trace.keys())
        plt.figure(figsize=(5,numLayers*2.5))
        for idx, layer in enumerate(self.hls4ml_trace.keys()):
            print(layer)
            if '_alpha' in layer:
                continue
            # plt.figure()
            plt.subplot(numLayers,1,idx+1)
            klayer = layer
            if '_linear' in layer:
                klayer = layer.replace('_linear', '')
            plt.scatter(self.hls4ml_trace[layer].flatten(), self.keras_trace[klayer].flatten(), s=0.2)
            min_x = min(np.amin(self.hls4ml_trace[layer]), np.amin(self.keras_trace[klayer]))
            max_x = max(np.amax(self.hls4ml_trace[layer]), np.amax(self.keras_trace[klayer]))
            plt.plot([min_x, max_x], [min_x, max_x], c='gray')
            plt.xlabel('hls4ml {}'.format(layer))
            plt.ylabel('QKeras {}'.format(klayer))
        self.closePlt("All layer traces")

    def plotHLSVerification(self):
        import matplotlib.colors as colors
        print(self.hls_model_predictions.shape,self.q_predictions.shape,self.yTest.shape)
        predictionsTogether = np.array([self.hls_model_predictions,self.q_predictions,self.yTest]).transpose()
        print(predictionsTogether[0][370])
        # plt.scatter(self.hls_model_predictions,self.q_predictions,c = self.yTest,marker=",")
        # plt.plot(self.hls_model_predictions,self.q_predictions,',')
        # plt.hist2d((self.hls_model_predictions[:]),(self.q_predictions[:]))
        plt.figure(figsize=(5,15))
        plt.subplot(511)
        plt.hist2d(self.hls_model_predictions[:,0],self.q_predictions[:,0],norm=colors.LogNorm(),bins=[100,100])
        plt.xlabel("hls prediction")
        plt.ylabel("qkeras prediction")
        plt.colorbar()
        plt.title(self.filepath)
        # plt.show()
        plt.subplot(512)
        plt.hist2d(self.hls_model_predictions[:,0],self.yTest[:,0],norm=colors.LogNorm(),bins=[100,4])
        plt.xlabel("hls prediction")
        plt.ylabel("Truth label")
        plt.colorbar()
        # plt.show()
        plt.subplot(513)
        plt.hist2d(self.q_predictions[:,0],self.yTest[:,0],norm=colors.LogNorm(),bins=[100,4])
        plt.xlabel("qkeras prediction")
        plt.ylabel("Truth label")
        plt.colorbar()
        plt.tight_layout()
        plt.subplot(514)
        plt.title("matrix for true bib")
        plt.hist2d(self.hls_model_predictions[self.yTest==0],self.q_predictions[self.yTest==0],norm=colors.LogNorm(),bins=[100,100])
        plt.xlabel("hls prediction")
        plt.ylabel("qkeras prediction")
        plt.colorbar()
        plt.subplot(515)
        plt.title("matrix for true sig")
        plt.hist2d(self.hls_model_predictions[self.yTest==1],self.q_predictions[self.yTest==1],norm=colors.LogNorm(),bins=[100,100])
        plt.xlabel("hls prediction")
        plt.ylabel("qkeras prediction")
        plt.colorbar()
        self.closePlt("output verification")
        plt.hist(self.hls_model_predictions[:,0]-self.q_predictions[:,0],bins='auto')
        plt.xlabel("hls prediction - qkeras prediction")
        plt.title("Distribution of difference between hls prediction and qkeras prediction")
        plt.ylabel("counts")
        plt.yscale('log')
        print(np.where(self.hls_model_predictions[:,0]-self.q_predictions[:,0]>0.2))
        self.closePlt("verification differences")
        # hls_model_predictions[self.yTest==0]
        # pd.DataFrame(predictionsTogether)



    def finishBuilding(self):
        if not self.buildModel:
            raise ValueError("Won't build model if buildModel is false")
            return
        if self.doingCatapult: 
            self.hls_model_ccs.build()
        if self.doingVitis: 
            qq = self.hls_model.build(csim=False, synth=True, cosim=True, validation=False, export=True, vsynth=True, reset=True )
        if self.doingCatapult:
            summary_file = self.output_dir_catapult + '/firmware/layer_summary.txt'
            print('Layer summary report: '+summary_file)
            with open(summary_file,'r') as f:
                print(f.read())
        if self.doingCatapult:
            import glob
            #Show the contents of the last report generated
            print(self.output_dir_catapult + '/Catapult*/myproject.v1/nnet_layer_results.txt')
            rpt_files = glob.glob(self.output_dir_catapult + '/Catapult*/myproject.v1/nnet_layer_results.txt')
            print('Latest report: '+rpt_files[-1])
            with open(rpt_files[-1],'r') as f:
                print(f.read())

    def closePlot(PLOT_DIR, interactivePlots, plotName,printOutputDir=True,transparent = False):
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, plotName), dpi=300, bbox_inches='tight',transparent=transparent)
        if interactivePlots:
            plt.show()
        else:
            plt.close()
        if printOutputDir:
            print(f"Plot saved as: {os.path.join(PLOT_DIR, plotName)}")
    def closePlt(self, plotName):
        hlsVerifier.closePlot(self.PLOT_DIR, self.interactivePlots, plotName)

    
    def writeTrace(self):
        old_target = sys.stdout
        logfilepath = os.path.join(self.PLOT_DIR, "traceLog.txt")
        # with open('noslice_hls4ml_vivado_trace_prj/tb_data/vivado_outputs.dat', 'w') as f:
        with open(logfilepath, 'w') as f:
            sys.stdout = f
            print(self.filepath)
            self.printHLSTrace()
            # f.write(' '.join(map(str, keras_trace['q_dense_2'][0].flatten())))
        sys.stdout = old_target