"""
Model3 Implementation using the SmartPixModel Abstract Base Class

This module implements Model3 as a concrete class inheriting from SmartPixModel.
Model3 is a CNN-based architecture that processes cluster data through Conv2D/MaxPooling,
then concatenates with a dense layer processing z_global and y_local features.

Architecture:
- Conv2D branch: cluster -> Conv2D (3x5 kernel, 32 filters) -> MaxPooling -> Flatten
- Scalar branch: z_global + y_local -> Concatenate -> Dense (32 units)
- Merge: Concatenate conv output with scalar dense output
- Head: Dense (200 units) -> Dense (100 units) -> Output (sigmoid)

Author: Eric
Date: 2024
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import pandas as pd

# Add the parent directory to path to import the base class and data generator
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/ryan/')

from Model_Classes import SmartPixModel
import OptimizedDataGenerator4 as ODG

# QKeras imports for quantized models
try:
    from qkeras import QDense, QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False


class Model3(SmartPixModel):
    """
    Model3: CNN-based architecture for smart pixel detector classification.
    
    This model processes cluster data (13x21 spatial, last timestamp) through Conv2D layers,
    then concatenates with z_global and y_local coordinates for binary classification.
    
    Features:
    - cluster: 13x21x20 (uses last timestamp only)
    - z_global: 1-dimensional global z coordinate
    - y_local: 1-dimensional local y coordinate
    """
    
    def __init__(self,
                 tfRecordFolder: str = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
                 nBits: list = None,
                 loadModel: bool = False,
                 modelPath: str = None,
                 conv_filters: int = 32,
                 kernel_rows: int = 3,
                 kernel_cols: int = 5,
                 scalar_dense_units: int = 32,
                 merged_dense_1: int = 200,
                 merged_dense_2: int = 100,
                 dropout_rate: float = 0.0,
                 initial_lr: float = 0.000871145,
                 end_lr: float = 5.3e-05,
                 power: int = 2,
                 bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
                 ):
        """
        Initialize Model3.
        
        Args:
            tfRecordFolder: Path to TFRecords directory
            nBits: List of bit configurations for quantization
            loadModel: Whether to load a pre-trained model
            modelPath: Path to saved model (if loadModel=True)
            conv_filters: Number of filters in Conv2D layer
            kernel_rows: Kernel height for Conv2D
            kernel_cols: Kernel width for Conv2D
            scalar_dense_units: Units in dense layer after concatenating z_global and y_local
            merged_dense_1: Units in first dense layer after merging conv and scalar branches
            merged_dense_2: Units in second dense layer before output
            dropout_rate: Dropout rate for regularization
            initial_lr: Initial learning rate
            end_lr: End learning rate for polynomial decay
            power: Power for polynomial decay
            bit_configs: List of (weight_bits, int_bits) tuples for quantization
        """
        super().__init__(tfRecordFolder, nBits, loadModel, modelPath)
        
        self.modelName = "Model3"
        
        # Architecture parameters
        self.conv_filters = conv_filters
        self.kernel_rows = kernel_rows
        self.kernel_cols = kernel_cols
        self.scalar_dense_units = scalar_dense_units
        self.merged_dense_1 = merged_dense_1
        self.merged_dense_2 = merged_dense_2
        self.dropout_rate = dropout_rate
        
        # Learning rate parameters
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.power = power
        
        # Model3 specific feature configuration
        self.x_feature_description = ['cluster', 'y_local', 'z_global']
        self.time_stamps = [19]  # Use only the last timestamp
        
        # Initialize data generators
        self.training_generator = None
        self.validation_generator = None
        
        # Results storage
        self.histories = {}
        
        # Initialize models dictionary
        self.models = {"Unquantized": None}
        self.bit_configs = bit_configs 
        for weight_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            self.models[config_name] = None
        
        # Load model if requested
        config_name = "Unquantized"
        if loadModel and modelPath:
            self.loadModel(modelPath, config_name)
    
    def loadTfRecords(self):
        """Load TFRecords using OptimizedDataGenerator4 for Model3 features."""
        trainDir = f"{self.tfRecordFolder}/tfrecords_train/"
        valDir = f"{self.tfRecordFolder}/tfrecords_validation/"
        
        print(f"Loading training data from: {trainDir}")
        print(f"Loading validation data from: {valDir}")
        
        # Determine batch size from directory name to match TFRecord format
        batch_size = 16384
        if "filtering_records16384" in self.tfRecordFolder:
            batch_size = 16384
        elif "filtering_records1024" in self.tfRecordFolder:
            batch_size = 1024
        
        print(f"Using batch_size={batch_size} to match TFRecord format")
        
        # Model3 uses cluster, y_local, z_global features with last timestamp only
        self.training_generator = ODG.OptimizedDataGenerator(
            load_records=True, 
            tf_records_dir=trainDir, 
            x_feature_description=self.x_feature_description,
            time_stamps=self.time_stamps,
            batch_size=batch_size
        )
        
        self.validation_generator = ODG.OptimizedDataGenerator(
            load_records=True, 
            tf_records_dir=valDir, 
            x_feature_description=self.x_feature_description,
            time_stamps=self.time_stamps,
            batch_size=batch_size
        )
        
        print(f"Training generator length: {len(self.training_generator)}")
        print(f"Validation generator length: {len(self.validation_generator)}")
        
        return self.training_generator, self.validation_generator
    
    def makeUnquantizedModel(self):
        """
        Build the unquantized Model3 architecture.
        
        Architecture:
        - Conv2D branch: cluster (13x21, last timestamp) -> Conv2D -> MaxPooling -> Flatten
        - Scalar branch: z_global + y_local -> Concatenate -> Dense
        - Merge: Concatenate conv output with scalar dense output
        - Head: Dense -> Dense -> Output
        """
        print("Building unquantized Model3...")
        print(f"  - Conv2D: {self.kernel_rows}x{self.kernel_cols} kernel, {self.conv_filters} filters")
        print(f"  - Scalar dense units: {self.scalar_dense_units}")
        print(f"  - Merged dense layers: {self.merged_dense_1} -> {self.merged_dense_2}")
        print(f"  - Dropout: {self.dropout_rate}")
        
        # Input layers
        # cluster is already (13, 21) after taking last timestamp in data generator
        cluster_input = Input(shape=(13, 21), name="cluster")
        z_global_input = Input(shape=(1,), name="z_global")
        y_local_input = Input(shape=(1,), name="y_local")
        
        # Conv2D branch
        # Add channel dimension for Conv2D: (13, 21) -> (13, 21, 1)
        conv_x = Reshape((13, 21, 1), name="add_channel")(cluster_input)
        conv_x = Conv2D(
            filters=self.conv_filters,
            kernel_size=(self.kernel_rows, self.kernel_cols),
            padding="same",
            activation="relu",
            name=f"conv2d_{self.kernel_rows}x{self.kernel_cols}"
        )(conv_x)
        conv_x = MaxPooling2D((2, 2), name="pool2d_1")(conv_x)
        conv_x = Flatten(name="flatten_vol")(conv_x)
        
        # Scalar branch: concatenate z_global and y_local, then dense
        scalar_concat = Concatenate(name="concat_scalars")([z_global_input, y_local_input])
        scalar_x = Dense(self.scalar_dense_units, activation="relu", name="dense_scalars")(scalar_concat)
        
        # Merge conv and scalar branches
        merged = Concatenate(name="concat_all")([conv_x, scalar_x])
        
        # Head layers
        h = Dense(self.merged_dense_1, activation="relu", name="merged_dense1")(merged)
        h = Dropout(self.dropout_rate, name="dropout_1")(h)
        h = Dense(self.merged_dense_2, activation="relu", name="merged_dense2")(h)
        
        # Output layer
        output = Dense(1, activation="sigmoid", name="output")(h)
        
        # Create and compile model
        self.models["Unquantized"] = Model(
            inputs=[cluster_input, z_global_input, y_local_input], 
            outputs=output, 
            name="model3_unquantized"
        )
        
        print("✓ Unquantized Model3 built successfully")
    
    def makeUnquatizedModelHyperParameterTuning(self, hp):
        """
        Build Model3 for hyperparameter tuning using Keras Tuner.
        
        Args:
            hp: Keras Tuner hyperparameter object
        """
        # Input layers
        cluster_input = Input(shape=(13, 21), name="cluster")
        z_global_input = Input(shape=(1,), name="z_global")
        y_local_input = Input(shape=(1,), name="y_local")
        
        # Hyperparameter search space
        conv_filters = hp.Int('conv_filters', min_value=16, max_value=64, step=16)
        kernel_rows = hp.Choice('kernel_rows', values=[3, 3])
        kernel_cols = hp.Choice('kernel_cols', values=[3, 3])
        scalar_dense_units = hp.Int('scalar_dense_units', min_value=16, max_value=64, step=16)
        merged_dense_1 = hp.Int('merged_dense_1', min_value=25, max_value=200, step=25)
        merged_dense_2 = hp.Int('merged_dense_2', min_value=15, max_value=150, step=15)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.2, step=0.1)
        
        # Conv2D branch
        conv_x = Reshape((13, 21, 1), name="add_channel")(cluster_input)
        conv_x = Conv2D(
            filters=conv_filters,
            kernel_size=(kernel_rows, kernel_cols),
            padding="same",
            activation="relu",
            name="conv2d"
        )(conv_x)
        conv_x = MaxPooling2D((2, 2), name="pool2d_1")(conv_x)
        conv_x = Flatten(name="flatten_vol")(conv_x)
        
        # Scalar branch: concatenate z_global and y_local, then dense
        scalar_concat = Concatenate(name="concat_scalars")([z_global_input, y_local_input])
        scalar_x = Dense(scalar_dense_units, activation="relu", name="dense_scalars")(scalar_concat)
        
        # Merge conv and scalar branches
        merged = Concatenate(name="concat_all")([conv_x, scalar_x])
        
        # Head layers
        h = Dense(merged_dense_1, activation="relu", name="merged_dense1")(merged)
        h = Dropout(dropout_rate, name="dropout_1")(h)
        h = Dense(merged_dense_2, activation="relu", name="merged_dense2")(h)
        
        # Output layer
        output = Dense(1, activation="sigmoid", name="output")(h)
        
        # Create model
        model = Model(
            inputs=[cluster_input, z_global_input, y_local_input], 
            outputs=output, 
            name="model3_hyperparameter_tuning"
        )
        
        # Compile with hyperparameter-tuned learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"]
        )
        
        return model
    
    def makeQuantizedModel(self):
        """
        Build quantized Model3 using QKeras.
        
        Args:
            bit_configs: List of (weight_bits, int_bits) tuples for quantization
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")
        
        # Default configurations
        for weight_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
        
        
            print(f"Building {config_name} model...")
            
            # Quantizers (use 8-bit activations regardless of weight bits)
            weight_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
            bias_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
            activation_quantizer = quantized_relu(8, 0)  # Always use 8-bit activations
            
            # Input layers
            cluster_input = Input(shape=(13, 21), name="cluster")
            z_global_input = Input(shape=(1,), name="z_global")
            y_local_input = Input(shape=(1,), name="y_local")
            
            # Conv2D branch with quantization
            conv_x = Reshape((13, 21, 1), name="add_channel")(cluster_input)
            conv_x = QConv2D(
                filters=self.conv_filters,
                kernel_size=(self.kernel_rows, self.kernel_cols),
                padding="same",
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name=f"conv2d_{self.kernel_rows}x{self.kernel_cols}"
            )(conv_x)
            conv_x = QActivation(activation_quantizer, name="conv2d_act")(conv_x)
            conv_x = MaxPooling2D((2, 2), name="pool2d_1")(conv_x)
            conv_x = Flatten(name="flatten_vol")(conv_x)
            
            # Scalar branch: concatenate z_global and y_local, then dense with quantization
            scalar_concat = Concatenate(name="concat_scalars")([z_global_input, y_local_input])
            scalar_x = QDense(
                self.scalar_dense_units, 
                kernel_quantizer=weight_quantizer, 
                bias_quantizer=bias_quantizer, 
                name="dense_scalars"
            )(scalar_concat)
            scalar_x = QActivation(activation_quantizer, name="dense_scalars_act")(scalar_x)
            
            # Merge conv and scalar branches
            merged = Concatenate(name="concat_all")([conv_x, scalar_x])
            
            # Head layers with quantization
            h = QDense(
                self.merged_dense_1, 
                kernel_quantizer=weight_quantizer, 
                bias_quantizer=bias_quantizer, 
                name="merged_dense1"
            )(merged)
            h = QActivation(activation_quantizer, name="merged_dense1_act")(h)
            h = Dropout(self.dropout_rate, name="dropout_1")(h)
            
            h = QDense(
                self.merged_dense_2, 
                kernel_quantizer=weight_quantizer, 
                bias_quantizer=bias_quantizer, 
                name="merged_dense2"
            )(h)
            h = QActivation(activation_quantizer, name="merged_dense2_act")(h)
            
            # Output layer with quantized_tanh
            output_dense = QDense(
                1, 
                kernel_quantizer=weight_quantizer, 
                bias_quantizer=bias_quantizer, 
                name="output_dense"
            )(h)
            output = QActivation("quantized_tanh", name="output")(output_dense)
            
            # Create model
            model = Model(
                inputs=[cluster_input, z_global_input, y_local_input], 
                outputs=output, 
                name=f"model3_{config_name}"
            )
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
                metrics=["binary_accuracy"]
            )
            
            self.models[config_name] = model
        
        # # Store quantized models
        # self.models["Quantized"] = quantized_models
        # self.quantized_model = quantized_models  # For backward compatibility
        
        print(f"✓ Built {len(self.bit_configs)} quantized Model3 variants")
        # return quantized_models
    
    def makeQuantizedModelHyperParameterTuning(self, hp, weight_bits, int_bits):
        """
        Build quantized Model3 for hyperparameter tuning using Keras Tuner and QKeras.
        
        Args:
            hp: Keras Tuner hyperparameter object
            weight_bits: Number of bits for weights
            int_bits: Number of integer bits
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")
        
        # Quantizers
        weight_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
        bias_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
        activation_quantizer = quantized_relu(8, 0)  # Always use 8-bit activations
        
        # Input layers
        cluster_input = Input(shape=(13, 21), name="cluster")
        z_global_input = Input(shape=(1,), name="z_global")
        y_local_input = Input(shape=(1,), name="y_local")
        
        # Hyperparameter search space
        conv_filters = hp.Int('conv_filters', min_value=16, max_value=48, step=8)
        kernel_rows = hp.Choice('kernel_rows', values=[3, 3])
        kernel_cols = hp.Choice('kernel_cols', values=[3, 3])
        scalar_dense_units = hp.Int('scalar_dense_units', min_value=16, max_value=64, step=16)
        merged_dense_1 = hp.Int('merged_dense_1', min_value=20, max_value=150, step=15)
        
        # Multiplier for second layer (0.4 to 1.0 of previous layer)
        merged_multiplier_2 = hp.Float('merged_multiplier_2', min_value=0.4, max_value=1.0, step=0.2)
        
        # Calculate layer size with rounding
        merged_dense_2 = int(round(merged_dense_1 * merged_multiplier_2))
        
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.1, step=0.1)
        
        # Conv2D branch with quantization
        conv_x = Reshape((13, 21, 1), name="add_channel")(cluster_input)
        conv_x = QConv2D(
            filters=conv_filters,
            kernel_size=(kernel_rows, kernel_cols),
            padding="same",
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="conv2d"
        )(conv_x)
        conv_x = QActivation(activation_quantizer, name="conv2d_act")(conv_x)
        conv_x = MaxPooling2D((2, 2), name="pool2d_1")(conv_x)
        conv_x = Flatten(name="flatten_vol")(conv_x)
        
        # Scalar branch: concatenate z_global and y_local, then dense with quantization
        scalar_concat = Concatenate(name="concat_scalars")([z_global_input, y_local_input])
        scalar_x = QDense(
            scalar_dense_units, 
            kernel_quantizer=weight_quantizer, 
            bias_quantizer=bias_quantizer, 
            name="dense_scalars"
        )(scalar_concat)
        scalar_x = QActivation(activation_quantizer, name="dense_scalars_act")(scalar_x)
        
        # Merge conv and scalar branches
        merged = Concatenate(name="concat_all")([conv_x, scalar_x])
        
        # Head layers with quantization
        h = QDense(
            merged_dense_1, 
            kernel_quantizer=weight_quantizer, 
            bias_quantizer=bias_quantizer, 
            name="merged_dense1"
        )(merged)
        h = QActivation(activation_quantizer, name="merged_dense1_act")(h)
        h = Dropout(dropout_rate, name="dropout_1")(h)
        
        h = QDense(
            merged_dense_2, 
            kernel_quantizer=weight_quantizer, 
            bias_quantizer=bias_quantizer, 
            name="merged_dense2"
        )(h)
        h = QActivation(activation_quantizer, name="merged_dense2_act")(h)
        
        # Output layer with quantized_tanh
        output_dense = QDense(
            1, 
            kernel_quantizer=weight_quantizer, 
            bias_quantizer=bias_quantizer, 
            name="output_dense"
        )(h)
        output = QActivation("quantized_tanh", name="output")(output_dense)
        
        # Create model
        model = Model(
            inputs=[cluster_input, z_global_input, y_local_input], 
            outputs=output, 
            name=f"model3_quantized_{weight_bits}w{int_bits}i_hyperparameter_tuning"
        )
        
        # Compile with hyperparameter-tuned learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"]
        )
        
        return model
    
    def _calculate_model_parameters(self, hyperparams):
        """
        Calculate model parameters from hyperparameters without loading the model.
        
        Model3 architecture:
        - Input: cluster (13x21), z_global (1), y_local (1)
        - Conv2D: kernel_size=(kernel_rows, kernel_cols), filters=conv_filters
        - MaxPooling2D (2,2) -> Flatten
        - Scalar branch: z_global + y_local (2 features) -> dense(scalar_dense_units)
        - Concatenate conv + scalar
        - merged_dense_1
        - merged_dense_2
        - output (1)
        
        Returns:
            dict: Model metadata including parameters and layer structure
        """
        conv_filters = hyperparams.get('conv_filters', 0)
        kernel_rows = hyperparams.get('kernel_rows', 0)
        kernel_cols = hyperparams.get('kernel_cols', 0)
        scalar_dense_units = hyperparams.get('scalar_dense_units', 0)
        merged_dense_1 = hyperparams.get('merged_dense_1', 0)
        merged_multiplier_2 = hyperparams.get('merged_multiplier_2', 1.0)
        merged_dense_2 = int(merged_dense_1 * merged_multiplier_2)
        
        # Calculate parameters for each layer
        # Conv2D: (kernel_rows * kernel_cols * input_channels * filters) + filters
        conv2d_params = (kernel_rows * kernel_cols * 1 * conv_filters) + conv_filters
        
        # After MaxPooling2D(2,2) on (13, 21) input: (6, 10, conv_filters) -> flattened to 60*conv_filters
        conv_flattened_size = 60 * conv_filters
        
        # Scalar dense: (2 * scalar_dense_units) + scalar_dense_units
        scalar_dense_params = (2 * scalar_dense_units) + scalar_dense_units
        
        # Concatenate: conv_flattened + scalar_dense_units
        concat_size = conv_flattened_size + scalar_dense_units
        
        # Merged dense 1: (concat_size * merged_dense_1) + merged_dense_1
        merged_dense_1_params = (concat_size * merged_dense_1) + merged_dense_1
        
        # Merged dense 2: (merged_dense_1 * merged_dense_2) + merged_dense_2
        merged_dense_2_params = (merged_dense_1 * merged_dense_2) + merged_dense_2
        
        # Output: (merged_dense_2 * 1) + 1
        output_params = merged_dense_2 + 1
        
        # Total parameters
        total_params = (conv2d_params + scalar_dense_params + 
                       merged_dense_1_params + merged_dense_2_params + 
                       output_params)
        
        # Layer structure
        layer_structure = [
            {'name': 'cluster_input', 'type': 'Input', 'shape': '(13, 21)'},
            {'name': 'z_global_input', 'type': 'Input', 'shape': 1},
            {'name': 'y_local_input', 'type': 'Input', 'shape': 1},
            {'name': 'conv2d', 'type': 'QConv2D', 'filters': conv_filters, 
             'kernel_size': f'{kernel_rows}x{kernel_cols}', 'parameters': conv2d_params},
            {'name': 'flatten', 'type': 'Flatten', 'units': conv_flattened_size},
            {'name': 'scalar_dense', 'type': 'QDense', 'units': scalar_dense_units, 'parameters': scalar_dense_params},
            {'name': 'concatenate', 'type': 'Concatenate', 'units': concat_size},
            {'name': 'merged_dense_1', 'type': 'QDense', 'units': merged_dense_1, 'parameters': merged_dense_1_params},
            {'name': 'merged_dense_2', 'type': 'QDense', 'units': merged_dense_2, 'parameters': merged_dense_2_params},
            {'name': 'output', 'type': 'QDense', 'units': 1, 'parameters': output_params}
        ]
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(total_params),
            'non_trainable_parameters': 0,
            'num_layers': len(layer_structure),
            'layer_structure': layer_structure
        }
    
    def runQuantizedHyperparameterTuning(self, bit_configs=None, max_trials=50, executions_per_trial=2, numEpochs=30):
        """
        Run hyperparameter tuning for quantized Model3 with specified bit configurations.
        
        Args:
            bit_configs: List of (weight_bits, int_bits) tuples for quantization. 
                        If None, uses self.bit_configs
            max_trials: Maximum number of trials for hyperparameter search
            executions_per_trial: Number of executions per trial
            numEpochs: Number of epochs for training
            
        Returns:
            Dictionary mapping config_name to (best_model, results, tuner)
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized hyperparameter tuning")
        
        if bit_configs is None:
            bit_configs = self.bit_configs
        
        print(f"Starting quantized hyperparameter tuning for Model3 with {len(bit_configs)} bit configurations...")
        
        # Load data if not already loaded
        if self.training_generator is None:
            self.loadTfRecords()
        
        # Store results for each configuration
        all_results = {}
        
        for weight_bits, int_bits in bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            print(f"\n{'='*60}")
            print(f"Starting hyperparameter tuning for {config_name}")
            print(f"{'='*60}")
            
            # Create a wrapper function that captures weight_bits and int_bits
            def model_builder(hp):
                return self.makeQuantizedModelHyperParameterTuning(hp, weight_bits, int_bits)
            
            # Create tuner with unique project name
            tuner = kt.RandomSearch(
                model_builder,
                objective="val_binary_accuracy",
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                project_name=f"model3_quantized_{weight_bits}w{int_bits}i_hyperparameter_search",
                directory="./hyperparameter_tuning"
            )
            
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
            
            # Run search
            print(f"Running hyperparameter search for {config_name}...")
            tuner.search(
                self.training_generator,
                validation_data=self.validation_generator,
                epochs=numEpochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Get all models and hyperparameters (not just the best)
            all_trials = list(tuner.oracle.trials.values())
            num_trials = len(all_trials)
            
            # Load models one by one, skipping trials with missing checkpoints
            completed_trials = []
            for trial in all_trials:
                if trial.status == 'COMPLETED' and trial.score is not None:
                    completed_trials.append(trial)
            
            # Sort by score (best first)
            completed_trials.sort(key=lambda t: t.score, reverse=True)
            
            all_models = []
            all_hyperparameters = []
            print(f"\nLoading models from {len(completed_trials)} completed trials (out of {num_trials} total)...")
            
            for idx, trial in enumerate(completed_trials):
                try:
                    # Load model for this specific trial
                    model = tuner.load_model(trial)
                    all_models.append(model)
                    all_hyperparameters.append(trial.hyperparameters)
                    if idx == 0:
                        print(f"  ✓ Loaded best model (trial {trial.trial_id}, score={trial.score:.4f})")
                    elif idx < 5:  # Print first 5 for visibility
                        print(f"  ✓ Loaded model {idx+1} (trial {trial.trial_id}, score={trial.score:.4f})")
                except Exception as e:
                    print(f"  ⚠ Failed to load model for trial {trial.trial_id}: {str(e)}")
            
            if len(completed_trials) > 5:
                print(f"  ... (loaded {len(all_models)} models total)")
            else:
                print(f"  ✓ Successfully loaded {len(all_models)} out of {len(completed_trials)} completed models")
            
            if len(all_models) == 0:
                raise RuntimeError(f"Failed to load any models for {config_name}. All trials may have missing checkpoints.")
            
            # Create directory for this configuration's models
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_dir = f"model3_{config_name}_hyperparameter_results_{timestamp}"
            os.makedirs(config_dir, exist_ok=True)
            print(f"\n✓ Created directory for {config_name} results: {config_dir}/")
            
            # Save all models and their hyperparameters
            model_files = []
            hyperparams_files = []
            
            for idx, (model, hyperparams) in enumerate(zip(all_models, all_hyperparameters)):
                # Save model
                model_filename = os.path.join(config_dir, f"model_trial_{idx:03d}.h5")
                model.save(model_filename)
                model_files.append(model_filename)
                
                # Save hyperparameters
                hyperparams_dict = hyperparams.values
                hyperparams_filename = os.path.join(config_dir, f"hyperparams_trial_{idx:03d}.json")
                with open(hyperparams_filename, 'w') as f:
                    json.dump(hyperparams_dict, f, indent=4)
                hyperparams_files.append(hyperparams_filename)
                
                if idx == 0:  # Print best model info
                    print(f"✓ Trial {idx} (BEST): saved to {model_filename}")
                    print(f"  Best hyperparameters: {hyperparams_dict}")
            
            print(f"✓ Saved {len(model_files)} models and hyperparameter files to {config_dir}/")
            
            # Also save a summary JSON with all trials (enriched with metadata)
            summary_filename = os.path.join(config_dir, "trials_summary.json")
            trials_summary = []
            for idx, hyperparams in enumerate(all_hyperparameters):
                trial = completed_trials[idx]
                
                # Basic trial info
                trial_info = {
                    'trial_id': idx,
                    'model_file': f"model_trial_{idx:03d}.h5",
                    'hyperparams_file': f"hyperparams_trial_{idx:03d}.json",
                    'hyperparameters': hyperparams.values,
                    'is_best': (idx == 0)
                }
                
                # Add validation accuracy
                if trial.score is not None:
                    trial_info['val_accuracy'] = float(trial.score)
                
                # Add metrics if available
                if hasattr(trial, 'metrics') and trial.metrics:
                    trial_info['metrics'] = trial.metrics
                
                # Calculate and add model parameters and structure
                param_metadata = self._calculate_model_parameters(hyperparams.values)
                trial_info.update(param_metadata)
                
                trials_summary.append(trial_info)
            
            with open(summary_filename, 'w') as f:
                json.dump(trials_summary, f, indent=4)
            print(f"✓ Saved enriched trials summary to: {summary_filename}")
            
            # Print results
            print(f"\n{config_name} Hyperparameter Tuning Completed!")
            print(f"Total trials: {num_trials}")
            
            # Store results
            all_results[config_name] = {
                'best_model': all_models[0],
                'best_hyperparameters': all_hyperparameters[0].values,
                'all_models': all_models,
                'all_hyperparameters': [hp.values for hp in all_hyperparameters],
                'tuner': tuner,
                'config_dir': config_dir,
                'model_files': model_files,
                'hyperparams_files': hyperparams_files,
                'summary_file': summary_filename,
                'num_trials': num_trials
            }
        
        print(f"\n{'='*60}")
        print(f"All quantized hyperparameter tuning completed!")
        print(f"{'='*60}")
        
        # Print summary
        print("\nSummary of saved results:")
        for config_name, results in all_results.items():
            print(f"  {config_name}:")
            print(f"    Results directory: {results['config_dir']}/")
            print(f"    Number of trials: {results['num_trials']}")
            print(f"    Best model: {results['model_files'][0]}")
            print(f"    Summary file: {results['summary_file']}")
        
        return all_results
    
    def runHyperparameterTuning(self, max_trials=75, executions_per_trial=2):
        """
        Run hyperparameter tuning for Model3.
        
        Args:
            max_trials: Maximum number of trials for hyperparameter search
            executions_per_trial: Number of executions per trial
        """
        print("Starting hyperparameter tuning for Model3...")
        
        # Load data if not already loaded
        if self.training_generator is None:
            self.loadTfRecords()
        
        # Create tuner
        tuner = kt.RandomSearch(
            self.makeUnquatizedModelHyperParameterTuning,
            objective="val_binary_accuracy",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            project_name="model3_hyperparameter_search",
            directory="./hyperparameter_tuning"
        )
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Run search
        tuner.search(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=30,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        self.hyperparameterModel = best_model
        
        # Save results
        results = tuner.results_summary()
        print("Hyperparameter tuning completed!")
        print(f"Best hyperparameters: {tuner.get_best_hyperparameters(num_trials=1)[0].values}")
        
        return best_model, results
    
    #trainModel() function moved to abstract class
    #evaluate() function moved to abstract class
    #plotModel() function moved to abstract class
    #runAllStuff() function moved to abstract class


def main():
    """Example usage of Model3"""
    print("=== Model3 Example Usage ===")
    
    # Initialize Model3
    model3 = Model3(
        tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
        conv_filters=32,
        kernel_rows=3,
        kernel_cols=5,
        scalar_dense_units=32,
        merged_dense_1=200,
        merged_dense_2=100,
        dropout_rate=0.0,
        initial_lr=0.000871145,
        end_lr=5.3e-05,
        power=2,
        bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
    )
    
    # Run complete pipeline
    results = model3.runAllStuff()
    
    print("Model3 quantization testing completed successfully!")


if __name__ == "__main__":
    main()

