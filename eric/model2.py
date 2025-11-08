"""
Model2 Implementation using the SmartPixModel Abstract Base Class

This module implements Model2 as a concrete class inheriting from SmartPixModel.
Model2 is a CNN-based architecture that processes x_profile, y_profile, z_global, and y_local features.

Architecture:
- x_profile + z_global branch (32 units)
- y_profile + y_local branch (32 units) 
- Merged dense layers: 128 -> 64 -> 32
- Output: 1 unit with sigmoid activation

Author: Eric
Date: 2024
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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
sys.path.append('../MuC_Smartpix_ML/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/ryan/')

from Model_Classes import SmartPixModel
import OptimizedDataGenerator4 as ODG

# QKeras imports for quantized models
try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False


class Model2(SmartPixModel):
    """
    Model2: CNN-based architecture for smart pixel detector classification.
    
    This model processes profile data (x_profile, y_profile) along with 
    global/local coordinates (z_global, y_local) for binary classification.
    
    Features:
    - x_profile: 21-dimensional profile data
    - z_global: 1-dimensional global z coordinate
    - y_profile: 13-dimensional profile data  
    - y_local: 1-dimensional local y coordinate
    """
    
    def __init__(self,
                 tfRecordFolder: str = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
                 nBits: list = None,
                 loadModel: bool = False,
                 modelPath: str = None,
                 xz_units: int = 32,
                 yl_units: int = 32,
                 merged_units_1: int = 128,
                 merged_units_2: int = 64,
                 merged_units_3: int = 32,
                 dropout_rate: float = 0.1,
                 initial_lr: float = 1e-3,
                 end_lr: float = 1e-4,
                 power: int = 2,
                 bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
                 ):
        """
        Initialize Model2.
        
        Args:
            tfRecordFolder: Path to TFRecords directory
            nBits: List of bit configurations for quantization
            loadModel: Whether to load a pre-trained model
            modelPath: Path to saved model (if loadModel=True)
            xz_units: Units in x_profile + z_global branch
            yl_units: Units in y_profile + y_local branch
            merged_units_1: Units in first merged dense layer
            merged_units_2: Units in second merged dense layer
            merged_units_3: Units in third merged dense layer
            dropout_rate: Dropout rate for regularization
            initial_lr: Initial learning rate
            end_lr: End learning rate for polynomial decay
            power: Power for polynomial decay
        """
        super().__init__(tfRecordFolder, nBits, loadModel, modelPath)
        
        self.modelName = "Model2"
        
        # Architecture parameters
        self.xz_units = xz_units
        self.yl_units = yl_units
        self.merged_units_1 = merged_units_1
        self.merged_units_2 = merged_units_2
        self.merged_units_3 = merged_units_3
        self.dropout_rate = dropout_rate
        
        # Learning rate parameters
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.power = power
        
        # Model2 specific feature configuration
        self.x_feature_description = ['x_profile', 'z_global', 'y_profile', 'y_local']
        
        # Initialize data generators
        self.training_generator = None
        self.validation_generator = None
        
        # Results storage
        # self.training_history = None
        self.histories = {}
        # self.evaluation_results = None
        
        # quantized_models = {}
        self.models = {"Unquantized": None}
        self.bit_configs = bit_configs 
        for weight_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            self.models[config_name] = None
        
        
        # Load model if requested
        #TODO decide whether to load unquantized or which quantized model
        config_name = "Unquantized"
        if loadModel and modelPath:
            self.loadModel(modelPath,config_name)
    
    def loadTfRecords(self):
        """Load TFRecords using OptimizedDataGenerator4 for Model2 features."""
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
        
        # Model2 uses x_profile, z_global, y_profile, y_local features
        self.training_generator = ODG.OptimizedDataGenerator(
            load_records=True, 
            tf_records_dir=trainDir, 
            x_feature_description=self.x_feature_description,
            batch_size=batch_size
        )
        
        self.validation_generator = ODG.OptimizedDataGenerator(
            load_records=True, 
            tf_records_dir=valDir, 
            x_feature_description=self.x_feature_description,
            batch_size=batch_size
        )
        
        print(f"Training generator length: {len(self.training_generator)}")
        print(f"Validation generator length: {len(self.validation_generator)}")
        
        return self.training_generator, self.validation_generator
    
    def makeUnquantizedModel(self):
        """
        Build the unquantized Model2 architecture.
        Architecture:
        - x_profile (21) + z_global (1) -> xz_units
        - y_profile (13) + y_local (1) -> yl_units  
        - Concatenate -> merged_units_1 -> merged_units_2 -> merged_units_3 -> 1
        """
        print("Building unquantized Model2...")
        print(f"  - XZ branch: {self.xz_units} units")
        print(f"  - YL branch: {self.yl_units} units")
        print(f"  - Merged dense layers: {self.merged_units_1} -> {self.merged_units_2} -> {self.merged_units_3}")
        print(f"  - Dropout: {self.dropout_rate}")
        
        # Input layers
        x_profile_input = Input(shape=(21,), name="x_profile")
        z_global_input = Input(shape=(1,), name="z_global")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input = Input(shape=(1,), name="y_local")
        
        # x_profile + z_global branch
        xz_concat = Concatenate(name="xz_concat")([x_profile_input, z_global_input])
        xz_dense = Dense(self.xz_units, activation="relu", name="xz_dense1")(xz_concat)
        
        # y_profile + y_local branch
        yl_concat = Concatenate(name="yl_concat")([y_profile_input, y_local_input])
        yl_dense = Dense(self.yl_units, activation="relu", name="yl_dense1")(yl_concat)
        
        # Merge both branches
        merged = Concatenate(name="merged_features")([xz_dense, yl_dense])
        merged_dense = Dense(self.merged_units_1, activation="relu", name="merged_dense1")(merged)
        merged_dense = Dropout(self.dropout_rate, name="dropout1")(merged_dense)
        merged_dense = Dense(self.merged_units_2, activation="relu", name="merged_dense2")(merged_dense)
        merged_dense = Dense(self.merged_units_3, activation="relu", name="merged_dense3")(merged_dense)
        
        # Output layer
        output = Dense(1, activation="sigmoid", name="output")(merged_dense)
        
        # Create and compile model
        self.models["Unquantized"] = Model(
            inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input], 
            outputs=output, 
            name="model2_unquantized"
        )
        
        # # Store in models dictionary
        # self.models["Unquantized"]
        
        print("✓ Unquantized Model2 built successfully")
        # return self.model
    
    def makeUnquatizedModelHyperParameterTuning(self, hp):
        """
        Build Model2 for hyperparameter tuning using Keras Tuner.
        
        Args:
            hp: Keras Tuner hyperparameter object
        """
        # Input layers
        x_profile_input = Input(shape=(21,), name="x_profile")
        z_global_input = Input(shape=(1,), name="z_global")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input = Input(shape=(1,), name="y_local")
        
        # Hyperparameter search space
        xz_units = hp.Int('xz_units', min_value=8, max_value=64, step=8)
        yl_units = hp.Int('yl_units', min_value=8, max_value=64, step=8)
        merged_units1 = hp.Int('merged_units1', min_value=32, max_value=256, step=32)
        merged_units2 = hp.Int('merged_units2', min_value=32, max_value=128, step=16)
        merged_units3 = hp.Int('merged_units3', min_value=16, max_value=64, step=8)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.1, step=0.1)
        
        # x_profile + z_global branch
        xz_concat = Concatenate(name="xz_concat")([x_profile_input, z_global_input])
        xz_dense = Dense(xz_units, activation="relu", name="xz_dense1")(xz_concat)
        
        # y_profile + y_local branch
        yl_concat = Concatenate(name="yl_concat")([y_profile_input, y_local_input])
        yl_dense = Dense(yl_units, activation="relu", name="yl_dense1")(yl_concat)
        
        # Merge both branches
        merged = Concatenate(name="merged_features")([xz_dense, yl_dense])
        merged_dense = Dense(merged_units1, activation="relu", name="merged_dense1")(merged)
        merged_dense = Dropout(dropout_rate, name="dropout1")(merged_dense)
        merged_dense = Dense(merged_units2, activation="relu", name="merged_dense2")(merged_dense)
        merged_dense = Dense(merged_units3, activation="relu", name="merged_dense3")(merged_dense)
        
        # Output layer
        output = Dense(1, activation="sigmoid", name="output")(merged_dense)
        
        # Create model
        model = Model(
            inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input], 
            outputs=output, 
            name="model2_hyperparameter_tuning"
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
        Build quantized Model2 using QKeras.
        
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
            x_profile_input = Input(shape=(21,), name="x_profile")
            z_global_input = Input(shape=(1,), name="z_global")
            y_profile_input = Input(shape=(13,), name="y_profile")
            y_local_input = Input(shape=(1,), name="y_local")
            
            # x_profile + z_global branch
            xz_concat = Concatenate(name="xz_concat")([x_profile_input, z_global_input])
            xz_dense = QDense(
                self.xz_units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="xz_dense1"
            )(xz_concat)
            xz_dense = QActivation(activation=activation_quantizer, name="xz_relu1")(xz_dense)
            
            # y_profile + y_local branch
            yl_concat = Concatenate(name="yl_concat")([y_profile_input, y_local_input])
            yl_dense = QDense(
                self.yl_units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="yl_dense1"
            )(yl_concat)
            yl_dense = QActivation(activation=activation_quantizer, name="yl_relu1")(yl_dense)
            
            # Merge both branches
            merged = Concatenate(name="merged_features")([xz_dense, yl_dense])
            merged_dense = QDense(
                self.merged_units_1,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="merged_dense1"
            )(merged)
            merged_dense = QActivation(activation=activation_quantizer, name="merged_relu1")(merged_dense)
            merged_dense = Dropout(self.dropout_rate, name="dropout1")(merged_dense)  # Add dropout to match non-quantized
            merged_dense = QDense(
                self.merged_units_2,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="merged_dense2"
            )(merged_dense)
            merged_dense = QActivation(activation=activation_quantizer, name="merged_relu2")(merged_dense)
            merged_dense = QDense(
                self.merged_units_3,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="merged_dense3"
            )(merged_dense)
            merged_dense = QActivation(activation=activation_quantizer, name="merged_relu3")(merged_dense)
            
            # Output layer
            output_dense = QDense(
                1,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="output_dense"
            )(merged_dense)
            output = QActivation("quantized_tanh", name="output")(output_dense)
            
            # Create model
            model = Model(
                inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input], 
                outputs=output, 
                name=f"model2_{config_name}"
            )
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
                metrics=["binary_accuracy"],
                run_eagerly=True
            )
            
            self.models[config_name] = model
        
        # # Store quantized models
        # self.models["Quantized"] = quantized_models
        # self.quantized_model = quantized_models  # For backward compatibility
        
        print(f"✓ Built {len(self.bit_configs)} quantized Model2 variants")
        # return quantized_models
    

    
    def runHyperparameterTuning(self, max_trials=50, executions_per_trial=2,numEpochs = 30):
        """
        Run hyperparameter tuning for Model2.
        
        Args:
            max_trials: Maximum number of trials for hyperparameter search
            executions_per_trial: Number of executions per trial
        """
        print("Starting hyperparameter tuning for Model2...")
        
        # Load data if not already loaded
        if self.training_generator is None:
            self.loadTfRecords()
        
        # Create tuner
        tuner = kt.RandomSearch(
            self.makeUnquatizedModelHyperParameterTuning,
            objective="val_binary_accuracy",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            project_name="model2_hyperparameter_search",
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
            epochs=numEpochs,
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
    
    def makeQuantizedModelHyperParameterTuning(self, hp, weight_bits, int_bits):
        """
        Build quantized Model2 for hyperparameter tuning using Keras Tuner and QKeras.
        
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
        x_profile_input = Input(shape=(21,), name="x_profile")
        z_global_input = Input(shape=(1,), name="z_global")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input = Input(shape=(1,), name="y_local")
        
        # Hyperparameter search space
        xz_units = hp.Int('xz_units', min_value=4, max_value=20, step=4)
        yl_units = hp.Int('yl_units', min_value=4, max_value=20, step=4)
        merged_units1 = hp.Int('merged_units1', min_value=16, max_value=64, step=8)
        merged_units2 = hp.Int('merged_units2', min_value=8, max_value=48, step=8)
        merged_units3 = hp.Int('merged_units3', min_value=4, max_value=32, step=8)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.1, step=0.1)
        
        # x_profile + z_global branch
        xz_concat = Concatenate(name="xz_concat")([x_profile_input, z_global_input])
        xz_dense = QDense(
            xz_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="xz_dense1"
        )(xz_concat)
        xz_dense = QActivation(activation=activation_quantizer, name="xz_relu1")(xz_dense)
        
        # y_profile + y_local branch
        yl_concat = Concatenate(name="yl_concat")([y_profile_input, y_local_input])
        yl_dense = QDense(
            yl_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="yl_dense1"
        )(yl_concat)
        yl_dense = QActivation(activation=activation_quantizer, name="yl_relu1")(yl_dense)
        
        # Merge both branches
        merged = Concatenate(name="merged_features")([xz_dense, yl_dense])
        merged_dense = QDense(
            merged_units1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="merged_dense1"
        )(merged)
        merged_dense = QActivation(activation=activation_quantizer, name="merged_relu1")(merged_dense)
        merged_dense = Dropout(dropout_rate, name="dropout1")(merged_dense)
        merged_dense = QDense(
            merged_units2,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="merged_dense2"
        )(merged_dense)
        merged_dense = QActivation(activation=activation_quantizer, name="merged_relu2")(merged_dense)
        merged_dense = QDense(
            merged_units3,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="merged_dense3"
        )(merged_dense)
        merged_dense = QActivation(activation=activation_quantizer, name="merged_relu3")(merged_dense)
        
        # Output layer with quantized_tanh
        output_dense = QDense(
            1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="output_dense"
        )(merged_dense)
        output = QActivation("quantized_tanh", name="output")(output_dense)
        
        # Create model
        model = Model(
            inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input], 
            outputs=output, 
            name=f"model2_quantized_{weight_bits}w{int_bits}i_hyperparameter_tuning"
        )
        
        # Compile with hyperparameter-tuned learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly=True
        )
        
        return model
    
    def runQuantizedHyperparameterTuning(self, bit_configs=None, max_trials=50, executions_per_trial=2, numEpochs=30):
        """
        Run hyperparameter tuning for quantized Model2 with specified bit configurations.
        
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
        
        print(f"Starting quantized hyperparameter tuning for Model2 with {len(bit_configs)} bit configurations...")
        
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
                project_name=f"model2_quantized_{weight_bits}w{int_bits}i_hyperparameter_search",
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
            # Filter out trials that don't have valid checkpoints
            # tuner.oracle.trials is a dictionary mapping trial_id to Trial objects
            all_trials = list(tuner.oracle.trials.values())
            num_trials = len(all_trials)
            
            # Load models one by one, skipping trials with missing checkpoints
            # Sort trials by objective value (best first)
            completed_trials = []
            for trial in all_trials:
                if trial.status == 'COMPLETED' and trial.score is not None:
                    completed_trials.append(trial)
            
            # Sort by score (best first, since objective is 'val_binary_accuracy' which should be maximized)
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
                    # Continue with other trials
            
            if len(completed_trials) > 5:
                print(f"  ... (loaded {len(all_models)} models total)")
            else:
                print(f"  ✓ Successfully loaded {len(all_models)} out of {len(completed_trials)} completed models")
            
            if len(all_models) == 0:
                raise RuntimeError(f"Failed to load any models for {config_name}. All trials may have missing checkpoints.")
            
            # Create directory for this configuration's models
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_dir = f"model2_{config_name}_hyperparameter_results_{timestamp}"
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
            
            # Also save a summary JSON with all trials
            summary_filename = os.path.join(config_dir, "trials_summary.json")
            trials_summary = []
            for idx, hyperparams in enumerate(all_hyperparameters):
                trial_info = {
                    'trial_id': idx,
                    'model_file': f"model_trial_{idx:03d}.h5",
                    'hyperparams_file': f"hyperparams_trial_{idx:03d}.json",
                    'hyperparameters': hyperparams.values,
                    'is_best': (idx == 0)
                }
                trials_summary.append(trial_info)
            
            with open(summary_filename, 'w') as f:
                json.dump(trials_summary, f, indent=4)
            print(f"✓ Saved trials summary to: {summary_filename}")
            
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
    
    #plotModel() function moved to abstract class
    #runAllStuff() function moved to abstract class


def main():
    """Example usage of Model2"""
    print("=== Model2 Example Usage ===")
    
    # Initialize Model2
    model2 = Model2(
        tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
        xz_units=8,
        yl_units=8,
        merged_units_1=64,
        merged_units_2=32,
        merged_units_3=16,
        dropout_rate=0.1,
        initial_lr=1e-3,
        end_lr=1e-4,
        power=2,
        bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
    )
    
    # Run complete pipeline
    results = model2.runAllStuff()
    
    print("Model2 quantization testing completed successfully!")


if __name__ == "__main__":
    main()