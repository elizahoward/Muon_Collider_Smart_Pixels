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
                 power: int = 2):
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
        self.training_history = None
        self.evaluation_results = None
        
        # Load model if requested
        if loadModel and modelPath:
            self.loadModel(modelPath)
    
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
        self.model = Model(
            inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input], 
            outputs=output, 
            name="model2_unquantized"
        )
        
        # Store in models dictionary
        self.models["Unquantized"] = self.model
        
        print("✓ Unquantized Model2 built successfully")
        return self.model
    
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
    
    def makeQuantizedModel(self, bit_configs=None):
        """
        Build quantized Model2 using QKeras.
        
        Args:
            bit_configs: List of (weight_bits, int_bits) tuples for quantization
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")
        
        if bit_configs is None:
            bit_configs = [(8, 0), (6, 0), (4, 0)]  # Default configurations
        
        quantized_models = {}
        
        for weight_bits, int_bits in bit_configs:
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
            
            quantized_models[config_name] = model
        
        # Store quantized models
        self.models["Quantized"] = quantized_models
        self.quantized_model = quantized_models  # For backward compatibility
        
        print(f"✓ Built {len(quantized_models)} quantized Model2 variants")
        return quantized_models
    
    def buildModel(self, model_type="unquantized", bit_configs=None):
        """
        Build the specified model type.
        
        Args:
            model_type: "unquantized" or "quantized"
            bit_configs: List of bit configurations for quantized models
        """
        if model_type == "unquantized":
            return self.makeUnquantizedModel()
        elif model_type == "quantized":
            return self.makeQuantizedModel(bit_configs)
        else:
            raise ValueError("model_type must be 'unquantized' or 'quantized'")
    
    def runHyperparameterTuning(self, max_trials=50, executions_per_trial=2):
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
    
    def trainModel(self, epochs=100, batch_size=32, learning_rate=None, 
                   save_best=True, early_stopping_patience=20):
        """
        Train the Model2.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training (not used, defined in data generator)
            learning_rate: Learning rate for optimizer (if None, uses polynomial decay)
            save_best: Whether to save the best model
            early_stopping_patience: Patience for early stopping
        """
        if self.model is None:
            raise ValueError("Model not built. Call buildModel() first.")
        
        if self.training_generator is None:
            self.loadTfRecords()
        
        print(f"Training Model2 for {epochs} epochs...")
        
        # Setup learning rate schedule
        if learning_rate is None:
            from tensorflow.keras.optimizers.schedules import PolynomialDecay
            decay_steps = 30 * 200
            lr_schedule = PolynomialDecay(
                initial_learning_rate=self.initial_lr,
                decay_steps=decay_steps,
                end_learning_rate=self.end_lr,
                power=self.power
            )
            optimizer = Adam(learning_rate=lr_schedule)
        else:
            optimizer = Adam(learning_rate=learning_rate)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["binary_accuracy"]
        )
        
        # Create callbacks
        callbacks = []
        
        if early_stopping_patience > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ))
        
        if save_best:
            # Use .h5 format instead of .keras to avoid the 'options' error
            callbacks.append(ModelCheckpoint(
                filepath=f'./{self.modelName}_best.h5',
                monitor='val_binary_accuracy',
                save_best_only=True,
                mode='max'
            ))
        
        # Train model
        self.training_history = self.model.fit(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Model2 training completed!")
        return self.training_history
    
    def evaluate(self, test_generator=None):
        """
        Evaluate the trained Model2.
        
        Args:
            test_generator: Optional test data generator
        """
        if self.model is None:
            raise ValueError("No model to evaluate. Train a model first.")
        
        # Use validation generator if no test generator provided
        eval_generator = test_generator if test_generator else self.validation_generator
        
        if eval_generator is None:
            self.loadTfRecords()
            eval_generator = self.validation_generator
        
        print("Evaluating Model2...")
        
        # Get predictions
        predictions = self.model.predict(eval_generator, verbose=1)
        
        # Get true labels
        true_labels = np.concatenate([y for _, y in eval_generator])
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(eval_generator, verbose=0)
        
        # Calculate ROC AUC
        fpr, tpr, thresholds = roc_curve(true_labels, predictions.ravel())
        roc_auc = auc(fpr, tpr)
        
        # Store results
        self.evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'roc_auc': float(roc_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        print(f"✓ Model2 evaluation completed!")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        return self.evaluation_results
    
    def plotModel(self, save_plots=True, output_dir="./plots"):
        """
        Plot training history and evaluation results.
        
        Args:
            save_plots: Whether to save plots to disk
            output_dir: Directory to save plots
        """
        if self.training_history is None:
            print("No training history available. Train a model first.")
            return
        
        # Create output directory
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.training_history.history['binary_accuracy'], label='Training')
        axes[0].plot(self.training_history.history['val_binary_accuracy'], label='Validation')
        axes[0].set_title('Model2 Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.training_history.history['loss'], label='Training')
        axes[1].plot(self.training_history.history['val_loss'], label='Validation')
        axes[1].set_title('Model2 Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/model2_training_history.png", dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {output_dir}/model2_training_history.png")
        
        plt.show()
        
        # Plot ROC curve if evaluation results available
        if self.evaluation_results is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(self.evaluation_results['fpr'], self.evaluation_results['tpr'], 
                    label=f"ROC Curve (AUC = {self.evaluation_results['roc_auc']:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Model2 ROC Curve')
            plt.legend()
            plt.grid(True)
            
            if save_plots:
                plt.savefig(f"{output_dir}/model2_roc_curve.png", dpi=300, bbox_inches='tight')
                print(f"ROC curve plot saved to {output_dir}/model2_roc_curve.png")
            
            plt.show()
    
    def runAllStuff(self):
        """
        Run the complete Model2 pipeline: build, train, evaluate, and plot for both quantized and non-quantized models.
        """
        print("=== Running Complete Model2 Pipeline with Quantization Testing ===")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"model2_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"All results will be saved to: {output_dir}/")
        
        # Create subdirectories
        models_dir = os.path.join(output_dir, "models")
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Load data
        print("1. Loading TFRecords...")
        self.loadTfRecords()
        
        # Results storage
        results = []
        
        # Test non-quantized model first
        print("\n2. Testing Non-quantized Model...")
        print("2a. Building unquantized model...")
        self.buildModel("unquantized")
        
        print("2b. Training unquantized model...")
        self.trainModel(epochs=1, early_stopping_patience=15, save_best=False)
        
        print("2c. Evaluating unquantized model...")
        eval_results = self.evaluate()
        
        # Save non-quantized model
        print("2d. Saving unquantized model...")
        model_save_path = os.path.join(models_dir, "model2_unquantized.h5")
        self.model.save(model_save_path)
        print(f"Unquantized model saved to: {model_save_path}")
        
        # Store non-quantized results
        results.append({
            'model_type': 'non_quantized',
            'weight_bits': 'N/A',
            'integer_bits': 'N/A',
            'test_accuracy': eval_results['test_accuracy'],
            'test_loss': eval_results['test_loss'],
            'roc_auc': eval_results['roc_auc'],
            'model_path': model_save_path
        })
        
        print(f"Non-quantized results: Acc={eval_results['test_accuracy']:.4f}, AUC={eval_results['roc_auc']:.4f}")
        
        # Plot non-quantized results
        print("2e. Plotting non-quantized results...")
        plot_dir_unquant = os.path.join(plots_dir, "non_quantized")
        self.plotModel(save_plots=True, output_dir=plot_dir_unquant)
        
        # Test quantized models
        bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
        
        for weight_bits, int_bits in bit_configs:
            print(f"\n3. Testing {weight_bits}-bit Quantized Model...")
            
            # Create completely fresh data generators for each quantized model to avoid state corruption
            print(f"3a. Creating fresh data generators...")
            trainDir = f"{self.tfRecordFolder}/tfrecords_train/"
            valDir = f"{self.tfRecordFolder}/tfrecords_validation/"
            
            # Determine batch size from directory name
            batch_size = 16384
            if "filtering_records16384" in self.tfRecordFolder:
                batch_size = 16384
            elif "filtering_records1024" in self.tfRecordFolder:
                batch_size = 1024
            
            # Create fresh generators
            fresh_train_gen = ODG.OptimizedDataGenerator(
                load_records=True, 
                tf_records_dir=trainDir, 
                x_feature_description=self.x_feature_description,
                batch_size=batch_size
            )
            
            fresh_val_gen = ODG.OptimizedDataGenerator(
                load_records=True, 
                tf_records_dir=valDir, 
                x_feature_description=self.x_feature_description,
                batch_size=batch_size
            )
            
            print(f"3b. Building {weight_bits}-bit quantized model...")
            self.buildModel("quantized", bit_configs=[(weight_bits, int_bits)])
            
            # Get the quantized model
            quantized_model = self.models["Quantized"][f"quantized_{weight_bits}w{int_bits}i"]
            
            print(f"3c. Training {weight_bits}-bit quantized model...")
            # Compile and train the quantized model
            from tensorflow.keras.optimizers.schedules import PolynomialDecay
            decay_steps = 30 * 200
            lr_schedule = PolynomialDecay(
                initial_learning_rate=self.initial_lr,
                decay_steps=decay_steps,
                end_learning_rate=self.end_lr,
                power=self.power
            )
            optimizer = Adam(learning_rate=lr_schedule)
            
            quantized_model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["binary_accuracy"],
                run_eagerly=True
            )
            
            # Train with early stopping using fresh generators
            history = quantized_model.fit(
                fresh_train_gen,
                validation_data=fresh_val_gen,
                epochs=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=15,
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )
            
            print(f"3d. Evaluating {weight_bits}-bit quantized model...")
            # Create another fresh validation generator for evaluation
            eval_val_gen = ODG.OptimizedDataGenerator(
                load_records=True, 
                tf_records_dir=valDir, 
                x_feature_description=self.x_feature_description,
                batch_size=batch_size
            )
            
            # Evaluate
            test_loss, test_accuracy = quantized_model.evaluate(eval_val_gen, verbose=0)
            
            # Create yet another fresh validation generator for predictions
            pred_val_gen = ODG.OptimizedDataGenerator(
                load_records=True, 
                tf_records_dir=valDir, 
                x_feature_description=self.x_feature_description,
                batch_size=batch_size
            )
            
            # Calculate ROC AUC
            predictions = quantized_model.predict(pred_val_gen, verbose=0).ravel()
            true_labels = np.concatenate([y for _, y in pred_val_gen])
            
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            roc_auc_score = auc(fpr, tpr)
            
            # Save quantized model
            print(f"3e. Saving {weight_bits}-bit quantized model...")
            model_save_path = os.path.join(models_dir, f"model2_quantized_{weight_bits}bit.h5")
            quantized_model.save(model_save_path)
            print(f"{weight_bits}-bit model saved to: {model_save_path}")
            
            # Store quantized results
            results.append({
                'model_type': 'quantized',
                'weight_bits': weight_bits,
                'integer_bits': int_bits,
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'roc_auc': float(roc_auc_score),
                'model_path': model_save_path
            })
            
            print(f"{weight_bits}-bit results: Acc={test_accuracy:.4f}, AUC={roc_auc_score:.4f}")
            
            # Plot quantized results
            print(f"3f. Plotting {weight_bits}-bit quantized results...")
            # Create a temporary model2 instance for plotting
            temp_model2 = Model2(
                tfRecordFolder=self.tfRecordFolder,
                xz_units=self.xz_units,
                yl_units=self.yl_units,
                merged_units_1=self.merged_units_1,
                merged_units_2=self.merged_units_2,
                merged_units_3=self.merged_units_3,
                dropout_rate=self.dropout_rate
            )
            temp_model2.model = quantized_model
            temp_model2.training_history = history
            temp_model2.evaluation_results = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'roc_auc': float(roc_auc_score),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            plot_dir_quant = os.path.join(plots_dir, f"{weight_bits}bit")
            temp_model2.plotModel(save_plots=True, output_dir=plot_dir_quant)
        
        # Create results summary
        print("\n4. Results Summary:")
        print("=" * 60)
        print(f"{'Model Type':<15} {'Bits':<8} {'Accuracy':<10} {'Loss':<12} {'ROC AUC':<10}")
        print("-" * 60)
        
        for result in results:
            model_type = result['model_type']
            bits = result.get('weight_bits', 'N/A')
            acc = result['test_accuracy']
            loss = result['test_loss']
            auc_score = result['roc_auc']
            
            print(f"{model_type:<15} {bits:<8} {acc:<10.4f} {loss:<12.4f} {auc_score:<10.4f}")
        
        # Find best configuration
        best_result = max(results, key=lambda x: x['test_accuracy'])
        print(f"\nBEST CONFIGURATION:")
        print(f"Model: {best_result['model_type']}")
        if best_result['model_type'] == 'quantized':
            print(f"Bits: {best_result['weight_bits']}-bit")
        print(f"Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"ROC AUC: {best_result['roc_auc']:.4f}")
        
        # Save results to CSV
        results_file = os.path.join(output_dir, "quantization_results.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        print(f"\n=== Model2 Quantization Pipeline Completed! ===")
        print(f"All outputs saved to: {output_dir}/")
        print(f"  - Models: {models_dir}/")
        print(f"  - Plots: {plots_dir}/")
        print(f"  - Results CSV: {results_file}")
        
        return results


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
        power=2
    )
    
    # Run complete pipeline
    results = model2.runAllStuff()
    
    print("Model2 quantization testing completed successfully!")


if __name__ == "__main__":
    main()