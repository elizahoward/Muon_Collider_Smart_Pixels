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
                 power: int = 2):
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
        self.training_history = None
        self.evaluation_results = None
        
        # Load model if requested
        if loadModel and modelPath:
            self.loadModel(modelPath)
    
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
        
        # Create model
        self.model = Model(
            inputs=[cluster_input, z_global_input, y_local_input], 
            outputs=output, 
            name="model3_unquantized"
        )
        
        # Store in models dictionary
        self.models["Unquantized"] = self.model
        
        print("✓ Unquantized Model3 built successfully")
        return self.model
    
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
    
    def makeQuantizedModel(self, bit_configs=None):
        """
        Build quantized Model3 using QKeras.
        
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
            
            # Output layer
            output = QDense(
                1, 
                activation="sigmoid", 
                kernel_quantizer=weight_quantizer, 
                bias_quantizer=bias_quantizer, 
                name="output"
            )(h)
            
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
                metrics=["binary_accuracy"],
                run_eagerly=True
            )
            
            quantized_models[config_name] = model
        
        # Store quantized models
        self.models["Quantized"] = quantized_models
        self.quantized_model = quantized_models  # For backward compatibility
        
        print(f"✓ Built {len(quantized_models)} quantized Model3 variants")
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
    
    def trainModel(self, epochs=200, batch_size=32, learning_rate=None, 
                   save_best=True, early_stopping_patience=35):
        """
        Train the Model3.
        
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
        
        print(f"Training Model3 for {epochs} epochs...")
        
        # Setup learning rate schedule
        if learning_rate is None:
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
            metrics=["accuracy"]
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
            # Use .h5 format to avoid compatibility issues
            callbacks.append(ModelCheckpoint(
                filepath=f'./{self.modelName}_best.h5',
                monitor='val_accuracy',
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
        
        print("✓ Model3 training completed!")
        return self.training_history
    
    def evaluate(self, test_generator=None):
        """
        Evaluate the trained Model3.
        
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
        
        print("Evaluating Model3...")
        
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
        
        print(f"✓ Model3 evaluation completed!")
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
        axes[0].plot(self.training_history.history['accuracy'], label='Training')
        axes[0].plot(self.training_history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model3 Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.training_history.history['loss'], label='Training')
        axes[1].plot(self.training_history.history['val_loss'], label='Validation')
        axes[1].set_title('Model3 Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/model3_training_history.png", dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {output_dir}/model3_training_history.png")
        
        plt.show()
        
        # Plot ROC curve if evaluation results available
        if self.evaluation_results is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(self.evaluation_results['fpr'], self.evaluation_results['tpr'], 
                    label=f"ROC Curve (AUC = {self.evaluation_results['roc_auc']:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Model3 ROC Curve')
            plt.legend()
            plt.grid(True)
            
            if save_plots:
                plt.savefig(f"{output_dir}/model3_roc_curve.png", dpi=300, bbox_inches='tight')
                print(f"ROC curve plot saved to {output_dir}/model3_roc_curve.png")
            
            plt.show()
    
    def runAllStuff(self):
        """
        Run the complete Model3 pipeline: build, train, evaluate, and plot for both quantized and non-quantized models.
        """
        print("=== Running Complete Model3 Pipeline with Quantization Testing ===")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"model3_results_{timestamp}"
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
        self.trainModel(epochs=20, early_stopping_patience=15, save_best=False)
        
        print("2c. Evaluating unquantized model...")
        eval_results = self.evaluate()
        
        # Save non-quantized model
        print("2d. Saving unquantized model...")
        model_save_path = os.path.join(models_dir, "model3_unquantized.h5")
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
                time_stamps=self.time_stamps,
                batch_size=batch_size
            )
            
            fresh_val_gen = ODG.OptimizedDataGenerator(
                load_records=True, 
                tf_records_dir=valDir, 
                x_feature_description=self.x_feature_description,
                time_stamps=self.time_stamps,
                batch_size=batch_size
            )
            
            print(f"3b. Building {weight_bits}-bit quantized model...")
            self.buildModel("quantized", bit_configs=[(weight_bits, int_bits)])
            
            # Get the quantized model
            quantized_model = self.models["Quantized"][f"quantized_{weight_bits}w{int_bits}i"]
            
            print(f"3c. Training {weight_bits}-bit quantized model...")
            # Compile and train the quantized model
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
                metrics=["accuracy"],
                run_eagerly=True
            )
            
            # Train with early stopping using fresh generators
            history = quantized_model.fit(
                fresh_train_gen,
                validation_data=fresh_val_gen,
                epochs=20,
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
                time_stamps=self.time_stamps,
                batch_size=batch_size
            )
            
            # Evaluate
            test_loss, test_accuracy = quantized_model.evaluate(eval_val_gen, verbose=0)
            
            # Create yet another fresh validation generator for predictions
            pred_val_gen = ODG.OptimizedDataGenerator(
                load_records=True, 
                tf_records_dir=valDir, 
                x_feature_description=self.x_feature_description,
                time_stamps=self.time_stamps,
                batch_size=batch_size
            )
            
            # Calculate ROC AUC
            predictions = quantized_model.predict(pred_val_gen, verbose=0).ravel()
            true_labels = np.concatenate([y for _, y in pred_val_gen])
            
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            roc_auc_score = auc(fpr, tpr)
            
            # Save quantized model
            print(f"3e. Saving {weight_bits}-bit quantized model...")
            model_save_path = os.path.join(models_dir, f"model3_quantized_{weight_bits}bit.h5")
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
            # Create a temporary model3 instance for plotting
            temp_model3 = Model3(
                tfRecordFolder=self.tfRecordFolder,
                conv_filters=self.conv_filters,
                kernel_rows=self.kernel_rows,
                kernel_cols=self.kernel_cols,
                scalar_dense_units=self.scalar_dense_units,
                merged_dense_1=self.merged_dense_1,
                merged_dense_2=self.merged_dense_2,
                dropout_rate=self.dropout_rate
            )
            temp_model3.model = quantized_model
            temp_model3.training_history = history
            temp_model3.evaluation_results = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'roc_auc': float(roc_auc_score),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            plot_dir_quant = os.path.join(plots_dir, f"{weight_bits}bit")
            temp_model3.plotModel(save_plots=True, output_dir=plot_dir_quant)
        
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
        
        print(f"\n=== Model3 Quantization Pipeline Completed! ===")
        print(f"All outputs saved to: {output_dir}/")
        print(f"  - Models: {models_dir}/")
        print(f"  - Plots: {plots_dir}/")
        print(f"  - Results CSV: {results_file}")
        
        return results


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
        dropout_rate=0.0
    )
    
    # Run complete pipeline
    results = model3.runAllStuff()
    
    print("Model3 quantization testing completed successfully!")


if __name__ == "__main__":
    main()

