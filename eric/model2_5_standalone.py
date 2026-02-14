"""
Model2.5 Standalone Implementation - Inherits from SmartPixModel but independent from Model2

Model2.5 is a simplified architecture with a single dense fusion layer and 
dedicated 8-bit (or 4-bit) projection for the z_global feature.

Architecture:
- Spatial features (x_profile + y_profile + y_local) -> spatial_units
- z_global feature -> z_global_units (with separate quantization)
- Concatenate -> dense2_units -> dense3_units -> output

Author: Eric
Date: February 2026
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve, auc

# Add paths for imports
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/ryan/')
sys.path.append('/local/d1/smartpixML/filtering_models/shuffling_data/')

from Model_Classes import SmartPixModel
import OptimizedDataGenerator4_data_shuffled_bigData as ODG2

# QKeras imports for quantized models
try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu, quantized_tanh
    from qkeras.utils import _add_supported_quantized_objects
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False


class Model2_5_Standalone(SmartPixModel):
    """
    Model2.5: Standalone implementation with single-hidden-layer architecture
    and dedicated quantization for z_global feature.
    
    Features:
    - x_profile: 21-dimensional profile data
    - y_profile: 13-dimensional profile data
    - y_local: 1-dimensional local y coordinate
    - z_global: 1-dimensional global z coordinate (separate branch)
    """
    
    def __init__(self,
                 tfRecordFolder: str = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData",
                 nBits: list = None,
                 loadModel: bool = False,
                 modelPath: str = None,
                 dense_units: int = 128,
                 z_global_units: int = 32,
                 dense2_units: int = 128,
                 dense3_units: int = 64,
                 dropout_rate: float = 0.1,
                 initial_lr: float = 1e-3,
                 end_lr: float = 1e-4,
                 power: int = 2,
                 bit_configs = [(4, 0)],
                 z_global_weight_bits: int = 4,
                 z_global_int_bits: int = 0):
        """
        Initialize Model2.5 Standalone.
        
        Args:
            tfRecordFolder: Path to TFRecords directory
            nBits: Legacy parameter for compatibility with SmartPixModel
            loadModel: Whether to load a pre-trained model
            modelPath: Path to saved model (if loadModel=True)
            dense_units: Units for spatial features branch
            z_global_units: Units for z_global branch
            dense2_units: Units in second dense layer
            dense3_units: Units in third dense layer
            dropout_rate: Dropout rate for regularization
            initial_lr: Initial learning rate
            end_lr: End learning rate for polynomial decay
            power: Power for polynomial decay
            bit_configs: List of (weight_bits, int_bits) tuples for quantization
            z_global_weight_bits: Bit width for z_global weights
            z_global_int_bits: Integer bits for z_global weights
        """
        # Call parent constructor
        super().__init__(
            tfRecordFolder=tfRecordFolder,
            nBits=nBits,
            loadModel=loadModel,
            modelPath=modelPath,
            initial_lr=initial_lr,
            end_lr=end_lr,
            power=power,
            bit_configs=bit_configs
        )
        
        self.modelName = "Model2.5"
        self.dense_units = dense_units
        self.z_global_units = z_global_units
        self.dense2_units = dense2_units
        self.dense3_units = dense3_units
        self.dropout_rate = dropout_rate
        self.z_global_weight_bits = z_global_weight_bits
        self.z_global_int_bits = z_global_int_bits
        
        # Feature configuration for Model2.5
        self.x_feature_description = ['x_profile', 'z_global', 'y_profile', 'y_local']
        
        # Data generators (will be set by loadTfRecords)
        self.training_generator = None
        self.validation_generator = None
        
        # Initialize models dict for quantized variants
        self.models = {"Unquantized": None}
        for weight_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            self.models[config_name] = None
        
        print("Model2.5 Standalone initialized")
    
    # Note: loadTfRecords is inherited from SmartPixModel but we override it
    # to use Model2.5-specific feature description
    def loadTfRecords(self):
        """Load TFRecords using OptimizedDataGenerator4 for Model2.5 features."""
        raise NotImplementedError("Eric, please use inheritance!!")
        trainDir = f"{self.tfRecordFolder}/tfrecords_train/"
        valDir = f"{self.tfRecordFolder}/tfrecords_validation/"
        
        print(f"Loading training data from: {trainDir}")
        print(f"Loading validation data from: {valDir}")
        
        # Determine batch size from directory name
        batch_size = 16384
        if "filtering_records16384" in self.tfRecordFolder:
            batch_size = 16384
        elif "filtering_records1024" in self.tfRecordFolder:
            batch_size = 1024
        
        print(f"Using batch_size={batch_size}")
        
        # Model2.5 uses x_profile, z_global, y_profile, y_local features
        self.training_generator = ODG2.OptimizedDataGeneratorDataShuffledBigData(
            load_records=True,
            tf_records_dir=trainDir,
            x_feature_description=self.x_feature_description,
            batch_size=batch_size
        )
        
        self.validation_generator = ODG2.OptimizedDataGeneratorDataShuffledBigData(
            load_records=True,
            tf_records_dir=valDir,
            x_feature_description=self.x_feature_description,
            batch_size=batch_size
        )
        
        print(f"Training batches: {len(self.training_generator)}")
        print(f"Validation batches: {len(self.validation_generator)}")
        
        return self.training_generator, self.validation_generator
    
    def makeUnquantizedModel(self):
        """Build the unquantized Model2.5 architecture."""
        print("Building unquantized Model2.5...")
        print(f"  - Spatial features branch: {self.dense_units} units")
        print(f"  - z_global branch: {self.z_global_units} units")
        print(f"  - Merged dense layers: {self.dense2_units} -> {self.dense3_units}")
        
        # Input layers
        x_profile_input = Input(shape=(21,), name="x_profile")
        z_global_input = Input(shape=(1,), name="z_global")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input = Input(shape=(1,), name="y_local")

        # Spatial features branch - concatenate in two stages for HLS compatibility
        xy_concat = Concatenate(name="xy_concat")([x_profile_input, y_profile_input])
        other_features = Concatenate(name="other_features")([xy_concat, y_local_input])
        other_dense = Dense(self.dense_units, activation="relu", name="other_dense")(other_features)

        # z_global branch
        z_dense = Dense(self.z_global_units, activation="relu", name="z_global_dense")(z_global_input)

        # Merge both branches
        merged = Concatenate(name="merged_features")([other_dense, z_dense])
        
        # Two more dense layers
        hidden = Dense(self.dense2_units, activation="relu", name="dense2")(merged)
        hidden = Dropout(self.dropout_rate, name="dropout1")(hidden)
        hidden = Dense(self.dense3_units, activation="relu", name="dense3")(hidden)
        
        # Output layer
        output = Dense(1, activation="tanh", name="output")(hidden)

        model = Model(
            inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input],
            outputs=output,
            name="model2_5_unquantized"
        )
        
        model.compile(
            optimizer=Adam(learning_rate=self.initial_lr),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"]
        )
        
        self.models["Unquantized"] = model
        print("✓ Unquantized Model2.5 built successfully")
        return model
    
    def makeQuantizedModel(self):
        """Build quantized Model2.5 variants for all bit configurations."""
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        for weight_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            print(f"Building Model2.5 {config_name}...")
            print(f"  - Spatial features branch: {self.dense_units} units ({weight_bits}-bit)")
            print(f"  - z_global branch: {self.z_global_units} units ({self.z_global_weight_bits}-bit)")

            # Quantizers
            weight_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
            z_weight_quantizer = quantized_bits(self.z_global_weight_bits, self.z_global_int_bits, alpha=1.0)

            # Input layers
            x_profile_input = Input(shape=(21,), name="x_profile")
            z_global_input = Input(shape=(1,), name="z_global")
            y_profile_input = Input(shape=(13,), name="y_profile")
            y_local_input = Input(shape=(1,), name="y_local")

            # Spatial features branch
            xy_concat = Concatenate(name="xy_concat")([x_profile_input, y_profile_input])
            other_features = Concatenate(name="other_features")([xy_concat, y_local_input])
            other_dense = QDense(
                self.dense_units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=weight_quantizer,
                name="other_dense"
            )(other_features)
            other_dense = QActivation("quantized_relu(8,0)", name="other_activation")(other_dense)

            # z_global branch
            z_dense = QDense(
                self.z_global_units,
                kernel_quantizer=z_weight_quantizer,
                bias_quantizer=z_weight_quantizer,
                name="z_global_dense"
            )(z_global_input)
            z_dense = QActivation("quantized_relu(8,0)", name="z_activation")(z_dense)

            # Merge both branches
            merged = Concatenate(name="merged_features")([other_dense, z_dense])
            
            # Two more dense layers
            hidden = QDense(
                self.dense2_units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=weight_quantizer,
                name="dense2"
            )(merged)
            hidden = QActivation("quantized_relu(8,0)", name="dense2_activation")(hidden)
            hidden = Dropout(self.dropout_rate, name="dropout1")(hidden)
            
            hidden = QDense(
                self.dense3_units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=weight_quantizer,
                name="dense3"
            )(hidden)
            hidden = QActivation("quantized_relu(8,0)", name="dense3_activation")(hidden)

            # Output layer
            output_dense = QDense(
                1,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=weight_quantizer,
                name="output"
            )(hidden)
            output = QActivation("quantized_tanh(8,0)", name="output_activation")(output_dense)

            model = Model(
                inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input],
                outputs=output,
                name=f"model2_5_{config_name}"
            )

            model.compile(
                optimizer=Adam(learning_rate=self.initial_lr),
                loss="binary_crossentropy",
                metrics=["binary_accuracy"],
                run_eagerly=True
            )

            self.models[config_name] = model
        
        print(f"✓ Built {len(self.bit_configs)} quantized Model2.5 variants")
    
    def makeQuantizedModelHyperParameterTuning(self, hp, weight_bits, int_bits):
        """Build quantized Model2.5 for hyperparameter tuning with progressive layer constraints."""
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        # Hyperparameter search space with progressive constraints
        spatial_units = hp.Int('spatial_units', min_value=16, max_value=150, step=16)
        z_global_units = hp.Int('z_global_units', min_value=2, max_value=16, step=2)
        
        # Calculate max size for dense2 (should not exceed concatenated size)
        concat_size = spatial_units + z_global_units
        dense2_max = min(256, concat_size)
        dense2_units = hp.Int('dense2_units', min_value=4, max_value=dense2_max, step=8)
        
        # Calculate max size for dense3 (should not exceed dense2)
        dense3_max = min(128, dense2_units)
        dense3_units = hp.Int('dense3_units', min_value=4, max_value=dense3_max, step=6)
        
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.3, step=0.05)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        # Quantizers
        weight_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
        z_weight_quantizer = quantized_bits(self.z_global_weight_bits, self.z_global_int_bits, alpha=1.0)

        # Input layers
        x_profile_input = Input(shape=(21,), name="x_profile")
        z_global_input = Input(shape=(1,), name="z_global")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input = Input(shape=(1,), name="y_local")

        # Spatial features branch
        xy_concat = Concatenate(name="xy_concat")([x_profile_input, y_profile_input])
        other_features = Concatenate(name="other_features")([xy_concat, y_local_input])
        other_dense = QDense(
            spatial_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="other_dense"
        )(other_features)
        other_dense = QActivation("quantized_relu(8,0)", name="other_activation")(other_dense)

        # z_global branch
        z_dense = QDense(
            z_global_units,
            kernel_quantizer=z_weight_quantizer,
            bias_quantizer=z_weight_quantizer,
            name="z_global_dense"
        )(z_global_input)
        z_dense = QActivation("quantized_relu(8,0)", name="z_activation")(z_dense)

        # Merge both branches
        merged = Concatenate(name="merged_features")([other_dense, z_dense])
        
        # Two more dense layers
        hidden = QDense(
            dense2_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="dense2"
        )(merged)
        hidden = QActivation("quantized_relu(8,0)", name="dense2_activation")(hidden)
        hidden = Dropout(dropout_rate, name="dropout1")(hidden)
        
        hidden = QDense(
            dense3_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="dense3"
        )(hidden)
        hidden = QActivation("quantized_relu(8,0)", name="dense3_activation")(hidden)

        # Output layer
        output_dense = QDense(
            1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="output"
        )(hidden)
        output = QActivation("quantized_tanh(8,0)", name="output_activation")(output_dense)

        model = Model(
            inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input],
            outputs=output,
            name=f"model2_5_quantized_{weight_bits}w{int_bits}i_hyperparameter_tuning"
        )

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly=True
        )

        return model
    
    def _calculate_model_parameters(self, hyperparams):
        """Calculate parameter counts for Model2.5 architecture."""
        spatial_units = hyperparams.get('spatial_units', self.dense_units)
        z_global_units = hyperparams.get('z_global_units', self.z_global_units)
        dense2_units = hyperparams.get('dense2_units', self.dense2_units)
        dense3_units = hyperparams.get('dense3_units', self.dense3_units)
        
        other_dim = 21 + 13 + 1  # x_profile + y_profile + y_local
        z_dim = 1

        # First layer: spatial features and z_global
        spatial_dense_params = (other_dim * spatial_units) + spatial_units
        z_dense_params = (z_dim * z_global_units) + z_global_units
        
        # Second layer: concatenated -> dense2_units
        concat_dim = spatial_units + z_global_units
        dense2_params = (concat_dim * dense2_units) + dense2_units
        
        # Third layer: dense2_units -> dense3_units
        dense3_params = (dense2_units * dense3_units) + dense3_units
        
        # Output layer: dense3_units -> 1
        output_params = dense3_units + 1

        total_params = spatial_dense_params + z_dense_params + dense2_params + dense3_params + output_params

        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(total_params),
            'non_trainable_parameters': 0
        }
    
    def runQuantizedHyperparameterTuning(
        self,
        bit_configs=None,
        max_trials=50,
        executions_per_trial=1,
        numEpochs=30,
        use_weighted_bkg_rej=False,
        bkg_rej_weights=None
    ):
        """
        Run hyperparameter tuning for quantized Model2.5.
        
        Args:
            bit_configs: List of (weight_bits, int_bits) tuples
            max_trials: Maximum number of trials
            executions_per_trial: Number of executions per trial
            numEpochs: Number of epochs for training
            use_weighted_bkg_rej: If True, optimize weighted background rejection
            bkg_rej_weights: Dict for BR weights, e.g. {0.95: 0.3, 0.98: 0.6, 0.99: 0.1}
            
        Returns:
            Dictionary mapping config_name to results
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized hyperparameter tuning")
        
        if bit_configs is None:
            bit_configs = self.bit_configs

        if bkg_rej_weights is None:
            bkg_rej_weights = {0.95: 0.3, 0.98: 0.6, 0.99: 0.1}

        def _bkg_rej_at_eff(y_true, y_score, target_eff):
            """Compute background rejection (1 - FPR) at fixed signal efficiency."""
            y_true = np.asarray(y_true).ravel()
            y_score = np.asarray(y_score).ravel()

            sig_scores = y_score[y_true == 1]
            bkg_scores = y_score[y_true == 0]

            if len(sig_scores) == 0 or len(bkg_scores) == 0:
                return np.nan

            threshold = np.quantile(sig_scores, 1.0 - target_eff)
            fpr = float(np.mean(bkg_scores >= threshold))
            return 1.0 - fpr
        
        print(f"Starting quantized hyperparameter tuning for Model2.5 with {len(bit_configs)} bit configurations...")
        if use_weighted_bkg_rej:
            print("Objective: val_weighted_bkg_rej")
            print(
                "  Weights: "
                f"BR95={bkg_rej_weights.get(0.95, 0.0):.3f}, "
                f"BR98={bkg_rej_weights.get(0.98, 0.0):.3f}, "
                f"BR99={bkg_rej_weights.get(0.99, 0.0):.3f}"
            )
        
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

            tuner_objective = (
                kt.Objective("val_weighted_bkg_rej", direction="max")
                if use_weighted_bkg_rej else "val_binary_accuracy"
            )
            objective_name = "val_weighted_bkg_rej" if use_weighted_bkg_rej else "val_binary_accuracy"
            
            # Create tuner with unique project name
            tuner = kt.RandomSearch(
                model_builder,
                objective=tuner_objective,
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                project_name=f"model2_5_quantized_{weight_bits}w{int_bits}i_hyperparameter_search",
                directory="./hyperparameter_tuning"
            )
            
            # Create directory for this configuration's models
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_dir = f"model2.5_{config_name}_hyperparameter_results_{timestamp}"
            os.makedirs(config_dir, exist_ok=True)
            print(f"\n✓ Created directory for {config_name} results: {config_dir}/")
            print(f"Models will be saved to this directory after each trial completes.\n")
            
            # Create a custom callback class that saves models after each trial
            class SaveModelAfterTrial(tf.keras.callbacks.Callback):
                def __init__(self, tuner, config_dir, config_name):
                    super().__init__()
                    self.tuner = tuner
                    self.config_dir = config_dir
                    self.config_name = config_name
                    self.saved_trials = set()
                
                def on_epoch_end(self, epoch, logs=None):
                    try:
                        completed_trials = [t for t in self.tuner.oracle.trials.values() 
                                          if t.status == 'COMPLETED' and t.trial_id not in self.saved_trials]
                        
                        for trial in completed_trials:
                            try:
                                model = self.tuner.load_model(trial)
                                
                                model_filename = os.path.join(self.config_dir, f"model_trial_{trial.trial_id}.h5")
                                model.save(model_filename)
                                
                                hyperparams_dict = trial.hyperparameters.values
                                hyperparams_filename = os.path.join(self.config_dir, f"hyperparams_trial_{trial.trial_id}.json")
                                with open(hyperparams_filename, 'w') as f:
                                    json.dump(hyperparams_dict, f, indent=4)
                                
                                self.saved_trials.add(trial.trial_id)
                                
                                print(f"\n✓ Trial {trial.trial_id} completed and saved to {model_filename}")
                                print(f"  {objective_name}: {trial.score:.4f}")
                                print(f"  Hyperparameters: {hyperparams_dict}\n")
                                
                            except Exception as e:
                                print(f"\n⚠ Warning: Failed to save model for trial {trial.trial_id}: {str(e)}\n")
                    except Exception as e:
                        pass

            class WeightedBackgroundRejectionCallback(tf.keras.callbacks.Callback):
                """Compute weighted BR objective at each epoch."""

                def __init__(self, validation_generator, weight_map):
                    super().__init__()
                    self.validation_generator = validation_generator
                    self.weight_map = weight_map
                    self.y_true = np.concatenate(
                        [np.asarray(y).ravel() for _, y in validation_generator],
                        axis=0
                    )

                def on_epoch_end(self, epoch, logs=None):
                    if logs is None:
                        logs = {}
                    y_pred = self.model.predict(self.validation_generator, verbose=0).ravel()

                    n = min(len(self.y_true), len(y_pred))
                    y_true_local = self.y_true[:n]
                    y_pred_local = y_pred[:n]

                    br95 = _bkg_rej_at_eff(y_true_local, y_pred_local, 0.95)
                    br98 = _bkg_rej_at_eff(y_true_local, y_pred_local, 0.98)
                    br99 = _bkg_rej_at_eff(y_true_local, y_pred_local, 0.99)

                    weighted = (
                        self.weight_map.get(0.95, 0.0) * br95 +
                        self.weight_map.get(0.98, 0.0) * br98 +
                        self.weight_map.get(0.99, 0.0) * br99
                    )

                    logs['val_bkg_rej_95'] = float(br95)
                    logs['val_bkg_rej_98'] = float(br98)
                    logs['val_bkg_rej_99'] = float(br99)
                    logs['val_weighted_bkg_rej'] = float(weighted)
                    print(
                        f"\n  [Weighted BR] BR95={br95:.4f}, BR98={br98:.4f}, "
                        f"BR99={br99:.4f}, weighted={weighted:.4f}"
                    )
            
            # Create callbacks
            save_callback = SaveModelAfterTrial(tuner, config_dir, config_name)
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                save_callback
            ]
            if use_weighted_bkg_rej:
                callbacks.insert(0, WeightedBackgroundRejectionCallback(self.validation_generator, bkg_rej_weights))
            
            # Run search
            print(f"Running hyperparameter search for {config_name}...")
            tuner.search(
                self.training_generator,
                validation_data=self.validation_generator,
                epochs=numEpochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Get completed trials
            all_trials = list(tuner.oracle.trials.values())
            num_trials = len(all_trials)
            
            completed_trials = []
            for trial in all_trials:
                if trial.status == 'COMPLETED' and trial.score is not None:
                    completed_trials.append(trial)
            
            completed_trials.sort(key=lambda t: t.score, reverse=True)
            
            all_models = []
            all_hyperparameters = []
            print(f"\nLoading models from {len(completed_trials)} completed trials (out of {num_trials} total)...")
            
            for idx, trial in enumerate(completed_trials):
                try:
                    model = tuner.load_model(trial)
                    all_models.append(model)
                    all_hyperparameters.append(trial.hyperparameters)
                    if idx == 0:
                        print(f"  ✓ Loaded best model (trial {trial.trial_id}, score={trial.score:.4f})")
                    elif idx < 5:
                        print(f"  ✓ Loaded model {idx+1} (trial {trial.trial_id}, score={trial.score:.4f})")
                except Exception as e:
                    print(f"  ⚠ Failed to load model for trial {trial.trial_id}: {str(e)}")
            
            if len(completed_trials) > 5:
                print(f"  ... (loaded {len(all_models)} models total)")
            
            if len(all_models) == 0:
                raise RuntimeError(f"Failed to load any models for {config_name}")
            
            # Generate ROC artifacts
            print(f"\nGenerating ROC artifacts for {config_name} models...")
            roc_dir = os.path.join(config_dir, "roc_analysis")
            os.makedirs(roc_dir, exist_ok=True)

            y_true_all = np.concatenate(
                [np.asarray(y).ravel() for _, y in self.validation_generator],
                axis=0
            )

            roc_rows = []
            for idx, model in enumerate(all_models):
                trial = completed_trials[idx]
                trial_id = trial.trial_id
                model_stem = f"model_trial_{trial_id}"

                try:
                    y_pred_all = model.predict(self.validation_generator, verbose=0).ravel()

                    n = min(len(y_true_all), len(y_pred_all))
                    y_true = y_true_all[:n]
                    y_pred = y_pred_all[:n]

                    fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=False)
                    roc_auc = auc(fpr, tpr)

                    # Save ROC data
                    roc_csv_path = os.path.join(roc_dir, f"{model_stem}_roc_data.csv")
                    pd.DataFrame({
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': thresholds
                    }).to_csv(roc_csv_path, index=False)

                    # Save ROC plot
                    roc_plot_path = os.path.join(roc_dir, f"{model_stem}_roc.png")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.4f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate', fontsize=12)
                    ax.set_ylabel('True Positive Rate', fontsize=12)
                    ax.set_title(f'ROC Curve: {model_stem}', fontsize=14, fontweight='bold')
                    ax.legend(loc="lower right")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)

                    br95 = _bkg_rej_at_eff(y_true, y_pred, 0.95)
                    br98 = _bkg_rej_at_eff(y_true, y_pred, 0.98)
                    br99 = _bkg_rej_at_eff(y_true, y_pred, 0.99)
                    weighted_br = (
                        bkg_rej_weights.get(0.95, 0.0) * br95 +
                        bkg_rej_weights.get(0.98, 0.0) * br98 +
                        bkg_rej_weights.get(0.99, 0.0) * br99
                    )

                    roc_rows.append({
                        'trial_id': trial_id,
                        'model_name': model_stem,
                        'auc': float(roc_auc),
                        'bkg_rej_95': float(br95),
                        'bkg_rej_98': float(br98),
                        'bkg_rej_99': float(br99),
                        'weighted_bkg_rej': float(weighted_br),
                        'roc_csv': os.path.join("roc_analysis", f"{model_stem}_roc_data.csv"),
                        'roc_plot': os.path.join("roc_analysis", f"{model_stem}_roc.png")
                    })
                    print(f"  ✓ ROC saved for trial {trial_id} (AUC={roc_auc:.4f})")
                except Exception as e:
                    print(f"  ⚠ Failed ROC generation for trial {trial_id}: {str(e)}")

            roc_summary_file = None
            if roc_rows:
                roc_metrics_df = pd.DataFrame(roc_rows).sort_values('weighted_bkg_rej', ascending=False)
                roc_summary_file = os.path.join(config_dir, "roc_metrics_summary.csv")
                roc_metrics_df.to_csv(roc_summary_file, index=False)
                print(f"✓ Saved ROC metrics summary to: {roc_summary_file}")
            
            # Save models and hyperparameters
            model_files = []
            hyperparams_files = []
            
            for idx, (model, hyperparams) in enumerate(zip(all_models, all_hyperparameters)):
                trial_id = completed_trials[idx].trial_id
                model_filename = os.path.join(config_dir, f"model_trial_{trial_id}.h5")
                hyperparams_filename = os.path.join(config_dir, f"hyperparams_trial_{trial_id}.json")
                
                if not os.path.exists(model_filename):
                    print(f"⚠ Model for trial {trial_id} was not saved during training, saving now...")
                    model.save(model_filename)
                    
                    hyperparams_dict = hyperparams.values
                    with open(hyperparams_filename, 'w') as f:
                        json.dump(hyperparams_dict, f, indent=4)
                
                model_files.append(model_filename)
                hyperparams_files.append(hyperparams_filename)
                
                if idx == 0:
                    print(f"✓ Best model (trial {trial_id}): {model_filename}")
                    print(f"  Best hyperparameters: {hyperparams.values}")
            
            print(f"✓ Total {len(model_files)} models saved to {config_dir}/")
            
            # Save trials summary
            summary_filename = os.path.join(config_dir, "trials_summary.json")
            trials_summary = []
            roc_by_trial_id = {row['trial_id']: row for row in roc_rows}
            
            for idx, hyperparams in enumerate(all_hyperparameters):
                trial = completed_trials[idx]
                trial_id = trial.trial_id
                
                trial_info = {
                    'trial_id': trial_id,
                    'rank': idx,
                    'model_file': f"model_trial_{trial_id}.h5",
                    'hyperparams_file': f"hyperparams_trial_{trial_id}.json",
                    'hyperparameters': hyperparams.values,
                    'is_best': (idx == 0),
                    'objective_name': objective_name
                }
                
                if trial.score is not None:
                    trial_info['objective_score'] = float(trial.score)
                    if use_weighted_bkg_rej:
                        trial_info['val_weighted_bkg_rej'] = float(trial.score)
                    else:
                        trial_info['val_accuracy'] = float(trial.score)

                if trial_id in roc_by_trial_id:
                    trial_info['roc'] = {
                        'auc': roc_by_trial_id[trial_id]['auc'],
                        'bkg_rej_95': roc_by_trial_id[trial_id]['bkg_rej_95'],
                        'bkg_rej_98': roc_by_trial_id[trial_id]['bkg_rej_98'],
                        'bkg_rej_99': roc_by_trial_id[trial_id]['bkg_rej_99'],
                        'weighted_bkg_rej': roc_by_trial_id[trial_id]['weighted_bkg_rej'],
                        'roc_csv': roc_by_trial_id[trial_id]['roc_csv'],
                        'roc_plot': roc_by_trial_id[trial_id]['roc_plot']
                    }
                
                param_metadata = self._calculate_model_parameters(hyperparams.values)
                trial_info.update(param_metadata)
                
                trials_summary.append(trial_info)
            
            with open(summary_filename, 'w') as f:
                json.dump(trials_summary, f, indent=4)
            print(f"✓ Saved trials summary to: {summary_filename}")
            
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
                'roc_summary_file': roc_summary_file,
                'num_trials': num_trials
            }
        
        print(f"\n{'='*60}")
        print(f"All quantized hyperparameter tuning completed!")
        print(f"{'='*60}")
        
        return all_results


def main():
    """Example usage of Model2.5 Standalone"""
    print("=== Model2.5 Standalone Example Usage ===")

    model25 = Model2_5_Standalone(
        tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData",
        dense_units=128,
        z_global_units=32,
        dense2_units=128,
        dense3_units=64,
        dropout_rate=0.1,
        initial_lr=1e-3,
        end_lr=1e-4,
        power=2,
        bit_configs=[(4, 0)],
        z_global_weight_bits=4,
        z_global_int_bits=0
    )

    results = model25.runQuantizedHyperparameterTuning(
        bit_configs=[(4, 0)],
        max_trials=8,
        executions_per_trial=1,
        numEpochs=15,
        use_weighted_bkg_rej=True,
        bkg_rej_weights={0.95: 0.3, 0.98: 0.6, 0.99: 0.1}
    )

    print("\nModel2.5 Standalone hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()
