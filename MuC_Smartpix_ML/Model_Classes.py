from abc import ABC, abstractmethod
# import all the necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay, ExponentialDecay, CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import keras_tuner as kt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
# sys.path.append("/local/d1/smartpixML/filtering_models/shuffling_data/") #TODO use the ODG from here
sys.path.append("../MuC_Smartpix_Data_Production/tfRecords")
import OptimizedDataGenerator4_data_shuffled_bigData_NewFormat as ODG2
import pandas as pd
from datetime import datetime

class WarmupThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Two-phase schedule for quantized models:
      1) Warmup with conservative LR for a few epochs
      2) Slightly higher LR that decays polynomially to the final value
    """
    def __init__(self, warmup_steps, warmup_lr, boosted_lr, total_steps, end_lr, power=1.0):
        super().__init__()
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.warmup_lr = tf.cast(warmup_lr, tf.float32)
        self.boosted_lr = tf.cast(boosted_lr, tf.float32)
        self.total_steps = tf.cast(tf.maximum(total_steps, 1), tf.float32)
        self.end_lr = tf.cast(end_lr, tf.float32)
        self.power = tf.cast(power, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        decay_steps = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        progress = tf.minimum(1.0, tf.maximum(0.0, (step - self.warmup_steps) / decay_steps))
        decay_lr = (self.boosted_lr - self.end_lr) * tf.pow(1.0 - progress, self.power) + self.end_lr
        return tf.where(step < self.warmup_steps, self.warmup_lr, decay_lr)

    def get_config(self):
        return {
            "warmup_steps": float(self.warmup_steps.numpy()),
            "warmup_lr": float(self.warmup_lr.numpy()),
            "boosted_lr": float(self.boosted_lr.numpy()),
            "total_steps": float(self.total_steps.numpy()),
            "end_lr": float(self.end_lr.numpy()),
            "power": float(self.power.numpy())
        }


class GradientMonitor(Callback):
    """
    Callback to monitor gradient magnitudes during training.
    Helps detect vanishing/exploding gradients.
    """
    def __init__(self, log_freq=5, training_generator=None):
        super().__init__()
        self.log_freq = log_freq
        self.training_generator = training_generator
        self.gradient_stats = {
            'epoch': [],
            'mean_grad_norm': [],
            'max_grad_norm': [],
            'min_grad_norm': [],
            'layer_stats': {}
        }
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_freq == 0 or epoch == 0:
            if self.training_generator is None:
                return
            
            try:
                # Get a sample batch from the generator
                # Reset generator to get a fresh batch
                batch_data = next(iter(self.training_generator))
                
                # Handle different batch formats
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    # Standard format: (x, y)
                    x, y = batch_data
                elif isinstance(batch_data, dict):
                    # Model2 format: dict with features, need to extract y separately
                    # The generator yields (features_dict, y) but might be wrapped
                    # Try to get y from the generator's structure
                    x = batch_data
                    # For Model2, we need to get y from the generator differently
                    # Create a temporary iterator to get both x and y
                    temp_iter = iter(self.training_generator)
                    temp_batch = next(temp_iter)
                    if isinstance(temp_batch, tuple) and len(temp_batch) == 2:
                        x, y = temp_batch
                    else:
                        # If we can't get y easily, skip gradient monitoring for this epoch
                        return
                else:
                    return
                
                # Get gradients using GradientTape
                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred = self.model(x, training=True)
                    
                    # Compute loss
                    if hasattr(self.model, 'compiled_loss') and self.model.compiled_loss is not None:
                        loss = self.model.compiled_loss(y, y_pred)
                    else:
                        loss = tf.keras.losses.binary_crossentropy(y, y_pred)
                
                # Compute gradients
                trainable_vars = self.model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                
                # Filter out None gradients
                gradients = [g for g in gradients if g is not None]
                
                if len(gradients) == 0:
                    return
                
                # Calculate gradient norms
                grad_norms = []
                for g in gradients:
                    if g is not None:
                        norm = tf.norm(g)
                        if tf.is_tensor(norm):
                            grad_norms.append(norm.numpy())
                        else:
                            grad_norms.append(float(norm))
                
                if len(grad_norms) == 0:
                    return
                
                mean_norm = np.mean(grad_norms)
                max_norm = np.max(grad_norms)
                min_norm = np.min(grad_norms)
                
                # Store statistics
                self.gradient_stats['epoch'].append(epoch)
                self.gradient_stats['mean_grad_norm'].append(float(mean_norm))
                self.gradient_stats['max_grad_norm'].append(float(max_norm))
                self.gradient_stats['min_grad_norm'].append(float(min_norm))
                
                # Print warning if gradients are vanishing
                if mean_norm < 1e-6:
                    print(f"\n⚠ WARNING: Vanishing gradients detected at epoch {epoch}!")
                    print(f"   Mean gradient norm: {mean_norm:.2e}")
                elif max_norm > 100:
                    print(f"\n⚠ WARNING: Large gradients detected at epoch {epoch}!")
                    print(f"   Max gradient norm: {max_norm:.2e}")
                
                # Log gradient stats
                print(f"\n[Gradient Monitor] Epoch {epoch}:")
                print(f"  Mean grad norm: {mean_norm:.2e}")
                print(f"  Max grad norm: {max_norm:.2e}")
                print(f"  Min grad norm: {min_norm:.2e}")
            except Exception as e:
                # Silently skip if gradient computation fails
                pass

class SmartPixModel(ABC):
    def __init__(self,                 
            # tfRecordFolder: str = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
            tfRecordFolder: str = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData",
            nBits: list = None, # just for fractional bits, integer bits 
            loadModel: bool = False,
            modelPath: str = None, # Only include if you are loading a model
            # dropout_rate: float = 0.1,
            initial_lr: float = 1e-3,
            end_lr: float = 1e-4,
            power: int = 2,
            bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)],  # Test 16, 8, 6, 4, 3, and 2-bit quantization
            ): 
        # Do we want to specify model, modelType, bitSize, etc.
        # Decide here if we want to load a pre-trained model or create a new one from scratch
        self.tfRecordFolder = tfRecordFolder
        self.modelName = "Base Model" # for other models, e.g., Model 1, Model 2, etc.
        self.models = {"Unquantized": None, "Quantized": None} # Maybe have a dictionary to store different versions of the model
        self.hyperparameterModel = None

        # Learning rate parameters
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.power = power
        return
    
    
    def loadTfRecords(self):
        """Load TFRecords using OptimizedDataGenerator4 for features."""
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
        
        self.training_generator = ODG2.OptimizedDataGeneratorDataShuffledBigData(
        # self.training_generator = ODG.OptimizedDataGenerator(
            load_records=True, 
            tf_records_dir=trainDir, 
            x_feature_description=self.x_feature_description,
            batch_size=batch_size
        )
        
        self.validation_generator = ODG2.OptimizedDataGeneratorDataShuffledBigData(
        # self.validation_generator = ODG.OptimizedDataGenerator(
            load_records=True, 
            tf_records_dir=valDir, 
            x_feature_description=self.x_feature_description,
            batch_size=batch_size
        )
        
        print(f"Training generator length: {len(self.training_generator)}")
        print(f"Validation generator length: {len(self.validation_generator)}")
        return 
    
    @abstractmethod
    def makeUnquantizedModel():
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def makeUnquatizedModelHyperParameterTuning(hp):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def makeQuantizedModel():
        raise NotImplementedError("Subclasses should implement this method.")

    
    def runHyperparameterTuning(self):
        tuner = kt.RandomSearch(
            self.makeUnquatizedModelHyperParameterTuning(),
            objective           = "val_binary_accuracy",
            max_trials          = 120,
            executions_per_trial = 2,
            project_name        = "new_hyperparam_search"
        )
        raise NotImplementedError("Subclasses should implement this method.")
    
    def buildModel(self, model_type="Unquantized"):
        """
        Build the specified model type.
        
        Args:
            model_type: "Unquantized" or "quantized"
            bit_configs: List of bit configurations for quantized models --Actually now a class field/attribute
        """
        if model_type == "Unquantized":
            return self.makeUnquantizedModel()
        elif model_type == "quantized":
            return self.makeQuantizedModel()
        else:
            raise ValueError("model_type must be 'Unquantized' or 'quantized'")
    """
    config_name = Unquantized or 
    config_name = f"quantized_{total_bits}w{int_bits}i"
    """
    def loadModel(self, file_path: str,config_name = "Unquantized"):
        self.models[config_name]=tf.keras.models.load_model(file_path, compile=False)

    """
    config_name = Unquantized or 
    config_name = f"quantized_{total_bits}w{int_bits}i"
    """
    def saveModel(self, file_path = None,overwrite=False,config_name = "Unquantized"):
        if file_path == None:
            file_path = Path(f'./{self.modelName}.keras').resolve()
        if not overwrite and os.path.exists(file_path):
            raise Exception("Model exists. To overwrite existing saved model, set overwrite to True.")
        self.models[config_name].save(file_path)
    
    def warmStartQuantizedModel(self, quantized_config_name, source_config_name="Unquantized"):
        """
        Warm-start a quantized model by copying weights from an Unquantized (or other) model.
        This implements quantization-aware fine-tuning by initializing the quantized model
        with pre-trained weights.
        
        Args:
            quantized_config_name: Name of the quantized model config (e.g., "quantized_4w0i")
            source_config_name: Name of the source model to copy from (default: "Unquantized")
        
        Note:
            - Only copies weights for layers with matching names
            - QDense layers can receive weights from Dense layers
            - Biases are copied if they exist in both models
            - Prints a summary of which layers were successfully copied
        """
        if self.models[source_config_name] is None:
            raise ValueError(f"Source model '{source_config_name}' not found. Train it first.")
        if self.models[quantized_config_name] is None:
            raise ValueError(f"Quantized model '{quantized_config_name}' not found. Build it first.")
        
        source_model = self.models[source_config_name]
        target_model = self.models[quantized_config_name]
        
        print(f"\n{'='*60}")
        print(f"Warm-starting {quantized_config_name} from {source_config_name}")
        print(f"{'='*60}")
        
        # Create a mapping of layer names to layers for the source model
        source_layers = {layer.name: layer for layer in source_model.layers}
        
        copied_layers = []
        skipped_layers = []
        
        for target_layer in target_model.layers:
            layer_name = target_layer.name
            
            # Skip input, concatenate, dropout, and activation layers (no weights)
            if len(target_layer.get_weights()) == 0:
                continue
            
            # Try to find matching source layer
            if layer_name in source_layers:
                source_layer = source_layers[layer_name]
                source_weights = source_layer.get_weights()
                
                # Check if weights are compatible
                target_weights = target_layer.get_weights()
                if len(source_weights) == len(target_weights):
                    # Check shapes match
                    shapes_match = all(sw.shape == tw.shape 
                                      for sw, tw in zip(source_weights, target_weights))
                    
                    if shapes_match:
                        target_layer.set_weights(source_weights)
                        copied_layers.append(layer_name)
                        print(f"  ✓ Copied weights: {layer_name} (shape: {source_weights[0].shape})")
                    else:
                        skipped_layers.append((layer_name, "shape mismatch"))
                        print(f"  ⚠ Skipped {layer_name}: shape mismatch")
                else:
                    skipped_layers.append((layer_name, "weight count mismatch"))
                    print(f"  ⚠ Skipped {layer_name}: weight count mismatch")
            else:
                skipped_layers.append((layer_name, "not found in source"))
                print(f"  ⚠ Skipped {layer_name}: not found in source model")
        
        print(f"\n{'='*60}")
        print(f"Warm-start summary:")
        print(f"  ✓ Copied: {len(copied_layers)} layers")
        print(f"  ⚠ Skipped: {len(skipped_layers)} layers")
        print(f"{'='*60}\n")
        
        return copied_layers, skipped_layers
    
    #dataset
    def trainModel(self, epochs=100, batch_size=32, learning_rate=None, 
                   save_best=True, early_stopping_patience=20,
                   run_eagerly = False, config_name = "Unquantized", clipnorm=None):
        """
        Train the {self.modelName}.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training (not used, defined in data generator)
            learning_rate: Learning rate for optimizer (if None, uses polynomial decay)
            save_best: Whether to save the best model
            early_stopping_patience: Patience for early stopping (<=0 disables it)
            run_eagerly: Whether to run model in eager mode (required for QKeras)
            config_name: Name of model configuration to train
            clipnorm: Gradient clipping by global norm. If None, no clipping. 
                     Recommended: 1.0 for quantized models to prevent gradient explosions.
        """
        if self.models[config_name] is None:
            raise ValueError("Model not built. Call buildModel() first.")
        
        print(self.models[config_name].summary())

        
        if self.training_generator is None:
            self.loadTfRecords()
        
        print(f"Training {self.modelName} for {epochs} epochs...")
        
        # Setup learning rate schedule
        if learning_rate is None:
            from tensorflow.keras.optimizers.schedules import PolynomialDecay
            # Calculate decay_steps based on actual training configuration
            steps_per_epoch = len(self.training_generator)
            decay_steps = epochs * steps_per_epoch
            
            # Quantized models get a warmup + boosted LR before decaying
            is_quantized = config_name.startswith("quantized_")
            if is_quantized:
                warmup_epochs = max(3, min(6, epochs // 10))  # ~10% of training
                warmup_epochs = min(warmup_epochs, max(1, epochs - 1))  # keep warmup < total epochs
                warmup_steps = warmup_epochs * steps_per_epoch
                warmup_lr = self.initial_lr * 0.4   # conservative start (e.g., 4e-4)
                boosted_lr = min(self.initial_lr * 0.9, self.initial_lr)  # slight boost (e.g., 9e-4)
                end_lr = self.end_lr * 0.1          # final fine-tune LR (e.g., 1e-5)
                
                print("Quantized model detected - using warmup + boosted polynomial decay")
                print(f"  Warmup: {warmup_epochs} epochs @ {warmup_lr:.2e}")
                print(f"  Boosted LR after warmup: {boosted_lr:.2e} (decays to {end_lr:.2e})")
                print(f"Learning rate decay: {decay_steps} steps ({epochs} epochs × {steps_per_epoch} steps/epoch)")
                
                lr_schedule = WarmupThenDecay(
                    warmup_steps=warmup_steps,
                    warmup_lr=warmup_lr,
                    boosted_lr=boosted_lr,
                    total_steps=decay_steps,
                    end_lr=end_lr,
                    power=self.power
                )
            else:
                initial_lr = self.initial_lr
                end_lr = self.end_lr
                print(f"Learning rate decay: {decay_steps} steps ({epochs} epochs × {steps_per_epoch} steps/epoch)")
                lr_schedule = PolynomialDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=decay_steps,
                    end_learning_rate=end_lr,
                    power=self.power
                )
            optimizer = Adam(learning_rate=lr_schedule, clipnorm=clipnorm)
        else:
            optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        
        # Compile model
        self.models[config_name].compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly=run_eagerly
        )
        
        # Create callbacks
        callbacks = []
        
        # Add gradient monitoring callback (pass training generator for batch access)
        gradient_monitor = GradientMonitor(log_freq=5, training_generator=self.training_generator)
        callbacks.append(gradient_monitor)
        
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
        self.histories[config_name] = self.models[config_name].fit(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✓ {self.modelName} {config_name} training completed!")
        
        # Print gradient statistics summary
        if hasattr(gradient_monitor, 'gradient_stats') and len(gradient_monitor.gradient_stats['epoch']) > 0:
            print(f"\n[Gradient Statistics Summary for {config_name}]")
            final_mean = gradient_monitor.gradient_stats['mean_grad_norm'][-1]
            final_max = gradient_monitor.gradient_stats['max_grad_norm'][-1]
            final_min = gradient_monitor.gradient_stats['min_grad_norm'][-1]
            print(f"  Final mean gradient norm: {final_mean:.2e}")
            print(f"  Final max gradient norm: {final_max:.2e}")
            print(f"  Final min gradient norm: {final_min:.2e}")
            
            if final_mean < 1e-6:
                print(f"  ⚠ WARNING: Very small gradients detected - may indicate vanishing gradient problem!")
        
        return self.histories[config_name]
  
    #plot based on history
    def plotModel(self, save_plots=True, output_dir="./plots",config_name = "Unquantized"):
        """
        Plot training history and evaluation results.
        
        Args:
            save_plots: Whether to save plots to disk
            output_dir: Directory to save plots
        """
        if self.histories[config_name] is None:
            print("No training history available. Train a model first.")
            return
        
        # Create output directory
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.histories[config_name].history['binary_accuracy'], label='Training')
        axes[0].plot(self.histories[config_name].history['val_binary_accuracy'], label='Validation')
        axes[0].set_title(f'{self.modelName} Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.histories[config_name].history['loss'], label='Training')
        axes[1].plot(self.histories[config_name].history['val_loss'], label='Validation')
        axes[1].set_title(f'{self.modelName} Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/{self.modelName}_training_history.png", dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {output_dir}/{self.modelName}_training_history.png")
        
        plt.show()
        
        # Plot ROC curve if evaluation results available
        if self.evaluation_results is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(self.evaluation_results['fpr'], self.evaluation_results['tpr'], 
                    label=f"ROC Curve (AUC = {self.evaluation_results['roc_auc']:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{self.modelName} ROC Curve')
            plt.legend()
            plt.grid(True)
            
            if save_plots:
                plt.savefig(f"{output_dir}/{self.modelName}_roc_curve.png", dpi=300, bbox_inches='tight')
                print(f"ROC curve plot saved to {output_dir}/{self.modelName}_roc_curve.png")
            
            plt.show()

    # Evaluate the model
    def compute_background_rejection_at_signal_eff(self, fpr, tpr, target_signal_efficiencies=[0.90, 0.98, 0.99]):
        """
        Compute background rejection at specified signal efficiencies.
        
        Background rejection = 1 - FPR, where FPR is the false positive rate.
        
        Args:
            fpr: False positive rate array from ROC curve
            tpr: True positive rate array (signal efficiency) from ROC curve
            target_signal_efficiencies: List of target signal efficiencies
        
        Returns:
            dict: Background rejection values and FPR at each target signal efficiency
        """
        results = {}
        
        for sig_eff in target_signal_efficiencies:
            # Find the index where TPR >= target signal efficiency
            idx = np.where(tpr >= sig_eff)[0]
            
            if len(idx) == 0:
                # Model doesn't reach target signal efficiency
                results[f'bkg_rej_at_{int(sig_eff*100)}pct'] = None
                results[f'fpr_at_{int(sig_eff*100)}pct'] = None
            else:
                # Use the first index where we reach the target
                # This gives us the operating point at the target signal efficiency
                fpr_at_target = fpr[idx[0]]
                
                # Background rejection = 1 - FPR
                bg_rej = 1.0 - fpr_at_target
                
                results[f'bkg_rej_at_{int(sig_eff*100)}pct'] = float(bg_rej)
                results[f'fpr_at_{int(sig_eff*100)}pct'] = float(fpr_at_target)
        
        return results
    
    def evaluate(self, test_generator=None, config_name="Unquantized", signal_efficiencies=[0.90, 0.98, 0.99]):
        """
        Evaluate the trained model with background rejection metrics.
        
        Args:
            test_generator: Optional test data generator
            config_name: Name of the model configuration to evaluate
            signal_efficiencies: List of target signal efficiencies for background rejection
        """
        if self.models[config_name] is None:
            raise ValueError("No model to evaluate. Train a model first.")
        
        # Use validation generator if no test generator provided
        eval_generator = test_generator if test_generator else self.validation_generator
        
        if eval_generator is None:
            self.loadTfRecords()
            eval_generator = self.validation_generator
        
        print(f"Evaluating {self.modelName} [{config_name}]...")
        
        # Get predictions
        predictions = self.models[config_name].predict(eval_generator, verbose=1)
        
        # Get true labels
        true_labels = np.concatenate([y for _, y in eval_generator])
        
        # Calculate metrics
        test_loss, test_accuracy = self.models[config_name].evaluate(eval_generator, verbose=0)
        
        # Calculate ROC AUC
        fpr, tpr, thresholds = roc_curve(true_labels, predictions.ravel())
        roc_auc = auc(fpr, tpr)
        
        # Compute background rejection at specific signal efficiencies
        bkg_rejection_metrics = self.compute_background_rejection_at_signal_eff(
            fpr, tpr, signal_efficiencies
        )
        
        # Store results
        self.evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'roc_auc': float(roc_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            **bkg_rejection_metrics  # Add background rejection metrics
        }
        
        print(f"✓ {self.modelName} evaluation completed!")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Print background rejection metrics
        print(f"\n  Background Rejection Metrics (Bkg Rej = 1 - FPR):")
        for sig_eff in signal_efficiencies:
            key = f'bkg_rej_at_{int(sig_eff*100)}pct'
            fpr_key = f'fpr_at_{int(sig_eff*100)}pct'
            
            bg_rej = bkg_rejection_metrics[key]
            fpr_val = bkg_rejection_metrics[fpr_key]
            
            if bg_rej is None:
                print(f"    @ {sig_eff:.0%} signal efficiency: NOT REACHED")
            else:
                print(f"    @ {sig_eff:.0%} signal efficiency: Bkg Rej = {bg_rej:.4f} (FPR = {fpr_val:.6f})")
        
        return self.evaluation_results
    
    def runAllStuff(self,numEpochs = 6):
        """
        Run the complete {self.modelName} pipeline: build, train, evaluate, and plot for both quantized and non-quantized models.
        """
        print(f"=== Running Complete {self.modelName} Pipeline with Quantization Testing ===")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.modelName}_results_{timestamp}"
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
        print("2a. Building Unquantized model...")
        self.buildModel("Unquantized")
        
        print("2b. Training Unquantized model...")
        self.trainModel(epochs=numEpochs, early_stopping_patience=15, save_best=False)
        
        print("2c. Evaluating Unquantized model...")
        eval_results = self.evaluate()
        
        # Save non-quantized model
        print("2d. Saving Unquantized model...")
        model_save_path = os.path.join(models_dir, f"{self.modelName}_Unquantized.h5")
        self.saveModel(file_path = model_save_path, config_name = "Unquantized")
        print(f"Unquantized model saved to: {model_save_path}")
        
        # Store non-quantized results
        results.append({
            'model_type': 'non_quantized',
            'weight_bits': 'N/A',
            'integer_bits': 'N/A',
            'test_accuracy': eval_results['test_accuracy'],
            'test_loss': eval_results['test_loss'],
            'roc_auc': eval_results['roc_auc'],
            'bkg_rej_90pct': eval_results.get('bkg_rej_at_90pct'),
            'bkg_rej_98pct': eval_results.get('bkg_rej_at_98pct'),
            'bkg_rej_99pct': eval_results.get('bkg_rej_at_99pct'),
            'model_path': model_save_path
        })
        
        print(f"Non-quantized results: Acc={eval_results['test_accuracy']:.4f}, AUC={eval_results['roc_auc']:.4f}")
        
        # Plot non-quantized results
        print("2e. Plotting non-quantized results...")
        plot_dir_unquant = os.path.join(plots_dir, "non_quantized")
        self.plotModel(save_plots=True, output_dir=plot_dir_unquant)
        
        # Test quantized models
        
        
        for weight_bits, int_bits in self.bit_configs:
            print(f"\n3. Testing {weight_bits}-bit Quantized Model...")
            
            # Create completely fresh data generators for each quantized model to avoid state corruption
            print(f"3a. Creating fresh data generators...")
            self.loadTfRecords()
            print("I'm not sure that's actually necessary")
            #TODO: See what happens without this line
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            
            print(f"3b. Building {weight_bits}-bit quantized model...")
            # self.buildModel("quantized", bit_configs=[(weight_bits, int_bits)])
            self.buildModel("quantized")
            print(self.models)

            
            # Warm-start: Copy weights from Unquantized model to quantized model
            print(f"3c. Warm-starting {weight_bits}-bit quantized model from Unquantized model...")
            self.warmStartQuantizedModel(config_name, source_config_name="Unquantized")
            
            print(f"3d. Training {weight_bits}-bit quantized model...")
            self.trainModel(epochs=numEpochs,early_stopping_patience=0,
                            config_name=config_name,
                            learning_rate=None,run_eagerly=True,clipnorm=1.0)
            # Compile and train the quantized model
            #deleted that code, because you have a function for it
            
            print(f"3e. Evaluating {weight_bits}-bit quantized model...")
            # Create another fresh validation generator for evaluation NO, don't need to
            evaluation_results = self.evaluate(test_generator = None, config_name = config_name)
            
            
            # Save quantized model
            print(f"3f. Saving {weight_bits}-bit quantized model...")
            model_save_path = os.path.join(models_dir, f"{self.modelName}_quantized_{weight_bits}bit.h5")
            self.saveModel(file_path = model_save_path, config_name = config_name)
            # quantized_model.save(model_save_path)
            print(f"{weight_bits}-bit model saved to: {model_save_path}")
            
            # Store quantized results
            results.append({
                'model_type': 'quantized',
                'weight_bits': weight_bits,
                'integer_bits': int_bits,
                'test_accuracy': evaluation_results["test_accuracy"],
                'test_loss': evaluation_results["test_loss"],
                'roc_auc': evaluation_results["roc_auc"],
                'bkg_rej_90pct': evaluation_results.get('bkg_rej_at_90pct'),
                'bkg_rej_98pct': evaluation_results.get('bkg_rej_at_98pct'),
                'bkg_rej_99pct': evaluation_results.get('bkg_rej_at_99pct'),
                'model_path': model_save_path
            })
            
            print(f'{weight_bits}-bit results: Acc={evaluation_results["test_accuracy"]:.4f}, AUC={evaluation_results["roc_auc"]:.4f}')
            
            # Plot quantized results
            print(f"3g. Plotting {weight_bits}-bit quantized results...")
            plot_dir_quant = os.path.join(plots_dir, f"{weight_bits}bit")
            self.plotModel(save_plots=True,output_dir=plot_dir_quant,config_name = config_name)
            # # Create a temporary {self.modelName} instance for plotting
        
        # Create results summary
        results.sort(key=lambda x: x['test_accuracy'], reverse=True)
        print("\n4. Results Summary:")
        print("=" * 120)
        print(f"{'Model Type':<15} {'Bits':<8} {'Accuracy':<10} {'Loss':<10} {'ROC AUC':<10} {'BkgRej@90%':<12} {'BkgRej@98%':<12} {'BkgRej@99%':<12}")
        print("-" * 120)
        
        for result in results:
            model_type = result['model_type']
            bits = result.get('weight_bits', 'N/A')
            acc = result['test_accuracy']
            loss = result['test_loss']
            auc_score = result['roc_auc']
            br90 = result.get('bkg_rej_90pct')
            br98 = result.get('bkg_rej_98pct')
            br99 = result.get('bkg_rej_99pct')
            
            br90_str = f"{br90:.4f}" if br90 is not None else "N/A"
            br98_str = f"{br98:.4f}" if br98 is not None else "N/A"
            br99_str = f"{br99:.4f}" if br99 is not None else "N/A"
            
            print(f"{model_type:<15} {bits:<8} {acc:<10.4f} {loss:<10.4f} {auc_score:<10.4f} {br90_str:<12} {br98_str:<12} {br99_str:<12}")
        
        # Find best configuration
        best_result = results[0]
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
        
        print(f"\n=== {self.modelName} Quantization Pipeline Completed! ===")
        print(f"All outputs saved to: {output_dir}/")
        print(f"  - Models: {models_dir}/")
        print(f"  - Plots: {plots_dir}/")
        print(f"  - Results CSV: {results_file}")
        
        return results