from abc import ABC, abstractmethod
# import all the necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay, ExponentialDecay, CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/local/d1/smartpixML/filtering_models/shuffling_data/") #TODO use the ODG from here
import OptimizedDataGenerator4_data_shuffled_bigData as ODG2
import pandas as pd
from datetime import datetime
sys.path.append("../ryan")
import OptimizedDataGenerator4 as ODG

class SmartPixModel(ABC):
    def __init__(self,
            tfRecordFolder: str = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
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
    
    def buildModel(self, model_type="unquantized"):
        """
        Build the specified model type.
        
        Args:
            model_type: "unquantized" or "quantized"
            bit_configs: List of bit configurations for quantized models --Actually now a class field/attribute
        """
        if model_type == "unquantized":
            return self.makeUnquantizedModel()
        elif model_type == "quantized":
            return self.makeQuantizedModel()
        else:
            raise ValueError("model_type must be 'unquantized' or 'quantized'")
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
    
    #dataset
    def trainModel(self, epochs=100, batch_size=32, learning_rate=None, 
                   save_best=True, early_stopping_patience=20,
                   run_eagerly = False,config_name = "Unquantized"):
        """
        Train the {self.modelName}.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training (not used, defined in data generator)
            learning_rate: Learning rate for optimizer (if None, uses polynomial decay)
            save_best: Whether to save the best model
            early_stopping_patience: Patience for early stopping
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
        self.models[config_name].compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly=run_eagerly
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
        self.histories[config_name] = self.models[config_name].fit(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✓ {self.modelName} {config_name} training completed!")
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
    def evaluate(self, test_generator=None,config_name = "Unquantized"):
        """
        Evaluate the trained {self.modelNam.
        
        Args:
            test_generator: Optional test data generator
        """
        if self.models[config_name] is None:
            raise ValueError("No model to evaluate. Train a model first.")
        
        # Use validation generator if no test generator provided
        eval_generator = test_generator if test_generator else self.validation_generator
        
        if eval_generator is None:
            self.loadTfRecords()
            eval_generator = self.validation_generator
        
        print(f"Evaluating {self.modelName}...")
        
        # Get predictions
        predictions = self.models[config_name].predict(eval_generator, verbose=1)
        
        # Get true labels
        true_labels = np.concatenate([y for _, y in eval_generator])
        
        # Calculate metrics
        test_loss, test_accuracy = self.models[config_name].evaluate(eval_generator, verbose=0)
        
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
        
        print(f"✓ {self.modelName} evaluation completed!")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
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
        print("2a. Building unquantized model...")
        self.buildModel("unquantized")
        
        print("2b. Training unquantized model...")
        self.trainModel(epochs=numEpochs, early_stopping_patience=15, save_best=False)
        
        print("2c. Evaluating unquantized model...")
        eval_results = self.evaluate()
        
        # Save non-quantized model
        print("2d. Saving unquantized model...")
        model_save_path = os.path.join(models_dir, f"{self.modelName}_unquantized.h5")
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

            
            # Get the quantized model
            # quantized_model = self.models[config_name]

            
            print(f"3c. Training {weight_bits}-bit quantized model...")
            self.trainModel(epochs=numEpochs,early_stopping_patience=15,
                            config_name=config_name,
                            learning_rate=None,run_eagerly=True,)
            # Compile and train the quantized model
            #deleted that code, because you have a function for it
            
            print(f"3d. Evaluating {weight_bits}-bit quantized model...")
            # Create another fresh validation generator for evaluation NO, don't need to
            evaluation_results = self.evaluate(test_generator = None, config_name = config_name)
            
            
            # Save quantized model
            print(f"3e. Saving {weight_bits}-bit quantized model...")
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
                'model_path': model_save_path
            })
            
            print(f'{weight_bits}-bit results: Acc={evaluation_results["test_accuracy"]:.4f}, AUC={evaluation_results["roc_auc"]:.4f}')
            
            # Plot quantized results
            print(f"3f. Plotting {weight_bits}-bit quantized results...")
            plot_dir_quant = os.path.join(plots_dir, f"{weight_bits}bit")
            self.plotModel(save_plots=True,output_dir=plot_dir_quant,config_name = config_name)
            # # Create a temporary {self.modelName} instance for plotting
        
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
        
        print(f"\n=== {self.modelName} Quantization Pipeline Completed! ===")
        print(f"All outputs saved to: {output_dir}/")
        print(f"  - Models: {models_dir}/")
        print(f"  - Plots: {plots_dir}/")
        print(f"  - Results CSV: {results_file}")
        
        return results