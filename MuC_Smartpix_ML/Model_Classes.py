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
import OptimizedDataGenerator4 as ODG

class SmartPixModel(ABC):
    def __init__(self,
            tfRecordFolder: str = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
            nBits: list = None, # just for fractional bits, integer bits 
            loadModel: bool = False,
            modelPath: str = None, # Only include if you are loading a model
            ): 
        # Do we want to specify model, modelType, bitSize, etc.
        # Decide here if we want to load a pre-trained model or create a new one from scratch
        self.tfRecordFolder = tfRecordFolder
        self.modelName = "Base Model" # for other models, e.g., Model 1, Model 2, etc.
        self.models = {"Unquantized": None, "Quantized": None} # Maybe have a dictionary to store different versions of the model
        self.hyperparameterModel = None
        return
    
    def runAllStuff(self,):
        return
    
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
    
    @abstractmethod
    def buildModel(self):
        raise NotImplementedError("Subclasses should implement this method.")

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
    def trainModel(self): # in the input, specify the learning rate scheduler, etc.
        raise NotImplementedError("We need to write this code")

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
        Evaluate the trained Model2.
        
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
        
        print(f"✓ Model2 evaluation completed!")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        return self.evaluation_results