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
import pandas as pd
from datetime import datetime
import json
from datetime import datetime
import keras_tuner as kt


sys.path.append(str(Path.cwd().parents[0]))

from MuC_Smartpix_ML.Model_Classes import SmartPixModel
print(SmartPixModel)


class SaveModelRandomSearch(kt.RandomSearch):
    def __init__(self, *args, save_dir=None, objective_name="val_binary_accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        if save_dir is None:
            raise ValueError("save_dir must be provided")
        self.save_dir = os.path.abspath(save_dir)
        self.objective_name = objective_name
        os.makedirs(self.save_dir, exist_ok=True)
        print("✅ Saving trial artifacts to:", self.save_dir)

    def run_trial(self, trial, *args, **kwargs):
        # Grab callbacks passed into tuner.search(...)
        callbacks = kwargs.pop("callbacks", [])
        callbacks = list(callbacks) if callbacks is not None else []

        # Save best weights for THIS trial directly into save_dir
        weights_path = os.path.join(self.save_dir, f"trial_{trial.trial_id}_best.weights.h5")
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor=self.objective_name,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        )
        callbacks.append(ckpt)
        kwargs["callbacks"] = callbacks

        # Run training for the trial
        history = super().run_trial(trial, *args, **kwargs)

        # Rebuild model, load best weights, save full model
        model = self.hypermodel.build(trial.hyperparameters)

        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            print(f"⚠️ No weights checkpoint found for trial {trial.trial_id} at {weights_path}")
            return history

        model_path = os.path.join(self.save_dir, f"model_trial_{trial.trial_id}.h5")
        model.save(model_path)

        # Save hyperparams + score
        hp_path = os.path.join(self.save_dir, f"hyperparams_trial_{trial.trial_id}.json")
        with open(hp_path, "w") as f:
            json.dump(trial.hyperparameters.values, f, indent=4)

        score_path = os.path.join(self.save_dir, f"score_trial_{trial.trial_id}.json")
        with open(score_path, "w") as f:
            json.dump(
                {
                    "trial_id": trial.trial_id,
                    "score": getattr(trial, "score", None),
                    "objective": self.objective_name,
                },
                f,
                indent=4,
            )

        print(f"\n✓ Saved trial {trial.trial_id}")
        print(f"  model: {model_path}")
        print(f"  best weights: {weights_path}")
        print(f"  HP: {trial.hyperparameters.values}\n")

        return history



class Model1(SmartPixModel):
    def __init__(self,
            tfRecordFolder: str = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData",
            # tfRecordFolder: str = "/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
            nBits: list = None, # just for fractional bits, integer bits
                                ## number of bits is the number of bits for each quantized model and then
                                ## run training should make one model for each bit size
            loadModel: bool = False,
            modelPath: str = None, # Only include if you are loading a model
                        # dropout_rate: float = 0.1,
            initial_lr: float = 1e-3,
            end_lr: float = 1e-4,
            power: int = 2,
            bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
            ): 
        self.tfRecordFolder = tfRecordFolder
        self.modelName = "Model1" # for other models, e.g., Model 1, Model 2, etc.
        # self.model = None
        self.histories = {}
        self.models = {"Unquantized": None}
        self.bit_configs = bit_configs
        for total_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{total_bits}w{int_bits}i"
            self.models[config_name] = None
        # self.quantized_model = None
        self.hyperparameterModel = None
        self.training_generator = None
        self.validation_generator = None
        self.x_feature_description: list = ['z_global','x_size', 'y_size', 'y_local']
        # Learning rate parameters
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.power = power
        return
 


    def makeUnquantizedModel(self):
        ## here i will be making a 4-layer neural network 
        ## Model 1: z-global, x size, y size, y local


        ## define the inputs
        input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
        input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
        input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
        input4 = tf.keras.layers.Input(shape=(1,), name="y_local")

        ## concatenate the inputs into one layer
        inputList = [input1, input2, input3, input4]
        inputs = tf.keras.layers.Concatenate()(inputList)


        ## here i will add the layers 

        stack1 = tf.keras.layers.Dense(2,activation='relu')(inputs)
        stack2 = tf.keras.layers.Dense(2,activation='relu')(stack1)
        output = tf.keras.layers.Dense(1,activation='sigmoid')(stack2)

        self.models["Unquantized"] = tf.keras.Model(inputs=inputList, outputs=output)


    def makeUnquatizedModelHyperParameterTuning(self):
        def model_builder(hp):
            # ── B) Architecture hyperparams ──────────────────────────────────────────
            # separately tune rows and cols

            row1nodes      = hp.Int("1",   1, 30, step=1)
            row2nodes      = hp.Int("2",   1, 30, step=1)
            row3nodes      = hp.Int("3",   1, 30, step=1)
            row4nodes      = hp.Int("4",   1, 30, step=1)
            row5nodes      = hp.Int("5",   1, 30, step=1)



            input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
            input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
            input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
            input4 = tf.keras.layers.Input(shape=(1,), name="y_local")

            ## concatenate the inputs into one layer
            inputList = [input1, input2, input3, input4]
            inputs = tf.keras.layers.Concatenate()(inputList)


            ## here i will add the layers 

            # layer 1
            x = tf.keras.layers.Dense(row1nodes,activation='relu')(inputs)
            x = tf.keras.layers.Dense(row5nodes, activation='relu')(x)
            output = tf.keras.layers.Dense(1,activation='sigmoid')(x)

            model = tf.keras.Model(inputs=inputList, outputs=output)

            model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly  = True 
            )
            return model

        tuner = kt.RandomSearch(
        model_builder, 
        objective           = "val_binary_accuracy",
        max_trials          = 120,
        executions_per_trial = 2,
        project_name        = "hp_search_1_30"
        )

        tuner.search(
            self.training_generator,
            validation_data = self.validation_generator,
            epochs          = 110,
            callbacks       = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )


    def makeUnquatizedModelHyperParameterTuning5(self):
        def model_builder(hp):
            # ── B) Architecture hyperparams ──────────────────────────────────────────
            # separately tune rows and cols

            rownodes      = hp.Int("1",   2, 11, step=1)
        
            input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
            input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
            input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
            input4 = tf.keras.layers.Input(shape=(1,), name="y_local")
            inputList = [input1, input2, input3, input4]


            x_concat1 = tf.keras.layers.Concatenate()([input1,input2])
            x_concat2 = tf.keras.layers.Concatenate()([x_concat1,input3])
            x_concat3 = tf.keras.layers.Concatenate()([x_concat2,input4])
            x=x_concat3


            ## here i will add the layers 

            # layer 1
            x = tf.keras.layers.Dense(rownodes,activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)

            output = tf.keras.layers.Dense(1,activation='sigmoid')(x)

            model = tf.keras.Model(inputs=inputList, outputs=output)

            model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly  = True 
            )
            return model

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"{self.modelName.lower()}_unquantized_hp5rows_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n✓ Trial artifacts will be saved in: {save_dir}/\n")

        tuner = SaveModelRandomSearch(
            hypermodel= model_builder,
            objective="val_binary_accuracy",
            max_trials=120,
            executions_per_trial=2,
            project_name="hp_search_5rows_matching",
            directory="./hyperparameter_tuning_5",   # keep KT logs in one place
            overwrite=True,                        # avoid weird resume behavior
            save_dir=save_dir,
            objective_name="val_binary_accuracy",
        )

        tuner.search(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=110,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
        )
        
        return tuner, save_dir


    def makeUnquatizedModelHyperParameterTuning4(self):
        def model_builder(hp):
            # ── B) Architecture hyperparams ──────────────────────────────────────────
            # separately tune rows and cols

            rownodes      = hp.Int("1",   2, 11, step=1)
        
            input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
            input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
            input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
            input4 = tf.keras.layers.Input(shape=(1,), name="y_local")
            inputList = [input1, input2, input3, input4]


            x_concat1 = tf.keras.layers.Concatenate()([input1,input2])
            x_concat2 = tf.keras.layers.Concatenate()([x_concat1,input3])
            x_concat3 = tf.keras.layers.Concatenate()([x_concat2,input4])
            x=x_concat3


            ## here i will add the layers 

            # layer 1
            x = tf.keras.layers.Dense(rownodes,activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)
            output = tf.keras.layers.Dense(1,activation='sigmoid')(x)

            model = tf.keras.Model(inputs=inputList, outputs=output)

            model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly  = True 
            )
            return model

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"{self.modelName.lower()}_unquantized_hp4rows_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n✓ Trial artifacts will be saved in: {save_dir}/\n")

        tuner = SaveModelRandomSearch(
            hypermodel= model_builder,
            objective="val_binary_accuracy",
            max_trials=120,
            executions_per_trial=2,
            project_name="hp_search_4rows_matching",
            directory="./hyperparameter_tuning_4",   # keep KT logs in one place
            overwrite=True,                        # avoid weird resume behavior
            save_dir=save_dir,
            objective_name="val_binary_accuracy",
        )


        tuner.search(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=110,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
        )
        
        return tuner, save_dir

    


    def makeUnquatizedModelHyperParameterTuning3(self):
        def model_builder(hp):
            # ── B) Architecture hyperparams ──────────────────────────────────────────
            # separately tune rows and cols

            rownodes      = hp.Int("1",   2, 11, step=1)
        
            input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
            input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
            input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
            input4 = tf.keras.layers.Input(shape=(1,), name="y_local")
            inputList = [input1, input2, input3, input4]


            x_concat1 = tf.keras.layers.Concatenate()([input1,input2])
            x_concat2 = tf.keras.layers.Concatenate()([x_concat1,input3])
            x_concat3 = tf.keras.layers.Concatenate()([x_concat2,input4])
            x=x_concat3


            ## here i will add the layers 

            # layer 1
            x = tf.keras.layers.Dense(rownodes,activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)
            output = tf.keras.layers.Dense(1,activation='sigmoid')(x)

            model = tf.keras.Model(inputs=inputList, outputs=output)

            model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly  = True 
            )
            return model

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"{self.modelName.lower()}_unquantized_hp3rows_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n✓ Trial artifacts will be saved in: {save_dir}/\n")

        tuner = SaveModelRandomSearch(
            hypermodel= model_builder,
            objective="val_binary_accuracy",
            max_trials=120,
            executions_per_trial=2,
            project_name="hp_search_3rows_matching",
            directory="./hyperparameter_tuning_3",   # keep KT logs in one place
            overwrite=True,                        # avoid weird resume behavior
            save_dir=save_dir,
            objective_name="val_binary_accuracy",
        )


        tuner.search(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=110,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
        )

        return tuner, save_dir

    def makeUnquatizedModelHyperParameterTuning2(self):
        def model_builder(hp):
            # ── B) Architecture hyperparams ──────────────────────────────────────────
            # separately tune rows and cols

            rownodes      = hp.Int("1",   2, 11, step=1)
        
            input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
            input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
            input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
            input4 = tf.keras.layers.Input(shape=(1,), name="y_local")
            inputList = [input1, input2, input3, input4]


            x_concat1 = tf.keras.layers.Concatenate()([input1,input2])
            x_concat2 = tf.keras.layers.Concatenate()([x_concat1,input3])
            x_concat3 = tf.keras.layers.Concatenate()([x_concat2,input4])
            x=x_concat3


            ## here i will add the layers 

            # layer 1
            x = tf.keras.layers.Dense(rownodes,activation='relu')(x)
            x = tf.keras.layers.Dense(rownodes, activation='relu')(x)
            output = tf.keras.layers.Dense(1,activation='sigmoid')(x)

            model = tf.keras.Model(inputs=inputList, outputs=output)

            model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly  = True 
            )
            return model

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"{self.modelName.lower()}_unquantized_hp2rows_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n✓ Trial artifacts will be saved in: {save_dir}/\n")

        tuner = SaveModelRandomSearch(
            hypermodel= model_builder,
            objective="val_binary_accuracy",
            max_trials=120,
            executions_per_trial=2,
            project_name="hp_search_2rows_matching",
            directory="./hyperparameter_tuning_2",   # keep KT logs in one place
            overwrite=True,                        # avoid weird resume behavior
            save_dir=save_dir,
            objective_name="val_binary_accuracy",
        )


        tuner.search(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=110,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
        )

        return tuner, save_dir


        
    def makeQuantizedModel(self):
        for total_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{total_bits}w{int_bits}i"
        
        
            print(f"Building {config_name} model...")
            self.makeQuantizedModel_withBits(total_bits=total_bits,int_bits=int_bits)

    def makeQuantizedModel_withBits(self, total_bits = 8,int_bits =0):
        """
        Build & compile your QKeras model with the given number of integer bits.
        """
        tf.keras.backend.clear_session()
        # inputs
        input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
        input2 = tf.keras.layers.Input(shape=(1,), name="x_size")
        input3 = tf.keras.layers.Input(shape=(1,), name="y_size")
        input4 = tf.keras.layers.Input(shape=(1,), name="y_local")
        x = tf.keras.layers.Concatenate()([input1, input2, input3, input4])

        ## I want to try this with 1 int bit and 7 fractional
        ## I want to try this with 0 int bit and 7 fractional
        
        # layer 1
        x = QDense(
            17,
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(total_bits, int_bits),
            name="q_relu1"
        )(x)

        # layer 2 (example—you can tweak per‐layer bits)
        x = QDense(
            20,
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(total_bits, int_bits),
            name="q_relu2"
        )(x)

        # layer 3
        x = QDense(
            9,
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(total_bits, int_bits),
            name="q_relu3"
        )(x)

        # layer 4
        x = QDense(
            16,
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(total_bits, int_bits),
            name="q_relu4"
        )(x)

        # layer 5
        x = QDense(
            8,
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            #kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
            ## adds sum of the activations squared to the loss function 
            #activity_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        x = QActivation(
            activation=quantized_relu(total_bits, int_bits),
            name="q_relu5"
        )(x)

        # output
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            bias_quantizer=quantized_bits(total_bits, int_bits, alpha=1),
            #kernel_regularizer=tf.keras.regularizers.L2(0.0001),
        )(x)
        out = QActivation("smooth_sigmoid")(x)
        config_name = f"quantized_{total_bits}w{int_bits}i"
        self.models[config_name] = tf.keras.Model(inputs=[input1, input2, input3, input4], outputs=out)


def main():

# import your Model1 class definition (wherever it lives)
# from model1 import Model1   # <-- if your class is in model1.py
# or paste/define Model1 once (but do NOT redefine SmartPixModel again)
    """

    MODEL_PATH = "model1_unquantized_hp2rows_results_20260213_105714/model_trial_008.h5"

    m1 = Model1()
    m1.loadTfRecords()

    # Load the saved model
    m1.models["Unquantized"] = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Ensure compiled (your evaluate() uses predict/evaluate; evaluate() calls model.evaluate)
    m1.models["Unquantized"].compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    m1.models["Unquantized"].summary()

    # Run your full evaluation (ROC, AUC, bkg rej, etc.)
    results = m1.evaluate(config_name="Unquantized")

    print(results["test_accuracy"], results["roc_auc"])
    """
    m1 = Model1()
    m1.loadTfRecords()
    m1.makeUnquatizedModelHyperParameterTuning2()
        
    m1.loadTfRecords()
    m1.makeUnquatizedModelHyperParameterTuning3()

    m1.loadTfRecords()
    m1.makeUnquatizedModelHyperParameterTuning4()

    m1.loadTfRecords()
    m1.makeUnquatizedModelHyperParameterTuning5()

    


if __name__ == "__main__":
    main()

