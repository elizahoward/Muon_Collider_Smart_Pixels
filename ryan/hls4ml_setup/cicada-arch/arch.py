# arch.py creates an Optuna study to search for the best hyperparameters

import os
import logging
import sys
import io
import re
import shlex
import time
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
import optuna
import tensorflow as tf

from pathlib import Path
from tensorflow import data
from tensorflow.keras import Model
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, AveragePooling2D, Flatten, Input, Reshape, UpSampling2D, Conv2DTranspose
from qkeras import QActivation, QConv2D, QDense, QDenseBatchnorm
from tqdm import tqdm
from larq.models import summary

from utils import IsValidFile, IsReadableDir, CreateFolder, save_to_npy, predict_single_image, save_args
from drawing import Draw
from generator import RegionETGenerator
from cicada_training import loss, quantize, get_student_targets
from models import CNN_Trial, Binary_Trial, Bit_Binary_Trial

# Load data from h5 files
def get_data(config):
    datasets = [i["path"] for i in config["background"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()

    X_train, X_val, X_test = gen.get_data_split(datasets)
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)
    outlier_train = gen.get_data(config["exposure"]["training"])
    outlier_val = gen.get_data(config["exposure"]["validation"])

    X_train = np.concatenate([X_train, outlier_train])
    X_val = np.concatenate([X_val, outlier_val])
    X_test = X_test.reshape(-1, 18, 14, 1)
    return gen, X_train, X_val, X_test, X_signal

# Load data from npy files
def get_data_npy(config):
    datasets = [i["path"] for i in config["evaluation"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()

    X_train = gen.get_data_npy(config["training"]["inputs"])
    y_train = gen.get_targets_npy(config["training"]["targets"])
    X_val = gen.get_data_npy(config["validation"]["inputs"])
    y_val = gen.get_targets_npy(config["validation"]["targets"])
    _, _, X_test = gen.get_data_split(datasets)
    X_test = X_test.reshape(-1, 18, 14, 1)
    X_test = X_test[:250000] # Will get killed otherwise; too much data
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)

    return gen, X_train, y_train, X_val, y_val, X_test, X_signal

# Get targets, given a generator and training, validation, and test data
def get_targets_from_teacher(gen, X_train, X_val):
    teacher = load_model("models/teacher")
    gen_train = get_student_targets(teacher, gen, X_train)
    gen_val = get_student_targets(teacher, gen, X_val)
    return gen_train, gen_val

# Get targets, given a generator and training, validation, and test data
def get_targets_from_npy(gen, X_train, y_train, X_val, y_val):
    gen_train = gen.get_generator(X_train.reshape((-1, 252, 1)), y_train, 1024, True)
    gen_val = gen.get_generator(X_val.reshape((-1, 252, 1)), y_val, 1024, True)
    return gen_train, gen_val

def train_model(
    model: Model,
    gen_train: data.Dataset,
    gen_val: data.Dataset,
    epochs: int = 1,
    callbacks=None,
    shuffle: bool = False, 
    verbose: bool = False,
) -> float:
    history = model.fit(
        gen_train,
        steps_per_epoch=len(gen_train),
        epochs=epochs,
        validation_data=gen_val,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle, 
    )
    return history

def main(args) -> None:

    def objective(trial, trial_id):

        start_time = time.time()

        to_save = { 
            'Mean Signal AUC (0.3-3 kHz)': np.array([]), 
            f'{labels[0]} AUC (0.3-3 kHz)': np.array([]),
            f'{labels[1]} AUC (0.3-3 kHz)': np.array([]),
            f'{labels[2]} AUC (0.3-3 kHz)': np.array([]),
            f'{labels[3]} AUC (0.3-3 kHz)': np.array([]),
            f'{labels[4]} AUC (0.3-3 kHz)': np.array([]),
            f'{labels[5]} AUC (0.3-3 kHz)': np.array([]),
            'Validation Loss': np.array([]), 
            'Model Size (number of parameters)': np.array([]), 
            'Model Size (b)': np.array([]), 
        }

        for i in tqdm(range(args.executions)):
            if (time.time()-start_time) > max_time_trial-max_time_execution: break
            if (time.time() - total_start_time > max_allocated_time - max_time_trial): break
            execution_id = i
            
            # Compile
            if args.type == 'cnn':
                model_gen = CNN_Trial((252,), execution_id)
                model, size_b = model_gen.get_trial(trial)
            elif args.type == 'bitbnn':
                model_gen = Bit_Binary_Trial((252,), args.type, execution_id)
                model, size_b = model_gen.get_trial(trial)
            elif args.type == 'bnn':
                model_gen = Binary_Trial((252,), execution_id)
                model, size_b = model_gen.get_trial(trial)
            if size_b < 50000 or size_b > 100000 or model==None: # Prune if too small or too large
                raise optuna.TrialPruned()
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
            es = EarlyStopping(monitor='val_loss', patience=3, baseline=10, start_from_epoch=10)
            log = CSVLogger(f"arch/{args.name}/models/{trial_id}-{execution_id}-training.log", append=True)

            # Train
            history = train_model(
                model,
                gen_train,
                gen_val,
                epochs=args.epochs,
                callbacks=[es, log], 
                verbose=args.verbose,
                shuffle=True, 
            )
            #log = pd.read_csv(f"arch/{args.name}/models/{trial_id}-{execution_id}-training.log")
            #draw_execution.plot_loss_history(log["loss"], log["val_loss"], f"training-history-{trial_id}-{execution_id}")
            
            # Evaluate the model on the test set
            auc, _ = get_aucs(model, trial_id, execution_id)
            mean_auc = np.mean(auc)
            n_params = model.count_params()

            for name, val in [
                [f'{labels[0]} AUC (0.3-3 kHz)', mean_auc], 
                [f'{labels[1]} AUC (0.3-3 kHz)', auc[0]], 
                [f'{labels[2]} AUC (0.3-3 kHz)', auc[1]], 
                [f'{labels[3]} AUC (0.3-3 kHz)', auc[2]], 
                [f'{labels[4]} AUC (0.3-3 kHz)', auc[3]], 
                [f'{labels[5]} AUC (0.3-3 kHz)', auc[4]], 
                ['Validation Loss', history.history["val_loss"][-1]], 
                ['Model Size (number of parameters)', n_params], 
                ['Model Size (b)', size_b], 
                ]:

                to_save[name] = np.append(to_save[name], val)
                pathname = f'arch/{args.name}/trial_metrics/{name}/{trial_id}.npy'
                save_to_npy(val, pathname)

        if to_save[f'{labels[0]} AUC (0.3-3 kHz)'].size > 0:
            med_auc = np.median(to_save[f'{labels[0]} AUC (0.3-3 kHz)'])
        else: med_auc = 0.
        if to_save["Validation Loss"].size > 0:
            med_val_loss = np.median(to_save["Validation Loss"])
        else: med_val_loss = 40.
        if to_save['Model Size (number of parameters)'].size > 0:
            n_params = np.median(to_save["Model Size (number of parameters)"])
        else: n_params = np.nan
        if to_save['Model Size (b)'].size > 0:
            size_b = np.median(to_save['Model Size (b)'])
        else: size_b = np.nan
        
        return med_auc, med_val_loss, n_params, size_b

    def get_aucs(model, trial_id, execution_id):
        y_loss_background = model.predict(X_test.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose)
        y_loss_background = np.nan_to_num(y_loss_background, nan=0.0, posinf=255, neginf=0)
        results = {'2023 Zero Bias' : y_loss_background}
        y_true, y_pred, inputs = [], [], []
        for name, data in X_signal.items():
            inputs.append(np.concatenate((data, X_test)))
            y_loss = model.predict(data.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose)
            y_loss = np.nan_to_num(y_loss, nan=0.0, posinf=255, neginf=0)
            results[name] = y_loss
            y_true.append(
                np.concatenate((np.ones(data.shape[0]), np.zeros(X_test.shape[0])))
            )
            y_pred.append(
                np.concatenate((y_loss, y_loss_background))
            )

        #draw_execution.plot_roc_curve(y_true, y_pred, [*X_signal], inputs, f"roc-{trial_id}-{execution_id}")
        roc_aucs, std_aucs = draw_execution.get_aucs(y_true, y_pred, use_cut_rate=True)
        return roc_aucs, std_aucs

    total_start_time=time.time()
    max_time_trial = 4 * args.epochs * args.executions * 4
    max_time_execution = 4 * args.epochs * 4

    # Parse args.jobflavour and args.trials
    if args.jobflavour=="espresso": max_allocated_time = 1200
    elif args.jobflavour=="microcentury": max_allocated_time = 3600
    elif args.jobflavour=="longlunch": max_allocated_time = 7200
    elif args.jobflavour=="workday": max_allocated_time = 28800
    elif args.jobflavour=="tomorrow": max_allocated_time = 86400
    elif args.jobflavour=="testmatch": max_allocated_time = 259200
    elif args.jobflavour=="nextweek": max_allocated_time = 604800
    max_allocated_time -= 300 # Allow for importing libraries and setting up environment
    if args.trials==-1: trials = int(1e6)
    else: trials = int(args.trials)

    # Get labels
    labels = [
        'Mean Signal', 
        'SUEP', 
        'H to Long Lived', 
        'VBHF to 2C', 
        'TT', 
        'SUSY ggHbb', 
    ]

    # Create folders
    for foldername in [
        'arch/', 
        f'arch/{args.name}/', 
        f'arch/{args.name}/study_plots/', 
        f'arch/{args.name}/trial_plots/', 
        f'arch/{args.name}/execution_plots/', 
        f'arch/{args.name}/models/', 
        f'arch/{args.name}/study_metrics/', 
        f'arch/{args.name}/trial_metrics/', 
        f'arch/{args.name}/trial_metrics/Mean Signal AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[0]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[1]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[2]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[3]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[4]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[5]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/Validation Loss/', 
        f'arch/{args.name}/trial_metrics/Model Size (number of parameters)/', 
        f'arch/{args.name}/trial_metrics/Model Size (b)/', 
        ]:
        if not os.path.exists(foldername):
            os.mkdir(foldername)

    # Move existing db if exists
    if Path(f'{args.name}.db').exists():
        shutil.move(f"{args.name}.db", f"arch/{args.name}/{args.name}.db")

    # Load data, get student targets
    config = yaml.safe_load(open(args.config))

    draw_execution = Draw(output_dir=f'arch/{args.name}/execution_plots/', interactive=args.interactive)

    gen, X_train, y_train, X_val, y_val, X_test, X_signal = get_data_npy(config)
    gen_train, gen_val = get_targets_from_npy(gen, X_train, y_train, X_val, y_val)

    # Add SQLite
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_name = f"sqlite:///arch/{args.name}/{args.name}.db"

    study = optuna.create_study(
            directions=['maximize', 'minimize', 'minimize', 'minimize'], 
            study_name=args.name, 
            storage=storage_name, 
            load_if_exists=True, 
        )

    all_existing_trials = study.get_trials()
    if len(all_existing_trials) == 2 and all_existing_trials[0].state.is_finished() == False and all_existing_trials[1].state.is_finished() == False:
        num_existing_trials = 0
    else: num_existing_trials = len(all_existing_trials)

    # Optuna study; if parallelized, reload study after each iteration
    for i in tqdm(range(trials)): # for parallelization
        if (time.time() - total_start_time > max_allocated_time - max_time_trial): break
        study = optuna.create_study(
            directions=['maximize', 'minimize', 'minimize', 'minimize'], 
            study_name=args.name, 
            storage=storage_name, 
            load_if_exists=True, 
        )
        #trial_id = len(study.get_trials())
        trial_id = i + num_existing_trials
        #if (args.type == 'cnn') and len(study.trials) < 2:
        #    study.enqueue_trial({
        #        "n_conv_layers": 0, 
        #        "n_dense_layers": 1, 
        #        "n_layers": 1, 
        #        "n_dense_units_0": 4, 
        #        "q_kernel_conv_bits": 12, 
        #        "q_kernel_conv_ints": 3, 
        #        "q_kernel_dense_bits": 8, 
        #        "q_kernel_dense_ints": 1, 
        #        "q_bias_dense_bits": 8, 
        #        "q_bias_dense_ints": 3, 
        #        "q_activation_bits": 10, 
        #        "q_activation_ints": 6, 
        #       "shortcut": False, 
        #        "dropout": 0., 
        #    }) # Include cicada_v1
        #    study.enqueue_trial({
        #        "n_conv_layers": 1, 
        #        "n_dense_layers": 1, 
        #        "n_layers": 2, 
        #        "n_filters_0": 4, 
        #        "kernel_width_0": 2, 
        #        "kernel_height_0": 2, 
        #        "stride_width_0": 2, 
        #        "stride_height_0": 2, 
        #        "use_bias_conv": False, 
        #        "n_dense_units_0": 4, 
        #        "q_kernel_conv_bits": 12, 
        #        "q_kernel_conv_ints": 3, 
        #        "q_kernel_dense_bits": 8, 
        #        "q_kernel_dense_ints": 1, 
        #        "q_bias_dense_bits": 8, 
        #        "q_bias_dense_ints": 3, 
        #        "q_activation_bits": 10, 
        #        "q_activation_ints": 6, 
        #        "shortcut": False, 
        #        "dropout": 0., 
        #    }) # Include cicada_v2
        #elif (args.type == 'bnn') and len(study.trials) < 2:
        #    study.enqueue_trial({
        #        "binary_type": "bnn", 
        #        "n_conv_layers": 0, 
        #        "n_dense_layers": 1, 
        #        "n_layers": 1, 
        #        "n_dense_units_0": 8, 
        #        "q_activation_bits": 10, 
        #        "q_activation_ints": 6, 
        #        "shortcut": False, 
        #        "dropout": 0., 
        #    }) # Include cicada_v1
        #    study.enqueue_trial({
        #        "binary_type": "bnn", 
        #        "n_conv_layers": 1, 
        #        "n_dense_layers": 1, 
        #        "n_layers": 2, 
        #        "n_filters_0": 8, 
        #        "kernel_width_0": 2, 
        #        "kernel_height_0": 2, 
        #        "stride_width_0": 2, 
        #        "stride_height_0": 2, 
        #        "use_bias_conv": False, 
        #        "n_dense_units_0": 7, 
        #        "q_activation_bits": 10, 
        #        "q_activation_ints": 6, 
        #        "shortcut": False, 
        #        "dropout": 0., 
        #    }) # Include cicada_v2
        study.optimize(lambda trial: objective(trial, trial_id), n_trials=1, n_jobs=args.parallels, show_progress_bar=False)
    print(f'Total time: {time.time() - total_start_time}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""CICADA training scripts""")
    parser.add_argument(
        "--config", "-c",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="example",
        help="Name of study",
    )
    parser.add_argument(
        "-y", "--type",
        type=str,
        default="cnn",
        help="Type of model. One of cnn, bnn, bitbnn.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively display plots as they are created",
        default=False,
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        help="Number of training epochs per execution",
        default=10,
    )
    parser.add_argument(
        "-x", "--executions",
        type=int,
        help="Number of executions per trial",
        default=10, 
    )
    parser.add_argument(
        "-t", "--trials",
        type=int,
        help="Number of trials. If -1, will continue until max time is hit.",
        default=-1, 
    )
    parser.add_argument(
        "-p", "--parallels",
        type=int,
        help="Number of parallel jobs",
        default=1, 
    )
    parser.add_argument(
        "-j", "--jobflavour",
        type=str,
        help="Maximum time for the job. Uses the JobFlavour times from https://batchdocs.web.cern.ch/local/submit.html",
        default="tomorrow", 
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    args = parser.parse_args()
    save_args(args)  # Save command-line arguments
    main(args)
