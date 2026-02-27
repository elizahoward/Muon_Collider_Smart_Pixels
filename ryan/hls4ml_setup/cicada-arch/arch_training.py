# arch-training.py retrieves the best performing trials on the Pareto front from an Optuna study and further trains it

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import yaml
import optuna

from pathlib import Path
from tensorflow import data
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from qkeras import quantized_bits
from tqdm import tqdm

from utils import IsValidFile, CreateFolder
from generator import RegionETGenerator
from models import TeacherAutoencoder, CicadaV1, CicadaV2, CNN_Trial, Binary_Trial, Bit_Binary_Trial
from cicada_training import loss, quantize, get_student_targets, train_model
from arch import get_data, get_data_npy, get_targets_from_teacher, get_targets_from_npy

def main(args) -> None:
    # Create folders
    for foldername in ['arch/', f'arch/{args.name}/', f'arch/{args.name}/plots/', f'arch/{args.name}/models/']:
        if not os.path.exists(foldername):
            os.mkdir(foldername)

    # Load data, get student targets
    config = yaml.safe_load(open(args.config))

    gen, X_train, y_train, X_val, y_val, _, _ = get_data_npy(config)
    gen_train, gen_val = get_targets_from_npy(gen, X_train, y_train, X_val, y_val)

    # Load study and best trials parameters (use if had search)
    #loaded_study = optuna.load_study(study_name=args.name, storage=f"sqlite:///arch/{args.name}/{args.name}.db")
    #pareto_trials = loaded_study.best_trials
    #pareto_params = [trial.params for trial in pareto_trials]
    
    # (use if want to train specific archs)
    pareto_params = [
    # for cnn
    #    {'n_conv_layers': 1, 'n_filters_0': 5, 'n_dense_layers': 1, 'n_dense_units_0': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 6, 'n_dense_layers': 1, 'n_dense_units_0': 13}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 7, 'n_dense_layers': 1, 'n_dense_units_0': 11}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 6, 'n_dense_layers': 1, 'n_dense_units_0': 13}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 2, 'n_dense_units_0': 15, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 5, 'n_dense_layers': 2, 'n_dense_units_0': 12, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 6, 'n_dense_layers': 2, 'n_dense_units_0': 11, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 7, 'n_dense_layers': 2, 'n_dense_units_0': 10, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 8, 'n_dense_layers': 2, 'n_dense_units_0': 9, 'n_dense_units_1': 16}, # Final model 1
    #    {'n_conv_layers': 1, 'n_filters_0': 9, 'n_dense_layers': 2, 'n_dense_units_0': 8, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 10, 'n_dense_layers': 2, 'n_dense_units_0': 7, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 2, 'n_dense_units_0': 16, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 3, 'n_dense_units_0': 16, 'n_dense_units_1': 16, 'n_dense_units_2': 16}, # Final model 2
    #    {'n_conv_layers': 0, 'n_dense_layers': 4, 'n_dense_units_0': 16, 'n_dense_units_1': 16, 'n_dense_units_2': 16, 'n_dense_units_3': 16}, 
    #    in cnn_skip, cnn_skip_1, cnn_skip_2, cnn_skip_3
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'shortcut': True}, 
    #    in cnn_drop
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 20}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 10}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 6}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 5}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 4}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 3}, 
    #    in cnn_quant
        {"q_kernel_conv": quantized_bits(1, 1, 0, alpha=1.0), "q_kernel_dense": quantized_bits(1, 1, 0, alpha=1.0), "q_bias_dense": quantized_bits(1, 1, 0, alpha=1.0), "q_activation": "quantized_relu(1, 1)"}, 
        {"q_kernel_conv": quantized_bits(12, 3, 1, alpha=1.0), "q_kernel_dense": quantized_bits(8, 1, 1, alpha=1.0), "q_bias_dense": quantized_bits(8, 3, 1, alpha=1.0), "q_activation": "quantized_relu(10, 6)"}, 
        {"q_kernel_conv": quantized_bits(8, 4, 1, alpha=1.0), "q_kernel_dense": quantized_bits(8, 4, 1, alpha=1.0), "q_bias_dense": quantized_bits(8, 4, 1, alpha=1.0), "q_activation": "quantized_relu(8, 4)"}, 
        {"q_kernel_conv": quantized_bits(16, 8, 1, alpha=1.0), "q_kernel_dense": quantized_bits(16, 8, 1, alpha=1.0), "q_bias_dense": quantized_bits(16, 8, 1, alpha=1.0), "q_activation": "quantized_relu(16, 8)"}, 
        {"q_kernel_conv": quantized_bits(32, 16, 1, alpha=1.0), "q_kernel_dense": quantized_bits(32, 16, 1, alpha=1.0), "q_bias_dense": quantized_bits(32, 16, 1, alpha=1.0), "q_activation": "quantized_relu(32, 16)"}, 
    # for bnn
    #    {'n_conv_layers': 0, 'n_dense_layers': 2, 'n_dense_units_0': 16, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 3, 'n_dense_units_0': 16, 'n_dense_units_1': 16, 'n_dense_units_2': 16}, 
    # for bitbnn
    #    in bitbnn_1
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 10}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 2, 'n_dense_units_0': 10, 'n_dense_units_1': 10}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 3, 'n_dense_units_0': 10, 'n_dense_units_1': 10, 'n_dense_units_2': 10}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 11}, 
    #    in bitbnn_2, bit_bnn_3
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 5}, # not in bitbnn_3
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 6}, # not in bitbnn_3
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 7}, # not in bitbnn_3
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 8}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 9}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 10}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 11}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 12}, # not in bitbnn_3
    #    in bitbnn_conv
    #    {'n_conv_layers': 1, 'n_filters': [4], 'n_dense_layers': 1, 'n_dense_units': [4], 'shortcut': True}, 
    #    {'n_conv_layers': 1, 'n_filters': [4], 'n_dense_layers': 1, 'n_dense_units': [6], 'shortcut': True}, 
    #    {'n_conv_layers': 1, 'n_filters': [4], 'n_dense_layers': 1, 'n_dense_units': [8], 'shortcut': True}, 
    #    {'n_conv_layers': 1, 'n_filters': [4], 'n_dense_layers': 1, 'n_dense_units': [10], 'shortcut': True}, 
    ]

    # Train best trials
    for params in tqdm(pareto_params):
        for i in tqdm(range(args.executions)):
            if args.type == 'cnn':
                model = CNN_Trial((252,)).get_model(params)
                model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
            elif args.type == 'vit':
                model = ViT_Trial((252,)).get_model(params)
                model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
            elif args.type == 'bitbnn':
                model = Bit_Binary_Trial((252,), args.type).get_model(params)
                model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
            elif args.type[0] == 'b':
                model = Binary_Trial((252,), args.type).get_model(params)
                model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
            model._name = '_'.join(str(x) + '_' + str(y) for x, y in params.items())
            model._name = model._name + f"_x_{i}"
            model._name = model._name.replace('[', '').replace(']', '').replace(',', '_').replace(' ', '').replace('(', '_').replace(')', '_').replace('_alpha=1.0', '')
            mc = ModelCheckpoint(f"arch/{args.name}/models/{model.name}", save_best_only=True)
            log = CSVLogger(f"arch/{args.name}/models/{model.name}/training.log", append=True)

            for epoch in tqdm(range(args.epochs)):
                train_model(
                    model,
                    gen_train,
                    gen_val,
                    epoch=epoch,
                    steps=1,
                    callbacks=[mc, log],
                    verbose=args.verbose,
                )


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
        help="Name of study to be loaded",
    )
    parser.add_argument(
        "-y", "--type",
        type=str,
        default="cnn",
        help="Type of model. One of cnn, vit, bnn, ban, bwn, bitbnn.",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        help="Number of training epochs",
        default=100,
    )
    parser.add_argument(
        "-x", "--executions",
        type=int,
        help="Number of executions per model",
        default=1, 
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    main(parser.parse_args())
