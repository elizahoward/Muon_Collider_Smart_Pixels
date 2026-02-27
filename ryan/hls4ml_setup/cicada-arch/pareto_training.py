# arch-training.py retrieves the best performing trials on the Pareto front from an Optuna study and further trains it

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import yaml
import optuna
import pandas as pd

from io import StringIO
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

    # Load data, get student targets
    config = yaml.safe_load(open(args.config))

    gen, X_train, y_train, X_val, y_val, X_test, X_signal = get_data_npy(config)
    gen_train, gen_val = get_targets_from_npy(gen, X_train, y_train, X_val, y_val)

    # Load study and best trials parameters (use if had search)
    #loaded_study = optuna.load_study(study_name=args.name, storage=f"sqlite:///arch/{args.name}/{args.name}.db")
    #pareto_trials = loaded_study.best_trials
    #pareto_params = [trial.params for trial in pareto_trials]

    with open(f'arch/{args.name}/{args.name}.csv', "r") as f:
        lines = f.readlines()

    signal_types = [line.strip().split("Signal Type:")[1].strip() for line in lines if line.startswith("Signal Type:")]
    metrics = ['Validation Loss', 'Adjusted ROC AUC', 'Model Size (b)']
    title_lines = [i for i, line in enumerate(lines) if line.startswith("Signal Type:")]
    title_lines.append(len(lines))  # Add EOF

    signal_dfs = []
    for i in range(len(title_lines) - 1):
        start = title_lines[i] + 1  # Skip the 'Signal Type:' line
        end = title_lines[i + 1]
        section_csv = StringIO("".join(lines[start:end]))
        df_section = pd.read_csv(section_csv)
        signal_dfs.append(df_section)

    # Associate each signal type with its corresponding list of pareto entries
    all_pareto_params = []
    for df in signal_dfs:
        param_cols = [col for col in df.columns if col not in metrics]
        pareto_data = []
        for i, (_, row) in enumerate(df.iterrows()):
            param_dict={}
            for k in param_cols:
                if isinstance(row[k], float):
                    if np.isnan(row[k]):
                        continue
                    elif k != "dropout":
                        param_dict[k] = int(row[k])
                else:
                    param_dict[k] = row[k]
            pareto_data.append([i, param_dict])
        seen = set()
        unique_pareto_data = []
        for id, param_dict in pareto_data:
            frozen = tuple(sorted(param_dict.items()))
            if frozen not in seen:
                seen.add(frozen)
                unique_pareto_data.append([id, param_dict])
        all_pareto_params.append(unique_pareto_data)

    # Train best trials
    for i in range(1):
        for id, params in tqdm(all_pareto_params[i]):
            for j in tqdm(range(args.executions)):
                if args.type == 'cnn':
                    model, _ = CNN_Trial((252,), id=42).get_model(params)
                    model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
                elif args.type == 'bnn':
                    model, _ = Binary_Trial((252,), args.type, id=42).get_model(params)
                    model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
                model._name = f'{i}-{id}-{j}'
                name = '_'.join(str(x) + '_' + str(y) for x, y in params.items())
                name = f"{signal_types[i]}_" + name + f"-{j}"
                name = name.replace('[', '').replace(']', '').replace(',', '_').replace(' ', '').replace('(', '_').replace(')', '_').replace('_alpha=1.0', '')
                mc = ModelCheckpoint(f"arch/{args.name}/models/{model.name}", save_best_only=True)
                log = CSVLogger(f"arch/{args.name}/models/{model.name}/training.log", append=True)
                if not os.path.exists(f"arch/{args.name}/models/{model.name}/"):
                    os.mkdir(f"arch/{args.name}/models/{model.name}/")
                with open(f"arch/{args.name}/models/{model.name}/params.yaml", "w") as f:
                    yaml.dump(params, f)

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
