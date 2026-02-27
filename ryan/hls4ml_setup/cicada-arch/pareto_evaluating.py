# arch-evaluating.py retrieves the best performing trials on the Pareto front from an Optuna study and evaluates them

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
import optuna
import shutil
import shlex
import re
import csv

from io import StringIO
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import load_model
from qkeras import quantized_bits

from utils import IsValidFile, IsReadableDir, CreateFolder, predict_single_image, load_args, save_to_npy
from drawing import Draw
from generator import RegionETGenerator
from models import TeacherAutoencoder, CicadaV1, CicadaV2, CNN_Trial, Binary_Trial, Bit_Binary_Trial
from cicada_evaluating import loss
from arch import get_data, get_data_npy, get_targets_from_teacher, get_targets_from_npy

def main(args):

    def evaluate_teacher(teacher):
        aucs, sizes, val_losses = [], [], []
        log = pd.read_csv(f"arch/{args.name}/models/{teacher.name}/training.log")
        draw_study.plot_loss_history(
            log["loss"], log["val_loss"], f"training-history-{teacher.name}"
        )

        y_pred_background_teacher = teacher.predict(X_test, batch_size=512, verbose=args.verbose)
        y_loss_background_teacher = loss(X_test, y_pred_background_teacher)

        teacher_results = dict()
        teacher_results["2024 Zero Bias"] = y_loss_background_teacher

        y_true = []
        y_pred_teacher = []
        inputs = []
        for name, data in X_signal.items():
            inputs.append(np.concatenate((data, X_test)))

            y_loss_signal_teacher = loss(
                data, teacher.predict(data, batch_size=512, verbose=args.verbose)
            )

            teacher_results[name] = y_loss_signal_teacher

            y_true.append(
                np.concatenate((np.ones(data.shape[0]), np.zeros(X_test.shape[0])))
            )
            y_pred_teacher.append(
                np.concatenate((y_loss_signal_teacher, y_loss_background_teacher))
            )

        draw_study.plot_anomaly_score_distribution(
            list(teacher_results.values()),
            [*teacher_results],
            f"anomaly-score-{teacher.name}",
        )
        draw_study.plot_roc_curve(y_true, y_pred_teacher, [*X_signal], inputs, f"roc-{teacher.name}")
        roc_aucs, _ = draw_study.get_aucs(y_true, y_pred_teacher, use_cut_rate=True)
        aucs.append(np.power([10.], np.mean(roc_aucs)))
        sizes.append(1000)
        val_losses.append(log["val_loss"].to_numpy()[-1])

        return aucs, sizes, val_losses

    def evaluate_students(students):
        aucs, sizes, val_losses = [], [], []
        for student in students:
            if student.name != "cicada-v1" and student.name != "cicada_v2":
                log = pd.read_csv(f"arch/{args.name}/models/{student.name}/training.log")
                draw_study.plot_loss_history(
                    log["loss"], log["val_loss"], f"training-history-{student.name}"
                )
            
            y_loss_background_student = student.predict(
                X_test.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
            )

            student_results = dict()
            student_results["2024 Zero Bias"] = y_loss_background_student

            y_true = []
            y_pred_student = []
            inputs = []
            for name, data in X_signal.items():
                inputs.append(np.concatenate((data, X_test)))

                y_loss_signal_student = student.predict(
                    data.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
                )

                student_results[name] = y_loss_signal_student

                y_true.append(
                    np.concatenate((np.ones(data.shape[0]), np.zeros(X_test.shape[0])))
                )

                y_pred_student.append(
                    np.concatenate((y_loss_signal_student, y_loss_background_student))
                )
        
            draw_study.plot_anomaly_score_distribution(
                list(student_results.values()),
                [*student_results],
                f"anomaly-score-{student.name}",
            )

            draw_study.plot_roc_curve(y_true, y_pred_student, [*X_signal], inputs, f"roc-{student.name}")
            roc_aucs, _ = draw_study.get_aucs(y_true, y_pred_student, use_cut_rate=True)
            aucs.append(np.power([10.], np.mean(roc_aucs)))
            sizes.append(student.count_params())
            val_losses.append(log["val_loss"].to_numpy()[-1])

        return aucs, sizes, val_losses

    # Get labels
    labels = [
        'Mean Signal', 
        'SUEP', 
        'H to Long Lived', 
        'VBHF to 2C', 
        'TT', 
        'SUSY ggHbb', 
    ]

    config = yaml.safe_load(open(args.config))

    draw_study = Draw(output_dir=f'arch/{args.name}/study_plots/', interactive=args.interactive)

    # Load old models
    for fromname, toname in [
        ['teacher', f'arch/{args.name}/models/teacher'], 
        ['cicada-v1', f'arch/{args.name}/models/cicada-v1'], 
        ['cicada-v2', f'arch/{args.name}/models/cicada-v2'], 
    ]:
        if not os.path.isdir(toname):
            shutil.copytree(f'{args.input}/{fromname}', toname)

    gen, X_train, y_train, X_val, y_val, X_test, X_signal = get_data_npy(config)

    # Load study and best trials (use if had search)
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

    all_trial_models = [[] for i in range(len(all_pareto_params))]
    sizes_b = [[] for i in range(len(all_pareto_params))]
    for i in range(1):
        for id, params in all_pareto_params[i]:
            for j in range(args.executions):
                model_name = f'{i}-{id}-{j}'
                all_trial_models[i].append(load_model(f'arch/{args.name}/models/{model_name}'))
                if args.type == 'cnn':
                    model, size_b = CNN_Trial((252,), id=42).get_model(params)
                elif args.type == 'bnn':
                    model, size_b = Binary_Trial((252,), args.type, id=42).get_model(params)
                sizes_b[i].append(size_b)

    # Load models
    teacher = load_model(f"arch/{args.name}/models/teacher")
    cicada_v1 = load_model(f"arch/{args.name}/models/cicada-v1")
    cicada_v2 = load_model(f"arch/{args.name}/models/cicada-v2")

    # Evaluate teacher
    #aucs_teacher, sizes_teacher, val_losses_teacher = evaluate_teacher(teacher)
    
    # Evaluate students
    aucs_trials, sizes_trials, val_losses_trials = [], [], []
    for trials in all_trial_models:
        aucs_trial, sizes_trial, val_losses_trial = evaluate_students(trials)
        aucs_trials.append(aucs_trial)
        sizes_trials.append(sizes_trial)
        val_losses_trials.append(val_losses_trial)

    aucs = aucs_trials[0]
    sizes = sizes_trials[0]
    sizes_b = sizes_b[0]
    val_losses = val_losses_trials[0]
    
    #if args.type == 'cnn':
    #    to_enumerate = ['Teacher', 'Cicada V1', 'Cicada V2']
    #else: to_enumerate = []
    to_enumerate = []

    draw_study.plot_2d(sizes, aucs, xlabel='Model Size', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-size-auc-all')
    draw_study.plot_2d(sizes, val_losses, xlabel='Model Size', ylabel='Validation Loss', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-size-loss-all')
    draw_study.plot_2d(val_losses, aucs, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-loss-auc-all')
    draw_study.plot_3d(val_losses, aucs, sizes, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', zlabel='Model Size', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-loss-auc-size-all')
    draw_study.plot_2d(sizes_b, aucs, xlabel='Model Size', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-size-b-auc-all')
    draw_study.plot_2d(sizes_b, val_losses, xlabel='Model Size', ylabel='Validation Loss', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-size-b-loss-all')
    draw_study.plot_2d(val_losses, aucs, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-loss-auc-all')
    draw_study.plot_3d(val_losses, aucs, sizes_b, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', zlabel='Model Size', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-loss-auc-size-b-all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""CICADA evaluation scripts""")
    parser.add_argument(
        "--input", "-i",
        action=IsReadableDir,
        type=Path,
        default="models/",
        help="Path to directory w/ trained models",
    )
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
        help="Type of model. One of cnn, bnn.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively display plots as they are created",
        default=False,
    )
    parser.add_argument(
        "-x", "--executions",
        type=int,
        help="Number of executions per trial",
        default=1, 
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    main(parser.parse_args())