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

from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import load_model
from qkeras import quantized_bits

from utils import IsValidFile, IsReadableDir, CreateFolder, predict_single_image, load_args, save_to_npy
from drawing import Draw
from generator import RegionETGenerator
from cicada_evaluating import loss
from arch import get_data, get_data_npy, get_targets_from_teacher, get_targets_from_npy

def main(args):

    def write_executions_to_csv(best_executions_global, output_filename=f"arch/{args.name}/{args.name}.csv"):
        with open(output_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            for i in range(len(best_executions_global)):
                signal_type = best_executions_global[i][1]
                executions = best_executions_global[i][2:]

                # Write block title row
                writer.writerow([f"Signal Type: {signal_type}"])
                
                # Prepare the header
                if executions:
                    # Get the union of all hyperparameter keys across all executions
                    all_keys = set()
                    for exec in executions:
                        all_keys.update(exec[3].keys())
                    hyperparam_keys = sorted(all_keys)

                    header = ['Validation Loss', 'Adjusted ROC AUC', 'Model Size (b)'] + hyperparam_keys
                    writer.writerow(header)

                    # Write each execution row
                    for exec in executions:
                        val_loss, auc, size, hyperparams = exec
                        row = [val_loss, auc, size]
                        # Add hyperparams in consistent order, with blanks if missing
                        row += [hyperparams.get(k, "") for k in hyperparam_keys]
                        writer.writerow(row)

                writer.writerow([])  # Add a blank line between blocks

    def compile_study_metrics(argname):
        if os.path.isfile(f"arch/{argname}/study_metrics/Min of Validation Losses.npy"): return
        study_metric_names = {
            'Max of H to Long Lived AUCs (0.3-3 kHz).npy',     'Median of VBHF to 2C AUCs (0.3-3 kHz).npy',
            'Max of Mean Signal AUCs (0.3-3 kHz).npy',         'Min of Validation Losses.npy',
            'Max of SUEP AUCs (0.3-3 kHz).npy',                'Model Size (b).npy',
            'Max of SUSY ggHbb AUCs (0.3-3 kHz).npy',          'Model Size (number of parameters).npy',
            'Max of TT AUCs (0.3-3 kHz).npy',                  'Standard Deviation of H to Long Lived AUCs (0.3-3 kHz).npy',
            'Max of VBHF to 2C AUCs (0.3-3 kHz).npy',          'Standard Deviation of Mean Signal AUCs (0.3-3 kHz).npy',
            'Median of H to Long Lived AUCs (0.3-3 kHz).npy',  'Standard Deviation of SUEP AUCs (0.3-3 kHz).npy',
            'Median of Mean Signal AUCs (0.3-3 kHz).npy',      'Standard Deviation of SUSY ggHbb AUCs (0.3-3 kHz).npy',
            'Median of SUEP AUCs (0.3-3 kHz).npy',             'Standard Deviation of TT AUCs (0.3-3 kHz).npy',
            'Median of SUSY ggHbb AUCs (0.3-3 kHz).npy',       'Standard Deviation of Validation Losses.npy',
            'Median of TT AUCs (0.3-3 kHz).npy',               'Standard Deviation of VBHF to 2C AUCs (0.3-3 kHz).npy',
            'Median of Validation Losses.npy',
        }
        trial_metric_names = {
            'H to Long Lived AUC (0.3-3 kHz)',  'Model Size (number of parameters)',  'TT AUC (0.3-3 kHz)',
            'Mean Signal AUC (0.3-3 kHz)',      'SUEP AUC (0.3-3 kHz)',               'Validation Loss',
            'Model Size (b)',                   'SUSY ggHbb AUC (0.3-3 kHz)',         'VBHF to 2C AUC (0.3-3 kHz)',
        }
        trial_names = os.listdir(f'arch/{argname}/trial_metrics/Validation Loss')
        trial_names_temp = np.array([int(trial_name.replace('.npy', '')) for trial_name in trial_names])
        trial_names_temp_ind = np.argsort(trial_names_temp)
        trial_names = [trial_names[i] for i in trial_names_temp_ind]

        for trial_name in trial_names:
            for study_metric_name in study_metric_names:
                for trial_metric_name in trial_metric_names:
                    if trial_metric_name.replace("AUC", "AUCs") in study_metric_name:
                        x = np.load(f'arch/{argname}/trial_metrics/{trial_metric_name}/{trial_name}').flatten()
                        if "Max" in study_metric_name:
                            save_to_npy(np.max(x), f'arch/{argname}/study_metrics/{study_metric_name}')
                        elif "Min" in study_metric_name:
                            save_to_npy(np.min(x), f'arch/{argname}/study_metrics/{study_metric_name}')
                        elif "Median" in study_metric_name:
                            save_to_npy(np.median(x), f'arch/{argname}/study_metrics/{study_metric_name}')
                        elif "Standard Deviation" in study_metric_name:
                            save_to_npy(np.std(x), f'arch/{argname}/study_metrics/{study_metric_name}')
                        elif "Model Size" in study_metric_name:
                            save_to_npy(np.median(x), f'arch/{argname}/study_metrics/{study_metric_name}')

    def compile_batch_study_metrics(argname, fromname, batchsize):
        def checkifrunexists(i):
            if not Path(f'arch/{fromname}{i}/{fromname}{i}.db').exists(): return False
            else: return True

        def checkiflossplotexists(id, j, k):
            if not Path(f'arch/{fromname}{id}/execution_plots/training-history-{j}-{k}.png').exists(): return False
            else: return True

        def checkifrocplotexists(id, j, k):
            if not Path(f'arch/{fromname}{id}/execution_plots/roc-{j}-{k}.png').exists(): return False
            else: return True

        # Create folders
        for foldername in [
            'arch/', 
            f'arch/{argname}/', 
            f'arch/{argname}/study_plots/', 
            f'arch/{argname}/trial_plots/', 
            f'arch/{argname}/execution_plots/', 
            f'arch/{argname}/models/', 
            f'arch/{argname}/study_metrics/', 
            f'arch/{argname}/trial_metrics/', 
            f'arch/{argname}/trial_metrics/Mean Signal AUC (0.3-3 kHz)/', 
            f'arch/{argname}/trial_metrics/{labels[0]} AUC (0.3-3 kHz)/', 
            f'arch/{argname}/trial_metrics/{labels[1]} AUC (0.3-3 kHz)/', 
            f'arch/{argname}/trial_metrics/{labels[2]} AUC (0.3-3 kHz)/', 
            f'arch/{argname}/trial_metrics/{labels[3]} AUC (0.3-3 kHz)/', 
            f'arch/{argname}/trial_metrics/{labels[4]} AUC (0.3-3 kHz)/', 
            f'arch/{argname}/trial_metrics/{labels[5]} AUC (0.3-3 kHz)/', 
            f'arch/{args.name}/trial_metrics/Validation Loss/', 
            f'arch/{args.name}/trial_metrics/Model Size (number of parameters)/', 
            f'arch/{args.name}/trial_metrics/Model Size (b)/', 
            ]:
            if not os.path.exists(foldername):
                os.mkdir(foldername)

        trial_metric_names = {
            'H to Long Lived AUC (0.3-3 kHz)',  'Model Size (number of parameters)',  'TT AUC (0.3-3 kHz)',
            'Mean Signal AUC (0.3-3 kHz)',      'SUEP AUC (0.3-3 kHz)',               'Validation Loss',
            'Model Size (b)',                   'SUSY ggHbb AUC (0.3-3 kHz)',         'VBHF to 2C AUC (0.3-3 kHz)',
        }
        
        if not args.lite:
            # Copy the first study
            first_id=0
            for i in range(batchsize):
                if checkifrunexists(i) == True:
                    first_id = i
                    break

            optuna.copy_study(from_study_name=f'{fromname}{first_id}', from_storage=f'sqlite:///arch/{fromname}{first_id}/{fromname}{first_id}.db', to_storage=f'sqlite:///arch/{argname}/{argname}.db', to_study_name=argname)
            base_study = optuna.load_study(study_name=argname, storage=f'sqlite:///arch/{argname}/{argname}.db')
            shutil.copyfile(f"arch/{fromname}{first_id}/args.txt", f"arch/{argname}/args.txt")
            
            # Dictionary to track trials based on hyperparameters
            existing_trials = {tuple(trial.params.items()): trial for trial in base_study.get_trials()}
            num_trials_to_add = len(existing_trials)

            # Copy initial trial metric files, execution plots
            trial_names = os.listdir(f'arch/{fromname}{first_id}/trial_metrics/Validation Loss')
            trial_names_temp = np.array([int(trial_name.replace('.npy', '')) for trial_name in trial_names])
            trial_names_temp_ind = np.argsort(trial_names_temp)
            trial_names = [trial_names[i] for i in trial_names_temp_ind]
            for trial_metric_name in trial_metric_names:
                for trial_name in trial_names:
                    shutil.copyfile(f"arch/{fromname}{first_id}/trial_metrics/{trial_metric_name}/{trial_name}", f"arch/{argname}/trial_metrics/{trial_metric_name}/{trial_name}")
            
            # Process remaining studies
            for i in range(first_id+1, batchsize):
                if checkifrunexists(i):
                    study_tmp = optuna.load_study(study_name=f'{fromname}{i}', storage=f'sqlite:///arch/{fromname}{i}/{fromname}{i}.db')

                    for j in range(len(study_tmp.get_trials())):
                        trial = study_tmp.get_trials()[j]
                        trial_params = tuple(trial.params.items())  # Convert params to tuple for hashing

                        if trial_params in existing_trials:
                            # Concatenate trial metric data instead of adding a new trial
                            existing_trial_id = existing_trials[trial_params].number # might not work
                            existing_trial_file = f"{existing_trial_id}.npy"

                            for trial_metric_name in trial_metric_names:
                                old_file = f"arch/{argname}/trial_metrics/{trial_metric_name}/{existing_trial_file}"
                                new_file = f"arch/{fromname}{i}/trial_metrics/{trial_metric_name}/{trial.number}.npy"

                                if os.path.exists(new_file):
                                    old_data = np.load(old_file) if os.path.exists(old_file) else np.array([])
                                    new_data = np.load(new_file)
                                    np.save(old_file, np.concatenate([old_data.flatten(), new_data.flatten()]))
                            
                        else:
                            # Copy metric files and add trial to study
                            new_trial_id = num_trials_to_add
                            existing_trials[trial_params] = trial

                            base_study.add_trial(trial)

                            for trial_metric_name in trial_metric_names:
                                from_file = f"arch/{fromname}{i}/trial_metrics/{trial_metric_name}/{trial.number}.npy"
                                to_file = f"arch/{argname}/trial_metrics/{trial_metric_name}/{new_trial_id}.npy"

                                if os.path.exists(from_file):
                                    shutil.copyfile(from_file, to_file)

                            num_trials_to_add += 1

    def search_plots():
        if not args.lite:
            all_names = [name for name in os.listdir(f'arch/{args.name}/study_metrics') if os.path.isfile(os.path.join(f'arch/{args.name}/study_metrics', name))]
        else:
            all_names = [name for name in os.listdir(f'arch/{args.fromname}0/study_metrics') if os.path.isfile(os.path.join(f'arch/{args.fromname}0/study_metrics', name))]
        min_names = [name for name in all_names if (('AUC' not in name) and ('Min' in name))]
        med_names = [name for name in all_names if (('AUC' not in name) and ('Median' in name))]
        std_names = [name for name in all_names if (('AUC' not in name) and ('Standard Deviation' in name))]
        size_names = [name for name in all_names if (('Size' in name))]
        max_auc_names = [name for name in all_names if (('AUC' in name) and ('Max' in name))]
        med_auc_names = [name for name in all_names if (('AUC' in name) and ('Median' in name))]
        std_auc_names = [name for name in all_names if (('AUC' in name) and ('Standard Deviation' in name))]
        name_triples = []
        std_triples = []

        for i in range(len(min_names)):
            for j in range(len(max_auc_names)):
                for k in range(len(size_names)):
                    name_triples.append((min_names[i], max_auc_names[j], size_names[k]))
                    std_triples.append((None, None, None))
                    name_triples.append((med_names[i], med_auc_names[j], size_names[k]))
                    std_triples.append((std_names[i], std_auc_names[j], None))

        if args.type == 'cnn':
            to_enumerate = ['Cicada V1 (search)', 'Cicada V2 (search)']

        pareto_3d_trials = []
        if not args.lite:
            loaded_study = optuna.load_study(study_name=args.name, storage=f"sqlite:///arch/{args.name}/{args.name}.db")
            for ((name_a, name_b, name_c), (std_name_a, std_name_b, std_name_c)) in zip(name_triples, std_triples):
                draw_study.plot_3d_pareto(name_a, name_b, name_c, std_name_a, std_name_b, std_name_c, args.name, label_seeds=False, name=f'{args.name}-all-and-pareto-{name_a.replace(".npy", "")}-{name_b.replace(".npy", "")}-{name_c.replace(".npy", "")}')
                pareto_3d_trials.append([name_a, name_b, name_c] + draw_study.get_3d_pareto(name_a, name_b, name_c, args.name, loaded_study))
        else:
            argnames = [f'{args.fromname}{i}' for i in range(args.batch)]
            loaded_studies = [optuna.load_study(study_name=f'{args.fromname}{i}', storage=f"sqlite:///arch/{args.fromname}{i}/{args.fromname}{i}.db") for i in range(args.batch)]
            for ((name_a, name_b, name_c), (std_name_a, std_name_b, std_name_c)) in zip(name_triples, std_triples):
                draw_study.plot_3d_pareto(name_a=name_a, name_b=name_b, name_c=name_c, std_name_a=std_name_a, std_name_b=std_name_b, std_name_c=std_name_c, argname=argnames, studies=loaded_studies, label_seeds=False, name=f'{args.name}-all-and-pareto-{name_a.replace(".npy", "")}-{name_b.replace(".npy", "")}-{name_c.replace(".npy", "")}')
                pareto_3d_trials.append([name_a, name_b, name_c] + draw_study.get_3d_pareto_lite(name_a, name_b, name_c, argnames, loaded_studies))

        return pareto_3d_trials

    def trial_plots():
        if not args.lite:
            names = [name for name in os.listdir(f'arch/{args.name}/trial_metrics') if (os.path.isdir(os.path.join(f'arch/{args.name}/trial_metrics', name)) and ('AUC' not in name) and ('Model Size' not in name))]
            auc_names = [name for name in os.listdir(f'arch/{args.name}/trial_metrics') if (os.path.isdir(os.path.join(f'arch/{args.name}/trial_metrics', name)) and ('AUC' in name) and ('Model Size' not in name))]
        else:
            names = [name for name in os.listdir(f'arch/{args.fromname}0/trial_metrics') if (os.path.isdir(os.path.join(f'arch/{args.fromname}0/trial_metrics', name)) and ('AUC' not in name) and ('Model Size' not in name))]
            auc_names = [name for name in os.listdir(f'arch/{args.fromname}0/trial_metrics') if (os.path.isdir(os.path.join(f'arch/{args.fromname}0/trial_metrics', name)) and ('AUC' in name) and ('Model Size' not in name))]
        name_pairs = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name_pairs.append((names[i], names[j]))
            for j in range(len(auc_names)):
                name_pairs.append((names[i], auc_names[j]))
        if not args.lite:
            trial_names = os.listdir(f'arch/{args.name}/trial_metrics/Validation Loss')
            trial_names_temp = np.array([int(trial_name.replace('.npy', '')) for trial_name in trial_names])
            trial_names_temp_ind = np.argsort(trial_names_temp)
            trial_names = [trial_names[i] for i in trial_names_temp_ind]
        else:
            trial_names=[]
            num_trials_in_study = []
            for i in range(args.batch):
                trial_names_study = os.listdir(f'arch/{args.fromname}{i}/trial_metrics/Validation Loss')
                trial_names_temp = np.array([int(trial_name.replace('.npy', '')) for trial_name in trial_names_study])
                num_trials_in_study.append(len(trial_names_temp))
                trial_names_temp_ind = np.argsort(trial_names_temp)
                trial_names = trial_names + [trial_names_study[i] for i in trial_names_temp_ind]

        pareto_3d_executions = []
        pareto_3d_executions_global = []
        if not args.lite:
            loaded_study = optuna.load_study(study_name=args.name, storage=f"sqlite:///arch/{args.name}/{args.name}.db")
            for (name_a, name_b) in name_pairs:
                for trial_name in trial_names:
                    draw_trial.plot_2d_pareto(name_x=name_a, name_y=name_b, trial_names=[trial_name], argname=args.name, label_seeds=False, show_non_pareto=True, show_legend=False, zoom = False, name=f'{args.name}-{trial_name}-all-and-pareto-{name_a}-{name_b}')
                    draw_trial.plot_2d_pareto(name_x=name_a, name_y=name_b, trial_names=[trial_name], argname=args.name, label_seeds=False, show_non_pareto=True, show_legend=False, zoom = True, name=f'{args.name}-{trial_name}-all-and-pareto-zoom-{name_a}-{name_b}')
                draw_trial.plot_2d_pareto(name_x=name_a, name_y=name_b, trial_names=trial_names, argname=args.name, min_pareto_length=3, label_seeds=False, show_non_pareto=False, show_legend=False, zoom=False, name=f'{args.name}-all-trials-all-and-pareto-{name_a}-{name_b}')
                draw_trial.plot_2d_pareto(name_x=name_a, name_y=name_b, trial_names=trial_names, argname=args.name, min_pareto_length=3, label_seeds=False, show_non_pareto=False, show_legend=False, zoom=True, name=f'{args.name}-all-trials-all-and-pareto-zoom-{name_a}-{name_b}')
                draw_trial.plot_3d_pareto_executions(name_x=name_a, name_y=name_b, name_z=f'Model Size (b)', trial_names=trial_names, argname=args.name, min_pareto_length=0, label_seeds=False, zoom=False, name=f'{args.name}-all-trials-and-pareto-over-all-trials-{name_a}-{name_b}')
                draw_trial.plot_3d_pareto_executions(name_x=name_a, name_y=name_b, name_z=f'Model Size (b)', trial_names=trial_names, argname=args.name, min_pareto_length=0, label_seeds=False, zoom=True, name=f'{args.name}-all-trials-and-pareto-over-all-trials-zoom-{name_a}-{name_b}')
                pareto_3d_executions.append([name_a, name_b] + draw_trial.get_3d_pareto_executions(name_x=name_a, name_y=name_b, trial_names=trial_names, argname=args.name, study=loaded_study))
                pareto_3d_executions_global.append([name_a] + [name_b] + draw_trial.get_pareto_executions(name_x=name_a, name_y=name_b, name_z=f'Model Size (b)', argname=args.name, study=loaded_study, trial_names=trial_names))
        else:
            argnames = [f'{args.fromname}{i}' for i in range(args.batch)]
            loaded_studies = [optuna.load_study(study_name=f'{args.fromname}{i}', storage=f"sqlite:///arch/{args.fromname}{i}/{args.fromname}{i}.db") for i in range(args.batch)]
            for (name_a, name_b) in name_pairs:
                draw_trial.plot_2d_pareto(name_x=name_a, name_y=name_b, trial_names=trial_names, num_trials_in_study=num_trials_in_study, argname=argnames, show_non_pareto=False, show_legend=False, zoom=False, name=f'{args.name}-all-trials-all-and-pareto-{name_a}-{name_b}')
                draw_trial.plot_2d_pareto(name_x=name_a, name_y=name_b, trial_names=trial_names, num_trials_in_study=num_trials_in_study, argname=argnames, show_non_pareto=False, show_legend=False, zoom=True, name=f'{args.name}-all-trials-all-and-pareto-zoom-{name_a}-{name_b}')
                print('finished plot_2d_pareto')
                draw_trial.plot_3d_pareto_executions(name_x=name_a, name_y=name_b, name_z=f'Model Size (b)', trial_names=trial_names, num_trials_in_study=num_trials_in_study, argname=argnames, min_pareto_length=0, label_seeds=False, zoom=False, name=f'{args.name}-all-trials-and-pareto-over-all-trials-{name_a}-{name_b}')
                draw_trial.plot_3d_pareto_executions(name_x=name_a, name_y=name_b, name_z=f'Model Size (b)', trial_names=trial_names, num_trials_in_study=num_trials_in_study, argname=argnames, min_pareto_length=0, label_seeds=False, zoom=True, name=f'{args.name}-all-trials-and-pareto-over-all-trials-zoom-{name_a}-{name_b}')
                print('finished plot_3d_pareto_executions')
                pareto_3d_executions.append([name_a, name_b] + draw_trial.get_3d_pareto_executions(name_x=name_a, name_y=name_b, trial_names=trial_names, num_trials_in_study=num_trials_in_study, argname=argnames, study=loaded_studies))
                print('finished get_3d_pareto_executions')
                pareto_3d_executions_global.append([name_a] + [name_b] + draw_trial.get_pareto_executions(name_x=name_a, name_y=name_b, name_z=f'Model Size (b)', argname=argnames, study=loaded_studies, trial_names=trial_names, num_trials_in_study=num_trials_in_study))
                print('finished get_pareto_executions')
        return pareto_3d_executions, pareto_3d_executions_global

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

    draw_trial = Draw(output_dir=f'arch/{args.name}/trial_plots/', interactive=args.interactive)
    draw_study = Draw(output_dir=f'arch/{args.name}/study_plots/', interactive=args.interactive)

    if args.batch != 0:
        compile_batch_study_metrics(args.name, args.fromname, args.batch)
    compile_study_metrics(args.name)
    best_executions, best_executions_global = trial_plots()
    return
    if not args.lite:
        best_trials = search_plots()
    write_executions_to_csv(best_executions_global)

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
        help="Type of model. One of cnn, vit, bnn, ban, bwn, bitbnn.",
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
        help="Number of trials",
        default=10, 
    )
    parser.add_argument(
        "-p", "--parallels",
        type=int,
        help="Number of trials",
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
    parser.add_argument(
        "-b", "--batch",
        type=int, 
        help="Number of batch jobs to parse. If not a batch job, 0.", 
        default=0, 
    )
    parser.add_argument(
        "--fromname", "-f",
        type=str,
        default="example",
        help="Name of studies to read from (w/o index)",
    )
    parser.add_argument(
        "--lite", "-l",
        type=int,
        default="0",
        help="To only plot trial and study plots, but skip combining studies and copying plots/metrics. Only use if is also a batch job.",
    )
    new_args = parser.parse_args()
    if new_args.batch != None: loaded_args = ['--config'] + ['misc/config.yml'] + ['--type'] + ['cnn'] + ['--epochs'] + ['25'] + ['--executions'] + ['50'] + ['--trials'] + ['-1'] + ['--parallels'] + ['1'] + ['--jobflavour'] + ['workday']
    else: loaded_args = load_args(new_args.name)
    main(parser.parse_args(['--name'] + [f"{new_args.name}"] + ['--fromname'] + [f'{new_args.fromname}'] + ['--batch'] + [f"{new_args.batch}"] + ['--lite'] + [f"{new_args.lite}"] + loaded_args))
