import os
import argparse
import logging
import sys
import optuna

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def main(args) -> None:

    labels = [
        'Mean Signal', 
        'SUEP', 
        'H to Long Lived', 
        'VBHF to 2C', 
        'TT', 
        'SUSY ggHbb', 
    ]
    
    for i in range(args.batch):

        for foldername in [
            'arch/', 
            f'../arch/{args.name}{i}/', 
            f'../arch/{args.name}{i}/study_plots/', 
            f'../arch/{args.name}{i}/trial_plots/', 
            f'../arch/{args.name}{i}/execution_plots/', 
            f'../arch/{args.name}{i}/models/', 
            f'../arch/{args.name}{i}/study_metrics/', 
            f'../arch/{args.name}{i}/trial_metrics/', 
            f'../arch/{args.name}{i}/trial_metrics/Mean Signal AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[0]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[1]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[2]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[3]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[4]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[5]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/Validation Loss/', 
            f'../arch/{args.name}{i}/trial_metrics/Model Size (number of parameters)/', 
            f'../arch/{args.name}{i}/trial_metrics/Model Size (b)/', 
            ]:
            if not os.path.exists(foldername):
                os.mkdir(foldername)

        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage_name = f"sqlite:///../arch/{args.name}{i}/{args.name}{i}.db"

        study = optuna.create_study(
                directions=['maximize', 'minimize', 'minimize', 'minimize'], 
                study_name=f"{args.name}{i}", 
                storage=storage_name, 
                load_if_exists=True, 
            )
        
        if (args.type == 'cnn') and (args.guided == 1):
            study.enqueue_trial({
                "n_conv_layers": 0, 
                "n_dense_layers": 1, 
                "n_layers": 1, 
                "n_dense_units_0": 4, 
                "q_kernel_conv_bits": 12, 
                "q_kernel_conv_ints": 3, 
                "q_kernel_dense_bits": 8, 
                "q_kernel_dense_ints": 1, 
                "q_bias_dense_bits": 8, 
                "q_bias_dense_ints": 3, 
                "q_activation_bits": 10, 
                "q_activation_ints": 6, 
                "shortcut": False, 
                "dropout": 0., 
            }) # Include cicada_v1
            study.enqueue_trial({
                "n_conv_layers": 1, 
                "n_dense_layers": 1, 
                "n_layers": 2, 
                "n_filters_0": 4, 
                "kernel_width_0": 2, 
                "kernel_height_0": 2, 
                "stride_width_0": 2, 
                "stride_height_0": 2, 
                "use_bias_conv": False, 
                "n_dense_units_0": 4, 
                "q_kernel_conv_bits": 12, 
                "q_kernel_conv_ints": 3, 
                "q_kernel_dense_bits": 8, 
                "q_kernel_dense_ints": 1, 
                "q_bias_dense_bits": 8, 
                "q_bias_dense_ints": 3, 
                "q_activation_bits": 10, 
                "q_activation_ints": 6, 
                "shortcut": False, 
                "dropout": 0., 
            }) # Include cicada_v2
        elif (args.type == 'bnn') and (args.guided == 1):
            study.enqueue_trial({
                "binary_type": "bnn", 
                "n_conv_layers": 0, 
                "n_dense_layers": 1, 
                "n_layers": 1, 
                "n_dense_units_0": 8, 
                "q_activation_bits": 10, 
                "q_activation_ints": 6, 
                "shortcut": False, 
                "dropout": 0., 
            }) # Include cicada_v1
            study.enqueue_trial({
                "binary_type": "bnn", 
                "n_conv_layers": 1, 
                "n_dense_layers": 1, 
                "n_layers": 2, 
                "n_filters_0": 8, 
                "kernel_width_0": 2, 
                "kernel_height_0": 2, 
                "stride_width_0": 2, 
                "stride_height_0": 2, 
                "use_bias_conv": False, 
                "n_dense_units_0": 7, 
                "q_activation_bits": 10, 
                "q_activation_ints": 6, 
                "shortcut": False, 
                "dropout": 0., 
            }) # Include cicada_v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Create dirs""")
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
        "-b", "--batch",
        type=int, 
        help="Number of batch jobs to create.", 
        default=0, 
    )
    parser.add_argument(
        "-g", "--guided",
        type=int, 
        help="Whether created study will be guided", 
        default=1,
    )
    main(parser.parse_args())
