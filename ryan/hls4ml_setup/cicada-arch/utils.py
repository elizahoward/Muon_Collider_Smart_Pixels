import argparse
import os
import shlex

import numpy as np
from pathlib import Path


class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid path".format(prospective_dir)
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                "{0} is not a readable directory".format(prospective_dir)
            )


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid file".format(prospective_file)
            )
        else:
            setattr(namespace, self.dest, prospective_file)


class CreateFolder(argparse.Action):
    """
    Custom action: create a new folder if not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    """

    def create_folder(self, folder_name):
        """
        Create a new directory if not exist. The action might throw
        OSError, along with other kinds of exception
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # folder_name = folder_name.rstrip(os.sep)
        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)
        setattr(namespace, self.dest, folders)


def save_to_npy(val, path):
    if os.path.exists(path):
        arr_temp = np.load(path)
        arr_temp = np.append(arr_temp, val)
    else:
        arr_temp = np.array(val)
    np.save(path, arr_temp)


def predict_single_image(model, image):
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image, verbose=False)
    return np.squeeze(pred, axis=0)

def save_args(args):
    """Save command-line arguments to a file, excluding 'name'."""
    folder = Path(f"arch/{args.name}")
    folder.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists
    args_file = folder / "args.txt"

    # Convert args to a dictionary and exclude 'name'
    arg_dict = {k: v for k, v in vars(args).items() if k != "name" and v is not False}

    # Construct argument list
    arg_list = []
    for k, v in arg_dict.items():
        arg_key = f"--{k.replace('_', '-')}"  # Convert underscores to dashes
        if isinstance(v, bool):  # Only store True flags
            if v:
                arg_list.append(arg_key)
        else:
            arg_list.extend([arg_key, str(v)])

    # Write args to file
    with args_file.open("w") as f:
        f.write(" ".join(arg_list) + "\n")

def load_args(name):
    """Load saved arguments for the given study."""
    args_file = Path(f"arch/{name}/args.txt")
    
    if not args_file.exists():
        raise FileNotFoundError(f"Arguments file not found: {args_file}")

    with args_file.open() as f:
        command_str = f.readline().strip()

    return shlex.split(command_str)