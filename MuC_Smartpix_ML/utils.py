import importlib
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import dgl
import signal

def buildFromConfig(conf, run_time_args = {}):
    device = run_time_args.get('device', 'cpu')
    if 'module' in conf:
        module = importlib.import_module(conf['module'])
        cls = getattr(module, conf['class'])
        args = conf['args'].copy()
        if 'weight' in args and isinstance(args['weight'], list):
            args['weight'] = torch.tensor(args['weight'], dtype=torch.float, device=device)
        # Remove device from run_time_args to not pass it to the class
        run_time_args = {k: v for k, v in run_time_args.items() if k != 'device'}
        return cls(**args, **run_time_args)
    else:
        print('No module specified in config. Returning None.')

def load_config(config_file):
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    include_config(conf)
    return conf