import sys
sys.path.append('../MuC_Smartpix_ML/')
sys.path.append('../daniel/')
sys.path.append('../ryan/')
sys.path.append('../eric/')
import argparse
import os

from utils import buildFromConfig, load_config

def main():

    parser = argparse.ArgumentParser(description="Evaluate SmartPix ML Models with Quantization")
    add_arg = parser.add_argument
    add_arg('--config', type=str, default=None, help='Path to model config')
    add_arg('--training_dir', type=str, default=None, help='Path to training directory')
    add_arg('--quantization', type=int, default=-1, help='Which version of quantization to use. -1 for no quantization')

    args = parser.parse_args()

    config = load_config(args.config)
    if args.training_dir is not None:
        config['TrainingDirectory'] = args.training_dir

    quantization_tag = 'unquantized' 
    if args.quantization > 0:
        quantization_tag = f'quantized_{args.quantization}bit'

    config['Model']['args']['loadModel'] = True
    config['Model']['args']['modelPath'] = os.path.join(config['TrainingDirectory'], f"models/{config['Model']['class']}_{quantization_tag}.h5")

    model = buildFromConfig(config['Model'])
    model.runEval(False)

if __name__== "__main__":
    main()