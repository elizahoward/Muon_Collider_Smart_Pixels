import sys
sys.path.append('../MuC_Smartpix_ML/')
sys.path.append('../daniel/')
sys.path.append('../ryan/')
sys.path.append('../eric/')
import argparse

from Model_Classes import SmartPixModel
from model1 import Model1
from model2 import Model2
from model3 import Model3
from utils import buildFromConfig, load_config

def main():

    parser = argparse.ArgumentParser(description="Evaluate SmartPix ML Models with Quantization")
    add_arg = parser.add_argument
    add_arg('--config', type=str, default=None, help='Path to model config')

    args = parser.parse_args()

    config = load_config(args.config)
    model  = buildFromConfig(config['Model'])
    results = model.runEval()
    print(results)

if __name__== "__main__":
    main()