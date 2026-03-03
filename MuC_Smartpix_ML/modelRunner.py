import os
import sys
sys.path.append('../MuC_Smartpix_ML/')
sys.path.append('../daniel/')
sys.path.append('../ryan/')
sys.path.append('../eric/')

from Model_Classes import SmartPixModel
from model1 import Model1
from model2 import Model2
from model2_5 import Model2_5
from model3 import Model3
import argparse
import yaml
from utils import buildFromConfig, load_config, save_config

def main():

    parser = argparse.ArgumentParser(description='Run Smartpix ML Models')
    add_arg = parser.add_argument
    add_arg('--run1', action='store_true', help='Run Model1')
    add_arg('--run2', action='store_true', help='Run Model2')
    add_arg('--run2_5', action='store_true', help='Run Model2_5')
    add_arg('--run3', action='store_true', help='Run Model3')
    add_arg('--n_layers', type=int, default=2, help='Number of layers for Model1')
    add_arg('--width', type=int, default=6, help='Width of layers for Model1')
    add_arg('--dense_units', type=int, default=32, help='Dense layer units for Model2_5')
    add_arg('--epochs', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()

    if args.run1:
        """Example usage of Model1"""
        print("=== Model1 Example Usage ===")

        config = load_config('model_configs/model1.yaml')
        config['Model']['args']['n_layers'] = args.n_layers
        config['Model']['args']['width'] = args.width
        config['ModelName'] = f'Model1_n{args.n_layers}_w{args.width}'

        model1 = buildFromConfig(config['Model'])
        
        # Run complete pipeline
        results, output_dir = model1.runAllStuff(numEpochs= args.epochs)
        config['TrainingDirectory'] = output_dir  # Update config with actual output directory
        save_config(config, output_dir)  # Save the config used for this run
        
        print("Model1 quantization testing completed successfully!")

    if args.run2:
        """Example usage of Model2"""
        print("=== Model2 Example Usage ===")

        config = load_config('model_configs/model2.yaml')
        model2 = buildFromConfig(config['Model'])  
        
        # Run complete pipeline
        results, output_dir = model2.runAllStuff(numEpochs= args.epochs)
        config['TrainingDirectory'] = output_dir  # Update config with actual output directory
        save_config(config, output_dir)  # Save the config used for this run
        
        print("Model2 quantization testing completed successfully!")

    if args.run2_5:
        """Example usage of Model2"""
        print("=== Model2 Example Usage ===")
        
        config = load_config('model_configs/model2_5.yaml')
        config['Model']['args']['dense_units'] = args.dense_units
        config['Model']['args']['z_global_units'] = args.dense_units // 4
        config['Model']['args']['dense2_units'] = args.dense_units // 2
        config['Model']['args']['dense3_units'] = args.dense_units // 4
        config['ModelName'] = f'Model2_5_dense{args.dense_units}'
        model2 = buildFromConfig(config['Model'])  
                
        # Run complete pipeline
        results, output_dir = model2.runAllStuff(numEpochs= args.epochs)
        config['TrainingDirectory'] = output_dir  # Update config with actual output directory
        save_config(config, output_dir)  # Save the config used for this run
        
        print("Model2 quantization testing completed successfully!")

    if args.run3:
        """Example usage of Model3"""
        print("=== Model3 Example Usage ===")

        config = load_config('model_configs/model3.yaml')
        model3 = buildFromConfig(config['Model'])
        
        # Run complete pipeline
        results, output_dir = model3.runAllStuff(numEpochs= args.epochs)
        config['TrainingDirectory'] = output_dir  # Update config with actual output directory
        save_config(config, output_dir)  # Save the config used for this run
        
        print("Model3 quantization testing completed successfully!")


if __name__ == "__main__":
    main()
