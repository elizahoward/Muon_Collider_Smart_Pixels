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

def main():

    parser = argparse.ArgumentParser(description="Evaluate SmartPix ML Models with Quantization")
    add_arg = parser.add_argument
    add_arg('--ckpt_path', type=str, default=None, help='Path to model checkpoint')
    add_arg('--model', type=str, choices=['model1', 'model2', 'model3'], help='Model to evaluate')

    args = parser.parse_args()

    if doModel1:
        """Example usage of Model1"""
        print("=== Model1 Example Usage ===")
        
        # Initialize Model1
        model1 = Model1(
            tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
            # tfRecordFolder = "../ryan/tf_records1000Daniel",
            initial_lr=1e-3, 
            end_lr=1e-4,
            power=2,
            bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
        )
        
        # Run complete pipeline
        results = model1.runAllStuff(numEpochs= numEpochs1)
        
        print("Model1 quantization testing completed successfully!")

    if doModel2:
        """Example usage of Model2"""
        print("=== Model2 Example Usage ===")
        
        # Initialize Model2
        model2 = Model2(
            tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
            xz_units=8,
            yl_units=8,
            merged_units_1=64,
            merged_units_2=32,
            merged_units_3=16,
            dropout_rate=0.1,
            initial_lr=1e-3,
            end_lr=1e-4,
            power=2,
            bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
        )
        
        # Run complete pipeline
        results = model2.runAllStuff(numEpochs= numEpochs2)
        
        print("Model2 quantization testing completed successfully!")

    if doModel3:
        """Example usage of Model3"""
        print("=== Model3 Example Usage ===")
        
        # Initialize Model3
        model3 = Model3(
            tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records16384_data_shuffled_single_bigData/",
            # xz_units=8,
            # yl_units=8,
            # merged_units_1=64,
            # merged_units_2=32,
            # merged_units_3=16,
            # dropout_rate=0.1,
            initial_lr=1e-3,
            end_lr=1e-4,
            power=2,
            bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
        )
        
        # Run complete pipeline
        results = model3.runAllStuff(numEpochs= numEpochs3)
        
        print("Model3 quantization testing completed successfully!")

if __name__== "__main__":
    main()