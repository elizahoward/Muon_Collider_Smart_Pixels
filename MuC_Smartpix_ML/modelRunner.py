##Common runner script

doModel1 = True
doModel2 = False
doModel3 = False
doModelAsic = False
numEpochs1 = 1;
numEpochs2 = 120;
numEpochs3 = 2;
numEpochsAsic = 100;


# import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# import 
import os
import sys
sys.path.append('../MuC_Smartpix_ML/')
sys.path.append('../daniel/')
sys.path.append('../ryan/')
sys.path.append('../eric/')

from Model_Classes import SmartPixModel
from model1 import Model1
from model2 import Model2
from model3 import Model3
if doModelAsic:
    import os
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    # import ASICModel
    from ASICModel import ModelASIC

if doModel1:
    """Example usage of Model1"""
    print("=== Model1 Example Usage ===")
    
    # Initialize Model1
    model1 = Model1(
        tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try3_eric/filtering_records16384_data_shuffled_single_bigData/",
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
        tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try3_eric/filtering_records16384_data_shuffled_single_bigData/",
        xz_units=20,
        yl_units=16,
        merged_units_1=32,
        merged_units_2=32,
        merged_units_3=16,
        dropout_rate=0.1,
        initial_lr=0.0008479354308396083,
        end_lr=1e-4,
        power=2,
        # bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)],  # Test 16, 8, 6, 4, 3, and 2-bit quantization
        bit_configs = [(4, 0)],  # Test 16, 8, 6, 4, 3, and 2-bit quantization
        trainUnquantized = False
    )
    
    # Run complete pipeline
    results = model2.runAllStuff(numEpochs= numEpochs2)
    
    print("Model2 quantization testing completed successfully!")

if doModel3:
    """Example usage of Model3"""
    print("=== Model3 Example Usage ===")
    
    # Initialize Model3
    model3 = Model3(
        tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try3_eric/filtering_records16384_data_shuffled_single_bigData/",
        conv_filters=24,
        kernel_rows=3,
        kernel_cols=3,
        scalar_dense_units=64,
        merged_dense_1=140,
        merged_dense_2=56,
        dropout_rate=0.1,
        initial_lr=0.0034854093215866385,
        end_lr=5.3e-05,
        power=2,
        bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]  # Test 16, 8, 6, 4, 3, and 2-bit quantization
    )
    
    # Run complete pipeline
    results = model3.runAllStuff(numEpochs= numEpochs3)
    
    print("Model3 quantization testing completed successfully!")

if doModelAsic:
    print("Model ASIC ")
    modelA = ModelASIC(
        tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try3_eric/filtering_records16384_data_shuffled_single_bigData/",
        initial_lr=1e-3, 
        end_lr=1e-4,
        power=2,
    )
    results = modelA.runAllStuff(numEpochs = numEpochsAsic)
    print("Model Asic all testing done")