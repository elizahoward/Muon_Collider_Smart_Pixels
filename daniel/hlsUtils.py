"""
Author: Daniel Abadjiev
Date: March 3, 2026
Description: Various utils I started writing for testCataputlNtbk.ipynb, but I think are in general useful
Perhaps I should change the script name
"""

import sys
sys.path.append('../MuC_Smartpix_ML/')
sys.path.append('../daniel/')
sys.path.append('../ryan/')
sys.path.append('../eric/')

from Model_Classes import SmartPixModel
from model1 import Model1
from model2 import Model2
from model3 import Model3
from model2_5 import Model2_5
from ASICModel import ModelASIC
import tensorflow as tf

tfRecordFolder = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData"

def flattenTfData(modelType, doTrain=True):
    if modelType not in ["1","2","2.5","3",1,2,3,2.5,"ASIC"]:
        raise TypeError("Not supported model type")
    if modelType in ["1", 1]:
        model = Model1(tfRecordFolder = tfRecordFolder)
    elif modelType in ["2",2]:
        model = Model2(tfRecordFolder = tfRecordFolder)    
    elif modelType in ["2.5", 2.5]:
        model = Model2_5(tfRecordFolder = tfRecordFolder)        
    elif modelType in ["3", 3]:
        model = Model3(tfRecordFolder = tfRecordFolder)    
    elif modelType in ["ASIC"]:
        model = ModelASIC(tfRecordFolder = tfRecordFolder)
    else:
        raise TypeError("Not supported model type")
    model.loadTfRecords()
    odgTrain = model.training_generator
    odgTest = model.validation_generator
    xTest, yTest = odgToVect(odgTest)
    xTestList = [xTest[key].numpy() for key in xTest.keys()]
    if doTrain:
        xTrain, yTrain = odgToVect(odgTrain)
        return xTest, yTest, xTestList, xTrain, yTrain
    else:
        return xTest, yTest, xTestList


def odgToVect(odg):
    qq = [odg.__getitem__(i) for i in range(odg.__len__())]
    y_test = tf.concat([qq[i][1] for i in range(odg.__len__())],0)
    # tf.concat([qq[i][0]["x_profile"] for i in range(odg.__len__())],0)
    x_test = {key: tf.concat([qq[i][0][key] for i in range(odg.__len__())],0) for key in qq[0][0].keys()}
    return x_test, y_test

