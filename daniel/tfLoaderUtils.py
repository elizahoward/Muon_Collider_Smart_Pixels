"""
Author: Daniel Abadjiev
Date: March 3, 2026
Description: Various utils I started writing for testCataputlNtbk.ipynb, but I think are in general useful
Perhaps I should change the script name
Update Apr 3: Actually to avoid a very long utils script, changed this script to tfLoaderUtils.py to be focused on loading the tf data
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
import numpy as np

import pandas as pd

def flattenTfData(modelType, doTrain=True, tfRecordFolder="",includenPix=False):
    if tfRecordFolder=="":
        tfRecordFolder = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData"
        tfRecordFolder = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/filtering_records16384_data_shuffled_single_bigData"

    if modelType not in ["1","2","2.5","3",1,2,3,2.5,"ASIC"]:
        raise TypeError("Not supported model type")
    if modelType in ["1", 1]:
        model = Model1(tfRecordFolder = tfRecordFolder)
        model.x_feature_description: list = ['z_global','x_size', 'y_size', 'y_local'] #remove nModule and x_local
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
    if includenPix:        
        model.x_feature_description = model.x_feature_description + ["nPix"]
    model.loadTfRecords()
    odgTrain = model.training_generator
    odgTest = model.validation_generator
    xTest, yTest = odgToVect(odgTest)
    xTestList = [xTest[key].numpy() for key in xTest.keys()]
    if doTrain:
        xTrain, yTrain = odgToVect(odgTrain)
        return xTest, yTest, xTestList, xTrain, yTrain, 
    else:
        return xTest, yTest, xTestList

def getNpixYtest(model):
    if "nPix" not in model.x_feature_description:
        model.x_feature_description = model.x_feature_description + ["nPix"]
        model.loadTfRecords()
    xTest, yTest = odgToVect(model.validation_generator)
    return xTest["nPix"], yTest

def getXYtest(model):
    if "nPix" not in model.x_feature_description:
        model.x_feature_description = model.x_feature_description + ["nPix"]
        model.loadTfRecords()
    xTest, yTest = odgToVect(model.validation_generator)
    return xTest, yTest

#     allowed_features = ['cluster', 'x_profile', 'y_profile', 'x_size', 'y_size', 'y_local', 'z_global', 'total_charge', 'adjusted_hit_time', 'adjusted_hit_time_30ps_gaussian', 'adjusted_hit_time_60ps_gaussian','nPix',"x_local","nModule"]
def getPredVarDF(model,predictions,keys=['x_size', 'y_size', "nModule","x_local","y_local", 'z_global',"nPix","pt"]):
    #, 'total_charge', 'adjusted_hit_time', 'adjusted_hit_time_30ps_gaussian', 'adjusted_hit_time_60ps_gaussian', #potentially add these, also add pt once it's in the tfRecords
    for key in keys:
        if key not in model.x_feature_description:
            model.x_feature_description = model.x_feature_description + [key]
    model.loadTfRecords()
    xTest, yTest = odgToVect(model.validation_generator)
    predVarDF = pd.DataFrame.from_dict({key:xTest[key].numpy() for key in keys})
    predVarDF["prediction"] = predictions.ravel()
    predVarDF["trueY"] = yTest.numpy()
    try:
        predVarDF = predVarDF.rename(columns={"x_size": "xSize", "y_size": "ySize", "y_local": "y-local", "z_global":"z-global"})
    except Exception as e:
        print("renaming columns of dataframe with xvariables, predictions, and truth failed because ")
        print(e.message, e.args)
    
    predVarDF["xSize"] = 21*(predVarDF["xSize"])
    predVarDF["ySize"] = 13*(predVarDF["ySize"])

    predVarDF["y-local"] = 8.5*(predVarDF["y-local"])
    predVarDF["z-global"] = 65*(predVarDF["z-global"])

    theta = np.arctan2(30, predVarDF['z-global'])
    predVarDF['eta'] = -np.log(np.tan(theta / 2))
    return predVarDF


def odgToVect(odg):
    qq = [odg.__getitem__(i) for i in range(odg.__len__())]
    y_test = tf.concat([qq[i][1] for i in range(odg.__len__())],0)
    # tf.concat([qq[i][0]["x_profile"] for i in range(odg.__len__())],0)
    x_test = {key: tf.concat([qq[i][0][key] for i in range(odg.__len__())],0) for key in qq[0][0].keys()}
    return x_test, y_test

