#Author: Daniel Abadjiev
#Date: Apr 28, 2026
#Description: Turn the dataRate.ipynb into a functions that can be called by Model_Classes.py in the .evaluate

import tensorflow as tf
from datetime import datetime
from tfLoaderUtils import *
import numpy as np
import qkeras
import matplotlib.pyplot as plt
from dataRateUtils import *

def pixPredictToDataRate(yTest,nPixes,predictions,cut=0.138891):
    backGnd = (yTest.numpy().ravel()==0)
    # print(backGnd.shape)
    # print(predictions.shape)
    # print(predictions[backGnd])
    backTotal = (predictions[backGnd]) #I think this can be cleaned up substantially?
    backAccept = np.where(predictions[backGnd]>cut)
    plt.hist(backTotal); plt.yscale('log'); plt.vlines([cut],0,10000,"red")
    print("accepted background samples: ",np.size(backAccept))
    print("total background samples: ",np.size(backTotal))
    print("backgorund acceptance: ",np.size(backAccept)/np.size(backTotal))

    numBackPixesTotal = np.sum(nPixes.numpy()[backGnd])
    print("Total background pixel count: ",numBackPixesTotal)
    numBackPixesPostFilter = np.sum(nPixes.numpy()[backGnd][backAccept])
    print("Accepted background pixel count: ",numBackPixesPostFilter)
    print("Data rate of background acceptance: ",numBackPixesPostFilter/numBackPixesTotal)
    return numBackPixesPostFilter,numBackPixesTotal,numBackPixesPostFilter/numBackPixesTotal

def genNpixAndGetDataRate(model,predictions):
    nPixes, yTest = getNpixYtest(model)
    return pixPredictToDataRate(yTest,nPixes,predictions)


#This is goint to be just for testing in the notebook
def modelSpecsToDataRate(modelType, quantizedModel,
                    tfRecordFolder = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/filtering_records16384_data_shuffled_single_bigData"
                    ):
    if modelType == 2.5:
        model = Model2_5(tfRecordFolder = tfRecordFolder)   
    configName = "justThisOne"
    model.models[configName] = quantizedModel
    model.evaluate(config_name = configName)
    predictions = model.models[configName].predict(model.validation_generator, verbose=1)
    print(predictions)
    predictions = predictions.ravel()
    return genNpixAndGetDataRate(model,predictions)
