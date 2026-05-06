#Author: Daniel Abadjiev
#Date: Apr 28, 2026
#Description: Turn the dataRate.ipynb into a functions that can be called by Model_Classes.py in the .evaluate

import tensorflow as tf
from datetime import datetime
import tfLoaderUtils
from tfLoaderUtils import *
import numpy as np
import qkeras
import matplotlib.pyplot as plt
from dataRateUtils import *


# I feel like this should be unnecessary...
import sys
sys.path.append('../MuC_Smartpix_ML/')
sys.path.append('../daniel/')
sys.path.append('../ryan/')
sys.path.append('../eric/')

# from Model_Classes import SmartPixModel
from model1 import Model1
from model2 import Model2
from model3 import Model3
from model2_5 import Model2_5


def pixPredictToDataRate(yTest,nPixes,predictions,cut=0.138891,doPrintExtra = True, doPrintData=True, doPlot = True):
    predictions = predictions.ravel() #Doesn't hurt to do it again
    backGnd = (yTest.numpy().ravel()==0)
    # print(backGnd.shape)
    # print(predictions.shape)
    # print(predictions[backGnd])
    backTotal = (predictions[backGnd]) #I think this can be cleaned up substantially?
    backAccept = np.where(predictions[backGnd]>cut)
    # backAccept = (np.logical_and(predictions>cut,(yTest.numpy().ravel()==0)))
    if doPlot:
        plt.hist(backTotal,bins=100,label="all background");
        plt.hist(predictions[backGnd][backAccept],alpha=0.4,label="accepted background")
        plt.legend()
        plt.yscale('log'); plt.vlines([cut],0,10000,"red")
        plt.title(f"prediction of background sample with thresh cut {cut}")
        plt.ylabel("N background samples")
        plt.xlabel("prediction")
        plt.show()
    if doPrintExtra:
        print()
        print("accepted background samples: ",np.size(backAccept))
        print("total background samples: ",np.size(backTotal))
        print("backgorund acceptance: ",np.size(backAccept)/np.size(backTotal))

    numBackPixesTotal = np.sum(nPixes.numpy()[backGnd])
    if doPrintData:
        print("Total background pixel count: ",numBackPixesTotal)
    pixOfBack = nPixes.numpy()[backGnd]
    # print(pixOfBack)
    # print(backAccept)
    # print(pixOfBack[backAccept])
    numBackPixesPostFilter = np.sum(pixOfBack[backAccept])
    if doPrintData:
        print("Accepted background pixel count: ",numBackPixesPostFilter)
        print("Data rate of background acceptance: ",numBackPixesPostFilter/numBackPixesTotal)
    return numBackPixesPostFilter,numBackPixesTotal,numBackPixesPostFilter/numBackPixesTotal

def genNpixAndGetDataRate(model,predictions,cut=0.138891,doPrintExtra = True, doPrintData=True, doPlot = True):
    # print(predictions)
    predictions = predictions.ravel() #Doesn't hurt to do it again
    # print(predictions)
    # predictions = predictions.ravel() #Doesn't hurt to do it again
    # print(predictions)
    nPixes, yTest = tfLoaderUtils.getNpixYtest(model)
    return pixPredictToDataRate(yTest,nPixes,predictions,cut=cut,doPrintExtra=doPrintExtra, doPrintData=doPrintData, doPlot=doPlot)


#This is goint to be just for testing in the notebook
def modelSpecsToDataRate(modelType, quantizedModel,cut=0.138891,
                    tfRecordFolder = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V2_Apr/TF_Records/filtering_records16384_data_shuffled_single_bigData"
                    ):
    if modelType == 2.5:
        model = Model2_5(tfRecordFolder = tfRecordFolder)   
    configName = "justThisOne"
    model.models[configName] = quantizedModel
    model.evaluate(config_name = configName)
    # #Actually the next part of the code is in .evaluate
    # print(".evaluate is done, so now doing it again for some reason")
    # predictions = model.models[configName].predict(model.validation_generator, verbose=1)
    # print(predictions)
    # predictions = predictions.ravel()
    # return genNpixAndGetDataRate(model,predictions,cut=cut)


def plotPredictVsAll(model,predictions):
    xTest, yTest = tfLoaderUtils.getXYtest(model)


    # print(xTest.keys())
    from matplotlib import colors
    keys = ["nModule","x_local","z_global","y_local","nPix"]
    keys =  list(set(keys) & set(xTest.keys()))
    print(keys)
    for idk,key in enumerate(keys):
        backX = (xTest[key][(yTest.numpy().ravel()==0)]).numpy()
        sigX = (xTest[key][(yTest.numpy().ravel()==1)]).numpy()
        predBack = predictions[(yTest.numpy().ravel()==0)]
        predSig = predictions[(yTest.numpy().ravel()==1)]
        plt.figure(figsize=(5,8))
        plt.subplot(4,1,1)
        plt.plot(predBack,backX,".",alpha=0.5,label="background",markersize=2)
        plt.plot(predSig,sigX,".",alpha=0.5,label="Signal",markersize=2)
        plt.legend()
        plt.ylabel(key)
        plt.title(f"{key} vs. neural network prediction")
        plt.subplot(4,1,2)
        plt.hist2d(predictions.ravel(),xTest[key].numpy(),norm=colors.LogNorm())
        plt.colorbar()
        plt.xlabel("prediction")
        plt.ylabel(key)
        plt.title("all test vectors")
        # plt.show()
        plt.subplot(4,1,3)
        plt.hist2d(predBack.ravel(),backX,norm=colors.LogNorm())
        plt.colorbar()
        plt.xlabel("prediction")
        plt.ylabel(key)
        plt.title("just background")
        # plt.show()
        plt.subplot(4,1,4)
        plt.hist2d(predSig.ravel(),sigX,norm=colors.LogNorm())
        plt.colorbar()
        plt.xlabel("prediction")
        plt.ylabel(key)
        plt.title("just signal")
        plt.tight_layout()
        plt.show()

    plt.hist2d(predictions.ravel(),yTest.numpy().ravel(),norm=colors.LogNorm())
    plt.xlabel("predictions")
    plt.ylabel("Bib vs. signal")
    plt.colorbar()
    plt.show()
    return