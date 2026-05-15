#Author: Daniel Abadjiev
#Date: May 8, 2026
#Description: Utilities for plotting variables like zglobal, xsize, nPix vs. the prediction by a model
#Converted from the dataRate.ipynb notebook

import tensorflow as tf
from datetime import datetime
from tfLoaderUtils import *
import numpy as np
import qkeras
import matplotlib.pyplot as plt
import dataRateUtils 
from typing import Callable, Any, Protocol
import functools
from validationPlots.plotUtils import * #double check this import will work if call from another folder
import pandas as pd
from pathlib import Path



tf.config.set_visible_devices([], 'GPU') #needed for multiprocessing

#This is kind of general, the first cell in the notebook, maybe generalize to something else later
def loadQModel(filepath = "", modelType=2):
    if filepath == "":
        filepath="../../Muon_Collider_Smart_Pixels/eric/model2.5_quantized_4w0i_hyperparameter_results_20260222_004048/model_trial_000.h5"
        filepath="../../Muon_Collider_Smart_Pixels/eric/model2.5_quantized_8w0i_hyperparameter_results_20260228_020952/model_trial_0.h5"
        filepath="./model_trial_0.h5"
        if modelType==2:
            filepath="../../Muon_Collider_Smart_Pixels/eric/model2.5_qi_4w0i_pareto_roc_selected/model_trial_25.h5"
            filepath="../eric/Model2_5_tahn/model2.5_quantizedinputs_8w0i_pareto_roc_selected/model_trial_065.h5"
            filepath="../eric/model2.5_quantizedinputs_quantized_3w0i_normalized_run_hyperparameter_results_20260430_185058/model_trial_004.h5"
        # filepath = "../../smart-pixels-ml/DanielModels/model2_20260325.keras"
        #Now trying an Ryan model
        if modelType==1:
            filepath="/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/ryan/old_quantization_res/model1_quantized_4w0i_pareto/model_trial_034.h5"
    co = {}       
    qkeras.utils._add_supported_quantized_objects(co)
    quantizedModel = tf.keras.models.load_model(filepath,custom_objects=co,compile=True)
    return quantizedModel

def getModelAndPredict(quantizedModel,
                        tfRecordFolder = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V3_May/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized",
                        configName = "justThisOne"
                       ):
    if "1" in quantizedModel.name[3:7]:
        modelType = 1
        model = Model1(tfRecordFolder = tfRecordFolder)   
    elif "2" in quantizedModel.name[3:7]:
        modelType = 2
        model = Model2_5(tfRecordFolder = tfRecordFolder)   
    elif "3" in quantizedModel.name[3:7]:
        modelType = 3
        model = Model3(tfRecordFolder = tfRecordFolder)   
    else:
        raise ValueError("Error processing model type")
    model.models[configName] = quantizedModel
    model.evaluate(config_name=configName,predictionPlots=False)
    predictions = model.models[configName].predict(model.validation_generator, verbose=1)
    return model, predictions, modelType

#########################################################################################################
#### Plottters
#############################################

###########template-like functions for general plotting
def plotVarPrediction(truthDF,varKey, varLabel,varBins=50,predBins = np.linspace(-0.1,1,50), title="",logColor = True,cmap="Blues"):
    mask = [True for i in range(len(truthDF))]
    counts, xedges, yedges, im = plot2dHistFromTruth(truthDF, "prediction",varKey, mask, predBins, varBins, cmap, "prediction by model",varLabel,title,logColor = logColor)
    return counts, xedges, yedges, im
class PlottingFunction(Protocol):
    def __call__(self, df: pd.DataFrame, title: str, *args: Any, **kwargs: Any) -> Any:
        ...
#Takes in the predVarDF which has the variable formated in plotUtils way, as well as a trueY for if bib/sig, and a prediction from model
#takes in the cut for the prediction
#takes in a plotting function, that must have 2 inputs (dataframe and title) to use to plot each subset of predVarDF
#revised by google ai
def plot3by3PredBibSig(
    predVarDF: pd.DataFrame, 
    # plot_func: Callable[[pd.DataFrame, str, ...], Any], #not allowed 
    plot_func: PlottingFunction, 
    cut: float = 0.51171875,
    genTitle = "",
    extendTitle = "",
    PLOT_DIR = "./ratePlots",
    interactivePlots = True,
    isHist2d = True, #means that plot_func should return counts, xedges, yedges, im
    *args: Any,
    **kwargs: Any
) -> None:
    """
    Plots a 3x3 grid of subsets of the data using an arbitrary plotting function.
    """
    plt.figure(figsize=(12, 10))
    
    configs = [
        (predVarDF, "all vectors"),
        (predVarDF.query("trueY == 0"), "all bib"),
        (predVarDF.query("trueY == 1"), "all sig"),
        
        (predVarDF.query("prediction > @cut"), "all vectors passing cut"),
        (predVarDF.query("trueY == 0 and prediction > @cut"), "all bib passing cut"),
        (predVarDF.query("trueY == 1 and prediction > @cut"), "all sig passing cut"),
        
        (predVarDF.query("prediction < @cut"), "all vectors rejected by cut"),
        (predVarDF.query("trueY == 0 and prediction < @cut"), "all bib rejected by cut"),
        (predVarDF.query("trueY == 1 and prediction < @cut"), "all sig rejected by cut")
    ]

    for i, (df_subset, title) in enumerate(configs, 1):
        plt.subplot(3, 3, i)
        # Pass the subset and title, then any extra args/kwargs
        plot_func(df_subset, title=title, *args, **kwargs)
    plt.suptitle(genTitle + "\n" + extendTitle)
    closePlot(PLOT_DIR,interactivePlots,plotName = f"predStratPlot_{genTitle}.png")

# a couple of declarations of plotters more explicitly
def plotZglobalXsizeJust1(truthDF, title="",binsZGlobal = 30,binsXSize = np.arange(0,22,1),cmap="Blues"):
    mask = [True for i in range(len(truthDF))]
    counts, xedges, yedges, im = plot2dHistFromTruth(truthDF,  "z-global","xSize", mask, binsZGlobal, binsXSize, cmap, 'z-global [mm]','x-size (# pixels)',title,logColor = True)
    return counts, xedges, yedges, im
def plotZglobalYsizeJust1(truthDF, title="",binsZGlobal = 30,binsYSize = np.arange(0,14,1),cmap="Blues"):
    mask = [True for i in range(len(truthDF))]
    counts, xedges, yedges, im = plot2dHistFromTruth(truthDF,  "z-global","ySize", mask, binsZGlobal, binsYSize, cmap, 'z-global [mm]','y-size (# pixels)',title,logColor = True)
    return counts, xedges, yedges, im
def plotYlocalXsizeJust1(truthDF, title="",binsYlocal = 30,binsXSize = np.arange(0,22,1),cmap="Blues"):
    mask = [True for i in range(len(truthDF))]
    counts, xedges, yedges, im = plot2dHistFromTruth(truthDF,  "y-local","xSize", mask, binsYlocal, binsXSize, cmap, 'y-local [μm]','x-size (# pixels)',title,logColor = True)
    return counts, xedges, yedges, im
def plotYlocalYsizeJust1(truthDF, title="",binsYlocal = 30,binsYSize = np.arange(0,14,1),cmap="Blues",):
    mask = [True for i in range(len(truthDF))]
    counts, xedges, yedges, im = plot2dHistFromTruth(truthDF,  "y-local","ySize", mask, binsYlocal, binsYSize, cmap, 'y-local [μm]','y-size (# pixels)',title,logColor = True)
    return counts, xedges, yedges, im

# Now to actually use the code
def runPredVarDFPlots(predVarDF,
                      cut = 0.51171875,PLOT_DIR = "./ratePlots",interactivePlots = True,extendTitle = "",cmap="Blues"):
    #TODO: COMMENT how the fancy callable stuff works
    varPredCallables = {"plotter": [functools.partial(plotVarPrediction,varKey = "pt", varLabel = "pT (MeV)",cmap=cmap),
                                    functools.partial(plotVarPrediction,varKey = "xSize", varLabel = "x-size (# pixels)",varBins=np.arange(0,22,1),cmap=cmap),
                                    functools.partial(plotVarPrediction,varKey = "ySize", varLabel = "y-size (# pixels)",varBins=np.arange(0,14,1),cmap=cmap),
                                    functools.partial(plotVarPrediction,varKey = "z-global", varLabel = "z-global [mm]",varBins = 20,cmap=cmap),
                                    functools.partial(plotVarPrediction,varKey = "y-local", varLabel = "y-local [μm]",varBins = 20,cmap=cmap),
                                    functools.partial(plotVarPrediction,varKey = "nPix", varLabel = "nPix [# pixels]",varBins = 20,cmap=cmap),
                                    functools.partial(plotZglobalXsizeJust1,cmap=cmap),
                                    functools.partial(plotZglobalYsizeJust1,cmap=cmap),
                                    functools.partial(plotYlocalXsizeJust1,cmap=cmap),
                                    functools.partial(plotYlocalYsizeJust1,cmap=cmap),
                                    ],
                        "genTitle":["PtVsPrediction","XSizeVsPrediction","YSizeVsPrediction","ZGlobalVsPrediction","YLocalVsPrediction",
                                    "nPixVsPrediction",
                                    "ZGlobalXSize",
                                    "ZglobalYsize",
                                    "YlocalXsize",
                                    "YlocalYsize",]}
    # varPredCallables[0](predVarDF,title="ptVsprediction")
    for i in range(len(varPredCallables["plotter"])):
        plot3by3PredBibSig(predVarDF,plot_func=varPredCallables["plotter"][i],genTitle=varPredCallables["genTitle"][i],extendTitle = extendTitle,cut=cut, PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots)

    plt.close()

def runModelPlots(filepath = "", modelType=2,
                  tfRecordFolder = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V3_May/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized",
                  #to pass through to later plotting functions:
                  sig_eff = 0.99,PLOT_DIR = "./ratePlots",interactivePlots = True,extendTitle = "",
                  modifyPlotDir = True,
                  ):
    quantizedModel = loadQModel(filepath,modelType)
    model, predictions,modelType = getModelAndPredict(quantizedModel,tfRecordFolder)
    if modelType == 1:
        cmap = "Purples"
    elif modelType == 2:
        cmap = "Blues"
    elif modelType == 3:
        cmap = "Greens"
    else:
        raise ValueError("why is modelType the wrong thing??? unrecognized model?")
    

    fpr_val = model.evaluation_results[f'fpr_at_{int(sig_eff*100)}pct']
    bg_rej = model.evaluation_results[f'bkg_rej_at_{int(sig_eff*100)}pct']
    fpr = model.evaluation_results["fpr"] #unfortunately once unwrap, no longer a np.array, so have to renp to index it
    thresholds = model.evaluation_results["thresholds"]
    fpr = np.array(fpr); thresholds = np.array(thresholds)
    threshVal = float(thresholds[fpr==fpr_val])
    perfSummary = f"Cut pred at {threshVal} which is sig effic: {sig_eff}, fpr: {fpr_val} and bckg rej: {bg_rej}"
    print(perfSummary)
    extendTitle = extendTitle + "\n" + perfSummary
    # print(fpr_val)
    # print(fpr)
    # print(thresholds)
    predVarDF = getPredVarDF(model,predictions)
    if modifyPlotDir:
        PLOT_DIR = PLOT_DIR[:-14]+f"b{bg_rej:.4f}"+PLOT_DIR[-14:]
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    runPredVarDFPlots(predVarDF,cut=threshVal, PLOT_DIR=PLOT_DIR,interactivePlots=interactivePlots,extendTitle=extendTitle,cmap=cmap)




################end of usefulness
# a by-hand function which is not totally great
def plotXYSizeLocPerPred(predVarDF, cut = 0.51171875,pltDirAll = "./ratePlots/allVect",plotDirPass="./ratePlots/passPredict",plotDirRej="./ratePlots/rejPredict" ):
    
    dummyArray = [True for i in range(len(predVarDF.query("trueY == 0")))]
    plotZglobalXYsize(predVarDF.query("trueY == 0"),predVarDF.query("trueY == 1"),None,None,dummyArray,dummyArray,dummyArray,dummyArray,PLOT_DIR=pltDirAll)
    # plotYlocalXYsize(truthbib, truthsig, xSizesSig, xSizesBib, ySizesSig, ySizesBib,mask_bib,mask_sig,PLOT_DIR="./plots",interactivePlots=False):
    plotYlocalXYsize(predVarDF.query("trueY == 0"),predVarDF.query("trueY == 1"),None,None,None,None,dummyArray,dummyArray,PLOT_DIR=pltDirAll)
    plotPtEta(predVarDF.query("trueY == 1"),predVarDF.query("trueY == 0"),PLOT_DIR=pltDirAll)
    plotPt(predVarDF.query("trueY == 1"),predVarDF.query("trueY == 0"),predVarDF.query("trueY == 0"),predVarDF.query("trueY == 0"),PLOT_DIR=pltDirAll)
    # plotPtLowHigh(predVarDF.query("trueY == 0"),predVarDF.query("trueY == 1"),None,None,PLOT_DIR=pltDirAll)
               
    dummyArrayB = [True for i in range(len(predVarDF.query("trueY == 0 and prediction > @cut")))]
    dummyArrayS = [True for i in range(len(predVarDF.query("trueY == 1 and prediction > @cut")))]
    plotZglobalXYsize(predVarDF.query("trueY == 0 and prediction > @cut"),predVarDF.query("trueY == 1 and prediction > @cut"),None,None,dummyArrayS,dummyArrayB,dummyArrayB,dummyArrayS,PLOT_DIR=plotDirPass)
    plotYlocalXYsize(predVarDF.query("trueY == 0 and prediction > @cut"),predVarDF.query("trueY == 1 and prediction > @cut"),None,None,None,None,dummyArrayB,dummyArrayS,PLOT_DIR=plotDirPass)
    plotPtEta(predVarDF.query("trueY == 1 and prediction > @cut"),predVarDF.query("trueY == 0 and prediction > @cut"),PLOT_DIR=plotDirPass)
    plotPt(predVarDF.query("trueY == 1 and prediction > @cut"),predVarDF.query("trueY == 0 and prediction > @cut"),predVarDF.query("trueY == 0 and prediction > @cut"),predVarDF.query("trueY == 0 and prediction > @cut"),PLOT_DIR=plotDirPass)
    # plotPtLowHigh(predVarDF.query("trueY == 0 and prediction > @cut"),predVarDF.query("trueY == 1 and prediction > @cut"),None,None,PLOT_DIR=plotDirPass)


    dummyArrayB = [True for i in range(len(predVarDF.query("trueY == 0 and prediction < @cut")))]
    dummyArrayS = [True for i in range(len(predVarDF.query("trueY == 1 and prediction < @cut")))]
    plotZglobalXYsize(predVarDF.query("trueY == 0 and prediction < @cut"),predVarDF.query("trueY == 1 and prediction < @cut"),None,None,dummyArrayS,dummyArrayB,dummyArrayB,dummyArrayS,PLOT_DIR=plotDirRej)
    plotYlocalXYsize(predVarDF.query("trueY == 0 and prediction < @cut"),predVarDF.query("trueY == 1 and prediction < @cut"),None,None,None,None,dummyArrayB,dummyArrayS,PLOT_DIR=plotDirRej)
    plotPtEta(predVarDF.query("trueY == 1 and prediction < @cut"),predVarDF.query("trueY == 0 and prediction < @cut"),PLOT_DIR=plotDirRej)
    plotPt(predVarDF.query("trueY == 1 and prediction < @cut"),predVarDF.query("trueY == 0 and prediction < @cut"),predVarDF.query("trueY == 0 and prediction < @cut"),predVarDF.query("trueY == 0 and prediction < @cut"),PLOT_DIR=plotDirRej)
   
