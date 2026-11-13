"""
Author: Daniel Abadjiev
Date: November 13, 2026
Description: quick script (coming from notebook) to convert the pkl dataframes into parquets (has to be run outside of container)
run this after doing make_tracklists_edm4hepContain.py

"""
import pandas as pd
import os

DEFAULT_KEY_NAMES ={"truthParticlesDf":"MCParticles",
                        "vtxBarrelSimHitsDf":"VertexBarrelCollection",
                        "vtxClusterToSimHitDf":"VXDBarrelRealHitsRelations",     
                        "vtxClustersDf":"VXDBarrelRealHits",       
                        "vtxPixelsToClustersDf":"VXDBarrelRealRawHitRelations",
                        "vtxPixelsDf":"VXDBarrelRealPixels",       
                        "truthParticlesDf2":"NewMCParticles",       
                        "vtxBarrelSimHitsDf2":"OverlayVertexBarrelCollection",}

 #https://towardsdatascience.com/saving-pandas-dataframes-efficiently-and-quickly-parquet-vs-feather-vs-orc-vs-csv-26051cc98f2e/
def convertFileStub(fileStub = "./edmTestFiles/digi_output_mu2.edm4hep.root",convertParquet=True,convertOrc = False,keyNamesToCollections=DEFAULT_KEY_NAMES):
    dfDict = {}
    for key in keyNamesToCollections.keys():
        if os.path.isfile(f"{fileStub}.{key}.csv"):
            dfDict[key] = pd.read_csv(f"{fileStub}.{key}.csv")
            # dfDict[key] = pd.read_csv(f"{fileStub}.{key}.csv")
            if convertParquet:
                dfDict[key].to_parquet(f"{fileStub}.{key}.parquet",compression="zstd")
            if convertOrc:
                dfDict[key].to_orc(f"{fileStub}.{key}.orc") #unable to do compression
            print("converted "+f"{fileStub}.{key}.csv")

def main():
    fileList = ["./edmTestFiles/sim_output.edm4hep.root","./edmTestFiles/sim_output2.edm4hep.root","./edmTestFiles/sim_output3.edm4hep.root","./edmTestFiles/digi_real.edm4hep.root","./edmTestFiles/sim_output_mu2.edm4hep.root","./edmTestFiles/digi_output_mu2.edm4hep.root","./edmTestFiles/sim_output_mu1.edm4hep.root","./edmTestFiles/sim_output_nu1.edm4hep.root","./edmTestFiles/sim_output_H1.edm4hep.root",]#"./edmTestFiles/gen_output.edm4hep.root",]
    for fileStub in fileList:
        convertFileStub(fileStub)

if __name__=="__main__":
    main()