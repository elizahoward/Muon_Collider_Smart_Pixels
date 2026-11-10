import numpy as np 
import pandas as pd 
import gc                            
import os 

datadir = '/home/mwells5/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb_copy_m/Parquet_Files/'
trackDir = '/home/mwells5/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb_copy_m/tl_with_moduleID_07_28_2026/'
outdir = '/home/mwells5/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb_copy_m/Parquet_files_w_ModuleID/'

max = 100
labels_bib_list = []
labels_sig_list = []
trackdata_list = []
trackHeader = ["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg", "moduleID"]

count=0
for file in os.listdir(datadir):
    if "labels" in file:
        if "bib" in file: 
            labels_bib_list.append(pd.read_parquet(f"{datadir}{file}")) 
            #truthbib = pd.concat([truthbib,pd.read_parquet(f"{datadir}{file}")])
            count+=1
    if count == max:
        break

countmp=0
for file in os.listdir(trackDir):
    if "bib_mp" in file: 
        trackdata_list.append(pd.read_csv(f"{trackDir}{file}", sep=' ', names=trackHeader))
        countmp+=1
    if countmp == max:
        break 

bib = pd.concat(labels_bib_list)
tracks = pd.concat(trackdata_list)

del labels_bib_list
del labels_sig_list
del trackdata_list
gc.collect()

i=0
errornum=0
moduleID_list = []
for i in (range(0, tracks.shape[0])):
    if tracks['cota'].iat[i] != bib['cotAlpha'].iat[i]:
        print("no match on cot(a)")
        print(tracks['cota'].iat[i], " ", bib['cotAlpha'].iat[i])
        errornum+=1
        continue
    elif tracks['cotb'].iat[i] != bib['cotBeta'].iat[i]:
        #print("no match on cot(b)")
        errornum+=1
        continue
    elif tracks['y-local'].iat[i] != bib['y-local'].iat[i]:
        #print("no match on y-local")
        errornum+=1
        continue
    elif tracks['z-global'].iat[i] != bib['z-global'].iat[i]:
        #print("no match on z-global")
        errornum+=1
        continue
    else:
        np.append(moduleID_list, tracks['moduleID'].iat[i])
    bib['moduleID'] = moduleID_list

if len(moduleID_list) != bib.shape[0]:
        print("Warning! These lists are not the same length")

bib.to_parquet(f"{outdir}/testblock.parquet")
print("number of skipped tracks: ", errornum)
    

     
