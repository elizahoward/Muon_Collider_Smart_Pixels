import numpy as np 
import pandas as pd 
import gc                            
import os 

datadir = '/home/mwells5/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb_copy_m/Parquet_Files/'
trackDir = '/home/mwells5/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb_copy_m/tl_with_moduleID_07_28_2026/'
outdir = '/home/mwells5/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb_copy_m/Parquet_files_w_ModuleID/'

truthbib = pd.DataFrame()
recon2Dbib = pd.DataFrame()
truthsig = pd.DataFrame()
recon2Dsig = pd.DataFrame()
trackData = pd.DataFrame()
recon3Dsig = pd.DataFrame()
recon3Dbib = pd.DataFrame()

#test = pd.DataFrame()
#test = pd.read_parquet("/home/mwells5/Muon_Collider_Smart_Pixels/Data_Files/Data_Set_2026Feb_copy_m/Parquet_Files/bib_mp_labels_42.parquet")

labels_bib_list = []
recon2D_bib_list = []
labels_sig_list = []
recon2D_sig_list = []
recon3D_bib_list = []
recon3D_sig_list = []
trackdata_list = []

trackHeader = ["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg", "moduleID"]

count=0
for file in os.listdir(datadir):
    if "labels" in file:
        if "bib" in file: 
            labels_bib_list.append(pd.read_parquet(f"{datadir}{file}")) 
            #truthbib = pd.concat([truthbib,pd.read_parquet(f"{datadir}{file}")])
            file = file.replace("labels","recon2D")
            recon2D_bib_list.append(pd.read_parquet(f"{datadir}{file}")) 
            #recon2Dbib = pd.concat([recon2Dbib,pd.read_parquet(f"{datadir}{file}")])
            file = file.replace("recon2D","recon3D")
            recon3D_bib_list.append(pd.read_parquet(f"{datadir}{file}"))
        elif "sig" in file: 
            labels_sig_list.append(pd.read_parquet(f"{datadir}{file}")) 
            #truthsig = pd.concat([truthsig,pd.read_parquet(f"{datadir}{file}")])
            file = file.replace("labels","recon2D")
            recon2D_sig_list.append(pd.read_parquet(f"{datadir}{file}")) 
            #recon2Dsig = pd.concat([recon2Dsig,pd.read_parquet(f"{datadir}{file}")])
            file = file.replace("recon2D","recon3D")
            recon3D_sig_list.append(pd.read_parquet(f"{datadir}{file}"))
            count+=1
        if count == 100:
            break

countmm=0
countmp=0
for file in os.listdir(trackDir):
    if "bib_mm" in file:
        trackdata_list.append(pd.read_csv(f"{trackDir}{file}", sep=' ', names=trackHeader))
        countmm+=1 
    elif "bib_mp" in file: 
        trackdata_list.append(pd.read_csv(f"{trackDir}{file}", sep=' ', names=trackHeader))
        countmp+=1
    if countmp+countmm==max*2:
        break 
            

truthbib = pd.concat(labels_bib_list)
recon2Dbib = pd.concat(recon2D_bib_list)
truthsig = pd.concat(labels_sig_list)
recon2Dsig = pd.concat(recon2D_sig_list)
recon3Dbib = pd.concat(recon3D_bib_list)
recon3Dsig = pd.concat(recon3D_sig_list)
trackData = pd.concat(trackdata_list)

del labels_bib_list
del recon2D_bib_list
del labels_sig_list
del recon2D_sig_list
del recon3D_sig_list
del recon3D_bib_list
del trackdata_list

gc.collect()

i=0
errornum=0
moduleID_list = []
for i in (range(trackData.shape[0]+1)):
    if trackData['cota'].iat[i] != truthbib['cotAlpha'].iat[i]:
        print("no match on cot(a)")
        print(trackData['cota'].iat[i], " ", truthbib['cotAlpha'].iat[i])
        errornum+=1
        continue
    elif trackData['cotb'].iat[i] != truthbib['cotBeta'].iat[i]:
        #print("no match on cot(b)")
        errornum+=1
        continue
    elif trackData['y-local'].iat[i] != truthbib['y-local'].iat[i]:
        #print("no match on y-local")
        errornum+=1
        continue
    elif trackData['z-global'].iat[i] != truthbib['z-global'].iat[i]:
        #print("no match on z-global")
        errornum+=1
        continue
    else:
        #add in here to match things up!

        np.append(moduleID_list, trackData['moduleID'].iat[i])
    truthbib['moduleID'] = moduleID_list

if len(moduleID_list) != truthbib.shape[0]:
        print("Warning! These lists are not the same length")

truthbib.to_parquet(f"{outdir}/testblock.parquet")
print("number of skipped tracks: ", errornum)