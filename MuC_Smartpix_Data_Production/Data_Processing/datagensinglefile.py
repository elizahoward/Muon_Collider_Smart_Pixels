import sys
import numpy as np
import pandas as pd
import math
import os

def split(outdir,tag,df1,df2,df3):

        df1.columns = df1.columns.astype(str)
        df2.columns = df2.columns.astype(str)
        df3.columns = df3.columns.astype(str)

        # unflipped, all charge                                                                         
        df1.to_parquet(f"{outdir}/labels{tag}.parquet")
        df2.to_parquet(f"{outdir}/recon2D{tag}.parquet")
        df3.to_parquet(f"{outdir}/recon3D{tag}.parquet")

def parseFile(filein,tag,nevents=-1):

        with open(filein) as f:
                lines = f.readlines()

        header = lines.pop(0).strip()
        pixelstats = lines.pop(0).strip()

        #print("Header: ", header)
        #print("Pixelstats: ", pixelstats)

        readyToGetTruth = False
        readyToGetTimeSlice = False

        clusterctr = 0
        cluster_truth =[]
        timeslice = 0
        cur_slice = []
        cur_cluster = []
        events = []
        
        for line in lines:
                ## Start of the cluster
                if "<cluster>" in line:
                        readyToGetTruth = True
                        readyToGetTimeSlice = False
                        clusterctr += 1
                        
                        # Create an empty cluster
                        cur_cluster = []
                        timeslice = 0
                        # move to next line
                        continue

                # the line after cluster is the truth
                if readyToGetTruth:
                        cluster_truth.append(line.strip().split())
                        readyToGetTruth = False

                        # move to next line
                        continue

                ## Put cluster information into np array
                if "time slice" in line:
                        readyToGetTimeSlice = True
                        cur_slice = []
                        timeslice += 1
                        # move to next line
                        continue

                if readyToGetTimeSlice:
                        cur_row = line.strip().split()
                        cur_slice += [float(item) for item in cur_row]

                        # When you have all elements of the 2D image:
                        if len(cur_slice) == 13*21:
                                cur_cluster.append(cur_slice)

                        # When you have all time slices:
                        if len(cur_cluster) == 20:
                                adjustCluster(cur_cluster,cluster_truth)

                                events.append(cur_cluster)
                                readyToGetTimeSlice = False

                                

        #print("Number of clusters = ", len(cluster_truth))
        #print("Number of events = ",len(events))
        #print("Number of time slices in cluster = ", len(events[0]))

        arr_truth = np.array(cluster_truth)
        arr_events = np.array( events )

        return arr_events, arr_truth


def adjustCluster(cluster, truth):
        # adjust y axis
        charge = np.zeros(13)
        pixelNo = np.arange(0,13) 
        lastSlice = cluster[19]
        for i in range(13):
                charge[i] = np.sum(lastSlice[i*21:(i+1)*21+1])
        center = int(np.round(np.sum(charge*pixelNo)/np.sum(charge)))

        # adjust array
        originalCenter = 6 
        nRows = center-originalCenter 
        
        if nRows>0: 
                for i in range(len(cluster)):
                        cluster[i] = cluster[i][nRows*21:]
                        for j in range(nRows*21):
                                cluster[i].append(0)
        if nRows<0: 
                for i in range(len(cluster)):
                        cluster[i] = cluster[i][:13*21+nRows*21]
                        for j in range(abs(nRows*21)):
                                cluster[i].insert(0,0)     
        # determine new y local
        truth[len(truth)-1][7]=str(float(truth[len(truth)-1][7])+nRows*25e-3)

        # adjust x axis

        charge = np.zeros(21)
        pixelNo = np.arange(0,21)
        for i in range(21):
                for j in range(13):
                        charge[i] += lastSlice[i+j*21]

        center = int(np.round(np.sum(charge*pixelNo)/np.sum(charge)))
        nCols = center-10

        if nCols>0:
                for k in range(len(cluster)):
                        for i in range(13):
                                for j in range(nCols):
                                        cluster[k].insert(20+i*21,0)
                                        cluster[k].pop(i*21)
        if nCols<0:
                for k in range(len(cluster)):
                        for i in range(13):
                                for j in range(abs(nCols)):
                                        cluster[k].pop(20+i*21)
                                        cluster[k].insert(i*21,0)

        # determine new z global
        truth[len(truth)-1][8]=str(float(truth[len(truth)-1][8])+nCols*25e-3)


def makeParquet(filename, tag, inputdir, outdir = None):
        if outdir == None:
            outdir = inputdir

        arr_events, arr_truth = parseFile(filein=f"{inputdir}/{filename}",tag=tag)

        #truth quantities - all are dumped to DF                                                                                                                           
        df = pd.DataFrame(arr_truth, columns = ['x-entry', 'y-entry','z-entry', 'n_x', 'n_y', 'n_z', 'number_eh_pairs', 'y-local', 'z-global', 'pt',  'hit_time', 'PID'])
        cols = df.columns
        for col in cols:
            df[col] = df[col].astype(float)

        df['cotAlpha'] = df['n_x']/df['n_z']
        df['cotBeta'] = df['n_y']/df['n_z']

        sensor_thickness = 100 #um                                                          
        df['y-midplane'] = df['y-entry'] + df['cotBeta']*(sensor_thickness/2 - df['z-entry'])
        df['x-midplane'] = df['x-entry'] + df['cotAlpha']*(sensor_thickness/2 - df['z-entry'])
        
        df['adjusted_hit_time'] = df['hit_time']-1e6*np.sqrt(df['z-global']**2+30**2)/299792458
        df['adjusted_hit_time_30ps_gaussian'] = df['adjusted_hit_time']+np.random.normal(loc=0,scale=30e-3,size=len(df['adjusted_hit_time']))
        df['adjusted_hit_time_60ps_gaussian'] = df['adjusted_hit_time']+np.random.normal(loc=0,scale=60e-3,size=len(df['adjusted_hit_time']))

        #print("The shape of the event array: ", arr_events.shape)
        #print("The ndim of the event array: ", arr_events.ndim)
        #print("The dtype of the event array: ", arr_events.dtype)
        #print("The size of the event array: ", arr_events.size)
    
        df2 = {}
        df2list = []

        df3 = {}
        df3list = []

        for i, e in enumerate(arr_events):

                # Only last time slice
                df2list.append(np.array(e[-1]).flatten())

                # All time slices
                df3list.append(np.array(e).flatten())

                max_val = np.amax(e)

        df2 = pd.DataFrame(df2list)
        df3 = pd.DataFrame(df3list)  

        # split into flipped/unflipped, pos/neg charge
        split(outdir, tag,df,df2,df3)

        print(f"\nConverted {filename}")