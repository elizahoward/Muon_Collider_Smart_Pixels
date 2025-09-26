import numpy as np
import pyLCIO
import ROOT
import glob
import os
import json
from math import *
import csv
import argparse

sensorAngles = np.arange(-np.pi,np.pi+2*np.pi/8,np.pi/8)

def getYlocalAndGamma(x,y):
    # Get exact angle gamma of the hit position
    gammaP=np.arctan2(y,x)

    # Get two sensor angles that are closest to the exact angle
    diff = np.abs(sensorAngles-gammaP)
    index1 = np.argmin(diff)
    gamma1=sensorAngles[index1]
    diff[index1]=3*np.pi
    index2 = np.argmin(diff)
    gamma2=sensorAngles[index2]

    # Rotate x coordinate of the point by each option for gamma
    x1=x*np.cos(-gamma1)-y*np.sin(-gamma1)
    y1=y*np.cos(-gamma1)+x*np.sin(-gamma1)
    x2=x*np.cos(-gamma2)-y*np.sin(-gamma2)
    y2=y*np.cos(-gamma2)+x*np.sin(-gamma2)

    # Determine which x is closest to expected value
    xTrue=30.16475324197002

    diff1=abs(x1-xTrue)
    diff2=abs(x2-xTrue)
    
    # If both x1 and x2 are really close to the ex
    if diff1 < 0.5 and diff2 < 0.5:
        if y1>8.5 or y1<-4.5:
            index=index2
        else:
            index=index1
            
    elif diff1<diff2:
        index=index1
    else:
        index=index2

    if index==index1:
        yentry=y1
    else:
        yentry=y2
    
    ylocal=-round(yentry/25e-3)*25e-3
    # at some point, add limits to possible ROIs

    if index==0:
        index=16
    if index==17:
        index=1
    
    gamma=sensorAngles[index]

    return ylocal, gamma

# user options
parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input_file", help="Input file", type=str, required=True)
parser.add_argument("-o", "--output_file", help="Output file", type=str, required=True)
parser.add_argument("-f", "--float_precision", help="Floating point precision", default=5, type=int)
parser.add_argument("-p", "--allowedPIDS", help="Allowed PIDs (comma-separated)", type=str, required=True)
parser.add_argument("-flp", "--flp", help="Direction of sensor (1 for FE side out, 0 for FE side down)", default=0, type=int)
ops = parser.parse_args()
    
# ############## SETUP #############################
# Prevent ROOT from drawing while you're running -- good for slow remote servers
ROOT.gROOT.SetBatch()

# check if output directory exists
if not os.path.isdir(ops.output_file):
    raise Exception(f"Directory {ops.output_file} does not exist")

# convert this to writing to a log file
print(f"Getting tracks from file: {ops.input_file}\n")
print(f"Allowed PIDs: {ops.allowedPIDS}\n")


if not ops.input_file.endswith(".slcio"):
    raise Exception("Input file must be a .slcio file")

tracks=[]
reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
reader.open(ops.input_file)

for ievt,event in enumerate(reader):
    print("Processing event %i."%ievt)

    # Print all the collection names in the event
    collection_names = event.getCollectionNames()

    # Get vertex barrel hits
    vtxBarrelHits = event.getCollection("VertexBarrelCollection")

    for hit in vtxBarrelHits: 
        position = hit.getPosition()
        
        x,y,hit_z = position[0],position[1],position[2]
        #rxy=(x**2+y**2)**0.5
        t=hit.getTime()

        # Get layer
        encoding = vtxBarrelHits.getParameters().getStringVal(pyLCIO.EVENT.LCIO.CellIDEncoding)
        decoder = pyLCIO.UTIL.BitField64(encoding)
        cellID = int(hit.getCellID0())
        decoder.setValue(cellID)
        #detector = decoder["system"].value()
        layer = decoder['layer'].value()
        #side = decoder["side"].value()

        if layer!=0: continue # first layer only

        # get the particle that caused the hit
        mcp = hit.getMCParticle()
        hit_pdg = mcp.getPDG() if mcp else None
        hit_id = mcp.id() if mcp else None

        if hit_id is None:
            print("No MCParticle associated with hit, skipping.")
            continue

        if abs(hit_pdg) not in ops.allowedPIDS: continue

        # momentum at production
        mcp_p = mcp.getMomentum()
        mcp_tlv = ROOT.TLorentzVector()
        mcp_tlv.SetPxPyPzE(mcp_p[0], mcp_p[1], mcp_p[2], mcp.getEnergy())

        # momentum at hit
        hit_p = hit.getMomentum()
        hit_tlv = ROOT.TLorentzVector()
        hit_tlv.SetPxPyPzE( hit_p[0], hit_p[1], hit_p[2], mcp.getEnergy())

        ylocal, gamma0 = getYlocalAndGamma(x,y)
        zglobal = round(hit_z/25e-3)*25e-3 # round to nearest pixel
        
        # Define unit vector of track at tracker edge with respect to barrel
        theta=hit_tlv.Theta()
        phi=hit_tlv.Phi()
        vx_global=np.sin(theta)*np.cos(phi)
        vy_global=np.sin(theta)*np.sin(phi)
        vz_global=np.cos(theta) 

        # Convert vector to sensor frame to calculate alpha
        vy_local=vx_global*np.sin(np.pi/2-gamma0)+vy_global*np.cos(np.pi/2-gamma0)
        vx_local=vz_global
        alpha=np.arctan2(vy_local,vx_local)
        
        # Calculate beta from phi and gamma0
        beta=phi-(gamma0-np.pi/2)

        # If we are unflipped, we must adjust alpha and beta, and flip y-local
        if ops.flp == 0:
            beta += np.pi
            alpha = 2*np.pi-alpha
            ylocal *= -1

        cota = 1./np.tan(alpha)
        cotb = 1./np.tan(beta)

        p = mcp_tlv.P()
        pt = mcp_tlv.Pt()
        track = [cota, cotb, p, ops.flp, ylocal, zglobal, pt, t, hit_pdg]
        tracks.append(track)

print(f"Writing {len(tracks)} tracks to {ops.output_file}\n") 

with open(ops.output_file, 'w') as out_file:
    for track in tracks:
        track = list(track)

        formatted_sublist = [f"{element:.{ops.float_precision}f}" if isinstance(element, float) else element for element in track]
        line = ' '.join(map(str, formatted_sublist)) + '\n'
        out_file.write(line)