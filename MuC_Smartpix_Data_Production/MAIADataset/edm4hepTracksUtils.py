"""
Author: Daniel Abadjiev
Date: November 4, 2026
Description: Utils functions moved out of make_tracklists_fromEdm4hep.py to be reused
"""

import pandas as pd
import numpy as np
# from make_tracklists import getYlocalAndGamma
######################
sensorAngles = np.arange(-np.pi-np.pi/8,np.pi+2*np.pi/8,np.pi/8)

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

    gamma=sensorAngles[index]
    # Shift range of gamma to 0 to 2 pi
    if gamma<0:
        gamma+=2*np.pi
    
    return ylocal, gamma
##########################
def hitDftoTracklistDf(hitDf: pd.DataFrame,flp=0):
    """
    convert to a df whos rows match Eliza's track = [cota, cotb, p, ops.flp, ylocal, zglobal, pt, t, hit_pdg]
    """
    if hitDf.empty:
        print("hit df empty, so empty tracklist df")
        return pd.DataFrame()
    hitDf["p"] = np.sqrt(hitDf["momentum.x"]**2 + hitDf["momentum.y"]**2 + hitDf["momentum.z"]**2 )
    hitDf["pt"] = np.sqrt(hitDf["momentum.x"]**2 + hitDf["momentum.y"]**2)
    hitDf["t"] = hitDf["time"]
    hitDf["hit_pdg"] = hitDf["MCParticles.PDG"]
    hitDf["zglobal"] = round(hitDf["position.z"]/25e-3)*25e-3 # round to nearest pixel
    out = [getYlocalAndGamma(x, y) for x, y in zip(hitDf["position.x"], hitDf["position.y"])]
    hitDf[["ylocal", "gamma0"]] = out
    hitDf["cota"] = hitDf["momentum.z"]/(hitDf["momentum.x"]*np.cos(hitDf["gamma0"])+hitDf["momentum.y"]*np.sin(hitDf["gamma0"]))
    hitDf["cotb"] = (hitDf["momentum.x"]*np.sin(hitDf["gamma0"])-hitDf["momentum.y"]*np.cos(hitDf["gamma0"]))/(hitDf["momentum.x"]*np.cos(hitDf["gamma0"])+hitDf["momentum.y"]*np.sin(hitDf["gamma0"]))
    if flp == 0:
        hitDf["ylocal"] *= -1
        hitDf["cota"] *= -1
    hitDf["flp"] = flp  
    tracklistDf = hitDf[["cota", "cotb", "p", "flp", "ylocal", "zglobal", "pt", "t", "hit_pdg"]].copy()
    return tracklistDf




"""
Author: Claude
cellid_decode.py -- decode a DD4hep cellID into system / side / layer / module / sensor (numpy, no DD4hep needed)

The same job as the slcio snippet
    decoder.setValue(cellID); layer = decoder['layer'].value()

    from cellid_decode import makeDecoder, MAIA_V0_TRACKER_ENCODING
    decode = makeDecoder(MAIA_V0_TRACKER_ENCODING)
    f = decode(simHits["cellID"].to_numpy())          # dict of arrays: f["layer"], f["sensor"], ...
    simHits["layer"] = f["layer"]
    inner = simHits[simHits["layer"] == 0]            # see the note on the double layer below

Encoding: MAIA_v0.xml (key4hep/k4geo, MuColl/MAIA/compact/MAIA_v0, commit 16658a9) defines
    GlobalTrackerReadoutID = "system:5,side:-2,layer:6,module:11,sensor:8"
and VertexBarrelCollection / VertexEndcapCollection use it, so layer = (cellID >> 7) & 0x3F.
I only checked the vertex file; the inner and outer tracker presumably use the same string but check
their <readout> entries. Another detector version can have another string: take it from its compact XML.

Vertex barrel layers in MAIA_v0 (ids 3, 5, 7 exist only for the endcap):
    layer 0 and 1 = a double layer at r ~ 30 / 32 mm,  layer 2 ~ 51 mm,  layer 4 ~ 74 mm,  layer 6 ~ 102 mm
"""
# import numpy as np

MAIA_V0_TRACKER_ENCODING = "system:5,side:-2,layer:6,module:11,sensor:8"


def makeDecoder(encoding=MAIA_V0_TRACKER_ENCODING):
    """DD4hep BitFieldCoder for an id string. Fields are packed from bit 0 upwards; a negative width is a
    signed field; 'name:offset:width' is also accepted."""
    fields, pos = {}, 0
    for part in encoding.split(","):
        bits = part.strip().split(":")
        name, nums = bits[0], [int(b) for b in bits[1:]]
        if len(nums) == 2:
            pos, width = nums
        else:
            width = nums[0]
        fields[name] = (pos, abs(width), width < 0)
        pos += abs(width)

    def decode(cellID):
        c = np.asarray(cellID, dtype=np.uint64)
        out = {}
        for name, (off, w, signed) in fields.items():
            v = ((c >> np.uint64(off)) & np.uint64((1 << w) - 1)).astype(np.int64)
            if signed:
                v = np.where(v >= (1 << (w - 1)), v - (1 << w), v)
            out[name] = v
        return out
    decode.fields = fields
    return decode