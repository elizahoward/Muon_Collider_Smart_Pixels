"""
Author: Claude (then Daniel Abadjiev added stuff)
Date: November 3, 2026
Description: Utils library with Row builders for EDM4hep objects read with podio (one dict per object).

Requires to be run inside the apptainer, e.g. 
apptainer run /cvmfs/unpacked.cern.ch/ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24\:v3.1-amd64
source /opt/setup_mucoll.sh
"""
import edm4hep
import podio
import pandas as pd

def eventToDfs(event, ievt):
    """
    Go through all the hits, links, etc. in the event and build them into lists of dictionaries and then pandas dataframes
    """
    collectionNames = event.getAvailableCollections()
    if "MCParticles" in collectionNames:
        truthParticles = event.get("MCParticles")
        truthParticlesRows = [mcParticleRow(hit,ievt) for hit in truthParticles]
        truthParticlesDf = pd.DataFrame(truthParticlesRows)
    else:
        truthParticlesDf = pd.DataFrame()
    if "VertexBarrelCollection" in collectionNames:
        vtxBarrelSimHits = event.get("VertexBarrelCollection")
        # vtxBarrelSimHitsRows = [simHitRow(hit,ievt) for hit in vtxBarrelSimHits]
        vtxBarrelSimHitsRows = [simHitRowWithMc(hit,ievt) for hit in vtxBarrelSimHits]
        vtxBarrelSimHitsDf = pd.DataFrame(vtxBarrelSimHitsRows)
    else:
        vtxBarrelSimHitsDf = pd.DataFrame()
    if "VXDBarrelRealHitsRelations" in collectionNames:
        vtxClusterToSimHit = event.get("VXDBarrelRealHitsRelations")
        vtxClusterToSimHitRows = [linkRow(hit,ievt) for hit in vtxClusterToSimHit]
        vtxClusterToSimHitDf = pd.DataFrame(vtxClusterToSimHitRows)
    else:
        vtxClusterToSimHitDf = pd.DataFrame()
    if "VXDBarrelRealHits" in collectionNames:
        vtxClusters = event.get("VXDBarrelRealHits")
        vtxClustersRows = [clusterRow(hit,ievt) for hit in vtxClusters]
        vtxClustersDf = pd.DataFrame(vtxClustersRows)
    else:
        vtxClustersDf = pd.DataFrame()
    if "VXDBarrelRealRawHitRelations" in collectionNames:
        vtxPixelsToClusters = event.get("VXDBarrelRealRawHitRelations")
        vtxPixelsToClustersRows = [linkRow(hit,ievt) for hit in vtxPixelsToClusters]
        vtxPixelsToClustersDf = pd.DataFrame(vtxPixelsToClustersRows)
    else:
        vtxPixelsToClustersDf = pd.DataFrame()
    if "VXDBarrelRealPixels" in collectionNames:
        vtxPixels = event.get("VXDBarrelRealPixels")
        vtxPixelsRows = [simHitRow(hit,ievt) for hit in vtxPixels]
        vtxPixelsDf = pd.DataFrame(vtxPixelsRows)
    else:
        vtxPixelsDf = pd.DataFrame()
    return {"truthParticlesDf":truthParticlesDf,"vtxBarrelSimHitsDf":vtxBarrelSimHitsDf,"vtxClusterToSimHitDf":vtxClusterToSimHitDf,"vtxClustersDf":vtxClustersDf,"vtxPixelsToClustersDf":vtxPixelsToClustersDf,"vtxPixelsDf":vtxPixelsDf,}

def idPair(obj):
    i = obj.id()
    return i.collectionID, i.index

def vec3(v):
    return v.x, v.y, v.z

def simHitRow(h, ievt):
    """edm4hep::SimTrackerHit (VertexBarrel, OuterTracker*, VXDBarrelRealPixels, ...)"""
    coll, idx = idPair(h)
    pos, mom = h.getPosition(), h.getMomentum()
    part = h.getParticle()
    pc, pi = (idPair(part) if part.isAvailable() else (-1, -1))
    return {
        "cellID": h.getCellID(), "eDep": h.getEDep(), "time": h.getTime(),
        "pathLength": h.getPathLength(), "quality": h.getQuality(),
        "position.x": pos.x, "position.y": pos.y, "position.z": pos.z,
        "momentum.x": mom.x, "momentum.y": mom.y, "momentum.z": mom.z,
        "particle_index": pi, "particle_collectionID": pc,           #add back in, just comment out for comparison to make_tracklists_fromEdm4hep.py
        "ievent": ievt, 
        "index": idx, "collectionID": coll,       #add back in, just comment out for comparison to make_tracklists_fromEdm4hep.py
        "secondary": h.isProducedBySecondary(), "overlay": h.isOverlay(), #add back in, just comment out for comparison to make_tracklists_fromEdm4hep.py
    }

def simHitRowWithMc(h, ievt):
    row = simHitRow(h, ievt)
    part = h.getParticle()
    if part.isAvailable():
        mc = mcParticleRow(part, ievt)
        row.update({f"MCParticles.{k}": v for k, v in mc.items() if k != "ievent"})
    else:
        raise ValueError("Can't use this function if the mc particle is unset")
    return row

def _cov(m):
    try:
        return [m[i] for i in range(6)]
    except Exception:
        return [m.getValue(i) for i in range(6)]        # check dir(cluster.getCovMatrix()) if both fail

def clusterRow(c, ievt):
    """edm4hep::TrackerHitPlane (VXDBarrelRealHits)"""
    coll, idx = idPair(c)
    pos, u, v = c.getPosition(), c.getU(), c.getV()
    row = {
        "cellID": c.getCellID(), "type": c.getType(), "quality": c.getQuality(),
        "time": c.getTime(), "eDep": c.getEDep(), "eDepError": c.getEDepError(),
        "position.x": pos.x, "position.y": pos.y, "position.z": pos.z,
        "u.a": u.a, "u.b": u.b, "v.a": v.a, "v.b": v.b,   # Vector2f members a,b (check dir(c.getU()))
        "du": c.getDu(), "dv": c.getDv(),
        "index": idx,"collectionID": coll, "ievent": ievt, 
    }
    for i, val in enumerate(_cov(c.getCovMatrix())):
        row[f"covMatrix[{i}]"] = val
    return row

def linkRow(l, ievt):
    """podio::LinkT<...> (VXDBarrelRealHitsRelations, VXDBarrelRealRawHitRelations)"""
    coll, idx = idPair(l)
    fc, fi = idPair(l.getFrom()); tc, ti = idPair(l.getTo())
    return {"ievent": ievt, "collectionID": coll, "index": idx,
            "from_collectionID": fc, "from_index": fi, "to_collectionID": tc, "to_index": ti,
            "weight": l.getWeight()}

def mcParticleRow(p, ievt):
    """edm4hep::MCParticle (MCParticles) -- getters not in your dir() output yet, check dir(truthParticles[0])"""
    coll, idx = idPair(p)
    vtx, end, mom, momE = p.getVertex(), p.getEndpoint(), p.getMomentum(), p.getMomentumAtEndpoint()
    return {
        "PDG": p.getPDG(), "generatorStatus": p.getGeneratorStatus(),
        "simulatorStatus": p.getSimulatorStatus(), "charge": p.getCharge(),
        "time": p.getTime(), "mass": p.getMass(), 
        "vertex.x": vtx.x, "vertex.y": vtx.y, "vertex.z": vtx.z,
        "endpoint.x": end.x, "endpoint.y": end.y, "endpoint.z": end.z,
        "momentum.x": mom.x, "momentum.y": mom.y, "momentum.z": mom.z,
        "momentumAtEndpoint.x": momE.x, "momentumAtEndpoint.y": momE.y, "momentumAtEndpoint.z": momE.z,
        "helicity": p.getHelicity(),
        "index": idx,"collectionID": coll, "ievent": ievt, 
    }

def mcParentRows(p, ievt):
    """one row per (child, parent) edge of the MCParticle tree"""
    _, child = idPair(p)
    return [{"ievent": ievt, "child_idx": child, "parent_idx": idPair(par)[1]} for par in p.getParents()]

def eventHeaderRow(event, ievt):
    hdr = event.get("EventHeader")[0]
    return {"ievent": ievt, "eventNumber": hdr.getEventNumber(), "runNumber": hdr.getRunNumber()}