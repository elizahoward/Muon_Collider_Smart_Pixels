#!/usr/bin/env python3
"""
Author: Claude
Date: November 5, 2026
Description:
linkedClusters.py -- loop over physical clusters, each with its pixels and its sim hits

Built on the tables saved by linkEdmHits.py (run that first; it also writes <fileStub>.linkManifest.json, which
says which saved table plays which role). Nothing is recomputed from keys here: the row numbers
(cluster_row, sim_row) in the saved link tables are used, which linkEdmHits.py verified before saving.

    clusters = LinkedClusters.fromFiles("./edmTestFiles/digi_output_mu2.edm4hep.root")
    print(len(clusters), clusters.summary())
    for c in clusters:                       # one LinkedCluster at a time, created when you ask for it
        if c.nPixels < 50:
            continue
        xy = c.pixels[["position.x", "position.y", "eDep"]]    # the pixels that are on (needs the pixel table)
        signal, bib = c.signalSimHits, c.bibSimHits            # sim hits of this cluster, split by overlay flag
        c.cluster                                              # the cluster itself (a pandas Series)
    c = clusters[1234]; c.asDict()           # {"cluster": Series, "simHits": DataFrame, "pixels": DataFrame}

What a LinkedCluster holds
    cluster      one row of the cluster table
    simHits      DataFrame of the sim hits linked to it (index labels = row numbers in the sim hit table).
                 A sim hit can belong to several clusters; the same hit then appears in each of them.
    pixels       DataFrame of its pixels (index labels = row numbers in the pixel table). Every pixel belongs
                 to exactly one cluster. Which pixel came from which sim hit is NOT known from the links.
    Nothing is dropped or capped: clusters with thousands of pixels or ~100 sim hits (delta rays, bad
    clustering) are objects like any other.

The collection is lazy: it keeps the tables and the link arrays and builds a LinkedCluster only when you index
or iterate, so a file with ~1M clusters does not create ~1M Python objects at once.
Use list(clusters[:100]) (or list(clusters)) if you want a real list.
"""
import json
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd


def _csr(owner, member, nOwners):
    """For every owner (0..nOwners-1) the members that point at it: members of owner r are
    memberSorted[offsets[r]:offsets[r+1]]; order says which input position each sorted entry came from."""
    order = np.argsort(owner, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(np.bincount(owner, minlength=nOwners))))
    return offsets, member[order], order


class LinkedCluster:
    """One cluster with its sim hits and pixels. Created by LinkedClusters."""
    __slots__ = ("_s", "row")

    def __init__(self, store, row):
        self._s, self.row = store, int(row)

    # ---- the cluster itself ------------------------------------------------------------------
    @property
    def ievent(self):
        return int(self._s.ievent[self.row])

    @property
    def index(self):
        """position of the cluster inside its event (the 'index' column)"""
        return int(self._s.index[self.row])

    @property
    def cluster(self):
        return self._s.clusters.iloc[self.row]

    # ---- sim hits ----------------------------------------------------------------------------
    @property
    def simHitRows(self):
        s = self._s
        return s.simRows[s.simOffsets[self.row]:s.simOffsets[self.row + 1]]

    @property
    def simHits(self):
        return self._s.simHits.iloc[self.simHitRows]

    @property
    def nSimHits(self):
        return int(self._s.simOffsets[self.row + 1] - self._s.simOffsets[self.row])

    @property
    def simWeights(self):
        s = self._s
        return s.linkWeight[s.simLinkOrder[s.simOffsets[self.row]:s.simOffsets[self.row + 1]]]

    @property
    def simDistancesMm(self):
        """distance between the cluster and each of its sim hits, same order as simHits"""
        s = self._s
        return s.linkDist[s.simLinkOrder[s.simOffsets[self.row]:s.simOffsets[self.row + 1]]]

    def _overlay(self):
        flag = self._s.overlay
        if flag is None:
            raise RuntimeError("the sim hit table has no 'overlay' column: cannot split signal and BIB")
        return flag[self.simHitRows]

    @property
    def signalSimHits(self):
        return self.simHits[~self._overlay()]

    @property
    def bibSimHits(self):
        return self.simHits[self._overlay()]

    @property
    def nSignal(self):
        return int((~self._overlay()).sum())

    @property
    def nBib(self):
        return int(self._overlay().sum())

    @property
    def hasSignal(self):
        return self.nSignal > 0

    # ---- pixels ------------------------------------------------------------------------------
    def _pixelSlice(self):
        s = self._s
        if s.pixOffsets is None:
            raise RuntimeError("no pixel links were saved for this file")
        return slice(s.pixOffsets[self.row], s.pixOffsets[self.row + 1])

    @property
    def nPixels(self):
        sl = self._pixelSlice()
        return int(sl.stop - sl.start)

    @property
    def pixelRows(self):
        """row numbers in the pixel table (only with a pixel table)"""
        if self._s.pixels is None:
            raise RuntimeError("the pixel table was not loaded (linkEdmHits.py had no vtxPixelsDf): use pixelIndices")
        return self._s.pixRows[self._pixelSlice()]

    @property
    def pixelIndices(self):
        """the pixels' 'index' within the event; works with and without a pixel table"""
        s = self._s
        rows = s.pixRows[self._pixelSlice()]
        return s.pixIndex[rows] if s.pixels is not None else rows

    @property
    def pixels(self):
        rows = self.pixelRows            # raises a clear error when there is no pixel table
        return self._s.pixels.iloc[rows]

    # ---- the whole thing ---------------------------------------------------------------------
    def asDict(self):
        d = {"cluster": self.cluster, "simHits": self.simHits}
        if self._s.pixels is not None:
            d["pixels"] = self.pixels
        else:
            d["pixelIndices"] = self.pixelIndices
        return d

    def __repr__(self):
        try:
            pix = f", {self.nPixels} pixels"
        except RuntimeError:
            pix = ""
        sig = ""
        if self._s.overlay is not None:
            sig = f" ({self.nSignal} signal, {self.nBib} BIB)"
        return f"LinkedCluster(ievent={self.ievent}, index={self.index}, row={self.row}: {self.nSimHits} sim hits{sig}{pix})"


class _Store:
    """The tables and link arrays shared by all LinkedCluster objects of one LinkedClusters."""
    pass


class LinkedClusters(Sequence):
    """All clusters of a file as a lazy sequence of LinkedCluster objects."""

    def __init__(self, clusters, simHits, simLinks, pixels=None, pixelEdges=None):
        """clusters, simHits: the <key>Linked tables; simLinks: vtxClusterSimHitLinksDf (cluster_row, sim_row,
        weight, dist_mm); pixels: vtxPixelsDfLinked (cluster_row); pixelEdges: vtxPixelClusterEdgesDf, used only
        when there is no pixel table."""
        s = _Store()
        n = len(clusters)
        s.clusters, s.simHits, s.pixels = clusters, simHits, pixels
        s.ievent, s.index = clusters["ievent"].to_numpy(), clusters["index"].to_numpy()
        s.overlay = simHits["overlay"].to_numpy(bool) if "overlay" in simHits else None

        cr, sr = simLinks["cluster_row"].to_numpy(np.int64), simLinks["sim_row"].to_numpy(np.int64)
        if len(cr) and (cr.min() < 0 or cr.max() >= n or sr.min() < 0 or sr.max() >= len(simHits)):
            raise ValueError("a row number in the sim hit link table is outside the cluster / sim hit table: "
                             "were the tables re-sorted or filtered after linkEdmHits.py saved them?")
        s.simOffsets, s.simRows, s.simLinkOrder = _csr(cr, sr, n)
        s.linkWeight = simLinks["weight"].to_numpy() if "weight" in simLinks else np.ones(len(cr))
        s.linkDist = simLinks["dist_mm"].to_numpy() if "dist_mm" in simLinks else np.full(len(cr), np.nan)

        s.pixOffsets = s.pixRows = None
        s.pixIndex = pixels["index"].to_numpy() if pixels is not None and "index" in pixels else None   # cached once
        if pixels is not None:
            owner = pixels["cluster_row"].to_numpy(np.int64)
            has = owner >= 0
            if has.any() and owner.max() >= n:
                raise ValueError("a cluster_row in the pixel table is outside the cluster table")
            s.pixOffsets, s.pixRows, _ = _csr(owner[has], np.flatnonzero(has), n)
        elif pixelEdges is not None:
            owner = pixelEdges["cluster_row"].to_numpy(np.int64)
            s.pixOffsets, s.pixRows, _ = _csr(owner, pixelEdges["pixel_index"].to_numpy(np.int64), n)
        self._s, self._n = s, n

    # ---- loading -----------------------------------------------------------------------------
    @staticmethod
    def _read(stub, name, columns=None):
        for ext in (".parquet", ".csv"):
            path = f"{stub}.{name}{ext}"
            if os.path.exists(path):
                if ext == ".parquet":
                    return pd.read_parquet(path, columns=columns)
                df = pd.read_csv(path, usecols=(lambda c: not c.startswith("Unnamed")) if columns is None else columns)
                return df
        raise FileNotFoundError(f"{stub}.{name}.parquet/.csv not found (was it saved by linkEdmHits.py?)")

    @classmethod
    def fromFiles(cls, fileStub, clusterColumns=None, simHitColumns=None, pixelColumns=None):
        """Load the tables saved by linkEdmHits.py for fileStub. The *Columns arguments restrict the columns
        that are loaded (to save memory); 'ievent', 'index', 'overlay' and the pointer columns are always loaded."""
        manifestPath = f"{fileStub}.linkManifest.json"
        if not os.path.exists(manifestPath):
            raise FileNotFoundError(f"{manifestPath} not found: run linkEdmHits.py on {fileStub} first")
        roles = json.load(open(manifestPath))["roles"]

        def cols(wanted, always):
            return None if wanted is None else list(dict.fromkeys(list(always) + list(wanted)))
        clusters = cls._read(fileStub, roles["clusters"], cols(clusterColumns, ["ievent", "index"]))
        simHits = cls._read(fileStub, roles["simHits"], cols(simHitColumns, ["ievent", "index", "overlay"]))
        links = cls._read(fileStub, roles["clusterSimHitLinks"], ["cluster_row", "sim_row", "weight", "dist_mm"])
        pixels = edges = None
        if roles.get("pixels"):
            pixels = cls._read(fileStub, roles["pixels"], cols(pixelColumns, ["ievent", "index", "cluster_row"]))
        elif roles.get("pixelEdges"):
            edges = cls._read(fileStub, roles["pixelEdges"], ["cluster_row", "pixel_index"])
        return cls(clusters, simHits, links, pixels=pixels, pixelEdges=edges)

    # ---- sequence ----------------------------------------------------------------------------
    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [LinkedCluster(self._s, r) for r in range(*i.indices(self._n))]
        if not isinstance(i, (int, np.integer)):
            raise TypeError("index with an int or a slice")
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("cluster index out of range")
        return LinkedCluster(self._s, i)

    def __iter__(self):
        for r in range(self._n):
            yield LinkedCluster(self._s, r)

    def inEvent(self, ievent):
        """the clusters of one event"""
        rows = np.flatnonzero(self._s.ievent == ievent)
        return [LinkedCluster(self._s, r) for r in rows]

    # ---- quick numbers without creating objects ----------------------------------------------
    @property
    def nSimHits(self):
        return np.diff(self._s.simOffsets)

    @property
    def nPixels(self):
        if self._s.pixOffsets is None:
            raise RuntimeError("no pixel links were saved for this file")
        return np.diff(self._s.pixOffsets)

    @property
    def nSignal(self):
        if self._s.overlay is None:
            raise RuntimeError("the sim hit table has no 'overlay' column")
        return np.bincount(np.repeat(np.arange(self._n), self.nSimHits), weights=(~self._s.overlay[self._s.simRows]).astype(float),
                           minlength=self._n).astype(int)

    def summary(self):
        out = {"clusters": self._n, "sim hit links": int(self.nSimHits.sum()), "max sim hits in a cluster": int(self.nSimHits.max())}
        if self._s.pixOffsets is not None:
            out.update({"pixels in clusters": int(self.nPixels.sum()), "max pixels in a cluster": int(self.nPixels.max())})
        if self._s.overlay is not None:
            out["clusters with a signal sim hit"] = int((self.nSignal > 0).sum())
        return out


if __name__ == "__main__":
    import sys
    stub = sys.argv[1] if len(sys.argv) > 1 else "./edmTestFiles/digi_output_mu2.edm4hep.root"
    cl = LinkedClusters.fromFiles(stub)
    print(cl.summary())
    print(cl[0])