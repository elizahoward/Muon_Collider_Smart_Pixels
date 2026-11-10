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
Use list(clusters[:100]) (or list(clusters)) if you want a real list of LinkedCluster objects.

Selecting clusters: clusters[5] is one LinkedCluster; every other kind of index gives a LinkedClusters again (same
tables, nothing copied) that you can select from further and use like the whole file:
    clusters = clusters[np.where(clusters.cellFields["layer"] == 0)]     # np.where's tuple works
    clusters = clusters[clusters.nSignal > 0]                            # a boolean mask, one entry per cluster
    clusters = clusters[[3, 17, -1]]  /  clusters[:1000]  /  clusters[positions]   # positions or a slice
Column accessors, counts, cellFields and clusterTable of a selection cover only its clusters, in its order.
LinkedCluster.row is always the row in the cluster table of the whole file. simHitTable and pixelTable stay the
tables of ALL sim hits / pixels (a sim hit can belong to clusters inside and outside your selection).

Where is a cluster? (system / side / layer / module / sensor)
    c.cellFields           {"system": 1, "side": 0, "layer": 0, "module": 3, "sensor": 1}
    c.cellIDsAgree         True if the cellIDs of the cluster, its sim hits and its pixels all decode to the same fields
    clusters.cellFields    the same for all clusters at once: {field: array with one value per cluster}
    clusters.cellIDsAgree  bool array
  All or nothing: if ANY of those cellIDs decodes differently from the cluster's own, EVERY field of that cluster is
  set to -1 (change it with mismatchValue=). Clusters without sim hits or pixels are compared on what they have.
  The sources are the cluster table, the sim hit table and the pixel table; a missing source is an ERROR, not skipped:
  a table without a 'cellID' column, or no pixel table at all. Choose the sources explicitly with
  cellSources=("simHits",), ("pixels",) or () (cluster cellID only).
  CAUTION: -1 is also a real value of a signed field such as 'side' (endcap, negative z): use cellIDsAgree to tell.
  How the cellID is cut up comes from the decoder: pass decoder=<callable: array of cellIDs -> {field: array}>
  (e.g. your own makeDecoder()); the default is cellid_decode.makeDecoder(MAIA_V0_TRACKER_ENCODING) from the file
  next to this one.

Whole columns at once (fast): a loop like  [c.simHits["cellID"] for c in clusters]  builds one pandas DataFrame per
cluster (~45 us each, tens of seconds for ~1M clusters). Ask for the column of ALL clusters instead:
    ids = clusters.simHitColumn("cellID")   # a Ragged: one numpy array per cluster, stored as ONE flat array + offsets
    ids[1234]                               # the cellIDs of the sim hits of cluster 1234 (a numpy view, no copy)
    ids.flat, ids.offsets, ids.counts       # the flat values and where every cluster starts
    ids.max(), ids.sum(), ids.mean()        # one number per cluster, vectorised
    ids.toAwkward(), ids.toList()           # an awkward array / a list of numpy arrays
    clusters.pixelColumn("eDep"), clusters.pixelIndices, clusters.simDistancesMm, clusters.simWeights    # same idea
    clusters.clusterTable, clusters.simHitTable, clusters.pixelTable                                      # the DataFrames
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


class Ragged:
    """One numpy array per cluster, stored as a single flat array plus offsets (no per-cluster Python objects).
    r[i] is the values of cluster i (a numpy view); the values of cluster i are flat[offsets[i]:offsets[i+1]]."""
    __slots__ = ("flat", "offsets")

    def __init__(self, flat, offsets):
        self.flat, self.offsets = flat, offsets

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if not isinstance(i, (int, np.integer)):
            raise TypeError("index with an int or a slice")
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError("cluster index out of range")
        return self.flat[self.offsets[i]:self.offsets[i + 1]]

    def __iter__(self):
        flat, off = self.flat, self.offsets
        for i in range(len(self)):
            yield flat[off[i]:off[i + 1]]

    @property
    def counts(self):
        """number of values of every cluster"""
        return np.diff(self.offsets)

    def _reduce(self, ufunc, empty):
        out = np.full(len(self), empty, dtype=float)
        nonEmpty = self.counts > 0
        if nonEmpty.any():
            out[nonEmpty] = ufunc.reduceat(self.flat, self.offsets[:-1][nonEmpty])
        return out

    def sum(self):
        return self._reduce(np.add, 0.0)

    def max(self):
        """per-cluster maximum (NaN for a cluster without values)"""
        return self._reduce(np.maximum, np.nan)

    def min(self):
        return self._reduce(np.minimum, np.nan)

    def mean(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.sum() / self.counts

    def toList(self):
        return list(self)

    def toAwkward(self):
        import awkward as ak
        return ak.unflatten(self.flat, self.counts)

    def __repr__(self):
        return f"Ragged({len(self)} clusters, {len(self.flat):,} values, dtype {self.flat.dtype})"


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

    @property
    def cellFields(self):
        """{field: value} decoded from the cellID of the cluster, its sim hits and its pixels. If any of them
        decodes differently from the cluster's own, every field is mismatchValue (-1)."""
        v = self._s.collection._cellInfo()["values"]
        return {k: int(a[self.row]) for k, a in v.items()}

    @property
    def cellIDsAgree(self):
        """True if the cluster, its sim hits and its pixels all decode to the same fields"""
        return bool(self._s.collection._cellInfo()["agree"][self.row])

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

    def __init__(self, clusters, simHits, simLinks, pixels=None, pixelEdges=None, decoder=None, mismatchValue=-1,
                 cellSources=("simHits", "pixels")):
        """clusters, simHits: the <key>Linked tables; simLinks: vtxClusterSimHitLinksDf (cluster_row, sim_row,
        weight, dist_mm); pixels: vtxPixelsDfLinked (cluster_row); pixelEdges: vtxPixelClusterEdgesDf, used only
        when there is no pixel table. decoder / mismatchValue: see 'Where is a cluster?' above."""
        s = _Store()
        s.decoder, s.mismatchValue, s.cellCache, s.cellSources = decoder, mismatchValue, None, tuple(cellSources)
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
        s.collection = self
        s.n = n                                   # number of clusters in the tables (a subset has fewer)
        self._s, self._n, self._rows = s, n, None   # _rows: None = all clusters, else the selected rows of the cluster table

    # ---- loading -----------------------------------------------------------------------------
    @staticmethod
    def _read(stub, name, columns=None, optional=()):
        """columns=None loads everything; otherwise only those columns plus the optional ones that exist."""
        for ext in (".parquet", ".csv"):
            path = f"{stub}.{name}{ext}"
            if os.path.exists(path):
                if columns is not None and optional:
                    if ext == ".parquet":
                        import pyarrow.parquet as pq
                        have = pq.read_schema(path).names
                    else:
                        have = list(pd.read_csv(path, nrows=0).columns)
                    columns = list(columns) + [o for o in optional if o in have and o not in columns]
                if ext == ".parquet":
                    return pd.read_parquet(path, columns=columns)
                df = pd.read_csv(path, usecols=(lambda c: not c.startswith("Unnamed")) if columns is None else columns)
                return df
        raise FileNotFoundError(f"{stub}.{name}.parquet/.csv not found (was it saved by linkEdmHits.py?)")

    @classmethod
    def fromFiles(cls, fileStub, clusterColumns=None, simHitColumns=None, pixelColumns=None, decoder=None, mismatchValue=-1,
                  cellSources=("simHits", "pixels")):
        """Load the tables saved by linkEdmHits.py for fileStub. The *Columns arguments restrict the columns
        that are loaded (to save memory); 'ievent', 'index', 'overlay' and the pointer columns are always loaded,
        and 'cellID' too where the table has it (cellFields needs it)."""
        manifestPath = f"{fileStub}.linkManifest.json"
        if not os.path.exists(manifestPath):
            raise FileNotFoundError(f"{manifestPath} not found: run linkEdmHits.py on {fileStub} first")
        roles = json.load(open(manifestPath))["roles"]

        def cols(wanted, always):
            return None if wanted is None else list(dict.fromkeys(list(always) + list(wanted)))
        clusters = cls._read(fileStub, roles["clusters"], cols(clusterColumns, ["ievent", "index"]), optional=["cellID"])
        simHits = cls._read(fileStub, roles["simHits"], cols(simHitColumns, ["ievent", "index", "overlay"]), optional=["cellID"])
        links = cls._read(fileStub, roles["clusterSimHitLinks"], ["cluster_row", "sim_row", "weight", "dist_mm"])
        pixels = edges = None
        if roles.get("pixels"):
            pixels = cls._read(fileStub, roles["pixels"], cols(pixelColumns, ["ievent", "index", "cluster_row"]), optional=["cellID"])
        elif roles.get("pixelEdges"):
            edges = cls._read(fileStub, roles["pixelEdges"], ["cluster_row", "pixel_index"])
        return cls(clusters, simHits, links, pixels=pixels, pixelEdges=edges, decoder=decoder, mismatchValue=mismatchValue,
                   cellSources=cellSources)

    # ---- sequence ----------------------------------------------------------------------------
    @classmethod
    def _view(cls, parent, rows):
        """a collection of some clusters of the same tables; nothing is copied"""
        v = cls.__new__(cls)
        v._s, v._rows, v._n = parent._s, np.asarray(rows, dtype=np.int64), len(rows)
        return v

    def _sel(self, fullLengthArray):
        """restrict a per-cluster array of the whole file to the clusters of this collection"""
        return fullLengthArray if self._rows is None else fullLengthArray[self._rows]

    def _row(self, position):
        return int(position) if self._rows is None else int(self._rows[position])

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        """int -> LinkedCluster. Anything that selects several clusters gives a LinkedClusters again (same tables,
        no copy): a slice, a list / array of positions (negative ones count from the end), a boolean mask with one
        entry per cluster, or what np.where(condition) returns (a tuple holding one array)."""
        if isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_)):
            if i < 0:
                i += self._n
            if not 0 <= i < self._n:
                raise IndexError("cluster index out of range")
            return LinkedCluster(self._s, self._row(i))
        if isinstance(i, slice):
            pos = np.arange(self._n)[i]
        else:
            if isinstance(i, tuple):                       # np.where(cond) gives (array,)
                if len(i) != 1:
                    raise TypeError(f"index with a 1-D selection, got a tuple of {len(i)} arrays (np.where of a 2-D condition?)")
                i = i[0]
            a = np.asarray(i)
            if a.ndim != 1:
                raise TypeError("index with a 1-D selection")
            if a.size == 0:
                pos = np.zeros(0, dtype=np.int64)
            elif a.dtype == bool:
                if len(a) != self._n:
                    raise IndexError(f"a boolean mask needs one entry per cluster ({self._n}), got {len(a)}")
                pos = np.flatnonzero(a)
            elif a.dtype.kind in "iu":
                pos = np.where(a < 0, a + self._n, a).astype(np.int64)
                if pos.min() < 0 or pos.max() >= self._n:
                    raise IndexError("cluster index out of range")
            else:
                raise TypeError(f"cannot index with {a.dtype}: use an int, a slice, positions or a boolean mask")
        return self._view(self, pos if self._rows is None else self._rows[pos])

    def __iter__(self):
        rows = range(self._n) if self._rows is None else self._rows
        for r in rows:
            yield LinkedCluster(self._s, int(r))

    def __repr__(self):
        sub = "" if self._rows is None else f" (selected from {self._s.n:,})"
        return f"LinkedClusters({self._n:,} clusters{sub})"

    def inEvent(self, ievent):
        """the clusters of one event (a LinkedClusters)"""
        return self[np.flatnonzero(self._sel(self._s.ievent) == ievent)]

    # ---- whole columns of all clusters at once ------------------------------------------------
    def _ragged(self, flatAll, offsetsAll):
        """Ragged of the clusters of this collection from the flat array and offsets of the whole file"""
        if self._rows is None:
            return Ragged(flatAll, offsetsAll)
        counts = np.diff(offsetsAll)[self._rows]
        offsets = np.concatenate(([0], np.cumsum(counts)))
        idx = np.arange(offsets[-1]) + np.repeat(offsetsAll[self._rows] - offsets[:-1], counts)
        return Ragged(flatAll[idx], offsets)

    @property
    def clusterTable(self):
        """the cluster table; for a selection only its rows (same order as this collection)"""
        return self._s.clusters if self._rows is None else self._s.clusters.iloc[self._rows]

    @property
    def simHitTable(self):
        """ALL sim hits of the file (not only those of a selection; sim hits are shared between clusters)"""
        return self._s.simHits

    @property
    def pixelTable(self):
        """ALL pixels of the file"""
        return self._s.pixels

    def simHitColumn(self, name):
        """Ragged: the values of sim hit column `name` for every cluster, in the order of LinkedCluster.simHits."""
        s = self._s
        return self._ragged(s.simHits[name].to_numpy()[s.simRows], s.simOffsets)

    @property
    def simDistancesMm(self):
        """Ragged: cluster <-> sim hit distance, same order as simHitColumn"""
        s = self._s
        return self._ragged(s.linkDist[s.simLinkOrder], s.simOffsets)

    @property
    def simWeights(self):
        s = self._s
        return self._ragged(s.linkWeight[s.simLinkOrder], s.simOffsets)

    def _needPixels(self):
        if self._s.pixOffsets is None:
            raise RuntimeError("no pixel links were saved for this file")

    def pixelColumn(self, name):
        """Ragged: the values of pixel column `name` for every cluster (needs the pixel table)."""
        self._needPixels()
        s = self._s
        if s.pixels is None:
            raise RuntimeError("the pixel table was not loaded (linkEdmHits.py had no vtxPixelsDf): only pixelIndices is available")
        return self._ragged(s.pixels[name].to_numpy()[s.pixRows], s.pixOffsets)

    @property
    def pixelIndices(self):
        """Ragged: the pixels' 'index' within the event; works with and without a pixel table"""
        self._needPixels()
        s = self._s
        return self._ragged(s.pixIndex[s.pixRows] if s.pixels is not None else s.pixRows, s.pixOffsets)

    # ---- where is the cluster: fields decoded from the cellIDs --------------------------------
    def _decoderFn(self):
        s = self._s
        if s.decoder is None:
            try:
                from edm4hepTracksUtils import makeDecoder, MAIA_V0_TRACKER_ENCODING
            except ImportError:
                raise RuntimeError("no decoder: pass decoder=<callable: array of cellIDs -> {field: array}> or put "
                                   "cellid_decode.py next to linkedClusters.py") from None
            s.decoder = makeDecoder(MAIA_V0_TRACKER_ENCODING)
        return s.decoder

    @staticmethod
    def _decodeUnique(dec, cells):
        """decode only the distinct values (a decoder may be slow), then spread back"""
        u, inv = np.unique(cells, return_inverse=True)
        d = dec(u)
        return {k: np.asarray(v)[inv] for k, v in d.items()}

    def _cellInfo(self):
        s = self._s
        if s.cellCache is not None:
            return s.cellCache
        if "cellID" not in s.clusters:
            raise RuntimeError("the cluster table has no 'cellID' column")
        bad = [x for x in s.cellSources if x not in ("simHits", "pixels")]
        if bad:
            raise ValueError(f"unknown cellSources {bad}; use 'simHits' and/or 'pixels'")
        hint = "; pass cellSources=('simHits',) (or ()) to compare without it"
        if "simHits" in s.cellSources and "cellID" not in s.simHits:
            raise RuntimeError("the sim hit table has no 'cellID' column" + hint)
        if "pixels" in s.cellSources:
            if s.pixels is None:
                raise RuntimeError("the pixel table was not loaded (linkEdmHits.py had no vtxPixelsDf), so the pixel cellIDs "
                                   "cannot be compared" + hint)
            if "cellID" not in s.pixels:
                raise RuntimeError("the pixel table has no 'cellID' column" + hint)
        dec, n = self._decoderFn(), s.n
        clCell = s.clusters["cellID"].to_numpy(np.int64)
        own = self._decodeUnique(dec, clCell)
        keys = list(own)
        differs = np.zeros(n, bool)                          # some sim hit / pixel decodes differently from its cluster
        sources = ["cluster"]

        def compare(memberCell, offsets):
            owner = np.repeat(np.arange(n), np.diff(offsets))
            diff = np.flatnonzero(memberCell != clCell[owner])      # equal cellIDs agree on every field: decode only the rest
            if len(diff):
                a = self._decodeUnique(dec, memberCell[diff])
                b = self._decodeUnique(dec, clCell[owner[diff]])
                for k in keys:
                    differs[owner[diff][a[k] != b[k]]] = True
        if "simHits" in s.cellSources:
            compare(s.simHits["cellID"].to_numpy(np.int64)[s.simRows], s.simOffsets)
            sources.append("simHits")
        if "pixels" in s.cellSources:
            compare(s.pixels["cellID"].to_numpy(np.int64)[s.pixRows], s.pixOffsets)
            sources.append("pixels")
        values = {k: np.where(differs, s.mismatchValue, own[k]) for k in keys}
        agree = ~differs
        for a in list(values.values()) + [agree]:
            a.setflags(write=False)
        s.cellCache = {"values": values, "agree": agree, "sources": tuple(sources)}
        return s.cellCache

    @property
    def cellFields(self):
        """{field: array, one value per cluster}; all fields are -1 (mismatchValue) where cluster, sim hits and
        pixels do not all decode to the same fields"""
        return {k: self._sel(v) for k, v in self._cellInfo()["values"].items()}

    @property
    def cellIDsAgree(self):
        """bool array: True where cluster, sim hits and pixels all decode to the same fields"""
        return self._sel(self._cellInfo()["agree"])

    @property
    def cellFieldSources(self):
        """which cellIDs were compared, e.g. ('cluster', 'simHits', 'pixels')"""
        return self._cellInfo()["sources"]

    # ---- quick numbers without creating objects ----------------------------------------------
    @property
    def nSimHits(self):
        return self._sel(np.diff(self._s.simOffsets))

    @property
    def nPixels(self):
        if self._s.pixOffsets is None:
            raise RuntimeError("no pixel links were saved for this file")
        return self._sel(np.diff(self._s.pixOffsets))

    @property
    def nSignal(self):
        s = self._s
        if s.overlay is None:
            raise RuntimeError("the sim hit table has no 'overlay' column")
        full = np.bincount(np.repeat(np.arange(s.n), np.diff(s.simOffsets)), weights=(~s.overlay[s.simRows]).astype(float),
                           minlength=s.n).astype(int)
        return self._sel(full)

    def summary(self):
        if self._n == 0:
            return {"clusters": 0}
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