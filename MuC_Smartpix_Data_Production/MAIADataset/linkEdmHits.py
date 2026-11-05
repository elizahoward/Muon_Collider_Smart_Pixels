#!/usr/bin/env python3
"""
Author: Claude
Date: November 5, 2026
Description:
linkEdmHits.py -- link and sanity-check the dataframes written by make_tracklists_edm4hepContain.py

Command line:
    python linkEdmHits.py [fileStub] [options]       (python linkEdmHits.py --help)
    e.g. python linkEdmHits.py ./edmTestFiles/digi_output_mu2.edm4hep.root --no-save

From python / a notebook:
    linker = EdmHitLinker("./edmTestFiles/digi_output_mu2.edm4hep.root", save=False, distWarnMm=5.0)
    exitCode = linker.runAll()
    linker.outputs["vtxClusterSimHitLinksDf"]        # the linked tables
    linker.problems                                  # the failed checks (list of str)

What it does
  1. Loads <fileStub>.<key>.parquet (or .csv) for every key of the key->collection dictionary that exists.
  2. Checks the tables: collection IDs against the names in the dictionary, unique object keys,
     'index' == row position within the event.
  3. Resolves the link tables into row numbers of the tables they point into:
        cluster -> sim hit   (vtxClusterToSimHitDf)
        cluster -> pixel     (vtxPixelsToClustersDf)
     Links whose ends are missing are reported (not silently dropped).
  4. Logs cardinalities, the physical distance between linked objects, linked/unlinked sim hits,
     and a list of odd clusters / sim hits.
  5. Saves the linked tables as <fileStub>.<name>.parquet / .csv, the log as <fileStub>.linkReport.txt and
     the table map as <fileStub>.linkManifest.json (the log also ends with that map).

Saved tables and how they link   (names for the digi_output_mu2 files; the log and the manifest list the real
names and row counts of every run)

  Two ways to point at an object:
    row number  = 0-based position in a SAVED table (use .iloc). Fast, valid while the saved tables are not
                  re-sorted or filtered.
    key         = (ievent, index). 'index' is the object's position inside its event and restarts at 0 in every
                  event, so it is NOT a row number of a table that holds several events.

  vtxClustersDfLinked        vtxClustersDf + n_sim_hits, n_signal_sim_hits, n_overlay_sim_hits, max_dist_to_sim_mm,
                             n_pixels
  vtxBarrelSimHitsDf2Linked  vtxBarrelSimHitsDf2 + n_clusters, linked, max_dist_to_cluster_mm
  vtxPixelsDfLinked          vtxPixelsDf + cluster_row, cluster_index    (only if the pixel table exists)
  vtxClusterSimHitLinksDf    one row per cluster <-> sim hit link:
                                cluster_row -> row of vtxClustersDfLinked
                                sim_row     -> row of vtxBarrelSimHitsDf2Linked
                                (ievent, cluster_index) / (ievent, sim_index) = the same two objects by key
                                + weight, dist_mm, n_sim_of_cluster, n_clusters_of_sim and copies of a few columns
  vtxPixelClusterEdgesDf     only if there is NO pixel table: one row per pixel with cluster_row / cluster_index
                             and the pixel's (ievent, pixel_index) key
  vtxBarrelSimHitsDf2Linked.particle_index (+ particle_collectionID) -> the MC particle by key (ievent, index);
                             -1 = no particle (all BIB hits)

  The original tables are never overwritten: a table that gets columns is saved as <key>Linked.
  cluster -> pixels is one-to-many (every pixel belongs to exactly one cluster), so it is a column on the pixel
  table. cluster <-> sim hits is many-to-many, so it needs the link table.

  Nothing is dropped or capped: clusters with thousands of pixels or ~100 sim hits (delta rays, bad clustering)
  are kept in every table; they are only listed in the log.

Collection IDs: podio (master) derives the ID of a collection as MurmurHash3_x86_32(name, seed 0).
This was checked against the IDs seen in digi_output_mu2 (MCParticles, VXDBarrelRealHits,
VXDBarrelRealPixels, VertexBarrelCollection). It is used here to tell which collection a link points into.

Exit code: 0 if no check failed, 1 otherwise (warnings do not fail the run).
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd


class EdmHitLinker:
    # ------------------------------------------------------------------------------------------
    # class constants (nothing below is changed while running)
    # ------------------------------------------------------------------------------------------
    DEFAULT_FILE_STUB = "./edmTestFiles/digi_output_mu2.edm4hep.root"
    DEFAULT_KEY_NAMES_TO_COLLECTIONS = {"truthParticlesDf": "MCParticles",
                                        "vtxBarrelSimHitsDf": "VertexBarrelCollection",
                                        "vtxClusterToSimHitDf": "VXDBarrelRealHitsRelations",
                                        "vtxClustersDf": "VXDBarrelRealHits",
                                        "vtxPixelsToClustersDf": "VXDBarrelRealRawHitRelations",
                                        "vtxPixelsDf": "VXDBarrelRealPixels",
                                        "truthParticlesDf2": "NewMCParticles",
                                        "vtxBarrelSimHitsDf2": "OverlayVertexBarrelCollection"}

    # which table plays which role
    CLUSTERS = "vtxClustersDf"
    CLUSTER_SIM_LINKS = "vtxClusterToSimHitDf"
    PIXEL_CLUSTER_LINKS = "vtxPixelsToClustersDf"
    PIXELS = "vtxPixelsDf"
    SIMHIT_TABLES = ("vtxBarrelSimHitsDf", "vtxBarrelSimHitsDf2")    # candidates; the one the links point into is chosen by collection ID

    # columns copied into the cluster<->sim hit link table
    DEFAULT_LINKED_CLUSTER_COLS = ("cellID", "eDep", "time", "position.x", "position.y", "position.z")
    DEFAULT_LINKED_SIM_COLS = ("cellID", "eDep", "time", "pathLength", "overlay", "secondary", "particle_index",
                               "MCParticles.PDG", "MCParticles.set")

    # names of the saved tables: a loaded table that gets columns is saved as <key><OUT_SUFFIX> (the original
    # files are never overwritten); the link tables are new tables
    OUT_SUFFIX = "Linked"
    OUT_CLUSTER_SIM_LINKS = "vtxClusterSimHitLinksDf"
    OUT_PIXEL_EDGES = "vtxPixelClusterEdgesDf"       # only written when there is no pixel table to add cluster_row to
    MANIFEST_SUFFIX = "linkManifest.json"

    PATH_LENGTH_BINS = (-np.inf, 0.02, 0.05, 0.08, 0.12, 0.2, 0.5, np.inf)    # mm, for the linked-fraction table
    KEY_STRIDE = 10 ** 9                                                       # (ievent, index) -> ievent * KEY_STRIDE + index

    # options that can be changed per run (constructor arguments / command line); these are the defaults
    DEFAULTS = {"convertParquet": True,         # a table read from csv is also written as parquet (if no parquet exists yet)
                "preferParquet": True,          # read parquet before csv when both exist
                "distWarnMm": 10.0,             # linked cluster and sim hit further apart than this are 'odd'
                "maxClustersPerSimWarn": 20,    # one sim hit feeding this many clusters or more is 'odd'
                "maxSimPerClusterWarn": 20,     # one cluster made from this many sim hits or more is 'odd'
                "topN": 10,                     # rows shown in the 'odd objects' lists
                "saveCsv": True,
                "csvMaxRows": 400_000,          # bigger tables are only written as parquet (csv row ~230 bytes, GitHub refuses > 100 MB)
                "saveParquet": True}

    # ------------------------------------------------------------------------------------------
    def __init__(self, fileStub=None, keyNamesToCollections=None, save=True,
                 linkedClusterCols=None, linkedSimCols=None, **options):
        unknown = set(options) - set(self.DEFAULTS)
        if unknown:
            raise TypeError(f"unknown option(s) {sorted(unknown)}; valid options: {sorted(self.DEFAULTS)}")
        opts = {**self.DEFAULTS, **options}
        self.fileStub = fileStub if fileStub is not None else self.DEFAULT_FILE_STUB
        self.keyNamesToCollections = dict(keyNamesToCollections if keyNamesToCollections is not None
                                          else self.DEFAULT_KEY_NAMES_TO_COLLECTIONS)
        self.save = save                                  # False: only check and log, write nothing
        self.linkedClusterCols = tuple(linkedClusterCols if linkedClusterCols is not None else self.DEFAULT_LINKED_CLUSTER_COLS)
        self.linkedSimCols = tuple(linkedSimCols if linkedSimCols is not None else self.DEFAULT_LINKED_SIM_COLS)
        self.convertParquet = opts["convertParquet"]
        self.preferParquet = opts["preferParquet"]
        self.distWarnMm = opts["distWarnMm"]
        self.maxClustersPerSimWarn = opts["maxClustersPerSimWarn"]
        self.maxSimPerClusterWarn = opts["maxSimPerClusterWarn"]
        self.topN = opts["topN"]
        self.saveCsv = opts["saveCsv"]
        self.csvMaxRows = opts["csvMaxRows"]
        self.saveParquet = opts["saveParquet"]

        self.idToName = {self.collectionId(n): n for n in self.keyNamesToCollections.values()}
        self.dfs = {}            # loaded tables (the big ones get the new columns added in place)
        self.outputs = {}        # the linked tables, filled by runAll()
        self.problems = []       # failed checks
        self.simKey = None       # the sim hit table the cluster links point into (set by runAll)
        self.log = logging.getLogger(f"linkEdmHits.{id(self)}")

    # ------------------------------------------------------------------------------------------
    # logging / small helpers
    # ------------------------------------------------------------------------------------------
    def _setupLogging(self):
        class Fmt(logging.Formatter):
            def format(self, rec):
                msg = super().format(rec)
                return msg if rec.levelno == logging.INFO else f"{rec.levelname}: {msg}"
        self.log.handlers.clear()
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        handlers = [logging.StreamHandler(sys.stdout)]
        if self.save:
            handlers.append(logging.FileHandler(f"{self.fileStub}.linkReport.txt", mode="w"))
        for h in handlers:
            h.setFormatter(Fmt("%(message)s"))
            self.log.addHandler(h)

    def _closeLogging(self):
        for h in list(self.log.handlers):
            h.flush()
            if isinstance(h, logging.FileHandler):
                h.close()
            self.log.removeHandler(h)

    def problem(self, msg):
        self.problems.append(msg)
        self.log.error(msg)

    def show(self, title, df):
        self.log.info("%s\n%s", title, df.to_string())

    @staticmethod
    def murmur3_32(data: bytes, seed: int = 0) -> int:
        c1, c2, M = 0xcc9e2d51, 0x1b873593, 0xFFFFFFFF
        h, n = seed, len(data)
        for i in range(0, n - n % 4, 4):
            k = int.from_bytes(data[i:i + 4], "little")
            k = (k * c1) & M; k = ((k << 15) | (k >> 17)) & M; k = (k * c2) & M
            h ^= k; h = ((h << 13) | (h >> 19)) & M; h = (h * 5 + 0xe6546b64) & M
        tail = data[n - n % 4:]
        k = 0
        for j in range(len(tail) - 1, -1, -1):
            k = (k << 8) | tail[j]
        if tail:
            k = (k * c1) & M; k = ((k << 15) | (k >> 17)) & M; k = (k * c2) & M; h ^= k
        h ^= n; h ^= h >> 16; h = (h * 0x85ebca6b) & M; h ^= h >> 13; h = (h * 0xc2b2ae35) & M; h ^= h >> 16
        return h

    @classmethod
    def collectionId(cls, name: str) -> int:
        return cls.murmur3_32(name.encode(), 0)

    @classmethod
    def linkedName(cls, key):
        """Name of the saved version of a loaded table."""
        return f"{key}{cls.OUT_SUFFIX}"

    def nameOf(self, collId):
        return self.idToName.get(int(collId), f"unknown({int(collId)})")

    @classmethod
    def keyOf(cls, ievent, index):
        return np.asarray(ievent, dtype=np.int64) * cls.KEY_STRIDE + np.asarray(index, dtype=np.int64)

    @staticmethod
    def histo(a, kmax=5):
        counts = np.bincount(np.minimum(np.asarray(a, dtype=np.int64), kmax), minlength=kmax + 1)
        return {(str(k) if k < kmax else f"{kmax}+"): int(v) for k, v in enumerate(counts)}

    class Lookup:
        """Row position of an object in its table from (ievent, collectionID, index); -1 if there is none."""

        def __init__(self, df):
            self.keys = pd.Index(EdmHitLinker.keyOf(df["ievent"], df["index"]))
            self.coll = df["collectionID"].to_numpy(np.int64)
            self.unique = self.keys.is_unique

        def find(self, ievent, coll, index):
            pos = self.keys.get_indexer(EdmHitLinker.keyOf(ievent, index))
            found = pos >= 0
            wrongColl = found & (self.coll[np.where(found, pos, 0)] != np.asarray(coll, dtype=np.int64))
            pos[wrongColl] = -1
            return pos

    # ------------------------------------------------------------------------------------------
    # loading / saving
    # ------------------------------------------------------------------------------------------
    def readTable(self, key):
        pqPath, csvPath = f"{self.fileStub}.{key}.parquet", f"{self.fileStub}.{key}.csv"
        for path in ([pqPath, csvPath] if self.preferParquet else [csvPath, pqPath]):
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]     # the saved row index
            if path.endswith(".csv") and self.convertParquet and len(df) > 0 and not os.path.exists(pqPath):
                df.to_parquet(pqPath, index=False)
                self.log.info("  %s: also written as %s", key, os.path.basename(pqPath))
            return df, path
        return None, None

    def loadAll(self):
        self.dfs = {}
        for key in self.keyNamesToCollections:
            df, path = self.readTable(key)
            if df is None:
                self.log.warning("%-24s no parquet/csv found -> skipped", key)
            elif len(df) == 0:
                self.log.info("%-24s empty table (%s) -> skipped", key, os.path.basename(path))
            else:
                self.dfs[key] = df.reset_index(drop=True)
                self.log.info("%-24s %11s rows, %2d columns  <- %s", key, f"{len(df):,}", df.shape[1], os.path.basename(path))

    def saveTable(self, df, name):
        written = []
        if self.saveParquet:
            df.to_parquet(f"{self.fileStub}.{name}.parquet", index=False)
            written.append("parquet")
        if self.saveCsv:
            if len(df) <= self.csvMaxRows:
                df.to_csv(f"{self.fileStub}.{name}.csv", index=False)
                written.append("csv")
            else:
                self.log.info("  %s: %s rows > csvMaxRows, csv skipped", name, f"{len(df):,}")
        self.log.info("saved %-28s %11s rows  (%s)", name, f"{len(df):,}",
                      ", ".join(written) if written else "nothing, saving is off")

    # ------------------------------------------------------------------------------------------
    # checks and linking
    # ------------------------------------------------------------------------------------------
    def checkTables(self):
        self.log.info("\n== 1. tables, collection IDs, object keys ==")
        for key, df in self.dfs.items():
            expected = self.collectionId(self.keyNamesToCollections[key])
            evs = df["ievent"].nunique() if "ievent" in df else "?"
            line = f"{key:24s} events {evs}"
            if "collectionID" in df:
                ids = df["collectionID"].unique()
                line += f" | collectionID -> {[self.nameOf(i) for i in ids]}"
                if set(int(i) for i in ids) != {expected}:
                    self.problem(f"{key}: collectionID {[int(i) for i in ids]} does not match "
                                 f"'{self.keyNamesToCollections[key]}' ({expected}); is the dictionary right?")
            self.log.info(line)
            if {"ievent", "index", "collectionID"} <= set(df.columns):
                dup = int(pd.Series(self.keyOf(df["ievent"], df["index"])).duplicated().sum())
                if dup:
                    self.problem(f"{key}: {dup} duplicate (ievent, index) keys; lookups into this table are ambiguous")
                pos = df.groupby("ievent").cumcount().to_numpy()
                if not np.array_equal(pos, df["index"].to_numpy()):
                    self.log.warning("%s: 'index' is not the row position within the event (rows reordered or filtered?)", key)

    def pickSimTable(self):
        """The sim hit table that the cluster -> sim hit links point into (by collection ID), or None."""
        toIds = set(int(i) for i in self.dfs[self.CLUSTER_SIM_LINKS]["to_collectionID"].unique())
        simKey = None
        for k in self.SIMHIT_TABLES:
            if k in self.dfs and set(int(i) for i in self.dfs[k]["collectionID"].unique()) == toIds:
                simKey = k
        if simKey is None:
            self.problem(f"cluster->sim hit links point into {[self.nameOf(i) for i in toIds]}, "
                         f"none of {list(self.SIMHIT_TABLES)} is that collection")
        return simKey

    def checkSimHitParticles(self, simKey):
        sim = self.dfs[simKey]
        self.log.info("\n== particle references in %s ==", simKey)
        if "particle_collectionID" not in sim:
            return
        unset = sim["particle_collectionID"] < 0
        byColl = {self.nameOf(k): int(v) for k, v in sim.loc[~unset, "particle_collectionID"].value_counts().items()}
        self.log.info("particle references: %s | unset (-1): %s of %s", byColl, f"{int(unset.sum()):,}", f"{len(sim):,}")
        if "overlay" in sim:
            sig = ~sim["overlay"].astype(bool)
            self.log.info("signal hits: %d, with unset particle: %d | overlay hits: %s, with unset particle: %s",
                          int(sig.sum()), int((sig & unset).sum()), f"{int((~sig).sum()):,}", f"{int((~sig & unset).sum()):,}")
            if (sig & unset).any():
                self.log.warning("%d signal sim hits have no MC particle", int((sig & unset).sum()))
        if "MCParticles.set" in sim:
            mismatch = int((sim["MCParticles.set"].astype(bool) == unset).sum())
            if mismatch:
                self.problem(f"{simKey}: 'MCParticles.set' disagrees with particle_collectionID in {mismatch} rows")

    def linkClusterSim(self, simKey):
        """Returns (cluster table, sim hit table, link table) with the new columns, or None."""
        cl, sim, links = self.dfs[self.CLUSTERS], self.dfs[simKey], self.dfs[self.CLUSTER_SIM_LINKS]
        log = self.log
        log.info("\n== 2. cluster -> sim hit links (%s -> %s) ==", self.CLUSTER_SIM_LINKS, simKey)
        ends = {self.nameOf(links["from_collectionID"].iloc[0]), self.nameOf(links["to_collectionID"].iloc[0])}
        log.info("link ends point into: %s", sorted(ends))
        cLook, sLook = self.Lookup(cl), self.Lookup(sim)
        if not (cLook.unique and sLook.unique):
            self.problem("cannot resolve cluster/sim hit links: object keys are not unique (see above)")
            return None
        ev = links["ievent"].to_numpy(np.int64)
        fpos = cLook.find(ev, links["from_collectionID"], links["from_index"])
        tpos = sLook.find(ev, links["to_collectionID"], links["to_index"])
        for label, pos, tab in (("cluster", fpos, self.CLUSTERS), ("sim hit", tpos, simKey)):
            n = int((pos < 0).sum())
            if n:
                self.problem(f"{n:,} of {len(links):,} links have a {label} end that is not in {tab}")
                self.show("  first few:", links.loc[pos < 0, ["ievent", "from_index", "to_index", "to_collectionID"]].head())
        ok = (fpos >= 0) & (tpos >= 0)
        fpos, tpos, ev = fpos[ok], tpos[ok], ev[ok]
        weight = links["weight"].to_numpy()[ok] if "weight" in links else np.ones(len(fpos))
        log.info("links resolved: %s of %s", f"{len(fpos):,}", f"{len(links):,}")
        if len(fpos) == 0:
            self.problem("no cluster -> sim hit link could be resolved; skipping the rest of the link checks")
            return None
        dup = int(pd.Series(fpos.astype(np.int64) * len(sim) + tpos).duplicated().sum())
        if dup:
            self.problem(f"{dup} duplicate (cluster, sim hit) links")
        w, wc = np.unique(weight, return_counts=True)
        log.info("weights: %s%s", dict(zip(np.round(w[:6], 4).tolist(), wc[:6].tolist())), "" if len(w) <= 6 else " ...")
        if len(w) > 1 or (len(w) == 1 and w[0] != 1.0):
            log.info("  (weights are not all 1.0, so they carry information)")

        nSim = np.bincount(fpos, minlength=len(cl))
        nCl = np.bincount(tpos, minlength=len(sim))
        log.info("sim hits per cluster:  %s", self.histo(nSim))
        log.info("clusters per sim hit:  %s", self.histo(nCl))
        log.info("max sim hits per cluster %d | max clusters per sim hit %d", nSim.max(), nCl.max())
        if (nSim == 0).any():
            log.warning("%s clusters have no sim hit link", f"{int((nSim == 0).sum()):,}")

        # physical distance between linked objects
        pos3 = ["position.x", "position.y", "position.z"]
        dist = np.full(len(fpos), np.nan)
        if set(pos3) <= set(cl.columns) and set(pos3) <= set(sim.columns):
            dist = np.linalg.norm(cl[pos3].to_numpy()[fpos] - sim[pos3].to_numpy()[tpos], axis=1)
            q = np.quantile(dist, [.5, .9, .99, 1])
            one = (nSim[fpos] == 1) & (nCl[tpos] == 1)
            q1 = np.quantile(dist[one], [.5, .99, 1]) if one.any() else [np.nan] * 3
            log.info("distance cluster<->sim hit [mm], all links  (median, 90%%, 99%%, max): %s", np.round(q, 3).tolist())
            log.info("                         one-to-one links  (median, 99%%, max):      %s", np.round(q1, 3).tolist())
            far = dist > self.distWarnMm
            log.info("links further apart than %g mm: %s (%.2f%%)", self.distWarnMm, f"{int(far.sum()):,}", 100 * far.mean())
        if "cellID" in cl and "cellID" in sim:
            same = cl["cellID"].to_numpy()[fpos] == sim["cellID"].to_numpy()[tpos]
            log.info("cluster and sim hit have the same cellID in %.1f%% of links", 100 * same.mean())

        # largest distance per cluster / per sim hit (-inf start: np.maximum would keep a NaN start forever)
        maxDistCl = np.full(len(cl), np.nan)
        maxDistSim = np.full(len(sim), np.nan)
        if not np.isnan(dist).all():
            tmpCl = np.full(len(cl), -np.inf); tmpSim = np.full(len(sim), -np.inf)
            np.maximum.at(tmpCl, fpos, dist); np.maximum.at(tmpSim, tpos, dist)
            maxDistCl = np.where(np.isinf(tmpCl), np.nan, tmpCl); maxDistSim = np.where(np.isinf(tmpSim), np.nan, tmpSim)
        overlay = sim["overlay"].to_numpy(bool) if "overlay" in sim else np.zeros(len(sim), bool)
        nSignalInCl = np.bincount(fpos, weights=(~overlay[tpos]).astype(float), minlength=len(cl)).astype(int)

        # the new columns are added to the loaded tables themselves (no copies: the big tables are ~1M rows)
        clOut = cl
        clOut["n_sim_hits"] = nSim
        clOut["n_signal_sim_hits"] = nSignalInCl
        clOut["n_overlay_sim_hits"] = nSim - nSignalInCl
        clOut["max_dist_to_sim_mm"] = maxDistCl
        simOut = sim
        simOut["n_clusters"] = nCl
        simOut["linked"] = nCl > 0
        simOut["max_dist_to_cluster_mm"] = maxDistSim

        linkOut = pd.DataFrame({"ievent": ev,
                                "cluster_row": fpos.astype(np.int64),            # row number in <clusters>Linked
                                "sim_row": tpos.astype(np.int64),                # row number in <simhits>Linked
                                "cluster_index": cl["index"].to_numpy()[fpos],   # the same objects by key (ievent, index)
                                "sim_index": sim["index"].to_numpy()[tpos],
                                "weight": weight, "dist_mm": dist,
                                "n_sim_of_cluster": nSim[fpos], "n_clusters_of_sim": nCl[tpos]})
        for c in self.linkedClusterCols:
            if c in cl:
                linkOut["cl_" + c] = cl[c].to_numpy()[fpos]
        for c in self.linkedSimCols:
            if c in sim:
                linkOut["sim_" + c] = sim[c].to_numpy()[tpos]
        return clOut, simOut, linkOut

    def linkPixels(self, clOut):
        """Cluster -> pixel links. Returns (name, table): the pixel table with cluster_row / cluster_index added
        when it is loaded, otherwise an edge table (ievent, cluster_row, cluster_index, pixel_index)."""
        links = self.dfs[self.PIXEL_CLUSTER_LINKS]
        log = self.log
        log.info("\n== 3. cluster -> pixel links (%s) ==", self.PIXEL_CLUSTER_LINKS)
        log.info("link ends point into: %s",
                 sorted({self.nameOf(links["from_collectionID"].iloc[0]), self.nameOf(links["to_collectionID"].iloc[0])}))
        cLook = self.Lookup(self.dfs[self.CLUSTERS])
        ev = links["ievent"].to_numpy(np.int64)
        fpos = cLook.find(ev, links["from_collectionID"], links["from_index"])
        n = int((fpos < 0).sum())
        if n:
            self.problem(f"{n:,} of {len(links):,} pixel links have a cluster end that is not in {self.CLUSTERS}")
        ok = fpos >= 0
        pxIdx = links["to_index"].to_numpy(np.int64)

        pixels = self.dfs.get(self.PIXELS)
        ppos = None                                    # row of each link's pixel in the pixel table
        if pixels is not None:
            pLook = self.Lookup(pixels)
            if not pLook.unique:
                self.problem(f"{self.PIXELS}: object keys are not unique")
            else:
                ppos = pLook.find(ev, links["to_collectionID"], pxIdx)
                m = int((ppos < 0).sum())
                if m:
                    self.problem(f"{m:,} pixel links have a pixel end that is not in {self.PIXELS}")
                else:
                    log.info("every pixel link end is a row of %s", self.PIXELS)
            if "particle_collectionID" in pixels:
                log.info("pixels with an unset MC particle: %.1f%%", 100 * (pixels["particle_collectionID"] < 0).mean())
        else:
            log.info("%s was not loaded: the pixel end is checked by ID and index continuity only", self.PIXELS)
            ids = links["to_collectionID"].unique()
            want = self.collectionId(self.keyNamesToCollections[self.PIXELS])
            if set(int(i) for i in ids) != {want}:
                self.problem(f"pixel link ends point into {[self.nameOf(i) for i in ids]}, "
                             f"expected {self.keyNamesToCollections[self.PIXELS]}")
        dupPx = int(pd.Series(self.keyOf(ev, pxIdx)).duplicated().sum())
        if dupPx:
            self.problem(f"{dupPx:,} pixels are linked to more than one cluster")
        g = links.groupby("ievent")["to_index"].agg(["nunique", "max"])
        if not (g["nunique"] == g["max"] + 1).all():
            log.warning("pixel indices are not contiguous 0..N-1 in every event")
        else:
            log.info("pixel indices per event are contiguous: %s", {int(k): int(v) for k, v in g["nunique"].items()})

        nPix = np.bincount(fpos[ok], minlength=len(clOut))
        clOut["n_pixels"] = nPix
        log.info("pixels per cluster: median %d | mean %.1f | max %d (kept) | clusters with no pixel link: %d",
                 int(np.median(nPix)), nPix.mean(), nPix.max(), int((nPix == 0).sum()))
        if (nPix == 0).any():
            log.warning("%d clusters have no pixel link", int((nPix == 0).sum()))
        nSim = clOut["n_sim_hits"].to_numpy()
        pairs = int((nPix.astype(np.int64) * nSim).sum())
        ambiguous = int(nPix[nSim > 1].sum())
        log.info("pixel<->sim hit pairs implied through the clusters: %s (not materialised); pixels in clusters with "
                 "more than one sim hit (truth ambiguous): %s of %s", f"{pairs:,}", f"{ambiguous:,}", f"{int(ok.sum()):,}")

        clIndex = self.dfs[self.CLUSTERS]["index"].to_numpy()
        if ppos is not None:
            # every pixel belongs to one cluster: the pointer is a column of the pixel table
            rowOf = np.full(len(pixels), -1, dtype=np.int64)
            good = ok & (ppos >= 0)
            rowOf[ppos[good]] = fpos[good]
            pixels["cluster_row"] = rowOf
            pixels["cluster_index"] = np.where(rowOf >= 0, clIndex[np.maximum(rowOf, 0)], -1)
            noCluster = int((rowOf < 0).sum())
            if noCluster:
                log.warning("%s pixels of %s belong to no cluster (cluster_row = -1)", f"{noCluster:,}", self.PIXELS)
            return self.linkedName(self.PIXELS), pixels
        edges = pd.DataFrame({"ievent": ev[ok].astype(np.int32),
                              "cluster_row": fpos[ok].astype(np.int64),
                              "cluster_index": clIndex[fpos[ok]].astype(np.int32),
                              "pixel_index": pxIdx[ok].astype(np.int32)})
        return self.OUT_PIXEL_EDGES, edges

    # ------------------------------------------------------------------------------------------
    # reports
    # ------------------------------------------------------------------------------------------
    def reportUnlinked(self, simKey, simOut):
        log = self.log
        log.info("\n== 4. sim hits without a cluster (%s) ==", simKey)
        linked = simOut["linked"]
        log.info("sim hits %s | linked %s | unlinked %s (%.1f%%)", f"{len(simOut):,}", f"{int(linked.sum()):,}",
                 f"{int((~linked).sum()):,}", 100 * (~linked).mean())
        if "overlay" in simOut:
            self.show("by overlay flag:", pd.crosstab(simOut["overlay"], linked.rename("linked"), margins=True))
            sigUnlinked = int((~simOut["overlay"].astype(bool) & ~linked).sum())
            if sigUnlinked:
                self.problem(f"{sigUnlinked} signal sim hits have no cluster")
        if "pathLength" in simOut:
            cut = pd.cut(simOut["pathLength"], list(self.PATH_LENGTH_BINS))
            self.show("fraction linked by path length in the sensor [mm]:",
                      simOut.groupby(cut, observed=True)["linked"].agg(n_hits="size", frac_linked="mean").round(3))
        if "cellID" in simOut:
            g = simOut.groupby("cellID")["linked"].agg(["size", "mean"])
            log.info("linked fraction per cellID (%d sensors): min %.2f, median %.2f, max %.2f; sensors with nothing linked: %d",
                     len(g), g["mean"].min(), g["mean"].median(), g["mean"].max(), int((g["mean"] == 0).sum()))
            if (g["mean"] == 0).any():
                log.warning("sensors where no sim hit is linked: %s", g.index[g["mean"] == 0].tolist()[:10])

    def reportOdd(self, simKey, clOut, simOut):
        log, top = self.log, self.topN
        log.info("\n== 5. odd objects ==")
        log.info("nothing is dropped or capped: all %s clusters and %s sim hits are kept; the lists below only report large or odd ones",
                 f"{len(clOut):,}", f"{len(simOut):,}")
        shared = simOut[simOut["n_clusters"] >= self.maxClustersPerSimWarn]
        log.info("sim hits feeding >= %d clusters: %d", self.maxClustersPerSimWarn, len(shared))
        simCols = [c for c in ["ievent", "index", "eDep", "pathLength", "time", "overlay", "cellID", "n_clusters",
                               "max_dist_to_cluster_mm"] if c in simOut]
        if len(shared):
            self.show(f"top {top} sim hits by number of clusters:", shared.nlargest(top, "n_clusters")[simCols])
        multi = clOut[clOut["n_sim_hits"] >= self.maxSimPerClusterWarn]
        log.info("clusters made from >= %d sim hits: %d", self.maxSimPerClusterWarn, len(multi))
        clCols = [c for c in ["ievent", "index", "cellID", "eDep", "n_pixels", "n_sim_hits", "n_signal_sim_hits",
                              "max_dist_to_sim_mm"] if c in clOut]
        if len(multi):
            self.show(f"top {top} clusters by number of sim hits:", multi.nlargest(top, "n_sim_hits")[clCols])
        if "n_pixels" in clOut:
            self.show(f"top {top} clusters by number of pixels:", clOut.nlargest(top, "n_pixels")[clCols])
        far = clOut[clOut["max_dist_to_sim_mm"] > self.distWarnMm]
        log.info("clusters with a sim hit more than %g mm away: %d", self.distWarnMm, len(far))
        if len(far):
            self.show(f"top {top} by distance:", far.nlargest(top, "max_dist_to_sim_mm")[clCols])
        if "n_signal_sim_hits" in clOut:
            withSignal = clOut[clOut["n_signal_sim_hits"] > 0]
            if len(withSignal):
                self.show("clusters containing a signal sim hit:", withSignal[clCols])

    # ------------------------------------------------------------------------------------------
    # the saved tables must agree with each other: use the row numbers like a user would
    # ------------------------------------------------------------------------------------------
    def verifyRowPointers(self):
        log = self.log
        log.info("\n== 6. row pointers in the saved tables ==")
        cl = self.outputs[self.linkedName(self.CLUSTERS)]
        sim = self.outputs[self.linkedName(self.simKey)]
        lk = self.outputs[self.OUT_CLUSTER_SIM_LINKS]

        def pointsTo(pointerTable, rowCol, idxCol, target):
            rows = pointerTable[rowCol].to_numpy()
            if len(rows) and (rows.min() < 0 or rows.max() >= len(target)):
                return False
            return bool(np.array_equal(target["ievent"].to_numpy()[rows], pointerTable["ievent"].to_numpy())
                        and np.array_equal(target["index"].to_numpy()[rows], pointerTable[idxCol].to_numpy()))
        okC = pointsTo(lk, "cluster_row", "cluster_index", cl)
        okS = pointsTo(lk, "sim_row", "sim_index", sim)
        for label, ok in (("cluster_row", okC), ("sim_row", okS)):
            if not ok:
                self.problem(f"{self.OUT_CLUSTER_SIM_LINKS}.{label} does not point at the object with the same (ievent, index)")
        msg = f"{self.OUT_CLUSTER_SIM_LINKS}: cluster_row and sim_row verified for all {len(lk):,} links"
        pixName = self.linkedName(self.PIXELS)
        for name in (pixName, self.OUT_PIXEL_EDGES):
            if name in self.outputs:
                t = self.outputs[name]
                has = t[t["cluster_row"] >= 0] if name == pixName else t
                if not pointsTo(has, "cluster_row", "cluster_index", cl):
                    self.problem(f"{name}.cluster_row does not point at the cluster with the same (ievent, cluster_index)")
                else:
                    msg += f"; {name}.cluster_row verified for {len(has):,} pixels"
        if not self.problems:
            log.info(msg)

    def linkMap(self):
        """The saved tables and which column of which table points where (names of this run)."""
        cl, sim = self.linkedName(self.CLUSTERS), self.linkedName(self.simKey)
        lk, pix, edges = self.OUT_CLUSTER_SIM_LINKS, self.linkedName(self.PIXELS), self.OUT_PIXEL_EDGES
        what = {cl: f"{self.CLUSTERS} + n_sim_hits, n_signal_sim_hits, n_overlay_sim_hits, max_dist_to_sim_mm, n_pixels",
                sim: f"{self.simKey} + n_clusters, linked, max_dist_to_cluster_mm",
                lk: "one row per cluster <-> sim hit link (a cluster can have many sim hits, a sim hit can feed many clusters)",
                pix: f"{self.PIXELS} + cluster_row, cluster_index (every pixel belongs to exactly one cluster; -1 = none)",
                edges: f"one row per pixel and the cluster it belongs to (written because {self.PIXELS} was not loaded)"}
        tables = {n: {"rows": int(len(self.outputs[n])), "what": w} for n, w in what.items() if n in self.outputs}
        pointers = []

        def ptr(table, column, to, how):
            if table in self.outputs:
                pointers.append({"table": table, "column": column, "points_to": to, "how": how})
        ptr(lk, "cluster_row", cl, "row number")
        ptr(lk, "sim_row", sim, "row number")
        ptr(lk, "(ievent, cluster_index)", cl, "key (ievent, index)")
        ptr(lk, "(ievent, sim_index)", sim, "key (ievent, index)")
        ptr(pix, "cluster_row", cl, "row number (-1 = no cluster)")
        ptr(pix, "(ievent, cluster_index)", cl, "key (ievent, index)")
        ptr(edges, "cluster_row", cl, "row number")
        ptr(edges, "(ievent, cluster_index)", cl, "key (ievent, index)")
        ptr(edges, "(ievent, pixel_index)", f"{self.PIXELS} (not loaded)", "key (ievent, index)")
        sims = self.outputs[sim]
        if "particle_collectionID" in sims:
            ids = set(int(i) for i in sims["particle_collectionID"].unique() if i >= 0)
            for key, df in self.dfs.items():
                if key in self.SIMHIT_TABLES or "collectionID" not in df:
                    continue
                if ids and set(int(i) for i in df["collectionID"].unique()) == ids:
                    ptr(sim, "(ievent, particle_index)", f"{key} (input table, not re-saved)",
                        "key (ievent, index); particle_index = -1 means no MC particle (all BIB hits)")
        roles = {"clusters": cl, "simHits": sim, "clusterSimHitLinks": lk,
                 "pixels": pix if pix in self.outputs else None, "pixelEdges": edges if edges in self.outputs else None}
        return {"fileStub": self.fileStub,
                "rowNumber": "0-based position in the saved table (use .iloc); valid while the tables are not re-sorted or filtered",
                "key": "(ievent, index): 'index' restarts at 0 in every event, it is NOT a row number",
                "roles": roles, "tables": tables, "pointers": pointers}

    def describeLinks(self):
        m = self.linkMap()
        log = self.log
        log.info("\n== 7. saved tables and how they link ==")
        log.info("row number = %s", m["rowNumber"])
        log.info("key        = %s", m["key"])
        log.info("tables:")
        w = max(len(n) for n in m["tables"])
        for name, t in m["tables"].items():
            log.info("  %-*s %11s rows  %s", w, name, f"{t['rows']:,}", t["what"])
        log.info("pointers (column of table A -> table B):")
        wa = max(len(f"{p['table']}.{p['column']}") for p in m["pointers"])
        for p in m["pointers"]:
            log.info("  %-*s -> %s  [%s]", wa, f"{p['table']}.{p['column']}", p["points_to"], p["how"])

    def writeManifest(self):
        path = f"{self.fileStub}.{self.MANIFEST_SUFFIX}"
        with open(path, "w") as f:
            json.dump(self.linkMap(), f, indent=2)
        self.log.info("saved table map: %s", os.path.basename(path))

    # ------------------------------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------------------------------
    def runAll(self):
        """Load, check, link, report, save. Returns 0 if every check passed, 1 otherwise."""
        self.problems, self.outputs, self.simKey = [], {}, None
        self._setupLogging()
        try:
            self._runAll()
        except Exception:
            self.log.exception("linkEdmHits crashed")
            raise
        finally:
            self._closeLogging()
        return 1 if self.problems else 0

    def _runAll(self):
        log = self.log
        log.info("linkEdmHits: %s", self.fileStub)
        self.loadAll()
        if not self.dfs:
            self.problem("no tables loaded")
            return
        self.checkTables()

        if self.CLUSTERS in self.dfs and self.CLUSTER_SIM_LINKS in self.dfs:
            self.simKey = self.pickSimTable()
            if self.simKey is not None:
                simKey = self.simKey
                self.checkSimHitParticles(simKey)
                res = self.linkClusterSim(simKey)
                if res is not None:
                    clOut, simOut, linkOut = res
                    self.outputs[self.OUT_CLUSTER_SIM_LINKS] = linkOut
                    if self.PIXEL_CLUSTER_LINKS in self.dfs:
                        name, table = self.linkPixels(clOut)
                        self.outputs[name] = table
                    else:
                        log.info("\n%s not loaded: no pixel links", self.PIXEL_CLUSTER_LINKS)
                    self.outputs[self.linkedName(self.CLUSTERS)] = clOut
                    self.outputs[self.linkedName(simKey)] = simOut
                    self.reportUnlinked(simKey, simOut)
                    self.reportOdd(simKey, clOut, simOut)
                    self.verifyRowPointers()
                    self.describeLinks()
        else:
            log.info("\nno cluster or no cluster->sim hit link table (non-overlay file?): nothing to link")

        if self.save and self.outputs:
            log.info("\n== saving ==")
            for name, df in self.outputs.items():
                self.saveTable(df, name)
            self.writeManifest()

        log.info("\n== summary ==")
        log.info("%d check(s) failed", len(self.problems))
        for p in self.problems:
            log.info("  - %s", p)


# ----------------------------------------------------------------------------------------------
def parseArgs(argv=None):
    d = EdmHitLinker.DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fileStub", nargs="?", default=EdmHitLinker.DEFAULT_FILE_STUB,
                    help="path of the saved tables without '.<key>.parquet/.csv' (default: %(default)s)")
    ap.add_argument("--no-save", action="store_true", help="only check and log, write nothing")
    ap.add_argument("--no-csv", action="store_true", help="do not write csv files")
    ap.add_argument("--no-parquet", action="store_true", help="do not write parquet files")
    ap.add_argument("--csv-max-rows", type=int, default=d["csvMaxRows"], help="larger tables get no csv (default: %(default)s)")
    ap.add_argument("--prefer-csv", action="store_true", help="read csv before parquet when both exist")
    ap.add_argument("--no-convert-parquet", action="store_true", help="do not write parquet for tables read from csv")
    ap.add_argument("--dist-warn-mm", type=float, default=d["distWarnMm"], help="(default: %(default)s)")
    ap.add_argument("--max-clusters-per-sim", type=int, default=d["maxClustersPerSimWarn"], help="(default: %(default)s)")
    ap.add_argument("--max-sim-per-cluster", type=int, default=d["maxSimPerClusterWarn"], help="(default: %(default)s)")
    ap.add_argument("--top-n", type=int, default=d["topN"], help="rows in the odd-object lists (default: %(default)s)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parseArgs(argv)                       # defaults collected from the class constants
    linker = EdmHitLinker(a.fileStub, save=not a.no_save,
                          convertParquet=not a.no_convert_parquet, preferParquet=not a.prefer_csv,
                          distWarnMm=a.dist_warn_mm, maxClustersPerSimWarn=a.max_clusters_per_sim,
                          maxSimPerClusterWarn=a.max_sim_per_cluster, topN=a.top_n,
                          saveCsv=not a.no_csv, csvMaxRows=a.csv_max_rows, saveParquet=not a.no_parquet)
    return linker.runAll()


if __name__ == "__main__":
    sys.exit(main())