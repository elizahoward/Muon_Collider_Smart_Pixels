# MAIA Dataset Generation: Signal + BIB with Realistic Vertex Digitization

How to generate a dataset with the MAIA detector (Muon Collider) that contains simulated hits, reconstructed hits (clusters), and individual fired pixels, with beam-induced background (BIB) overlaid.

Based on the MAIA/Fermilab 2026 tutorial plus a modified version of `MuonCVXDDigitiser` (`brr_devel` branch) that can store individual pixels.

- Tutorial setup: https://mcd-wiki.web.cern.ch/software/tutorials/fermilab2026/setup/
- Tutorial digitization/reco: https://mcd-wiki.web.cern.ch/software/tutorials/fermilab2026/digireco/
- Digitizer repo: https://github.com/rroberts194/MuonCVXDDigitiser (branch `brr_devel`)

## Overview

| Step | What | Output |
|------|------|--------|
| 1 | Generation (particle gun or physics process) | `gen_output.edm4hep.root` |
| 1.5 | Detector simulation with `ddsim` | `sim_output.edm4hep.root` |
| 2 | Build `MuonCVXDDigitiser`, test standalone on signal only | `digi.edm4hep.root`, `digi_real.edm4hep.root` |
| 3 | Full digitization with BIB overlay | `digi_output.edm4hep.root` |


### Choosing what to generate

- **Signal:** use a muon gun. This sample is also what gets overlaid with BIB.
- **BIB only:** use a neutrino gun (no signal particle, so only background shows up).
- **Realistic physics:** use a physics process from the MAIA tutorial or pythia, madgraph instead of a muon gun (to be tested by us, but it's in the Mucoll tutorial).

## 0. Environment

Follow the instructions from https://mcd-wiki.web.cern.ch/software/tutorials/fermilab2026/setup/

The instructions in the wiki are copied here, but they are exactly the same, so can consult wiki for up to date version and more details:

Get to your working folder on OSG /scratch/$USER

Start the container (bind-mount the data area):

```bash
apptainer run -B /ospool/uc-shared/project/futurecolliders/data/ \
    /cvmfs/unpacked.cern.ch/ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.1-amd64
```

Then, **once per container session**:

```bash
source /opt/setup_mucoll.sh
```

> **Do not source `setup_mucoll.sh` twice in the same session.** It appends to `CMAKE_PREFIX_PATH` each time; the second time the environment becomes so long that commands fail with `Argument list too long` (seen with `cmake`, `realpath`, and `k4run`). If this happens, `exit` and start a fresh container.

Cloning the mucoll-benchmarks repository:
```bash
git clone --recurse-submodules --branch tutorial_20260817 \
    https://github.com/MuonColliderSoft/mucoll-benchmarks.git
cd mucoll-benchmarks
```
Then selecting geometry
```bash
source setup_config.sh . MAIA_v0
```

Every time you re-enter the container you must redo the `source /opt/setup_mucoll.sh` above (**once**), the `cd mucoll-benchmarks;source setup_config.sh . MAIA_v0` call, and the digitizer environment exports (step 2).

## 1. Generation and simulation
Follow the instructions from https://mcd-wiki.web.cern.ch/software/tutorials/fermilab2026/sim/

The instructions in the wiki are copied here, but they are exactly the same, so can consult wiki for up to date version and more details:

Generate muons with a muon gun:

```bash
python generation/pgun/pgun_edm4hep.py \
    -p 1 -e 10 --pdg 13 --pt 0 50 --theta 10 170 \
    -- gen_output.edm4hep.root
```

The number of events is set to `-e 10`, but can also generate more events.


For more realistic signal, consult wiki for alternate signal generation.

### 1.5 Simulation  
Simulate with GEANT4 (https://mcd-wiki.web.cern.ch/software/tutorials/fermilab2026/sim/#detector-simulation). This is needed for signal; the BIB is already simulated.

```bash
ddsim --steeringFile simulation/steer_baseline.py \
      --inputFiles gen_output.edm4hep.root \
      --outputFile sim_output.edm4hep.root \
      --numberOfEvents 10
```

The number of events is set to `--numberOfEvents 10`, but can also simulate more events.

Expect ~40 s of startup plus ~1 s/event. Stop here (end of the https://mcd-wiki.web.cern.ch/software/tutorials/fermilab2026/sim/ page) and do not continue with the tutorial's Digitization and Reconstruction page yet.

The output of this file is what was later passed into our tracklist generation and pixelav (although this is the edm4hep updated format).

## 2. Build the modified digitizer
Following the README of https://github.com/rroberts194/MuonCVXDDigitiser which is forked from https://github.com/MuonColliderSoft/MuonCVXDDigitiser
Clone MuonCVXDDigitiser inside `mucoll-benchmarks` and **switch to `brr_devel` before configuring**:

```bash
#if not already, go inside mucoll-benchmarks 
#cd path/to/mucoll-benchmarks
git clone https://github.com/rroberts194/MuonCVXDDigitiser.git
cd MuonCVXDDigitiser
git switch brr_devel
```

> The default `master` branch fails at configure time (`Could not find a package configuration file provided by "ILCUTIL"`). `brr_devel` which is a based on the gaudification branch of https://github.com/MuonColliderSoft/MuonCVXDDigitiser/tree/gaudification configures fine. However, it allegedly has a bug currently. 

Build and install (see the repo README Building section for details):

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install
cmake --build build -j4
cmake --install build
```

Make the plugins visible. **Run these from inside the `mucoll-benchmarks/MuonCVXDDigitiser` directory** (they use `$PWD`), and repeat in every new session:

```bash
export LD_LIBRARY_PATH=$PWD/install/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/install/python:$PYTHONPATH
export GAUDI_PLUGIN_PATH=$PWD/install/lib:$GAUDI_PLUGIN_PATH
```

> If you export from the wrong directory, `k4run` later fails with
> `ImportError: cannot import name 'MuonCVXDRealDigitiser' from 'Gaudi.Configurables'`.

### Test
(The Testing section of the repo README)

```bash
cd build
ctest -j4 --output-on-failure
cd ..
```

One test is expected to fail for now: `check_MuonCVXDDigitiser` (reports `pixels stored although not requested`). The other 16 should pass (~3 min total).

### Standalone run on signal only

(The Running section of the repo README)

This runs only the digitization on the sim hits. It works for signal-only testing and for checking the build, but it **cannot include BIB**. The two commands differ by the digitizer algorithm: the first does the whole tracker, the second is just the vertex barrel with realistic digitization.

Run from `mucoll-benchmarks/` (the README paths need an extra `MuonCVXDDigitiser/` prefix when run from there), and point `--IOSvc.Input` at your simulation output from step 1.5:

```bash
k4run MuonCVXDDigitiser/MuonCVXDDigitiser/options/runMuonCVXDDigitiser.py \
      --IOSvc.Input sim_output.edm4hep.root --IOSvc.Output digi.edm4hep.root

k4run MuonCVXDDigitiser/MuonCVXDDigitiser/options/runMuonCVXDRealDigitiser.py \
      --IOSvc.Input sim_output.edm4hep.root --IOSvc.Output digi_real.edm4hep.root
```

Inspect with `podio-dump digi_real.edm4hep.root`. Collections of interest:

| Collection | Meaning |
|------------|---------|
| `VertexBarrelCollection` | Sim hits (what was previously passed to PixelAV) |
| `VXDBarrelRealHits` | Reco hits = clusters. The least realistic option would be a Gaussian smearing of the sim hit; the realistic digitizer clusters pixels and builds reco hits from the clusters |
| `VXDBarrelRealPixels` | Individual fired pixels (one entry per pixel) |
| `VXDBarrelRealHitsRelations` | Links reco hit ↔ sim hit |
| `VXDBarrelRealRawHitRelations` | Links pixels ↔ clusters |

The original repository default throws away the individual pixels; this branch default keeps them (`StoreFiredPixels True`).

## 3. Digitization with BIB overlay

### 3a. Make sure the config uses the realistic digitizer

`digiAlgList.py` points to `TrackerDigi/tracking_vertex.py` which we need to update to use the RealDigitiser, i.e. `configs/MAIAConfig/MAIAConfig/TrackerDigi/tracking_vertex.py`. In that file:

- import the realistic digitization by editing ln 2 `from Configurables import DDPlanarDigi, MuonCVXDRealDigitiser, MuonCVXDDigitiser`
- change `MuonCVXDDigitiser` to `MuonCVXDRealDigitiser` in the def _vertex_digitiser(...) fucntion
- pass the class the option to store fired pixels

Or get the updated version of the file from Ryan (in the future we should put in some git somewhere, or make a fork). The updated file looks like 

```
from GaudiKernel.Constants import INFO, WARNING
from Configurables import DDPlanarDigi, MuonCVXDRealDigitiser, MuonCVXDDigitiser
from Common.overlay_utils import overlay_input

def VXDBarrel_cfg(args):
    """
    Create a new vertex barrel instance with the given parameters.
    """
    inputHitCollections = overlay_input("VertexBarrelCollection", args)
    # return DDPlanarDigi(
    #     "VXDBarrelDigitiser",
    #     CorrectTimesForPropagation = True,
    #     IsStrip = False,
    #     ResolutionT = [0.03],
    #     ResolutionU = [0.005],
    #     ResolutionV = [0.005],
    #     SubDetectorName = "Vertex",
    #     TimeWindowMax = [0.15],
    #     TimeWindowMin = [-0.09],
    #     UseTimeWindow = True,
    #     SimTrackHitCollectionName = inputHitCollections,
    #     SimTrkHitRelCollection = ["VXDBarrelHitsRelations"],
    #     TrackerHitCollectionName = ["VXDBarrelHits"],
    #     ForceHitsOntoSurface = True,
    #     OutputLevel = INFO
    # )

    # return MuonCVXDDigitiser(
    #     "VXDBarrelRealDigitiser",
    #     SubDetectorName="VertexBarrel",
    #     LayerIDs=[0, 1, 2, 4, 6],
    #     CollectionName=inputHitCollections,
    #     EventHeader=["EventHeader"],
    #     OutputCollectionName=["VXDBarrelHits"],
    #     RelationColName=["VXDBarrelHitsRelations"],
    #     SimHitLocCollectionName=["VXDBarrelPixels"],
    #     RawHitsLinkColName=["VXDBarrelRawHitRelations"],
    #     OutputLevel=INFO,
    #     StoreFiredPixels=True,
    # )

    return MuonCVXDRealDigitiser(
        "VXDBarrelRealDigitiser",
        SubDetectorName="VertexBarrel",
        LayerIDs=[0, 1, 2, 4, 6],
        SensorType=1,
        CreateStats=True,
        CollectionName=inputHitCollections,
        EventHeader=["EventHeader"],
        OutputCollectionName=["VXDBarrelRealHits"],
        RelationColName=["VXDBarrelRealHitsRelations"],
        SimHitLocCollectionName=["VXDBarrelRealPixels"],
        RawHitsLinkColName=["VXDBarrelRealRawHitRelations"],
        OutputLevel=INFO,
        StoreFiredPixels=True,
    )


def VXDEndcap_cfg(args):
    """
    Create a new vertex endcap instance with the given parameters.
    """
    inputHitCollections = overlay_input("VertexEndcapCollection", args)
    return DDPlanarDigi(
        "VXDEndcapDigitiser",
        CorrectTimesForPropagation = True,
        IsStrip = False,
        ResolutionT = [0.03],
        ResolutionU = [0.005],
        ResolutionV = [0.005],
        SubDetectorName = "Vertex",
        TimeWindowMax = [0.15],
        TimeWindowMin = [-0.09],
        UseTimeWindow = True,
        SimTrackHitCollectionName = inputHitCollections,
        SimTrkHitRelCollection = ["VXDEndcapHitsRelations"],
        TrackerHitCollectionName = ["VXDEndcapHits"],
        ForceHitsOntoSurface = True,
        OutputLevel = INFO
    )
```

In the checkout used here (tag `tutorial_20260817`) the file already imports `MuonCVXDRealDigitiser` and the algorithm sequence already contains `VXDBarrelRealDigitiser` with `StoreFiredPixels True`, so check the file before editing. There is no `vi` in the container; so edit tracking_vertex.py from outside it (the filesystem is shared) or use `sed`/`python`.

### 3b. Run the overlay
Back in the MAIA tutorial's BIB-overlay section (https://mcd-wiki.web.cern.ch/software/tutorials/fermilab2026/digireco/#beam-induced-background). From `mucoll-benchmarks/`, with the container set up (see step 0), `setup_config.sh . MAIA_v0` sourced, and the digitizer exports in place:

```bash
export BPATH=/ospool/uc-shared/project/futurecolliders/data/fmeloni/DataMuC_MAIA_v0/v9/BIB10TeV

k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_steer.py" \
      --doOverlayFull \
      --OverlayFullNumberBackground 5 \
      --OverlayFullPathToMuPlus  $BPATH/sim_mp \
      --OverlayFullPathToMuMinus $BPATH/sim_mm
```

Alternative BIB location, with 10 background events:

```bash
k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_steer.py" \
      --doOverlayFull \
      --OverlayFullNumberBackground 10 \
      --OverlayFullPathToMuPlus  /ospool/uc-shared/public/futurecolliders/gregorypenn/BIB_v9/bib_mp/ \
      --OverlayFullPathToMuMinus /ospool/uc-shared/public/futurecolliders/gregorypenn/BIB_v9/bib_mm/
```

Full realistic digitization of the vertex, inner and outer trackers with a large number of background events (using the magic number from the wiki):

```bash
k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_steer.py" \
      --doOverlayFull --doRealisticDigiVertex --doRealisticDigiInner --doRealisticDigiOuter \
      --OverlayFullNumberBackground 1666 \
      --OverlayFullPathToMuPlus  /ospool/uc-shared/public/futurecolliders/gregorypenn/BIB_v9/bib_mp/ \
      --OverlayFullPathToMuMinus /ospool/uc-shared/public/futurecolliders/gregorypenn/BIB_v9/bib_mm/
```

The default output is `digi_output.edm4hep.root`. It should contain sim hits (labelled as BIB or signal), reco hits (clusters), and individual pixel hits, with links sim hit ↔ reco hit and pixel ↔ reco hit.

Then we can open that root file, and find out how to turn it into a parquet and then tensorflow record. 

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Argument list too long` | `setup_mucoll.sh` was sourced more than once. Exit and restart the container. |
| `Could not find ... ILCUTIL` | Digitizer is on `master`. `git switch brr_devel`. |
| `cannot import name 'MuonCVXDRealDigitiser'` | Library/Python paths not exported, or exported from the wrong directory. Re-run the three `export`s from inside `MuonCVXDDigitiser/`. |
| `No such file or directory: 'MuonCVXDDigitiser/options/...'` | Wrong working directory. The script lives at `MuonCVXDDigitiser/MuonCVXDDigitiser/options/` relative to `mucoll-benchmarks/`. |
| `Input file 'sim.edm4hep.root' does not exist` | The README example filename; use your own, e.g. `sim_output.edm4hep.root`. |
| `FileNotFoundError: ... digi_steer.py\xa0--doOverlayFull...` or a path starting with `“` | Pasted command contains curly quotes or non-breaking spaces (common when copying from notes/Word/Notion). Retype the quotes and spaces, or paste into a plain-text editor first. Also make sure there is no space after a line-ending `\`. |
| `vi: command not found` | Not installed in the container; edit files from outside it. |
| `source setup_config.sh` says benchmark directory does not exist | You are not in the `mucoll-benchmarks` directory (or the environment is already over-long, see the first row). |

## Known issues / not yet verified

- **BIB overlay run has not completed successfully yet.** The overlay run using `/ospool/uc-shared/public/futurecolliders/gregorypenn/BIB_v9/...` reported
  `directory iterator cannot open directory: No such file or directory` for both paths, then segfaulted on event 0. Most likely the `public/` area is not visible inside the container, because the `apptainer run -B` only binds `.../project/futurecolliders/data/`. Try adding a bind such as `-B /ospool/uc-shared/public/futurecolliders/`, or use the `$BPATH` location under `project/.../data/`. (This is a hypothesis, not confirmed; the segfault may simply follow from the empty background list.)
- The `$BPATH` (`fmeloni`) command has not been run successfully possibly because of some copy/paste problems above.
- The `1666`-background-event command with all `--doRealisticDigi*` flags is untested.
- The contents of `digi_output.edm4hep.root` (`podio-dump`) have not been checked yet. After a successful run, confirm that the pixel and link collections listed above are present and that sim hits are labelled BIB vs. signal.
- `ctest` failure `check_MuonCVXDDigitiser` is known and accepted for now.
