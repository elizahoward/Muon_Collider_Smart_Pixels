# First look at MAIA vertex digitization

These plots use the two-event `digi_output_mu2` sample with BIB overlay. They do
not require its ROOT file. The committed Parquet tables contain the clusters,
overlay sim hits, cluster-to-sim-hit relations, and pixel-to-cluster relations.
The individual BIB pixel table is absent. A separate, small `digi_real` sample
provides actual fired-pixel positions and charges for the example pixel maps.

## Reproduce locally

Run from the repository root with Python packages `numpy`, `pandas`, `pyarrow`,
and `matplotlib`. The source tables are read-only; derived files go under
`local_only`, which is ignored by the repository.

```bash
mkdir -p local_only/maia_linked
MAIA_SRC="$PWD/MuC_Smartpix_Data_Production/MAIADataset/edmTestFiles/digi_output_mu2.edm4hep.root"
MAIA_DST="$PWD/local_only/maia_linked/digi_output_mu2.edm4hep.root"

for suffix in vtxClustersDf.parquet vtxClusterToSimHitDf.parquet \
              vtxBarrelSimHitsDf2.parquet truthParticlesDf2.parquet; do
  ln -sfn "${MAIA_SRC}.${suffix}" "${MAIA_DST}.${suffix}"
done

python3 MuC_Smartpix_Data_Production/MAIADataset/linkEdmHits.py \
  "$MAIA_DST" --no-csv --no-convert-parquet \
  > local_only/maia_linked/link.log 2>&1
tail -n 12 local_only/maia_linked/link.log

python3 MuC_Smartpix_Data_Production/MAIADataset/plot_maia_digitization.py
```

The linker should end with `0 check(s) failed`. It intentionally does not load
the 6.9 million pixel-to-cluster links at this stage. The plotting script reads
those links in Parquet batches to count pixels per cluster, without retaining
all link rows in memory. It saves PNGs, `pixel_counts.npy`, and `summary.json`
in `local_only/maia_visualizations`. Use `--linked-stub`, `--pixel-stub`,
`--pixel-links`, and `--output-dir` to point it at other locations.

## What the plots show

![BIB cluster occupancy in the vertex barrel with signal-linked clusters marked](figures/maia_occupancy.png)

The BIB-only cluster density traces the barrel layers. The eight cyan markers
are clusters linked to a non-overlay sim hit. Positions are in millimetres.

![BIB cluster energy and pixel-link count, with the eight signal-linked clusters marked](figures/maia_cluster_features.png)

The blue BIB curves restrict the reference to the sensors containing a signal
cluster. Red marks show the eight signal clusters individually; they are too few
for a meaningful signal distribution. Their pixel-link counts are
`[5, 5, 2, 3, 4, 3, 3, 3]`.

![Cluster to sim-hit link distance and sensor-ID mismatch](figures/maia_link_quality.png)

All 865,866 cluster-to-sim-hit links resolve by object key, but 10,466 span more
than 10 mm and 10,097 join different cell IDs. Structural link success does not
establish that each distant association is physically sensible.

![Four pixel charge maps from the separate signal-only digitized sample](figures/maia_signal_only_pixel_examples.png)

These pixel maps come from `digi_real.edm4hep.root`, **without BIB overlay**.
Their axes are sensor pixel column and row, not global detector coordinates.
Pixel colour shows its stored `eDep`; the digitizer's pixel storage convention
should be checked before assigning a physical unit to that value.

## Limits and next experiment

- The BIB-overlaid sample contains two events, 807,826 clusters, and only eight
  signal-linked clusters. It supports examples and validation, not performance
  claims about signal–BIB separation.
- The pixel-to-cluster relation allows pixel counts, but it does not contain the
  BIB pixels' coordinates or charges. Pixel-level truth is also ambiguous when
  a cluster links to multiple sim hits.
- Cluster `time` is almost always 25 in this sample, so no timing comparison is
  plotted.
- To compare signal and BIB *pixel shapes in the same overlaid event*, produce
  a new digitized sample from the existing valid `sim_output_mu2` file, first
  with a small BIB overlay count. The MAIA guide's step 3 describes the custom
  digitizer and BIB configuration. Run substantial cluster processing as a
  scheduled job rather than on the OSG Access Point.
