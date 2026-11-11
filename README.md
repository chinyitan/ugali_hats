[![Build](https://github.com/DarkEnergySurvey/ugali/actions/workflows/ci.yml/badge.svg)](https://github.com/DarkEnergySurvey/ugali/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ugali.svg)](https://pypi.python.org/pypi/ugali)
[![Release](https://img.shields.io/github/release/DarkEnergySurvey/ugali.svg)](../../releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../)
[![Github Downloads](https://img.shields.io/github/downloads/DarkEnergySurvey/ugali/total.svg)](../../releases)

Overview
--------

The ultra-faint galaxy likelihood (`ugali`) toolkit provides a set of python modules developed for maximum-likelihood-based studies of resolved satellite galaxies. The primary inputs are stellar object catalogs derived from optical photometric surveys and the coverage masks of those surveys. In addition, `ugali` ships with a set of synthetic isochrone libraries and catalogs of known resolved stellar systems.

Authors: [Keith Bechtol](https://github.com/bechtol) & [Alex Drlica-Wagner](https://github.com/kadrlica)

Installation
------------

There are several ways to install `ugali`.

The most robust way is to follow the installation procedure for the automated GitHub Actions build documented in [ci.yml](.github/workflows/ci.yml). This installation creates a `conda` environment with the necessary dependencies and installs `ugali`.
```bash
# Create and activate conda environment
conda create -n ugali-env numpy scipy matplotlib astropy healpy pyyaml emcee nose fitsio corner -c conda-forge -c kadrlica
source activate ugali-env

# Clone source code from the parent repository
git clone https://github.com/DarkEnergySurvey/ugali.git && cd ugali

# Install just the python source code
python setup.py install 

# Install source code with a minimal set of isochrone and catalog libraries
python setup.py install --isochrones --catalogs
```

In theory, the easiest way to get a stable release of `ugali` is through [PyPi](https://pypi.python.org/pypi) using `pip`:
```bash
# Install just the source code
pip install ugali

# Install source code with a minimal set of isochrone and catalog libraries
pip install ugali --install-option "--isochrones" --install-option "--catalogs"
```

By default, the minimal isochrone and catalog libraries are installed into the directory specified by the `$UGALIDIR` environment variable (default: `$HOME/.ugali`). The download and unpacking of the isochrone and catalog files might make it appear that your `pip` installation has stalled. Unfortunately, `pip` [may not display a progress bar](https://github.com/pypa/pip/issues/2732#issuecomment-97119093) during this delay.

Auxiliary Libraries
-------------------

The `ugali` source code is distributed with several auxiliary libraries for isochrone generation and catalog matching. These libraries can be downloaded directly from the [releases](../../releases) page, and unpacked in your `$UGALIDIR`.

The following isochrone libraries are available. All cover ages of 1.0 - 13.5 Gyr in 0.1 Gyr steps. The metallicity grid depends on the model: the PARSEC libraries (`bressan2012`, `marigo2017`) cover Z = 1e-4 - 1e-3 in 1e-5 steps (11466 isochrones), while the MIST libraries (`dotter2016`) extend an order of magnitude lower, Z = 1e-5 - 1e-3 in the same steps (12600 isochrones):

| survey   | filter system                                  | distributed models                                        |
| -------- | ---------------------------------------------- | --------------------------------------------------------- |
| `des`    | DECam *ugrizY*                                 | `bressan2012`, `marigo2017`, `dotter2008`, `dotter2016`   |
| `ps1`    | Pan-STARRS1 *grizyw*                           | `bressan2012`, `marigo2017`, `dotter2008`, `dotter2016`   |
| `sdss`   | SDSS *ugriz*                                   | `bressan2012`, `marigo2017`, `dotter2008`, `dotter2016`   |
| `lsst`   | LSST *ugrizy* (R1.9 throughputs, Sept 2023)    | `bressan2012`, `marigo2017`, `dotter2016`                 |
| `roman`  | Roman 2021 filters (*F062* - *F213*)           | `bressan2012`, `marigo2017`, `dotter2016`                 |
| `euclid` | Euclid VIS+NISP (*VIS*, *Y*, *J*, *H*)         | `bressan2012`, `marigo2017`, `dotter2016`                 |

A minimal subset of all of these libraries (two ages at two metallicities) is what gets installed by the `--isochrones` option. 

Additional LSST filter systems can be downloaded with `ugali/scratch/download_isochrones.py`, but are not distributed: `lsst_dp0` (Oct 2017 throughputs, used for the DP0/DC2 simulations) and `lsst_2012` (the original March 2012 throughputs). 

For example, to install the [Bressan et al. 2012](http://adsabs.harvard.edu/abs/2012MNRAS.427..127B) isochrones for the DES survey:

```
cd $UGALIDIR
wget https://github.com/DarkEnergySurvey/ugali/releases/download/v1.9.0/ugali-des-bressan2012.tar.gz
tar -xzf ugali-des-bressan2012.tar.gz
```

The `UGALIDIR` environment variable is used to point to the isochrone and catalog libraries. If you install the isochrones in a non-standard location be sure to set `UGALIDIR` so `ugali` can find them:

```
export UGALIDIR=<PATH>
```

An experimental interface for downloading the isochrone and catalog libraries also exists through `setup.py`:
```
# To install the Bressan+ 2012 isochrones for the DES survey
python setup.py isochrones --survey des --model bressan2012

# If you have isochrones already installed, you may need to force
python setup.py isochrones --survey des --model bressan2012 --force

# To install all available DES isochrones
python setup.py isochrones --survey des

# To install all available Bressan+ 2012 isochrones
python setup.py isochrones --model bressan2012

# To install the LSST isochrones
python setup.py isochrones --survey lsst

# To install the catalog libraries
python setup.py catalogs
```

Usage Examples
--------------

Several examples can be found as jupyter notebooks in the [notebooks](notebooks/) directory.

Additional Information
----------------------

#### Dependencies
These should mostly be taken care of by PyPi with a `pip install`.
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](http://matplotlib.org/)
* [astropy](http://www.astropy.org/)
* [healpy](https://github.com/healpy/healpy)
* [emcee](http://dan.iel.fm/emcee/current/)
* [pyyaml](http://pyyaml.org/)
* [fitsio](https://github.com/esheldon/fitsio)

#### Isochrones
The isochrones used by `ugali` come dominantly from:
* PARSEC isochrones from the Padova group (http://stev.oapd.inaf.it/cgi-bin/cmd)
* Dartmouth isochrones (http://stellar.dartmouth.edu/models/isolf_new.html)
* MESA isochrones (http://waps.cfa.harvard.edu/MIST/interp_isos.html)

More information can be found in the [isochrone](ugali/isochrone) module.

#### HATS catalog input and parallel scans
Catalogs in the [HATS](https://hats.readthedocs.io) format (healpix-partitioned
parquet) can be used in place of the per-pixel FITS catalog files by replacing
`dirname`/`basename` in the `catalog` section of the config with:
```yaml
catalog:
  hats:
    path: /path/to/hats_catalog   # directory containing partition_info.csv
    max_tiles: 64                 # partitions kept in the per-process LRU cache
  # ...the usual lon_field, mag_1_field, selection, ... keys
```
Partitions are read with `pyarrow` through a least-recently-used cache
(`ugali.observation.hats_catalog`), so neighbouring ROIs share IO; the
selection cut is applied per partition before caching. The rows loaded for a
ROI are the same as those the FITS path reads, so results are identical to
float32 precision; because the rows arrive in a different order, the
per-object likelihood sum can differ in its last double-precision bits, which
shows up as a 1-ulp change of LOG_LIKELIHOOD in a few percent of pixels where
TS < 1e-4 (5 of 704 pixels in the DES Y6 reference region). The mask files
are still read from `mask: dirname`.

Depth masks can likewise be read from [HealSparse](https://healsparse.readthedocs.io)
maps instead of the per-pixel mask files, by replacing `dirname`/`basename_*`
in the `mask` section with:
```yaml
mask:
  healsparse:
    maglim_1: [/path/g_depth_a.hsp, /path/g_depth_b.hsp]   # band-1 depth maps, in priority order
    maglim_2: [/path/r_depth_a.hsp, /path/r_depth_b.hsp]
    # fracdet_1 / fracdet_2: optional detection-fraction maps; by default the
    # fraction of valid map sub-pixels is used
    reduction: mean       # how finer map pixels are combined into nside_pixel
  minimum_solid_angle: 0.1
```
Only the coverage pixels an ROI touches are read (through a per-process
cache), maps finer than `nside_pixel` are degraded over their valid
sub-pixels, pixel existence and the survey footprint come from the maps'
coverage and valid pixels (`ugali.observation.healsparse_mask`). A scan with
maps built from the DES Y6 mask files is bit-identical to the file-based one
(`tests/test_healsparse_mask.py`).

`ugali/analysis/scan_parallel.py` runs the likelihood scan for many pixels on
one machine with a pool of long-lived workers (pixels are sorted by NESTED
index so that neighbours share a worker's cache). It is built for long,
resumable runs: outputs are written atomically and existing ones are skipped
unless `--force`, per-pixel timing/cache statistics are appended to
`--stats-file` as pixels finish, a worker that dies (e.g. OOM) triggers a pool
restart with the unfinished work resubmitted, and the default `-n -1` is
limited by available memory (~4 GB per worker):
```
python ugali/analysis/scan_parallel.py config.yaml --pixels 639648 640671 -n 4 --stats-file stats.csv
python ugali/analysis/scan_parallel.py config.yaml --hpxlist pixels.txt -n -1
```
The driver pins every worker to one BLAS/OpenMP thread (a multi-threaded
BLAS in each of 48 workers was measured 3x slower); set `OMP_NUM_THREADS`
yourself to override. The stats file records the setup (catalog/mask load)
and search time of every pixel separately.

Two stalls seen on a shared 256-core node, with their fixes:
* *Memory compaction.* numpy requests transparent huge pages for every
  array over 4 MB; with a large fragmented page cache each request can stall
  in direct compaction. In dense (bulge-direction) fields 96 workers ran at
  96% system CPU and 0.5 pixels/s; with the request disabled they ran at
  15-30 pixels/s. The driver now sets `NUMPY_MADVISE_HUGEPAGE=0` before
  importing numpy (an explicit value in the environment wins). Diagnose with
  `vmstat` (system vs user time) and `/proc/vmstat` `compact_stall`.
* *Slow home directory.* ugali calls `mkdir` on its data directory and reads
  the isochrone files for every pixel. If `~/.ugali` is on a slow or
  overloaded NFS server every worker sits in `mkdir` (state D). Point
  `UGALIDIR` at a copy of the isochrones on fast storage.
To stop a run, kill the driver's process tree; a `pkill -f` whose pattern
also appears in the calling shell's command line kills the caller.

Several steps of the grid search are reorganized so that they do strictly
less work with the same arithmetic, and the outputs are bit-for-bit those of
the original loops (verified on the 704-pixel DES Y6 reference region):
* the spatial signal term does not depend on the distance modulus, so
  `LogLikelihood` memoizes it per kernel position/extension
  (`likelihood: cache_spatial: True`, default), the kernel radii of a centroid
  are shared by its extensions, and the Plummer and King kernels memoize their
  normalization integral per extension;
* the richness maximization evaluates the likelihood directly from the synced
  spatial/color terms instead of re-syncing them at every richness
  (`likelihood: fast_richness: True`, default), and all target pixels of an
  extension are fit in one vectorized call (`scan: batch_pixels: True`,
  default; a C-contiguous row of a 2D array sums exactly like a 1D array);
* the observable fraction is evaluated only for the unique mask values that
  occur in the interior region, with a masked sum;
* the isochrone pdf finds the contributing (object, isochrone bin) pairs
  (a few percent of all pairs) from the magnitude windows instead of building
  dense distance arrays.
Mask files are read once per process into an LRU cache shared by
neighbouring ROIs, and the isochrone directory grid is memoized.
Together these reduce the per-pixel scan time from ~41 s to ~3-4 s (DES Y6).
Set `cache_spatial: False` / `fast_richness: False` to fall back to the
original code paths for the first two; `tests/test_optimizations.py` checks
each step against the original formula, and
`notebooks/scan_optimizations.ipynb` walks through every change with the
exactness argument and the full-region verification. The regression
references in `tests/data/desy6` are made with numpy 2 (numpy 1 differs from
numpy 2 at the richness-fit tolerance, ~1e-5 in log-likelihood).

One further option is not bit-exact: `likelihood: color_pdf_sparse: True`
accumulates the (object, isochrone bin) contributions of the color pdf
directly in float64 with `np.bincount` instead of summing dense float32
arrays. The result differs from the dense sum only at the float32 rounding
level (|dTS| <= 3e-5 on the DELVE test pixels, richness differences <= 1e-4
relative), but it removes the memory-bound step that dominates large ROIs:
on DELVE DR3 (nside_pixel 1024, 21 moduli x 3 kernels) the per-pixel time
fell from 42 s to 4.6 s, and the scan then scales to 96 workers on a
2x64-core node where the dense path saturates the memory bandwidth at 48.
With it the full DELVE DR3 footprint (312,736 nside-256 pixels) scans in
about 3 hours.

#### Further work

Ideas that came out of the DES Y6 / DELVE DR3 runs, roughly in order of
expected payoff:

* *Share per-object terms between neighbouring ROIs.* The color pdf of an
  object depends only on the object and the distance modulus, not on the
  target pixel, and the observable fraction and background density depend
  only on the mask pixel. With a 2 deg ROI and 0.23 deg target pixels every
  object is recomputed ~50 times; a per-worker memo keyed on catalog
  partition and distance modulus (pixels are already scanned in NESTED
  order) would remove most of the remaining per-pixel cost.
* *Validate on a known stream.* Before trusting a stream search, check that
  the scan recovers a stream the survey clearly covers (e.g. Jhelum, Phoenix
  or ATLAS in DELVE) at the right modulus and kernel size.
* *A stream-aware peak finder.* `ugali.analysis.search` looks for compact
  peaks; streams appear as linear ridges in the TS map of one distance
  modulus, so a Hough or matched-filter pass over the merged per-modulus
  maps is needed to find them.
* *Elongated kernels.* The round Plummer kernels used so far are a poor match
  to 0.1-0.3 deg half-width streams; an `EllipticalPlummer` with ellipticity
  ~0.9 scanned over a few position angles would be more sensitive at a few
  times the cost per kernel.
* *Flag ROIs with a biased background.* In surveys with tile-level
  incompleteness (DELVE single-epoch g/r tiles detect <50% of the bright
  objects, which no per-band depth map captures) the annulus background can
  be badly over-estimated. Besides masking such tiles through an explicit
  `fracdet` map, the ratio of the interior to annulus bright-object density
  per target pixel is a cheap post-hoc flag.
* *Collation.* `ugali/analysis/collate.py` reads the per-pixel FITS
  outputs with a process pool into one `.npz` of (cell, modulus) arrays
  (313k files in about a minute on 48 workers); the stock serial merge step
  does not scale to a full survey. The search step still needs to be
  pointed at the collated arrays.

#### Cleaning the collated maps

`ugali/analysis/postproc.py` applies the post-hoc remedies that were
validated on the DELVE DR3 scans (hotspot study of the Galactic-north
rescan, 164k targets, checked against the galstreams tracks and the LVDB
satellites in the footprint):

* a scan-level fix first: `fractype: binary` plus a `fracdet` map zeroed
  where the scan-selected star counts fall below half the local level (and
  inside holes around bright globular clusters) removed the ~2 deg ROI
  hotspots: cells above TS 25 fell from 7.8% to 4.5% of the region, above
  TS 10 from 20% to 14%;
* `roi_density_ratio` + a threshold of 1.10 on the bright selected-star
  counts masks the remaining interior/annulus mismatches, and
  `median_subtract` removes the modulus-independent excess (together: 2.8%
  of the cells masked, TS > 25 down to 3.0%);
* the "all-modulus" hotspots (TS > 10 at every modulus, 0.7% of the cells,
  TS in the thousands) are dusty bulge-direction fields: the SFD
  dereddening over-corrects there and shifts the stellar locus 0.01-0.02
  mag blue relative to the annulus, which the likelihood reads as signal at
  every distance. Their rate rises from 0.2% of cells at E(B-V) < 0.05 to
  13% at E(B-V) > 0.5. `ebv_mask` at E(B-V) > 0.2 costs 6% of the cells and
  removes 60% of them; a faint-end (22.5 < g < 23.5) density-ratio mask at
  1.25 catches most of the rest. Things that did *not* fix them: capping
  the magnitude range (g < 23 or 22.5 cures 5 of 40 and worsens others), a
  per-cell stellar-locus colour correction (the locus scatters by 0.04 mag
  between neighbouring cells inside the dusty fields), and a sector-median
  background estimate (it inflates clean fields);
* `protect_objects` keeps the cells around catalogued dwarfs and clusters,
  which raise their own interior counts.
* the density ratios must be weighted by the detection fraction the scan
  used (`--fracdet`): tiles the scan masked have anomalous counts and
  otherwise bias the annulus. Weighted, the whole recipe masks 8.5% of the
  north region and removes 74% of the all-modulus hotspot cells; unweighted
  it masks 11.8% for 77%.

The remaining small round spots at large modulus are Poisson noise of the
few faint objects per kernel and need a shape criterion rather than a
mask; a directional (line) filter is being evaluated.

See `notebooks/hats_io_validation.ipynb` for the validation against the FITS
path and timings, and `tests/test_des_scan.py` / `tests/test_hats_catalog.py`.

#### Abbreviations
* CMD: color-magnitude diagram
* ROI: region of interest
* PDF: probability distribution function
* LKHD: likelihood
* IMF: initial mass function
* LUT: look-up table
