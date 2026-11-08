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
  (`likelihood: fast_richness: True`, default);
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

See `notebooks/hats_io_validation.ipynb` for the validation against the FITS
path and timings, and `tests/test_des_scan.py` / `tests/test_hats_catalog.py`.

#### Abbreviations
* CMD: color-magnitude diagram
* ROI: region of interest
* PDF: probability distribution function
* LKHD: likelihood
* IMF: initial mass function
* LUT: look-up table
