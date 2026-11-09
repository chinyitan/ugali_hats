#!/usr/bin/env python
"""
Magnitude-limit masks read from HealSparse maps.

The file-based mask (ugali.observation.mask.MaskBand) needs, for every
ROI pixel at `coords: nside_pixel`, a magnitude limit and a detection
fraction. This module provides the same two arrays from HealSparse depth
maps, read through the maps' coverage pixels so that only the sky an ROI
touches is loaded, with a process-wide cache shared by neighbouring ROIs.

Config block (replaces mask: dirname/basename_1/basename_2):

    mask:
      healsparse:
        maglim_1: [map1.hsp, map2.hsp]   # depth maps of band 1, in order of priority
        maglim_2: [map1.hsp, map2.hsp]   # depth maps of band 2
        fracdet_1: [frac.hsp]            # optional: detection-fraction maps; default:
        fracdet_2: [frac.hsp]            #   fraction of valid map sub-pixels
        reduction: mean                  # how finer map pixels are combined (mean|median|min|max)
        cache_size: 512                  # coverage-pixel sub-maps kept per process
      fractype: binary                   # as for file masks
      fracmin: 0.5
      minimum_solid_angle: 0.1

Where maps overlap, the first one listed that has a valid value wins.
A map finer than nside_pixel is degraded with `reduction` over its valid
sub-pixels (the fraction of valid sub-pixels is the default detection
fraction); a coarser map is upgraded by repetition.
"""
from collections import OrderedDict as odict
import warnings

import numpy as np
import healpy as hp

from ugali.utils.logger import logger

try:
    import healsparse as hsp
except ImportError:  # pragma: no cover
    hsp = None

REDUCTIONS = {
    'mean': np.nanmean, 'median': np.nanmedian, 'min': np.nanmin, 'max': np.nanmax,
}

# Process-wide caches: coverage objects per file, and per-coverage-pixel
# sub-maps keyed by (filename, coverage pixel)
_COVERAGE_CACHE = {}
_SUBMAP_CACHE = odict()
SUBMAP_CACHE_SIZE = 512


def _require():
    if hsp is None:
        raise ImportError("healsparse is required for mask: healsparse")


def _log2ratio(a, b):
    """ 2*log2(a/b) for healpix nsides a >= b (bit shift between NEST indices). """
    return 2 * int(round(np.log2(a // b)))


def read_coverage(filename):
    """ HealSparseCoverage of a map (cached). """
    _require()
    cov = _COVERAGE_CACHE.get(filename)
    if cov is None:
        cov = _COVERAGE_CACHE[filename] = hsp.HealSparseCoverage.read(filename)
    return cov


def read_submap(filename, covpix, cache_size=None):
    """
    Sub-map of `filename` restricted to one coverage pixel (NEST index at
    the map's nside_coverage), read through the process cache. Returns
    None for a coverage pixel without data.
    """
    _require()
    cache_size = cache_size or SUBMAP_CACHE_SIZE
    key = (filename, int(covpix))
    if key in _SUBMAP_CACHE:
        _SUBMAP_CACHE.move_to_end(key)
        return _SUBMAP_CACHE[key]
    cov = read_coverage(filename)
    sub = hsp.HealSparseMap.read(filename, pixels=[int(covpix)]) if cov.coverage_mask[covpix] else None
    _SUBMAP_CACHE[key] = sub
    while len(_SUBMAP_CACHE) > cache_size:
        _SUBMAP_CACHE.popitem(last=False)
    return sub


def clear_cache():
    _SUBMAP_CACHE.clear()
    _COVERAGE_CACHE.clear()


def _valid(submap, values):
    valid = values != submap.sentinel
    if not submap.is_integer_map:
        valid &= np.isfinite(values)
    return valid


def values_at(filename, pixels_nest, nside, reduction='mean', cache_size=None):
    """
    Map values at NEST pixels of resolution `nside`, and the fraction of
    valid map sub-pixels. A finer map is reduced over its valid
    sub-pixels, a coarser one is repeated. Returns (values, fraction),
    with NaN values where the map has no valid data.
    """
    cov = read_coverage(filename)
    nside_sparse, nside_cov = cov.nside_sparse, cov.nside_coverage
    if nside < nside_cov:
        raise ValueError("nside %i is coarser than the coverage nside %i of %s"
                         % (nside, nside_cov, filename))
    pixels_nest = np.asarray(pixels_nest, dtype=np.int64)
    values = np.full(len(pixels_nest), np.nan)
    fraction = np.zeros(len(pixels_nest))
    if len(pixels_nest) == 0:
        return values, fraction

    covpix = pixels_nest >> _log2ratio(nside, nside_cov)
    finer = nside_sparse >= nside
    if finer:
        shift = _log2ratio(nside_sparse, nside)
        nsub = 1 << shift
        reduce = REDUCTIONS[reduction]
    else:
        shift = _log2ratio(nside, nside_sparse)

    for p in np.unique(covpix):
        sub = read_submap(filename, p, cache_size)
        if sub is None:
            continue
        sel = np.nonzero(covpix == p)[0]
        pix = pixels_nest[sel]
        if finer:
            fine = (pix[:, None] << shift) + np.arange(nsub)[None, :]
            v = sub.get_values_pix(fine.ravel(), nest=True).reshape(fine.shape).astype(np.float64)
            valid = _valid(sub, v)
            nvalid = valid.sum(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                red = reduce(np.where(valid, v, np.nan), axis=1)
            values[sel] = np.where(nvalid > 0, red, np.nan)
            fraction[sel] = nvalid / float(nsub)
        else:
            v = sub.get_values_pix(pix >> shift, nest=True).astype(np.float64)
            valid = _valid(sub, v)
            values[sel] = np.where(valid, v, np.nan)
            fraction[sel] = valid.astype(float)
    return values, fraction


def _first_valid(files, pixels_nest, nside, reduction, cache_size):
    """ Values (and sub-pixel fractions) from the first map with data, per pixel. """
    out = np.full(len(pixels_nest), np.nan)
    frac = np.zeros(len(pixels_nest))
    for f in np.atleast_1d(files):
        todo = np.isnan(out)
        if not todo.any():
            break
        v, fr = values_at(f, pixels_nest[todo], nside, reduction, cache_size)
        idx = np.nonzero(todo)[0]
        got = ~np.isnan(v)
        out[idx[got]] = v[got]
        frac[idx[got]] = fr[got]
    return out, frac


def band_sparse(maglim_files, pixels_ring, nside, fracdet_files=None,
                reduction='mean', cache_size=None):
    """
    Magnitude limit and detection fraction at RING pixels of resolution
    `nside` from a priority-ordered list of depth maps.

    Returns (maglim, fracdet): maglim is 0 where no map has data; fracdet
    is the fraction of valid map sub-pixels of the map that provided the
    value, or, when `fracdet_files` is given, the value of those maps
    (0 where they have no data).
    """
    pixels_ring = np.asarray(pixels_ring, dtype=np.int64)
    pixels_nest = hp.ring2nest(nside, pixels_ring)
    maglim, frac = _first_valid(maglim_files, pixels_nest, nside, reduction, cache_size)
    if fracdet_files is not None:
        fd, _ = _first_valid(fracdet_files, pixels_nest, nside, reduction, cache_size)
        frac = np.where(np.isnan(fd), 0.0, fd)
    maglim = np.where(np.isnan(maglim), 0.0, maglim)
    return maglim, frac


def covered_pixels(files, nside, nest=False):
    """
    Boolean array over all pixels of resolution `nside` (RING unless
    `nest`) inside the coverage of at least one of the maps.
    """
    npix = hp.nside2npix(nside)
    out = np.zeros(npix, dtype=bool)
    for f in np.atleast_1d(files):
        cov = read_coverage(f)
        mask = np.asarray(cov.coverage_mask, dtype=bool)  # NEST at nside_coverage
        if cov.nside_coverage == nside:
            m = mask
        elif cov.nside_coverage < nside:
            m = np.repeat(mask, (nside // cov.nside_coverage) ** 2)
        else:
            m = mask.reshape(npix, -1).any(axis=1)
        out |= m
    if not nest:
        out = out[hp.ring2nest(nside, np.arange(npix))]
    return out


def valid_superpixels(band_files, nside_out, catalog_pixels=None, nside_catalog=None,
                      cache_size=None):
    """
    RING pixels of resolution `nside_out` that contain at least one valid
    pixel of some map of every band (`band_files` is a list of per-band
    map lists, e.g. [maglim_1_files, maglim_2_files]), restricted to the
    `catalog_pixels` (RING at `nside_catalog`) if given. This is the
    survey footprint used to select likelihood pixels.
    """
    result = None
    for files in band_files:
        band = set()
        for f in np.atleast_1d(files):
            cov = read_coverage(f)
            nside_cov, nside_sparse = cov.nside_coverage, cov.nside_sparse
            covpix = np.nonzero(cov.coverage_mask)[0]
            if catalog_pixels is not None:
                want = hp.ring2nest(nside_catalog, np.asarray(catalog_pixels, dtype=np.int64))
                if nside_catalog == nside_cov:
                    covpix = np.intersect1d(covpix, want)
                elif nside_catalog < nside_cov:
                    covpix = covpix[np.isin(covpix >> _log2ratio(nside_cov, nside_catalog), want)]
                else:
                    covpix = np.intersect1d(covpix, want >> _log2ratio(nside_catalog, nside_cov))
            logger.debug("Footprint of %s: %i coverage pixels" % (f, len(covpix)))
            for p in covpix:
                sub = read_submap(f, p, cache_size)
                if sub is None:
                    continue
                valid = sub.valid_pixels
                if nside_sparse >= nside_out:
                    band.update(np.unique(valid >> _log2ratio(nside_sparse, nside_out)).tolist())
                else:
                    shift = _log2ratio(nside_out, nside_sparse)
                    children = (valid[:, None] << shift) + np.arange(1 << shift)[None, :]
                    band.update(children.ravel().tolist())
        result = band if result is None else (result & band)
    if not result:
        return np.zeros(0, dtype=np.int64)
    out = np.array(sorted(result), dtype=np.int64)
    return np.sort(hp.nest2ring(nside_out, out))
