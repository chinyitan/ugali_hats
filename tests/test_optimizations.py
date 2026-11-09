#!/usr/bin/env python
"""
Bit-exactness tests for the performance optimizations of the likelihood
scan. Each optimized code path is compared (with np.array_equal / ==, never
approximately) with the original formula, on the single-pixel DES Y6
fixture in tests/data/desy6.

    fast_richness           LogLikelihood.fit_richness fast path
    cache_spatial           memoized / geometry-reusing spatial term
    observableFractionCMD   unique-interior columns + masked add.reduce
    IsochroneModel.pdf      sparse (object, bin) pair search
    Mask.mask_roi_unique/digi memoization
"""
import os

import numpy as np
import pytest

import ugali.utils.binning
from ugali.utils.healpix import pix2ang
from ugali.utils.stats import norm_cdf
from ugali.utils.config import Config
from ugali.analysis.scan import createGridSearch

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'desy6')
CONFIG = os.path.join(FIXTURE, 'config.yaml')
NSIDE = 256
FIXTURE_PIX = 639648


def make_grid(scan=None, **likelihood):
    lon, lat = pix2ang(NSIDE, FIXTURE_PIX)
    cfg = Config(CONFIG)
    for k, v in likelihood.items():
        cfg['likelihood'][k] = v
    for k, v in (scan or {}).items():
        cfg['scan'][k] = v
    return createGridSearch(cfg, lon, lat)


@pytest.fixture(scope='module', autouse=True)
def fixture_cwd():
    """The fixture config uses paths relative to its own directory."""
    old = os.getcwd()
    os.chdir(FIXTURE)
    yield
    os.chdir(old)


@pytest.fixture(scope='module')
def grid(fixture_cwd):
    return make_grid()


@pytest.fixture(scope='module')
def loglike(grid):
    return grid.loglike


def grid_points(grid):
    """A handful of (distance_modulus, lon, lat, extension) grid points."""
    dms = grid.distance_modulus_array
    exts = grid.extension_array
    lons = grid.roi.pixels_target.lon
    lats = grid.roi.pixels_target.lat
    pts = []
    for i, (d, e) in enumerate([(0, 0), (len(dms)//2, 1), (len(dms)-1, len(exts)-1),
                                (len(dms)//3, len(exts)//2)]):
        k = (i * 7) % len(lons)
        pts.append((dms[d], lons[k], lats[k], exts[e]))
    return pts


def set_point(loglike, pt):
    dm, lon, lat, ext = pt
    loglike.set_params(distance_modulus=dm, lon=lon, lat=lat, extension=ext)
    loglike.sync_params()


# ---------------------------------------------------------------- richness
def test_fit_richness_fast_path_identical(grid, loglike):
    """fit_richness gives identical (loglike_max, richness_max) and final
    source.richness with fast_richness True and False."""
    old = loglike.fast_richness
    try:
        for pt in grid_points(grid):
            res, rich = {}, {}
            for mode in (True, False):
                loglike.fast_richness = mode
                set_point(loglike, pt)
                ll, r, _ = loglike.fit_richness()
                res[mode] = (ll, r)
                rich[mode] = loglike.source.richness
            assert res[True] == res[False]
            assert rich[True] == rich[False]
    finally:
        loglike.fast_richness = old


# ----------------------------------------------------------------- spatial
def test_spatial_geom_matches_original(grid, loglike):
    """calc_signal_spatial_geom == calc_signal_spatial, for both the
    returned u_spatial and surface_intensity_sparse."""
    for pt in grid_points(grid):
        set_point(loglike, pt)
        u_ref = np.array(loglike.calc_signal_spatial())
        s_ref = np.array(loglike.surface_intensity_sparse)
        u_new = np.array(loglike.calc_signal_spatial_geom())
        s_new = np.array(loglike.surface_intensity_sparse)
        assert np.array_equal(u_ref, u_new)
        assert np.array_equal(s_ref, s_new)
        # a second extension at the same centroid reuses the cached geometry
        loglike.set_params(extension=pt[3]*1.5)
        loglike.sync_params()
        u_ref = np.array(loglike.calc_signal_spatial())
        s_ref = np.array(loglike.surface_intensity_sparse)
        u_new = np.array(loglike.calc_signal_spatial_geom())
        s_new = np.array(loglike.surface_intensity_sparse)
        assert np.array_equal(u_ref, u_new)
        assert np.array_equal(s_ref, s_new)


def test_cached_spatial_hit_matches_original(grid, loglike):
    """calc_signal_spatial_cached (miss and hit) == calc_signal_spatial."""
    pt = grid_points(grid)[1]
    set_point(loglike, pt)
    u_ref = np.array(loglike.calc_signal_spatial())
    s_ref = np.array(loglike.surface_intensity_sparse)
    for _ in range(2):
        u = np.array(loglike.calc_signal_spatial_cached())
        assert np.array_equal(u_ref, u)
        assert np.array_equal(s_ref, loglike.surface_intensity_sparse)


def test_full_scan_all_optimizations_off_identical():
    """Whole grid search with cache_spatial, fast_richness and batch_pixels
    off gives arrays identical to the default (optimized) run."""
    outs = []
    for kw in ({}, dict(cache_spatial=False, fast_richness=False, scan=dict(batch_pixels=False))):
        g = make_grid(**kw)
        assert g.loglike.cache_spatial == kw.get('cache_spatial', True)
        assert g.loglike.fast_richness == kw.get('fast_richness', True)
        assert g.batch_pixels == kw.get('scan', {}).get('batch_pixels', True)
        g.search()
        outs.append([np.array(a) for a in (g.loglike_array, g.richness_array,
                                           g.fraction_observable_array)])
    for a, b in zip(*outs):
        assert np.array_equal(a, b)


def test_fit_richness_batch_matches_rows(grid, loglike):
    """fit_richness_batch over a stack of synced positions equals
    fit_richness run position by position (bit-exact)."""
    pts = grid_points(grid)[:6]
    u, f, expected = [], [], []
    for pt in pts:
        set_point(loglike, pt)
        u.append(np.array(loglike.u)); f.append(loglike.f)
        ll, rich, _ = loglike.fit_richness()
        expected.append((ll, rich))
    ll_b, rich_b = loglike.fit_richness_batch(np.array(u), np.array(f))
    for i, (ll, rich) in enumerate(expected):
        assert ll_b[i] == ll and rich_b[i] == rich


# -------------------------------------------------- observable fraction
def reference_observable_fraction(iso, mask, distance_modulus, mass_min=0.1):
    mass_init,mass_pdf,mass_act,mag_1,mag_2 = iso.sample(mass_min=mass_min,full_data_range=False)
    mag = mag_1 if iso.band_1_detection else mag_2
    color = mag_1 - mag_2
    mag_1_mask,mag_2_mask = mask.mask_roi_unique.T
    cmd_cut = ugali.utils.binning.take2D(mask.solid_angle_cmd,color,mag+distance_modulus,
                                         mask.roi.bins_color, mask.roi.bins_mag) > 0
    mass_pdf_cut = mass_pdf*cmd_cut
    mask_1_cut = (mag_1+distance_modulus)[:,np.newaxis] < mag_1_mask
    mask_2_cut = (mag_2+distance_modulus)[:,np.newaxis] < mag_2_mask
    mask_cut_repeat = (mask_1_cut & mask_2_cut)
    observable_fraction = (mass_pdf_cut[:,np.newaxis]*mask_cut_repeat).sum(axis=0)
    return observable_fraction[mask.mask_roi_digi[mask.roi.pixel_interior_cut]] * mask.frac_interior_sparse


@pytest.mark.parametrize('dm', [16.0, 19.0, 22.0])
def test_observable_fraction_cmd_identical(loglike, dm):
    """observableFractionCMD == dense reference formula, exactly."""
    iso = loglike.isochrone.isochrones[0]
    mask = loglike.mask
    new = iso.observableFractionCMD(mask, dm)
    ref = reference_observable_fraction(iso, mask, dm)
    assert new.shape == ref.shape
    assert np.array_equal(new, ref)


# ---------------------------------------------------------------- iso pdf
def reference_pdf(iso, mag_1, mag_2, mag_err_1, mag_err_2,
                  distance_modulus=None, delta_mag=0.01, steps=10000):
    """Original (dense) IsochroneModel.pdf."""
    nsigma = 5.0
    if distance_modulus is None:
        distance_modulus = iso.distance_modulus
    mag_err_1 = np.sqrt(mag_err_1**2 + 0.01**2)
    mag_err_2 = np.sqrt(mag_err_2**2 + 0.01**2)
    histo_pdf,bins_mag_1,bins_mag_2 = iso.histogram2d(distance_modulus,delta_mag,steps)
    mag_1_mesh, mag_2_mesh = np.meshgrid(bins_mag_2[1:], bins_mag_1[1:])
    mag_1_max = np.max(mag_1+nsigma*mag_err_1)
    mag_1_min = np.min(mag_1-nsigma*mag_err_1)
    mag_2_max = np.max(mag_2+nsigma*mag_err_2)
    mag_2_min = np.min(mag_2-nsigma*mag_err_2)
    in_mag_space = ((mag_1_mesh>=mag_1_min)&(mag_1_mesh<=mag_1_max))
    in_mag_space*= ((mag_2_mesh>=mag_2_min)&(mag_2_mesh<=mag_2_max))
    histo_pdf *= in_mag_space
    idx_mag_1, idx_mag_2 = np.nonzero(histo_pdf)
    isochrone_pdf = histo_pdf[idx_mag_1, idx_mag_2]
    n_catalog = len(mag_1)
    n_isochrone_bins = len(idx_mag_1)

    mag_1 = mag_1.reshape([n_catalog, 1]); mag_err_1 = mag_err_1.reshape([n_catalog, 1])
    mag_2 = mag_2.reshape([n_catalog, 1]); mag_err_2 = mag_err_2.reshape([n_catalog, 1])
    dist_mag_1_hi = (mag_1-bins_mag_1[idx_mag_1])/mag_err_1
    dist_mag_1_lo = (mag_1-bins_mag_1[idx_mag_1+1])/mag_err_1
    dist_mag_2_hi = (mag_2-bins_mag_2[idx_mag_2])/mag_err_2
    dist_mag_2_lo = (mag_2-bins_mag_2[idx_mag_2+1])/mag_err_2
    idx_nonzero_0,idx_nonzero_1 = np.nonzero((dist_mag_1_hi > -nsigma)*(dist_mag_1_lo < nsigma)*(dist_mag_2_hi > -nsigma)*(dist_mag_2_lo < nsigma))
    pdf_mag_1 = np.zeros([n_catalog, n_isochrone_bins],dtype=np.float32)
    pdf_mag_1[idx_nonzero_0,idx_nonzero_1] = norm_cdf(dist_mag_1_hi[idx_nonzero_0,idx_nonzero_1]) - norm_cdf(dist_mag_1_lo[idx_nonzero_0,idx_nonzero_1])
    pdf_mag_2 = np.zeros([n_catalog, n_isochrone_bins],dtype=np.float32)
    pdf_mag_2[idx_nonzero_0,idx_nonzero_1] = norm_cdf(dist_mag_2_hi[idx_nonzero_0,idx_nonzero_1]) - norm_cdf(dist_mag_2_lo[idx_nonzero_0,idx_nonzero_1])
    u_color = np.sum(pdf_mag_1 * pdf_mag_2 * isochrone_pdf, axis=1)
    u_color /= delta_mag**2
    return u_color.astype(np.float32)


@pytest.mark.parametrize('dm', [17.0, 19.5, 22.0])
def test_isochrone_pdf_sparse_identical(loglike, dm):
    """Sparse IsochroneModel.pdf == dense reference on the fixture catalog."""
    iso, cat = loglike.isochrone, loglike.catalog
    args = (cat.mag_1, cat.mag_2, cat.mag_err_1, cat.mag_err_2)
    new = iso.pdf(*args, distance_modulus=dm, delta_mag=loglike.delta_mag)
    ref = reference_pdf(iso, *[np.array(a) for a in args], distance_modulus=dm,
                        delta_mag=loglike.delta_mag)
    assert new.dtype == ref.dtype
    assert np.array_equal(new, ref)
    assert np.any(new > 0)


def test_isochrone_pdf_out_of_range_object_is_zero(loglike):
    """Objects whose magnitude window misses the isochrone give exactly 0
    in both implementations (mixed with in-range objects so the
    isochrone window is not trivially empty)."""
    iso, cat = loglike.isochrone, loglike.catalog
    n = 50
    m1 = np.concatenate([np.array(cat.mag_1)[:n], [10.0]])
    m2 = np.concatenate([np.array(cat.mag_2)[:n], [10.0]])
    e1 = np.concatenate([np.array(cat.mag_err_1)[:n], [0.01]])
    e2 = np.concatenate([np.array(cat.mag_err_2)[:n], [0.01]])
    new = iso.pdf(m1, m2, e1, e2, distance_modulus=19.0, delta_mag=loglike.delta_mag)
    ref = reference_pdf(iso, m1, m2, e1, e2, distance_modulus=19.0,
                        delta_mag=loglike.delta_mag)
    assert np.array_equal(new, ref)
    assert new[-1] == 0 and ref[-1] == 0


def test_isochrone_pdf_empty_catalog(loglike):
    """Empty input: if the dense reference raises (np.max of empty), the
    sparse version is not required to work; otherwise results match."""
    iso = loglike.isochrone
    e = np.zeros(0)
    try:
        ref = reference_pdf(iso, e, e, e, e, distance_modulus=19.0,
                            delta_mag=loglike.delta_mag)
    except ValueError:
        pytest.skip('dense reference raises on empty input')
    new = iso.pdf(e, e, e, e, distance_modulus=19.0, delta_mag=loglike.delta_mag)
    assert np.array_equal(new, ref)


# ------------------------------------------------------------------- mask
def test_mask_roi_unique_digi_memoized(loglike):
    """mask_roi_unique / mask_roi_digi are stable across calls and
    unique[digi] reproduces the per-pixel (mag_1, mag_2) mask exactly."""
    mask = loglike.mask
    u1, u2 = mask.mask_roi_unique, mask.mask_roi_unique
    d1, d2 = mask.mask_roi_digi, mask.mask_roi_digi
    assert np.array_equal(u1, u2)
    assert np.array_equal(d1, d2)
    full = np.vstack([mask.mask_1.mask_roi_sparse, mask.mask_2.mask_roi_sparse]).T
    assert np.array_equal(mask.mask_roi_unique[mask.mask_roi_digi], full)


# -------------------------------------------------- sparse color pdf (opt-in, not bit-exact)
def test_isochrone_pdf_sparse_accumulation_close(loglike):
    """likelihood: color_pdf_sparse accumulates the same (object, bin)
    contributions in float64; it must agree with the dense float32 sum to
    float32 rounding (relative 1e-5 of the largest value)."""
    iso, cat = loglike.isochrone, loglike.catalog
    for dm in (17.0, 20.0):
        dense = iso.pdf(cat.mag_1, cat.mag_2, cat.mag_err_1, cat.mag_err_2, dm, loglike.delta_mag, 10000)
        sparse = iso.pdf(cat.mag_1, cat.mag_2, cat.mag_err_1, cat.mag_err_2, dm, loglike.delta_mag, 10000, sparse=True)
        assert dense.shape == sparse.shape and dense.dtype == sparse.dtype
        assert np.array_equal(dense > 0, sparse > 0)
        assert np.abs(dense.astype(float) - sparse.astype(float)).max() <= 1e-5 * dense.max()
