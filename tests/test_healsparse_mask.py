#!/usr/bin/env python
"""
Tests of the HealSparse magnitude-limit masks (ugali.observation.healsparse_mask).

The DES Y6 fixture mask files are converted to HealSparse maps and the
mask arrays, the likelihood scan and the footprint are compared bit-exactly
with the file-based mask. Synthetic maps test the degrade/upgrade logic.
"""
import glob
import os
import shutil

import numpy as np
import healpy as hp
import pytest

hsp = pytest.importorskip('healsparse')
fitsio = pytest.importorskip('fitsio')

from ugali.observation import healsparse_mask as hm
from ugali.utils.config import Config
from ugali.utils.healpix import pix2ang
from ugali.utils.skymap import inFootprint
from ugali.analysis.scan import createGridSearch

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'desy6')
NSIDE = 256
FIXTURE_PIX = 639648

MASK_SECTION = """mask:
  healsparse:
    maglim_1: [{d}/maglim_g.hsp]
    maglim_2: [{d}/maglim_r.hsp]
    fracdet_1: [{d}/fracdet_g.hsp]
    fracdet_2: [{d}/fracdet_r.hsp]
  minimum_solid_angle: 0.1
"""


def _replace_mask_section(text, new):
    """Replace the top-level `mask:` block of a yaml text."""
    lines = text.splitlines(True)
    out, skipping = [], False
    for line in lines:
        if line.startswith('mask:'):
            skipping = True
            out.append(new)
            continue
        if skipping:
            if line.strip() == '' or line.startswith((' ', '\t', '#')):
                continue
            skipping = False
        out.append(line)
    return ''.join(out)


@pytest.fixture(scope='module')
def hsp_fixture(tmp_path_factory):
    """Copy of the fixture dir with a healsparse config; returns (dir, config path)."""
    hm.clear_cache()
    workdir = str(tmp_path_factory.mktemp('hspfix') / 'desy6')
    shutil.copytree(FIXTURE, workdir)
    for band in 'gr':
        files = sorted(glob.glob(os.path.join(FIXTURE, 'mask', 'maglim_%s_hpx*.fits' % band)))
        d = np.concatenate([fitsio.read(f) for f in files])
        nest = hp.ring2nest(4096, d['PIXEL'])
        for col, name in [('MAGLIM', 'maglim'), ('FRACDET', 'fracdet')]:
            m = hsp.HealSparseMap.make_empty(32, 4096, np.float64)
            m[nest] = d[col].astype(np.float64)
            m.write(os.path.join(workdir, '%s_%s.hsp' % (name, band)), clobber=True)
    with open(os.path.join(FIXTURE, 'config.yaml')) as f:
        text = f.read()
    new = _replace_mask_section(text, MASK_SECTION.format(d=workdir))
    assert 'healsparse' in new and 'basename_1' not in new
    cfg = os.path.join(workdir, 'config_healsparse.yaml')
    with open(cfg, 'w') as f:
        f.write(new)
    yield workdir, cfg
    hm.clear_cache()


def _build(workdir, cfgfile):
    old = os.getcwd()
    os.chdir(workdir)
    try:
        lon, lat = pix2ang(NSIDE, FIXTURE_PIX)
        return createGridSearch(Config(cfgfile), lon, lat)
    finally:
        os.chdir(old)


@pytest.fixture(scope='module')
def grid_file():
    return _build(FIXTURE, os.path.join(FIXTURE, 'config.yaml'))


@pytest.fixture(scope='module')
def grid_hsp(hsp_fixture):
    return _build(*hsp_fixture)


def _search(grid, workdir):
    old = os.getcwd()
    os.chdir(workdir)
    try:
        grid.search()
    finally:
        os.chdir(old)
    return grid


# ------------------------------------------------------------ fixture data
def test_mask_arrays_identical(grid_file, grid_hsp):
    mf, mh = grid_file.loglike.mask, grid_hsp.loglike.mask
    for attr in ['mask_1.mask_roi_sparse', 'mask_2.mask_roi_sparse',
                 'mask_1.frac_roi_sparse', 'mask_2.frac_roi_sparse',
                 'frac_roi_sparse', 'solid_angle_cmd']:
        a, b = mf, mh
        for part in attr.split('.'):
            a, b = getattr(a, part), getattr(b, part)
        assert a.shape == b.shape, attr
        assert np.array_equal(a, b), attr
    assert np.array_equal(grid_file.roi.getCatalogPixels(), grid_hsp.roi.getCatalogPixels())
    assert len(grid_file.loglike.catalog) == len(grid_hsp.loglike.catalog)


def test_scan_identical(grid_file, grid_hsp, hsp_fixture):
    _search(grid_file, FIXTURE)
    _search(grid_hsp, hsp_fixture[0])
    for name in ['loglike_array', 'richness_array', 'fraction_observable_array']:
        a, b = getattr(grid_file, name), getattr(grid_hsp, name)
        assert np.array_equal(a, b), name


def test_footprint_identical(hsp_fixture):
    workdir, cfg = hsp_fixture
    lon, lat = pix2ang(NSIDE, FIXTURE_PIX)
    vec = hp.ang2vec(lon, lat, lonlat=True)
    pixels = hp.query_disc(NSIDE, vec, np.radians(1.0), inclusive=True)
    far = hp.ang2pix(NSIDE, 0., -89., lonlat=True)
    pixels = np.append(pixels, far)
    old = os.getcwd()
    os.chdir(workdir)
    try:
        a = inFootprint(os.path.join(FIXTURE, 'config.yaml'), pixels)
        b = inFootprint(cfg, pixels)
    finally:
        os.chdir(old)
    assert a.dtype == bool and b.dtype == bool
    assert np.array_equal(a, b)
    assert a.any() and not a[-1]


# --------------------------------------------------------------- synthetic
@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    hm.clear_cache()
    d = tmp_path_factory.mktemp('synth')
    nside_s = 64
    # nside-16 pixel P fully valid (16 sub-pixels), Q partially (4 of 16)
    P, Q = 100, 700
    shift = 2 * int(np.log2(nside_s // 16))  # nside 64 -> 16
    sub = 1 << shift  # 16 fine pixels per nside-16 pixel
    full_vals = np.arange(sub, dtype=float) + 20.0
    part_vals = np.array([10., 12., 14., 30.])
    full = (P << shift) + np.arange(sub)
    part = (Q << shift) + np.arange(4)
    lone = np.array([(2000 << shift) + 3])  # nside-16 pixel with a single valid sub-pixel

    m1 = hsp.HealSparseMap.make_empty(4, nside_s, np.float64)
    m1[np.concatenate([full, part, lone])] = np.concatenate([full_vals, part_vals, [5.0]])
    f1 = str(d / 'm1.hsp')
    m1.write(f1, clobber=True)

    # second map: overlaps P with other values (must lose), fills pixel R and the rest of Q
    R = 300
    m2 = hsp.HealSparseMap.make_empty(4, nside_s, np.float64)
    rest_q = (Q << shift) + np.arange(4, sub)
    r = (R << shift) + np.arange(sub)
    m2[np.concatenate([full, rest_q, r])] = np.concatenate(
        [np.full(sub, -99.), np.full(len(rest_q), 50.), np.full(sub, 7.)])
    f2 = str(d / 'm2.hsp')
    m2.write(f2, clobber=True)
    yield dict(f1=f1, f2=f2, P=P, Q=Q, R=R, shift=shift, sub=sub,
               full_vals=full_vals, part_vals=part_vals, m1=m1, m2=m2)
    hm.clear_cache()


def test_band_sparse_degrade_and_upgrade(synth):
    s = synth
    nest16 = np.array([s['P'], s['Q'], 2000, 5, 3000])
    ring16 = hp.nest2ring(16, nest16)
    mag, frac = hm.band_sparse([s['f1']], ring16, nside=16)
    assert mag[0] == pytest.approx(s['full_vals'].mean()) and frac[0] == 1.0
    assert mag[1] == pytest.approx(s['part_vals'].mean()) and frac[1] == 4 / 16.
    assert mag[2] == 5.0 and frac[2] == 1 / 16.
    assert np.all(mag[3:] == 0) and np.all(frac[3:] == 0)

    mag, frac = hm.band_sparse([s['f1']], ring16, nside=16, reduction='max')
    assert mag[0] == s['full_vals'].max()
    assert mag[1] == s['part_vals'].max()
    assert np.all(mag[3:] == 0)
    mag, frac = hm.band_sparse([s['f1']], ring16, nside=16, reduction='min')
    assert mag[0] == s['full_vals'].min() and mag[1] == s['part_vals'].min()

    # explicit fracdet maps override the valid-sub-pixel fraction
    mag, frac = hm.band_sparse([s['f1']], ring16, nside=16, fracdet_files=[s['f2']])
    assert frac[0] == pytest.approx(-99.)  # taken from the fracdet map itself
    assert frac[3] == 0

    # upgrade: nside 128 from the nside-64 map -> parent's value, fracdet 1
    pix64 = (s['P'] << s['shift']) + np.arange(s['sub'])
    pix64 = np.append(pix64, 12345)  # invalid
    child = (pix64[:, None] << 2) + np.arange(4)[None, :]
    ring128 = hp.nest2ring(128, child.ravel())
    mag, frac = hm.band_sparse([s['f1']], ring128, nside=128)
    mag, frac = mag.reshape(child.shape), frac.reshape(child.shape)
    assert np.array_equal(mag[:-1], np.repeat(s['full_vals'][:, None], 4, axis=1))
    assert np.all(frac[:-1] == 1.0)
    assert np.all(mag[-1] == 0) and np.all(frac[-1] == 0)


def test_band_sparse_priority(synth):
    s = synth
    nest16 = np.array([s['P'], s['Q'], s['R']])
    ring16 = hp.nest2ring(16, nest16)
    mag, frac = hm.band_sparse([s['f1'], s['f2']], ring16, nside=64 // 4)
    # P: first map wins; R: only in second; Q: pixel-wise mix is NOT done at
    # nside 16 (the first map has data for Q so it wins as a whole)
    assert mag[0] == pytest.approx(s['full_vals'].mean())
    assert mag[2] == pytest.approx(7.0) and frac[2] == 1.0
    assert mag[1] == pytest.approx(s['part_vals'].mean())
    # at the native nside 64, second map fills the gaps of the first
    q64 = (s['Q'] << s['shift']) + np.arange(s['sub'])
    mag, frac = hm.band_sparse([s['f1'], s['f2']], hp.nest2ring(64, q64), nside=64)
    assert np.array_equal(mag[:4], s['part_vals'])
    assert np.all(mag[4:] == 50.0) and np.all(frac == 1.0)
    p64 = (s['P'] << s['shift']) + np.arange(s['sub'])
    mag, _ = hm.band_sparse([s['f1'], s['f2']], hp.nest2ring(64, p64), nside=64)
    assert np.array_equal(mag, s['full_vals'])
    mag, _ = hm.band_sparse([s['f2'], s['f1']], hp.nest2ring(64, p64), nside=64)
    assert np.all(mag == -99.)


def test_covered_pixels_and_valid_superpixels(synth):
    s = synth
    for f in (s['f1'], s['f2']):
        m = s['m1'] if f == s['f1'] else s['m2']
        cov = np.nonzero(np.asarray(m.coverage_mask))[0]
        assert len(cov) > 0
        for nside in (4, 16):
            expect = np.zeros(hp.nside2npix(nside), dtype=bool)
            if nside == 4:
                expect[cov] = True
            else:
                expect[(cov[:, None] * 16 + np.arange(16)[None, :]).ravel()] = True
            got_nest = hm.covered_pixels(f, nside, nest=True)
            assert np.array_equal(got_nest, expect)
            got_ring = hm.covered_pixels(f, nside)
            assert np.array_equal(got_ring, expect[hp.ring2nest(nside, np.arange(len(expect)))])
        # coarser than coverage
        nest2 = hm.covered_pixels(f, 2, nest=True)
        exp2 = np.zeros(hp.nside2npix(2), dtype=bool)
        exp2[cov >> 2] = True
        assert np.array_equal(nest2, exp2)

    both = hm.valid_superpixels([[s['f1']], [s['f2']]], nside_out=8)
    v1 = np.unique(s['m1'].valid_pixels >> 6)  # nside 64 -> 8
    v2 = np.unique(s['m2'].valid_pixels >> 6)
    expect = np.sort(hp.nest2ring(8, np.intersect1d(v1, v2)))
    assert len(expect) > 0
    assert np.array_equal(both, expect)
    one = hm.valid_superpixels([[s['f1']], [s['f1']]], nside_out=8)
    assert np.array_equal(one, np.sort(hp.nest2ring(8, v1)))
