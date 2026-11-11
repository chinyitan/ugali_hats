"""Synthetic tests of ugali.analysis.postproc."""
import numpy as np
import healpy as hp
import pytest

from ugali.analysis import postproc as pp


def _region(nside=256, ra0=180.0, dec0=0.0, radius=4.0):
    """RING pixels of a disc, sorted."""
    pix = hp.query_disc(nside, hp.ang2vec(ra0, dec0, lonlat=True), np.radians(radius))
    return np.sort(pix)


def test_median_subtract_removes_flat_offset_keeps_peak():
    ts = np.full((5, 21), 7.0, np.float32)
    ts[2, 10] = 30.0
    out = pp.median_subtract(ts)
    assert np.all(out[:, :10] == 0) and out[2, 10] == pytest.approx(23.0)
    assert out.min() >= 0


def test_roi_density_ratio_flags_overdense_interior():
    nside, nside_t = 256, 64
    pix = _region(nside)
    footprint = np.zeros(hp.nside2npix(nside), bool)
    footprint[hp.ring2nest(nside, pix)] = True
    counts = np.where(footprint, 10.0, 0.0)             # NESTED full-sky map, uniform
    targets, it = pp.targets_of(pix, nside, nside_t)
    # double the counts within 0.4 deg of one target centre
    t0 = targets[len(targets) // 2]
    ra, dec = hp.pix2ang(nside_t, t0, lonlat=True, nest=True)
    hot = hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(0.4), nest=True)
    counts[hot] *= 2
    r = pp.roi_density_ratio(counts, nside, footprint, targets, nside_t, r_in=0.5, r_out=2.0)
    i0 = np.searchsorted(targets, t0)
    assert r[i0] > 1.4
    far = np.abs(np.arange(len(targets)) - i0) > 0
    # targets whose annulus contains the hot spot get a ratio below one, the rest sit at one
    assert np.nanmedian(r[far]) == pytest.approx(1.0, abs=0.05)


def test_roi_density_ratio_weights_ignore_masked_tiles():
    """A masked tile with anomalous counts in the annulus must not bias the ratio when weighted."""
    nside, nside_t = 256, 64
    pix = _region(nside)
    footprint = np.zeros(hp.nside2npix(nside), bool)
    footprint[hp.ring2nest(nside, pix)] = True
    counts = np.where(footprint, 10.0, 0.0)
    targets, it = pp.targets_of(pix, nside, nside_t)
    t0 = targets[len(targets) // 2]
    ra, dec = hp.pix2ang(nside_t, t0, lonlat=True, nest=True)
    # an "incomplete tile" 1 deg away with a fifth of the counts, as a scan-masked region
    bad = hp.query_disc(nside, hp.ang2vec(ra + 1.0, dec, lonlat=True), np.radians(0.5), nest=True)
    counts[bad] = 2.0
    w = footprint.astype(float); w[bad] = 0.0
    i0 = np.searchsorted(targets, t0)
    r_unw = pp.roi_density_ratio(counts, nside, footprint, targets, nside_t)[i0]
    r_w = pp.roi_density_ratio(counts, nside, footprint, targets, nside_t, weights=w)[i0]
    assert r_unw > 1.05 and r_w == pytest.approx(1.0, abs=0.02)


def test_ebv_mask_and_protect():
    nside = 256
    pix = _region(nside)
    ebv = np.zeros(hp.nside2npix(64))
    ra, dec = hp.pix2ang(nside, pix, lonlat=True)
    ebv[hp.ang2pix(64, 180.0, 0.0, lonlat=True)] = 0.5    # one dusty nside-64 pixel at the centre
    m = pp.ebv_mask(pix, nside, ebv, 0.2)
    assert 0 < m.sum() < len(pix)
    sep = hp.rotator.angdist(np.array([np.radians(90 - dec), np.radians(ra)]),
                             np.array([np.radians(90.0), np.radians(180.0)]))
    assert np.all(np.degrees(sep[m]) < 1.5)
    prot = pp.protect_objects(pix, nside, np.array([180.0]), np.array([0.0]), radius=1.0)
    assert prot.sum() > 0 and np.all(np.degrees(sep[prot]) < 1.0 + 0.3)


def test_clean_roundtrip(tmp_path):
    nside = 256
    pix = _region(nside, radius=3.0)
    mus = np.arange(15, 19.1, 0.2)
    rng = np.random.default_rng(0)
    ts = rng.exponential(1.0, (len(pix), len(mus))).astype(np.float32)
    fn = tmp_path / 'cube.npz'
    np.savez(fn, pixel=pix, nside=nside, ordering='RING', distance_modulus=mus, log_likelihood=ts / 2)
    pixel, ns, m, t = pp.load_cube(fn)
    assert ns == nside and np.allclose(t, ts, atol=1e-6)
    out, mask = pp.clean(pixel, ns, t, subtract_median=True)
    assert not mask.any() and np.all(np.nanmin(out, axis=1) == 0)
    pp.save_cube(tmp_path / 'clean.npz', pixel, ns, m, out, keep=~mask)
    assert len(np.load(tmp_path / 'clean.npz')['pixel']) == len(pix)
