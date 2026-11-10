#!/usr/bin/env python
"""
Post-hoc cleaning of collated likelihood-scan maps.

The maps produced by `ugali/analysis/collate.py` hold TS = 2 log L per output
cell and distance modulus. Three classes of artefact were identified in the
DELVE DR3 scans and each has a cheap post-hoc remedy that was validated
against known streams and satellites (see the README):

* *ROI-normalisation hotspots* (~2 deg, peaked in modulus): targets whose
  interior star density does not match their annulus. Flag them with the
  interior/annulus density ratio of the scan-selected stars
  (`roi_density_ratio`), mask the targets above a threshold, and subtract the
  per-cell median TS over modulus (`median_subtract`).
* *All-modulus hotspots* (TS > 10 at every modulus): dusty bulge-direction
  fields where the SFD dereddening over-corrects and shifts the stellar locus
  relative to the annulus, and tile-scale depth/classification contrasts.
  Mask cells above an E(B-V) threshold (`ebv_mask`) and targets whose
  faint-end selected-object density ratio is high (`roi_density_ratio` on a
  faint count map).
* Known objects are protected: cells within a radius of a catalogued dwarf
  or cluster are never masked (`protect_objects`).

All functions work on the arrays of the collated .npz (RING pixel indices at
`nside`) and on full-sky HEALPix count maps; nothing here reads catalogues.
A command line applies the whole recipe::

    python ugali/analysis/postproc.py ts_maps.npz cleaned.npz \
        --counts selected_counts_n1024.npz:sel_bright --ratio-max 1.10 \
        --counts selected_counts_n1024.npz:sel_faint --ratio-max 1.25 \
        --ebv ebv_sfd98_nside512_ring.fits --ebv-max 0.2 \
        --protect objects.csv --protect-radius 1.0 --median-subtract

Masked cells are dropped from the output so that downstream tools treat them
as unobserved.
"""
import numpy as np
import healpy as hp
from scipy.spatial import cKDTree


def chord(deg):
    """Chord length of an angle (deg) on the unit sphere."""
    return 2 * np.sin(np.radians(deg) / 2)


def load_cube(npz):
    """Return (pixel RING, nside, distance_modulus, TS[ncell, nmod]) from a collated .npz."""
    d = np.load(npz)
    ts = np.maximum(2 * d['log_likelihood'], 0).astype(np.float32)
    return d['pixel'], int(d['nside']), d['distance_modulus'], ts


def save_cube(npz, pixel, nside, mus, ts, keep=None):
    """Write a cube in the collate.py format; `keep` drops the masked cells."""
    if keep is not None:
        pixel, ts = pixel[keep], ts[keep]
    np.savez_compressed(npz, pixel=pixel, nside=nside, ordering='RING',
                        distance_modulus=mus, log_likelihood=(ts / 2).astype(np.float32))


def median_subtract(ts):
    """TS minus the per-cell median over modulus, clipped at zero.

    Removes the broad, modulus-independent excess of a mis-normalised
    background while keeping peaked signals."""
    return np.maximum(ts - np.median(ts, axis=1, keepdims=True), 0)


def targets_of(pixel, nside, nside_target):
    """NESTED target index of every cell and the sorted unique targets."""
    nest = hp.ring2nest(nside, pixel)
    t = nest >> (2 * int(np.log2(nside // nside_target)))
    targets = np.unique(t)
    return targets, np.searchsorted(targets, t)


def roi_density_ratio(counts, nside_counts, footprint, targets, nside_target,
                      r_in=0.5, r_out=2.0, workers=1):
    """Interior/annulus density ratio of a full-sky NESTED count map per target.

    `footprint` is a boolean full-sky NESTED map at `nside_counts` of the
    cells the scan covered (only those count towards the areas). Returns an
    array aligned with `targets` (NaN where the annulus is empty)."""
    cells = np.where(footprint)[0]
    tree = cKDTree(np.array(hp.pix2vec(nside_counts, cells, nest=True)).T)
    q = np.array(hp.pix2vec(nside_target, targets, nest=True)).T
    out = np.full(len(targets), np.nan)
    for a in range(0, len(targets), 20000):
        sums = []
        for R in (r_in, r_out):
            nbr = tree.query_ball_point(q[a:a + 20000], chord(R), workers=workers, return_sorted=False)
            n = np.array([len(x) for x in nbr])
            flat = cells[np.concatenate(nbr).astype(np.int64)] if n.sum() else np.zeros(0, int)
            seg = np.repeat(np.arange(len(n)), n)
            sums.append((np.bincount(seg, counts[flat], minlength=len(n)), n.astype(float)))
        (ci, ni), (co, no) = sums
        with np.errstate(all='ignore'):
            out[a:a + len(ni)] = (ci / ni) / ((co - ci) / (no - ni))
    return out


def ebv_mask(pixel, nside, ebv_map, ebv_max=0.2, nest=False):
    """Cells whose E(B-V) (full-sky map, RING unless `nest`) exceeds `ebv_max`."""
    ra, dec = hp.pix2ang(nside, pixel, lonlat=True)
    return ebv_map[hp.ang2pix(hp.npix2nside(len(ebv_map)), ra, dec, lonlat=True, nest=nest)] > ebv_max


def protect_objects(pixel, nside, ra, dec, radius=1.0):
    """Cells within `radius` deg of any of the given positions."""
    out = np.zeros(len(pixel), bool)
    if len(ra) == 0:
        return out
    tree = cKDTree(np.array(hp.pix2vec(nside, pixel)).T)
    idx = tree.query_ball_point(np.array(hp.ang2vec(ra, dec, lonlat=True)).reshape(-1, 3), chord(radius))
    hit = np.concatenate([np.asarray(i, int) for i in idx]) if len(idx) else np.zeros(0, int)
    out[np.unique(hit)] = True
    return out


def clean(pixel, nside, ts, ratio_masks=(), ebv=None, ebv_max=0.2,
          protect=None, protect_radius=1.0, nside_target=256, subtract_median=True):
    """Apply the recipe. `ratio_masks` is a list of (counts map NESTED at `nside`, threshold).

    Returns (cleaned TS with NaN in masked cells, mask)."""
    targets, it = targets_of(pixel, nside, nside_target)
    footprint = np.zeros(hp.nside2npix(nside), bool)
    footprint[hp.ring2nest(nside, pixel)] = True
    mask = np.zeros(len(pixel), bool)
    for counts, thr in ratio_masks:
        r = roi_density_ratio(counts, nside, footprint, targets, nside_target)
        mask |= ((r > thr) | ~np.isfinite(r))[it]
    if protect is not None:
        mask &= ~protect_objects(pixel, nside, protect[0], protect[1], protect_radius)
    if ebv is not None:
        mask |= ebv_mask(pixel, nside, ebv, ebv_max)
    out = median_subtract(ts) if subtract_median else ts.copy()
    out[mask] = np.nan
    return out, mask


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('infile', help='collated .npz (collate.py)')
    p.add_argument('outfile', help='cleaned .npz (masked cells dropped)')
    p.add_argument('--counts', action='append', default=[],
                   help='NPZFILE:KEY of a full-sky NESTED count map at the cube nside (repeatable)')
    p.add_argument('--ratio-max', action='append', type=float, default=[],
                   help='interior/annulus ratio threshold for the corresponding --counts')
    p.add_argument('--ebv', help='full-sky RING E(B-V) HEALPix map (fits)')
    p.add_argument('--ebv-max', type=float, default=0.2)
    p.add_argument('--protect', help='CSV with ra,dec columns of objects to keep')
    p.add_argument('--protect-radius', type=float, default=1.0)
    p.add_argument('--nside-target', type=int, default=256)
    p.add_argument('--median-subtract', action='store_true')
    a = p.parse_args()
    if len(a.counts) != len(a.ratio_max):
        p.error('give one --ratio-max per --counts')
    pixel, nside, mus, ts = load_cube(a.infile)
    ratio_masks = []
    for spec, thr in zip(a.counts, a.ratio_max):
        fn, key = spec.rsplit(':', 1)
        ratio_masks.append((np.load(fn)[key].astype(float), thr))
    ebv = hp.read_map(a.ebv) if a.ebv else None
    protect = None
    if a.protect:
        import pandas as pd
        t = pd.read_csv(a.protect)
        protect = (t['ra'].values, t['dec'].values)
    out, mask = clean(pixel, nside, ts, ratio_masks, ebv, a.ebv_max, protect, a.protect_radius,
                      a.nside_target, a.median_subtract)
    save_cube(a.outfile, pixel, nside, mus, out, keep=~mask)
    print('masked %.2f%% of %d cells; wrote %s' % (100 * mask.mean(), len(mask), a.outfile))


if __name__ == '__main__':
    main()
