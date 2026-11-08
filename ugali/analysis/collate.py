#!/usr/bin/env python
"""
Collate the per-pixel outputs of a likelihood scan into dense arrays.

Each scan_parallel output file holds the likelihood of the nside_pixel cells
of one target pixel at every distance modulus. This script reads all of them
in parallel and writes one .npz with

  pixel             : HEALPix index (RING, nside = NSIDE) of every cell
  distance_modulus  : the scanned distance moduli
  log_likelihood    : (n_cells, n_modulus) float32  (TS = 2 * log_likelihood)
  richness          : (n_cells, n_modulus) float32
  fraction_observable : (n_cells, n_modulus) float32
  nside, ordering   : of `pixel`

which is what the search / plotting steps consume. Reading ~300k FITS files
serially takes hours; with 48 workers it takes a few minutes.

  python ugali/analysis/collate.py scan-stream/ ts_maps.npz -n 48
"""
import os, sys, glob, time, argparse
import numpy as np
from multiprocessing import Pool

import fitsio

COLUMNS = ['LOG_LIKELIHOOD', 'RICHNESS', 'FRACTION_OBSERVABLE']


def read_one(filename):
    try:
        with fitsio.FITS(filename) as f:
            h = f['PIX_DATA'].read_header()
            d = f['PIX_DATA'].read()
            dm = f['DISTANCE_MODULUS'].read()['DISTANCE_MODULUS']
        return (filename, int(h['NSIDE']), h['ORDERING'], d['PIXEL'],
                [d[c] for c in COLUMNS], dm, None)
    except Exception as e:
        return (filename, None, None, None, None, None, '%s: %s' % (type(e).__name__, e))


def collate(filenames, njobs=1, chunksize=64):
    pixels, cols, dm0, nside0, order0 = [], [[] for _ in COLUMNS], None, None, None
    failed = []
    t0 = time.time()
    with Pool(njobs) as pool:
        for i, (fn, nside, order, pix, data, dm, err) in enumerate(
                pool.imap_unordered(read_one, filenames, chunksize=chunksize)):
            if err:
                failed.append((fn, err)); continue
            if dm0 is None:
                dm0, nside0, order0 = dm, nside, order
            else:
                assert nside == nside0 and order == order0, fn
                assert np.array_equal(dm, dm0), fn
            pixels.append(pix)
            for c, d in zip(cols, data): c.append(d)
            if (i + 1) % 20000 == 0:
                print('  %d/%d files (%.0f s)' % (i + 1, len(filenames), time.time() - t0), flush=True)
    pixels = np.concatenate(pixels)
    out = {c.lower(): np.concatenate(d).astype(np.float32) for c, d in zip(COLUMNS, cols)}
    # sort by pixel so that neighbours are adjacent
    idx = np.argsort(pixels, kind='stable')
    out = {k: v[idx] for k, v in out.items()}
    out.update(pixel=pixels[idx], distance_modulus=dm0, nside=nside0, ordering=order0)
    return out, failed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('likedir', help='directory of scan_*.fits files (or a glob)')
    p.add_argument('outfile', help='output .npz')
    p.add_argument('-n', '--n-jobs', type=int, default=8)
    args = p.parse_args()

    pattern = args.likedir if '*' in args.likedir else os.path.join(args.likedir, 'scan_*.fits')
    filenames = sorted(glob.glob(pattern))
    print('%d files' % len(filenames))
    t0 = time.time()
    out, failed = collate(filenames, args.n_jobs)
    np.savez(args.outfile, **out)
    print('wrote %s: %d cells x %d moduli, nside %d %s (%.0f s); %d unreadable files' % (
        args.outfile, len(out['pixel']), len(out['distance_modulus']), out['nside'],
        out['ordering'], time.time() - t0, len(failed)))
    for fn, err in failed[:20]:
        print('  FAILED', fn, err)
