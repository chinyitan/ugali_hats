#!/usr/bin/env python
"""
Run the likelihood scan for many pixels in parallel on a single machine.

Workers are forked once and live for the whole run, so the per-process HATS
tile cache (ugali.observation.hats_catalog) persists across pixels.  Pixels
are sorted by NESTED index and handed out in contiguous chunks so that
neighbouring pixels (which share HATS partitions) go to the same worker.
"""
import os
import sys
import csv
import time
import multiprocessing as mp
from os.path import exists, dirname

import numpy as np
import healpy as hp

import ugali.utils.config
import ugali.utils.skymap
from ugali.utils.healpix import pix2ang, ang2vec, query_disc
from ugali.utils.projector import gal2cel
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

STAT_FIELDS = ['pix', 'lon', 'lat', 'runtime_s', 'hits', 'misses',
               'bytes_read', 'rows_read', 'read_time', 'pid', 'status', 'error']

# Per-process state, set by _init (before fork, inherited by workers).
_STATE = {}


def select_pixels(configfile, coords=None, pixels=None, minpix=None,
                  maxpix=None, hpxlist=None):
    """ Select likelihood pixels, mirroring Farm.submit_all. """
    from ugali.analysis.farm import Farm
    farm = Farm(configfile)
    config = farm.config
    nside = farm.nside_likelihood

    if pixels is not None:
        pixels = np.asarray(pixels, dtype=int)
    elif coords is None:
        pixels = np.arange(hp.nside2npix(nside))
    else:
        lon, lat, radius = coords['lon'], coords['lat'], coords['radius']
        if config['coords']['coordsys'].lower() == 'cel':
            lon, lat = gal2cel(lon, lat)
        vec = ang2vec(lon, lat)
        pixels = np.zeros(0, dtype=int)
        for v, r in zip(vec, radius):
            pixels = np.hstack([pixels, query_disc(nside, v, r, inclusive=True, fact=32)])
        pixels = np.unique(pixels)

    inside = ugali.utils.skymap.inFootprint(config, pixels)
    if inside.sum() != len(pixels):
        logger.warning("Ignoring pixels outside survey footprint:\n" + str(pixels[~inside]))
    pixels = pixels[inside]

    if minpix is not None: pixels = pixels[pixels >= minpix]
    if maxpix is not None: pixels = pixels[pixels < maxpix]
    if hpxlist is not None:
        if hpxlist.endswith('.npy'): hpx = np.load(hpxlist)
        else: hpx = np.loadtxt(hpxlist, dtype=int, ndmin=1)
        pixels = pixels[np.isin(pixels, hpx)]
    return pixels


def sort_nested(nside, pixels):
    """ Sort RING pixels by NESTED index (spatial locality). """
    pixels = np.asarray(pixels, dtype=int)
    nest = hp.ring2nest(nside, pixels)
    return pixels[np.argsort(nest, kind='stable')]


def _cache_snapshot(config):
    """ Return reader.stats copy, or None if the config has no HATS catalog. """
    if 'hats' not in (config['catalog'] or {}):
        return None
    from ugali.observation.hats_catalog import get_reader
    return dict(get_reader(config).stats)


def scan_pixel(pix):
    """ Scan one pixel using the process-global state. Never raises. """
    from ugali.analysis.scan import createGridSearch
    config = _STATE['config']
    nside = _STATE['nside']
    force = _STATE['force']

    outfile = config.likefile % (pix, config['coords']['coordsys'].lower())
    rec = dict(pix=int(pix), lon=np.nan, lat=np.nan, runtime_s=0.0, hits=0,
               misses=0, bytes_read=0, rows_read=0, read_time=0.0,
               pid=os.getpid(), status='ok', error='')
    if exists(outfile) and not force:
        rec['status'] = 'skipped'
        return rec

    t0 = time.time()
    try:
        # Same convention as scan.py --hpx: pix2ang passed straight through.
        lon, lat = pix2ang(nside, pix)
        rec['lon'], rec['lat'] = float(lon), float(lat)
        before = _cache_snapshot(config)
        # Pass the Config object (not the filename) so that overrides such
        # as --max-tiles reach the workers and the YAML is parsed only once.
        grid = createGridSearch(config, lon, lat)
        grid.search()
        mkdir(dirname(outfile))
        grid.write(outfile)
        after = _cache_snapshot(config)
        if before is not None:
            for k in ('hits', 'misses', 'bytes_read', 'rows_read', 'read_time'):
                rec[k] = after[k] - before[k]
    except Exception as e:
        import traceback
        logger.error("Pixel %i failed:\n%s" % (pix, traceback.format_exc()))
        rec['status'] = 'failed'
        rec['error'] = '%s: %s' % (type(e).__name__, e)
    rec['runtime_s'] = time.time() - t0
    return rec


def run_scan(configfile, pixels, n_jobs=1, chunksize=None, force=False,
             max_tiles=None, stats_file=None):
    """ Scan `pixels` with n_jobs workers. Returns list of per-pixel records. """
    config = ugali.utils.config.Config(configfile)
    if max_tiles is not None:
        if 'hats' not in (config['catalog'] or {}):
            logger.warning("--max-tiles ignored: no catalog.hats block")
        else:
            config['catalog']['hats']['max_tiles'] = int(max_tiles)
    nside = config['coords']['nside_likelihood']
    if n_jobs is None or n_jobs < 0:
        n_jobs = os.cpu_count() or 1

    pixels = sort_nested(nside, pixels)
    npix = len(pixels)
    n_jobs = max(1, min(n_jobs, npix)) if npix else 1
    if chunksize is None:
        chunksize = max(1, npix // (n_jobs * 4))

    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, '1')
    _STATE.update(configfile=configfile, config=config, nside=nside, force=force)

    logger.info("Scanning %i pixels with %i worker(s), chunksize=%i" % (npix, n_jobs, chunksize))
    t0 = time.time()
    records = []
    if n_jobs == 1:
        it = map(scan_pixel, pixels)
        pool = None
    else:
        pool = mp.get_context('fork').Pool(n_jobs)
        it = pool.imap_unordered(scan_pixel, pixels, chunksize=chunksize)
    try:
        for i, rec in enumerate(it, 1):
            records.append(rec)
            logger.info("[%i/%i] pix=%i %s %.1fs (hits=%i misses=%i) elapsed=%.0fs" % (
                i, npix, rec['pix'], rec['status'], rec['runtime_s'],
                rec['hits'], rec['misses'], time.time() - t0))
    finally:
        if pool is not None:
            pool.close(); pool.join()
    total = time.time() - t0

    if stats_file:
        mkdir(dirname(os.path.abspath(stats_file)))
        with open(stats_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=STAT_FIELDS)
            w.writeheader()
            for r in sorted(records, key=lambda r: r['pix']):
                w.writerow(r)
        logger.info("Wrote stats to %s" % stats_file)

    summarize(records, total)
    return records


def summarize(records, total):
    ran = [r for r in records if r['status'] == 'ok']
    nfail = sum(r['status'] == 'failed' for r in records)
    nskip = sum(r['status'] == 'skipped' for r in records)
    hits = sum(r['hits'] for r in ran); misses = sum(r['misses'] for r in ran)
    mb = sum(r['bytes_read'] for r in ran) / 1e6
    rate = hits / float(hits + misses) if (hits + misses) else float('nan')
    print("=== Scan summary ===")
    print("pixels: %i ok, %i skipped, %i failed" % (len(ran), nskip, nfail))
    print("total time: %.1fs  (%.4f pixels/s)" % (total, len(ran) / total if total > 0 else 0))
    if ran:
        print("mean per-pixel runtime: %.1fs" % np.mean([r['runtime_s'] for r in ran]))
    print("cache: hits=%i misses=%i hit_rate=%.3f  read=%.1f MB" % (hits, misses, rate, mb))
    for r in records:
        if r['status'] == 'failed':
            print("FAILED pix=%i: %s" % (r['pix'], r['error']))


if __name__ == "__main__":
    import ugali.utils.parser
    parser = ugali.utils.parser.Parser(description=__doc__)
    parser.add_config()
    parser.add_verbose()
    parser.add_pixels()
    parser.add_coords(required=False, radius=True, targets=True)
    parser.add_argument('--pixels', nargs='+', type=int, default=None,
                        help='Explicit likelihood pixel indices (RING).')
    parser.add_argument('-n', '--n-jobs', type=int, default=1,
                        help='Number of worker processes (-1 = all CPUs).')
    parser.add_argument('--chunksize', type=int, default=None,
                        help='Pixels per task (default npix//(4*n_jobs)).')
    parser.add_argument('--max-tiles', type=int, default=None,
                        help='Override catalog.hats.max_tiles.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Re-run pixels whose output already exists.')
    parser.add_argument('--stats-file', default=None,
                        help='Write per-pixel stats CSV here.')
    opts = parser.parse_args()

    pixels = select_pixels(opts.config, coords=opts.coords, pixels=opts.pixels,
                           minpix=opts.minpix, maxpix=opts.maxpix, hpxlist=opts.hpxlist)
    logger.info("Selected %i pixels" % len(pixels))
    if opts.pixfile is not None:
        np.savetxt(opts.pixfile, pixels, fmt='%d')
        logger.info("Wrote pixels to %s" % opts.pixfile)
        sys.exit(0)
    if len(pixels) == 0:
        logger.warning("No pixels to scan.")
        sys.exit(0)
    recs = run_scan(opts.config, pixels, n_jobs=opts.n_jobs, chunksize=opts.chunksize,
                    force=opts.force, max_tiles=opts.max_tiles, stats_file=opts.stats_file)
    sys.exit(1 if any(r['status'] == 'failed' for r in recs) else 0)
