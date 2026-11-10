#!/usr/bin/env python
"""
Run the likelihood scan for many pixels in parallel on a single machine.

Workers are forked once and live for the whole run, so the per-process HATS
tile cache (ugali.observation.hats_catalog) persists across pixels.  Pixels
are sorted by NESTED index and handed out in contiguous chunks so that
neighbouring pixels (which share HATS partitions) go to the same worker.

Designed for long, resumable runs:
  * each pixel's output is written atomically (temp file + rename), and
    pixels whose output exists are skipped unless --force;
  * per-pixel statistics are appended to the stats file as pixels finish;
  * a worker that dies (OOM kill, segfault) does not hang the run: the pool
    is restarted and the unfinished chunks are resubmitted (up to
    --max-restarts times), relying on the resume logic to skip what is done;
  * the default number of workers is limited by the available memory.
"""
import os
# numpy asks the kernel for transparent huge pages on every allocation >= 4 MB.
# On a shared node with a large, fragmented page cache every such fault can
# stall in direct memory compaction: in dense fields 96 workers were measured
# at ~96% system CPU and 0.5 pixels/s, versus 15-30 pixels/s with the request
# disabled (/proc/vmstat: compact_stall, thp_fault_alloc). Must be set before
# numpy is imported; an explicit value in the environment is respected.
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import sys
import csv
import time
import signal
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from os.path import exists, dirname

# One BLAS/OpenMP thread per worker: the scan is made of small numpy
# operations (e.g. the 3x3 solves of the richness parabola), and a
# multi-threaded BLAS in every worker oversubscribes the machine (measured
# 3x slower with 48 workers). Set before numpy is imported so that it takes
# effect when this module is run as a script; an explicit setting in the
# environment is respected.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import healpy as hp

import ugali.utils.config
import ugali.utils.skymap
from ugali.utils.healpix import pix2ang, ang2vec, query_disc
from ugali.utils.projector import gal2cel
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

STAT_FIELDS = ['pix', 'lon', 'lat', 'runtime_s', 'setup_s', 'search_s',
               'hits', 'misses', 'bytes_read', 'rows_read', 'read_time',
               'pid', 'status', 'error']

# Memory needed per worker for a DES-like scan (measured ~3.6 GB RSS)
MEMORY_PER_WORKER_GB = 4.0

# Per-process state, set before forking and inherited by the workers.
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
        logger.warning("Ignoring %i pixels outside survey footprint" % (~inside).sum())
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


def available_memory_gb():
    """ Available system memory in GB (Linux /proc/meminfo), or None. """
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return float(line.split()[1]) / 1024**2
    except (IOError, ValueError):
        pass
    return None


def default_n_jobs(requested=None):
    """ Number of workers: requested, or all CPUs limited by memory. """
    ncpu = os.cpu_count() or 1
    n = ncpu if (requested is None or requested < 0) else max(1, requested)
    mem = available_memory_gb()
    if mem is not None:
        nmem = max(1, int(mem // MEMORY_PER_WORKER_GB))
        if requested is None or requested < 0:
            n = min(n, nmem)
        elif n > nmem:
            logger.warning("%i workers x ~%.0f GB exceeds available memory (%.0f GB)"
                           % (n, MEMORY_PER_WORKER_GB, mem))
    return n


def _cache_snapshot(config):
    """ Return a copy of reader.stats, or None if the config has no HATS catalog. """
    if 'hats' not in (config['catalog'] or {}):
        return None
    from ugali.observation.hats_catalog import get_reader
    return dict(get_reader(config).stats)


def _outfile(config, pix):
    return config.likefile % (pix, config['coords']['coordsys'].lower())


def _append_stats(rec):
    """ Append one record to the stats file (workers share the lock). """
    stats_file = _STATE.get('stats_file')
    if not stats_file:
        return
    lock = _STATE.get('lock')
    if lock is not None: lock.acquire()
    try:
        with open(stats_file, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=STAT_FIELDS).writerow(rec)
            f.flush()
    finally:
        if lock is not None: lock.release()


def scan_pixel(pix):
    """ Scan one pixel using the process-global state. Never raises. """
    from ugali.analysis.scan import createGridSearch
    config = _STATE['config']
    nside = _STATE['nside']
    force = _STATE['force']

    outfile = _outfile(config, pix)
    rec = dict(pix=int(pix), lon=np.nan, lat=np.nan, runtime_s=0.0,
               setup_s=0.0, search_s=0.0, hits=0, misses=0, bytes_read=0,
               rows_read=0, read_time=0.0, pid=os.getpid(), status='ok', error='')
    if exists(outfile) and not force:
        rec['status'] = 'skipped'
        return rec

    t0 = time.time()
    tmpfile = outfile + '.tmp'
    try:
        # Same convention as scan.py --hpx: pix2ang passed straight through.
        lon, lat = pix2ang(nside, pix)
        rec['lon'], rec['lat'] = float(lon), float(lat)
        before = _cache_snapshot(config)
        # The Config object (not the filename) is passed so that overrides
        # such as --max-tiles reach the workers.
        grid = createGridSearch(config, lon, lat)
        rec['setup_s'] = time.time() - t0
        t1 = time.time()
        grid.search()
        rec['search_s'] = time.time() - t1
        mkdir(dirname(outfile))
        # Atomic write: a crash can't leave a truncated output behind
        grid.write(tmpfile)
        os.replace(tmpfile, outfile)
        after = _cache_snapshot(config)
        if before is not None:
            for k in ('hits', 'misses', 'bytes_read', 'rows_read', 'read_time'):
                rec[k] = after[k] - before[k]
    except Exception as e:
        logger.error("Pixel %i failed:\n%s" % (pix, traceback.format_exc()))
        rec['status'] = 'failed'
        rec['error'] = '%s: %s' % (type(e).__name__, e)
        if exists(tmpfile):
            try: os.remove(tmpfile)
            except OSError: pass
    rec['runtime_s'] = time.time() - t0
    _append_stats(rec)
    return rec


def scan_chunk(pixels):
    """ Scan a contiguous chunk of pixels in this worker; returns the records. """
    return [scan_pixel(pix) for pix in pixels]


def _init_worker():
    # Let the parent handle Ctrl-C; workers are cleaned up by the executor.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Pin the BLAS thread pool inherited from the parent to one thread
    # (the environment variables above only help if numpy was imported
    # after they were set, e.g. when this module is run as a script).
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass


def _make_pool(n_jobs, max_tasks_per_child=None):
    kwargs = dict(max_workers=n_jobs, mp_context=mp.get_context('fork'),
                  initializer=_init_worker)
    if max_tasks_per_child:
        try:
            return ProcessPoolExecutor(max_tasks_per_child=max_tasks_per_child, **kwargs)
        except TypeError:  # python < 3.11
            logger.warning("max_tasks_per_child requires python >= 3.11; ignored")
    return ProcessPoolExecutor(**kwargs)


def _open_stats(stats_file, resume):
    """ Create the stats file with a header unless resuming an existing one. """
    if not stats_file:
        return
    mkdir(dirname(os.path.abspath(stats_file)))
    if resume and exists(stats_file) and os.path.getsize(stats_file) > 0:
        logger.info("Appending to existing stats file %s" % stats_file)
        return
    with open(stats_file, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=STAT_FIELDS).writeheader()


def run_scan(configfile, pixels, n_jobs=1, chunksize=None, force=False,
             max_tiles=None, stats_file=None, max_restarts=3,
             max_tasks_per_child=None):
    """ Scan `pixels` with n_jobs workers. Returns the list of per-pixel records.

    Parameters
    ----------
    configfile : config file (or Config/dict)
    pixels     : RING likelihood pixel indices
    n_jobs     : worker processes (1 = serial in-process; <0 = memory-limited CPU count)
    chunksize  : pixels per task (default npix // (4*n_jobs))
    force      : re-run pixels whose output exists
    max_tiles  : override catalog.hats.max_tiles
    stats_file : CSV appended per pixel (kept when resuming)
    max_restarts : pool restarts allowed after a worker dies
    max_tasks_per_child : recycle workers after this many chunks (python >= 3.11)
    """
    config = ugali.utils.config.Config(configfile)
    if max_tiles is not None:
        if 'hats' not in (config['catalog'] or {}):
            logger.warning("--max-tiles ignored: no catalog.hats block")
        else:
            config['catalog']['hats']['max_tiles'] = int(max_tiles)
    nside = config['coords']['nside_likelihood']

    pixels = sort_nested(nside, pixels)
    npix = len(pixels)
    n_jobs = default_n_jobs(n_jobs)
    n_jobs = max(1, min(n_jobs, npix)) if npix else 1
    if chunksize is None:
        chunksize = max(1, npix // (n_jobs * 4))

    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, '1')
    _open_stats(stats_file, resume=not force)
    _STATE.update(config=config, nside=nside, force=force, stats_file=stats_file,
                  lock=mp.get_context('fork').Lock() if n_jobs > 1 else None)

    ndone = int(np.sum([exists(_outfile(config, p)) for p in pixels])) if not force else 0
    logger.info("Scanning %i pixels (%i already done) with %i worker(s), chunksize=%i"
                % (npix, ndone, n_jobs, chunksize))
    t0 = time.time()
    records = []

    def report(rec, i):
        logger.info("[%i/%i] pix=%i %s %.1fs (hits=%i misses=%i) elapsed=%.0fs" % (
            i, npix, rec['pix'], rec['status'], rec['runtime_s'],
            rec['hits'], rec['misses'], time.time() - t0))

    if n_jobs == 1:
        for i, pix in enumerate(pixels, 1):
            rec = scan_pixel(pix)
            records.append(rec)
            report(rec, i)
    else:
        chunks = [pixels[i:i + chunksize] for i in range(0, npix, chunksize)]
        pending = list(range(len(chunks)))
        restarts = 0
        while pending:
            pool = _make_pool(n_jobs, max_tasks_per_child)
            futures = {pool.submit(scan_chunk, chunks[i]): i for i in pending}
            try:
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        i = futures.pop(fut)
                        recs = fut.result()   # raises BrokenProcessPool if a worker died
                        records.extend(recs)
                        pending.remove(i)
                        for rec in recs:
                            report(rec, len(records))
                pool.shutdown(wait=True)
            except BrokenProcessPool:
                pool.shutdown(wait=False)
                restarts += 1
                if restarts > max_restarts:
                    logger.error("A worker died %i times; giving up. Re-run to resume "
                                 "(%i chunks unfinished)." % (restarts, len(pending)))
                    break
                logger.error("A worker died (OOM/segfault?); restarting the pool "
                             "(%i/%i) and resubmitting %i unfinished chunks"
                             % (restarts, max_restarts, len(pending)))
                time.sleep(5)
            except KeyboardInterrupt:
                logger.warning("Interrupted; cancelling remaining work (re-run to resume)")
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    total = time.time() - t0

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
        print("mean per-pixel runtime: %.1fs (setup %.1fs, search %.1fs)" % (
            np.mean([r['runtime_s'] for r in ran]),
            np.mean([r.get('setup_s', 0.) for r in ran]),
            np.mean([r.get('search_s', 0.) for r in ran])))
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
                        help='Worker processes (-1 = all CPUs, limited by memory).')
    parser.add_argument('--chunksize', type=int, default=None,
                        help='Pixels per task (default npix//(4*n_jobs)).')
    parser.add_argument('--max-tiles', type=int, default=None,
                        help='Override catalog.hats.max_tiles.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Re-run pixels whose output already exists.')
    parser.add_argument('--stats-file', default=None,
                        help='Per-pixel stats CSV, appended as pixels finish.')
    parser.add_argument('--max-restarts', type=int, default=3,
                        help='Pool restarts allowed after a worker dies.')
    parser.add_argument('--max-tasks-per-child', type=int, default=None,
                        help='Recycle a worker after this many chunks (python >= 3.11).')
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
                    force=opts.force, max_tiles=opts.max_tiles, stats_file=opts.stats_file,
                    max_restarts=opts.max_restarts, max_tasks_per_child=opts.max_tasks_per_child)
    nfail = sum(r['status'] == 'failed' for r in recs)
    sys.exit(1 if (nfail or len(recs) < len(pixels)) else 0)
