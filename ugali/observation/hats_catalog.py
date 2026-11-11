"""
Read object catalogs stored in the HATS format (hierarchical healpix
partitioned parquet, https://hats.readthedocs.io) with a per-process
LRU cache of partitions.

This is an alternative *input* path for ugali: when the configuration
contains a ``catalog: hats:`` block, :class:`ugali.observation.catalog.Catalog`
loads its data through :func:`load_catalog_pixels` instead of the
per-pixel FITS files. Everything downstream (ROI cuts, masks, likelihood)
is unchanged, and the rows returned are exactly the rows that the FITS
path would read for the same set of catalog pixels (the selection cut
is applied per partition before caching, which is equivalent to ugali's
usual post-load selection).

Only ``pyarrow`` and ``healpy`` are required; the catalog layout
(``<path>/partition_info.csv`` and
``<path>/dataset/Norder=<o>/Dir=<d>/Npix=<p>.parquet``) is read
directly so that the reader works with any ``hats`` library version.

Configuration::

    catalog:
      hats:
        path: /astro/store/shire/hats/catalogs/des/des_y6_gold/des_y6_gold
        columns: null   # optional explicit column list (default: the
                        # fields named in the catalog section)
        max_tiles: 64   # LRU size in partitions (None = unbounded)
        derived:        # optional columns computed per partition with numpy
          G_DERED: "g_psfMag - 3.64 * ebv"   # (before the selection cut)
      ... the usual catalog keys (lon_field, mag_1_field, selection, ...)

Usage::

    reader = get_reader(config)             # one per process, cached
    data = reader.load_pixels(pixels, nside) # rows in those healpix pixels
    reader.stats                             # hits/misses/bytes/time
"""
import os
import re
import time
import threading
from collections import OrderedDict

import numpy as np
import healpy as hp

from ugali.utils.logger import logger


def tile_path(catalog_path, order, pixel):
    """Path of a HATS partition file (Dir groups 10,000 pixels)."""
    d = (int(pixel) // 10000) * 10000
    return os.path.join(catalog_path, 'dataset',
                        'Norder=%d' % order, 'Dir=%d' % d, 'Npix=%d.parquet' % pixel)


def read_partition_info(catalog_path):
    """Return (orders, pixels) int arrays of the partitions in a HATS catalog."""
    fname = os.path.join(catalog_path, 'partition_info.csv')
    if not os.path.exists(fname):
        raise IOError("Not a HATS catalog (no partition_info.csv): %s" % catalog_path)
    data = np.genfromtxt(fname, delimiter=',', names=True, dtype=int)
    data = np.atleast_1d(data)
    return data['Norder'].astype(int), data['Npix'].astype(int)


class _SelectionData(object):
    """Minimal stand-in for ``self`` when evaluating a config selection string."""
    def __init__(self, data):
        self.data = data


def pa_types():
    import pyarrow.types
    return pyarrow.types


_IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def derived_inputs(derived):
    """Identifiers referenced by the ``derived`` expressions (candidate input columns)."""
    names = []
    for expr in (derived or {}).values():
        for n in _IDENT.findall(str(expr)):
            if n not in names: names.append(n)
    return names


def add_derived(data, derived):
    """Append columns computed from ``derived`` = {name: expression}.

    Each expression is evaluated with numpy (available as ``np``) and the
    columns of `data` as variables, in the order given, so later expressions
    may use earlier derived columns. Returns a new structured array.
    """
    if not derived:
        return data
    names = list(data.dtype.names)
    env = {name: data[name] for name in names}
    new = []
    for name, expr in derived.items():
        # Invalid inputs (e.g. log10 of a non-positive flux) legitimately give
        # NaN; silence the floating-point warnings, whose machinery would
        # otherwise need __import__, which the restricted builtins remove.
        with np.errstate(all='ignore'):
            val = eval(str(expr), {'np': np, '__builtins__': {}}, env)
        val = np.broadcast_to(np.asarray(val), (len(data),))
        env[name] = val
        new.append((name, val))
    dtype = [(n, data.dtype[n]) for n in names if n not in dict(new)]
    dtype += [(n, v.dtype) for n, v in new]
    out = np.empty(len(data), dtype=dtype)
    for n in names:
        if n not in dict(new): out[n] = data[n]
    for n, v in new:
        out[n] = v
    return out


def table_to_recarray(table):
    """Convert a pyarrow Table to a numpy structured array (native byte order)."""
    arrays, dtype = [], []
    for name in table.column_names:
        col = table.column(name)
        if col.null_count and pa_types().is_boolean(col.type):
            # null flags would become an object array; treat them as False
            arr = col.fill_null(False).to_numpy(zero_copy_only=False)
        elif col.null_count:
            arr = col.to_pandas().to_numpy()
        else:
            arr = col.to_numpy(zero_copy_only=False)
        arr = np.ascontiguousarray(arr)
        arrays.append(arr)
        dtype.append((name, arr.dtype))
    out = np.empty(table.num_rows, dtype=dtype)
    for name, arr in zip([d[0] for d in dtype], arrays):
        out[name] = arr
    return out


class HatsCatalogReader(object):
    """Read HATS partitions with a least-recently-used cache.

    Parameters
    ----------
    path      : HATS catalog directory (the one containing partition_info.csv)
    columns   : list of column names to read (None = all)
    selection : optional ugali-style selection string evaluated on each
                partition before caching, e.g. "(self.data['FLAGS_GOLD'] == 0)"
    max_tiles : maximum number of partitions kept in the cache (None = unbounded)
    derived   : optional {name: numpy expression} of columns computed from
                the loaded columns of each partition before the selection
                (see :func:`add_derived`); `columns` must include their inputs
    """

    def __init__(self, path, columns=None, selection=None, max_tiles=64, derived=None):
        self.path = path
        self.derived = dict(derived) if derived else None
        self.columns = list(columns) if columns is not None else None
        self.selection = selection
        self.max_tiles = max_tiles

        self.orders, self.pixels = read_partition_info(path)
        self._keys = set(zip(self.orders.tolist(), self.pixels.tolist()))
        self._orders = np.unique(self.orders)

        self._cache = OrderedDict()   # (order,pixel) -> structured array
        self._lock = threading.Lock()
        self.reset_stats()

    # ------------------------------------------------------------------
    # Partition lookup (mixed-order aware)
    # ------------------------------------------------------------------
    def partitions_for_pixels(self, pixels, nside, nest=False):
        """HATS partitions overlapping the given healpix pixels.

        Works for catalogs with partitions at several orders: a partition
        overlaps a pixel if one is an ancestor (or equal) of the other in
        the NESTED scheme.

        Parameters
        ----------
        pixels : healpix pixel indices
        nside  : nside of those pixels
        nest   : whether `pixels` are in the NESTED scheme (default RING)

        Returns
        -------
        keys : sorted list of (order, pixel) tuples
        """
        pixels = np.unique(np.atleast_1d(pixels))
        if len(pixels) == 0:
            return []
        if not nest:
            pixels = hp.ring2nest(nside, pixels)
        order = int(np.log2(nside))
        keys = set()
        for o in self._orders:
            o = int(o)
            if o <= order:
                # partition is an ancestor of (or equal to) the pixel
                parents = np.unique(pixels >> (2 * (order - o)))
                for p in parents.tolist():
                    if (o, p) in self._keys:
                        keys.add((o, p))
            else:
                # partition is a descendant of the pixel
                sel = (self.orders == o)
                parents_of_parts = self.pixels[sel] >> (2 * (o - order))
                hit = np.isin(parents_of_parts, pixels)
                for p in self.pixels[sel][hit].tolist():
                    keys.add((o, p))
        return sorted(keys)

    def partitions_for_cone(self, lon, lat, radius):
        """HATS partitions overlapping a cone (deg) around (lon,lat) (deg)."""
        order = int(self._orders.max())
        nside = hp.order2nside(order)
        vec = hp.ang2vec(lon, lat, lonlat=True)
        pix = hp.query_disc(nside, vec, np.radians(radius), inclusive=True, fact=4, nest=True)
        return self.partitions_for_pixels(pix, nside, nest=True)

    # ------------------------------------------------------------------
    # Reading and caching
    # ------------------------------------------------------------------
    def _read_tile(self, key):
        import pyarrow.parquet as pq
        order, pixel = key
        fname = tile_path(self.path, order, pixel)
        t0 = time.time()
        pf = pq.ParquetFile(fname)
        table = pf.read(columns=self.columns)
        data = table_to_recarray(table)
        data = add_derived(data, self.derived)
        nbytes = table.nbytes  # bytes of the columns actually read (uncompressed)
        if self.selection:
            sel = eval(self.selection, {'np': np}, {'self': _SelectionData(data)})
            data = data[sel]
        self.stats['misses'] += 1
        self.stats['bytes_read'] += nbytes
        self.stats['rows_read'] += len(data)
        self.stats['read_time'] += time.time() - t0
        logger.debug("Read HATS tile %s: %i rows (%.2fs)" % (str(key), len(data), time.time() - t0))
        return data

    def get_tile(self, key):
        """Return the (selected) rows of one partition, from cache if possible."""
        key = (int(key[0]), int(key[1]))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.stats['hits'] += 1
                return self._cache[key]
        data = self._read_tile(key)
        with self._lock:
            self._cache[key] = data
            self._cache.move_to_end(key)
            self._evict()
        return data

    def _evict(self):
        if self.max_tiles is None:
            return
        while len(self._cache) > self.max_tiles:
            self._cache.popitem(last=False)
            self.stats['evictions'] += 1

    def load_tiles(self, keys):
        """Concatenate the rows of several partitions (sorted by key)."""
        keys = sorted(set((int(o), int(p)) for o, p in keys))
        if self.max_tiles is not None and len(keys) > self.max_tiles:
            logger.warning("Requested %i tiles exceeds cache size %i; cache will thrash"
                           % (len(keys), self.max_tiles))
        parts = [self.get_tile(k) for k in keys]
        parts = [p for p in parts if len(p)]
        if not parts:
            if keys:
                # keep the dtype of an (empty) partition
                return self.get_tile(keys[0])[:0]
            return None
        return np.concatenate(parts)

    def load_pixels(self, pixels, nside, nest=False):
        """Rows located in the given healpix pixels.

        Partitions overlapping the pixels are loaded (through the cache)
        and the rows are then restricted to the pixels themselves, so the
        result does not depend on the partition order of the catalog.

        Returns
        -------
        data : structured array (possibly empty); None if no partition overlaps
        """
        t0 = time.time()
        keys = self.partitions_for_pixels(pixels, nside, nest=nest)
        data = self.load_tiles(keys)
        if data is None or len(data) == 0:
            return data
        lon, lat = data[self.lon_field], data[self.lat_field]
        pix = hp.ang2pix(nside, lon, lat, nest=nest, lonlat=True)
        data = data[np.isin(pix, np.atleast_1d(pixels))]
        self.stats['requests'] += 1
        self.stats['request_time'] += time.time() - t0
        return data

    # The coordinate columns are set by `get_reader` from the config; defaults
    # follow the HATS convention.
    lon_field = 'RA'
    lat_field = 'DEC'

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def reset_stats(self):
        self.stats = dict(hits=0, misses=0, evictions=0, requests=0,
                          bytes_read=0, rows_read=0, read_time=0.0, request_time=0.0)

    @property
    def hit_rate(self):
        n = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / float(n) if n else np.nan

    def __len__(self):
        return len(self._cache)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __repr__(self):
        return "%s(path=%s, tiles=%i/%s, hits=%i, misses=%i, read=%.1f MB in %.1fs)" % (
            self.__class__.__name__, self.path, len(self._cache), self.max_tiles,
            self.stats['hits'], self.stats['misses'], self.stats['bytes_read'] / 1e6,
            self.stats['read_time'])


# ----------------------------------------------------------------------
# Config integration
# ----------------------------------------------------------------------
_READERS = {}


def hats_config(config):
    """Return the ``catalog: hats:`` block of a config, or None."""
    cat = config['catalog'] if 'catalog' in config else {}
    return cat.get('hats') or None


def default_columns(config):
    """Catalog columns needed by ugali for this config."""
    cat = config['catalog']
    cols = [cat[k] for k in ['objid_field', 'lon_field', 'lat_field',
                             'mag_1_field', 'mag_err_1_field',
                             'mag_2_field', 'mag_err_2_field']]
    for k in ['mc_source_id_field']:
        if cat.get(k): cols.append(cat[k])
    # Columns referenced by the selection string
    selection = cat.get('selection') or ''
    import re
    cols += re.findall(r"self\.data\['([^']+)'\]", selection)
    cols += re.findall(r'self\.data\["([^"]+)"\]', selection)
    # Derived columns are computed, not read: read their inputs instead
    # (identifiers that are not catalog columns are dropped by get_reader)
    derived = (cat.get('hats') or {}).get('derived') or {}
    cols = [c for c in cols if c not in derived] + derived_inputs(derived)
    # Keep order, drop duplicates
    out = []
    for c in cols:
        if c not in out and c not in derived and c != 'np': out.append(c)
    return out


def get_reader(config):
    """Reader for the config's HATS catalog, created once per process.

    The cache is keyed by the catalog path and the column list, so
    repeated Scan/Catalog objects in one process share the LRU cache.
    """
    hcfg = hats_config(config)
    if hcfg is None:
        raise ValueError("Config has no 'catalog: hats:' block")
    path = os.path.expandvars(hcfg['path'])
    columns = tuple(hcfg.get('columns') or default_columns(config))
    derived = hcfg.get('derived') or None
    key = (path, columns, config['catalog'].get('selection') or None,
           hcfg.get('max_tiles', 64),
           tuple((k, str(v)) for k, v in derived.items()) if derived else None)
    if key not in _READERS:
        # Columns that may not exist in the catalog (e.g. MC_SOURCE_ID) are
        # added by Catalog._defineVariables; drop them here.
        import pyarrow.parquet as pq
        orders, pixels = read_partition_info(path)
        schema = pq.ParquetFile(tile_path(path, orders[0], pixels[0])).schema_arrow
        inputs = set(derived_inputs(derived))
        missing = [c for c in columns if c not in schema.names and c not in inputs]
        if missing:
            logger.warning("Columns not in HATS catalog (ignored): %s" % missing)
        use = [c for c in columns if c in schema.names]
        reader = HatsCatalogReader(path, columns=use, selection=key[2], max_tiles=key[3],
                                   derived=derived)
        reader.lon_field = config['catalog']['lon_field']
        reader.lat_field = config['catalog']['lat_field']
        _READERS[key] = reader
        logger.info("Created HATS reader: %s (%i partitions, orders %s)"
                    % (path, len(orders), np.unique(orders).tolist()))
    return _READERS[key]


def load_catalog_pixels(config, pixels):
    """Catalog rows for the given ``nside_catalog`` pixels (RING), via the cache.

    This is the HATS equivalent of reading ``catalog_hpx%04i.fits`` files.
    """
    reader = get_reader(config)
    nside = config['coords']['nside_catalog']
    return reader.load_pixels(pixels, nside, nest=False)
