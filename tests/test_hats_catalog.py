"""Portable unit tests for ugali.observation.hats_catalog (synthetic HATS catalog)."""
import os

import numpy as np
import pytest

pytest.importorskip('pyarrow')
import pyarrow as pa
import pyarrow.parquet as pq
import healpy as hp

from ugali.observation import hats_catalog as hc
from ugali.observation.hats_catalog import HatsCatalogReader

NPTS = 50
HIGH = 10  # order used to generate random points inside a partition


def _partitions():
    """Mixed-order partitions: (order, pixel)."""
    parts = [(2, 17)]                                   # order 2
    parts += [(3, 4 * 40 + i) for i in range(4)]        # children of order-2 pixel 40
    parts += [(4, 16 * 5 + 4 * 1 + i) for i in (0, 1, 3)]   # grandchildren of order-2 pixel 5
    return parts


def _overlap(o, p, oq, q):
    if o >= oq:
        return (p >> (2 * (o - oq))) == q
    return (q >> (2 * (oq - o))) == p


@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('hats'))
    rng = np.random.RandomState(42)
    parts = _partitions()
    with open(os.path.join(root, 'partition_info.csv'), 'w') as f:
        f.write('Norder,Npix\n')
        for o, p in parts:
            f.write('%d,%d\n' % (o, p))
    rows = {}
    nid = 0
    for o, p in parts:
        k = HIGH - o
        sub = p * 4 ** k + rng.randint(0, 4 ** k, NPTS)
        ra, dec = hp.pix2ang(2 ** HIGH, sub, nest=True, lonlat=True)
        ids = np.arange(nid, nid + NPTS, dtype=np.int64)
        nid += NPTS
        flag = rng.randint(0, 2, NPTS).astype(np.int64)
        tab = pa.table({'RA': ra.astype('f8'), 'DEC': dec.astype('f8'),
                        'ID': ids, 'FLAG': flag})
        fn = hc.tile_path(root, o, p)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        pq.write_table(tab, fn)
        rows[(o, p)] = (ra, dec, ids, flag)
    return root, parts, rows


def test_partitions_for_pixels_matches_brute_force(synth):
    root, parts, _ = synth
    reader = HatsCatalogReader(root)
    rng = np.random.RandomState(1)
    for nside in [1, 2, 4, 8, 16, 32, 64]:
        oq = int(np.log2(nside))
        for nest in (False, True):
            for _ in range(10):
                q = np.unique(rng.randint(0, hp.nside2npix(nside), rng.randint(1, 20)))
                qn = hp.ring2nest(nside, q) if not nest else q
                expect = sorted((o, p) for o, p in parts
                                if any(_overlap(o, p, oq, int(x)) for x in qn))
                assert reader.partitions_for_pixels(q, nside, nest=nest) == expect
    # whole sky at nside 1 returns everything
    allp = reader.partitions_for_pixels(np.arange(12), 1, nest=True)
    assert allp == sorted(parts)


def test_partitions_for_cone(synth):
    root, parts, rows = synth
    reader = HatsCatalogReader(root)
    o, p = (3, 4 * 40 + 2)
    ra, dec = hp.pix2ang(2 ** o, p, nest=True, lonlat=True)
    keys = reader.partitions_for_cone(ra, dec, 0.01)
    assert keys == [(o, p)]
    # larger cone: every returned partition must overlap the cone's pixels
    keys = reader.partitions_for_cone(ra, dec, 5.0)
    assert (o, p) in keys
    vec = hp.ang2vec(ra, dec, lonlat=True)
    pix = hp.query_disc(2 ** 4, vec, np.radians(5.0), inclusive=True, fact=4, nest=True)
    for ko, kp in keys:
        assert any(_overlap(ko, kp, 4, int(x)) for x in pix)


def test_load_pixels_rows(synth):
    root, parts, rows = synth
    reader = HatsCatalogReader(root)
    ra = np.concatenate([r[0] for r in rows.values()])
    dec = np.concatenate([r[1] for r in rows.values()])
    ids = np.concatenate([r[2] for r in rows.values()])
    nside = 16
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    populated = np.unique(pix)
    for q in populated[:5]:
        data = reader.load_pixels([q], nside)
        assert np.array_equal(np.sort(data['ID']), np.sort(ids[pix == q]))
    # several pixels at once
    q = populated[:4]
    data = reader.load_pixels(q, nside)
    assert np.array_equal(np.sort(data['ID']), np.sort(ids[np.isin(pix, q)]))
    # a pixel overlapping a partition but containing no rows
    empty = None
    for cand in range(hp.nside2npix(nside)):
        if cand in populated:
            continue
        if reader.partitions_for_pixels([cand], nside):
            empty = cand
            break
    if empty is not None:
        data = reader.load_pixels([empty], nside)
        assert data is not None and len(data) == 0
    # a pixel with no overlapping partition
    nopart = next(c for c in range(hp.nside2npix(nside))
                  if not reader.partitions_for_pixels([c], nside))
    data = reader.load_pixels([nopart], nside)
    assert data is None or len(data) == 0


def test_selection_applied(synth):
    root, parts, rows = synth
    reader = HatsCatalogReader(root, selection="(self.data['FLAG'] == 0)")
    key = parts[0]
    data = reader.get_tile(key)
    assert len(data) > 0 and np.all(data['FLAG'] == 0)
    assert len(data) == int(np.sum(rows[key][3] == 0))
    allrows = reader.load_tiles(parts)
    assert np.all(allrows['FLAG'] == 0)


def test_lru_eviction_and_stats(synth):
    root, parts, _ = synth
    reader = HatsCatalogReader(root, max_tiles=2)
    a, b, c = parts[0], parts[1], parts[2]
    for k in (a, b, c):
        reader.get_tile(k)
    assert len(reader) == 2
    assert reader.stats['misses'] == 3 and reader.stats['hits'] == 0
    assert reader.stats['evictions'] == 1
    reader.get_tile(c)                     # most recent: hit
    assert reader.stats['hits'] == 1 and reader.stats['misses'] == 3
    assert reader.hit_rate == pytest.approx(1 / 4.)
    reader.get_tile(a)                     # evicted earlier: miss
    assert reader.stats['misses'] == 4
    reader.reset_stats()
    assert reader.stats['hits'] == 0 and reader.stats['misses'] == 0
    assert np.isnan(reader.hit_rate)
    reader.clear()
    assert len(reader) == 0


def test_table_to_recarray():
    ra = np.array([1.5, 2.5, 3.5])
    ids = np.array([1, 2, 3], dtype=np.int64)
    tab = pa.table({'RA': ra, 'ID': ids, 'F': np.array([1, 0, 1], dtype=np.int32)})
    rec = hc.table_to_recarray(tab)
    assert rec.dtype.names == ('RA', 'ID', 'F')
    for name in rec.dtype.names:
        assert rec.dtype[name].isnative
    np.testing.assert_array_equal(rec['RA'], ra)
    np.testing.assert_array_equal(rec['ID'], ids)
    assert rec['F'].dtype == np.int32


def test_get_reader_config(synth):
    root, parts, _ = synth
    config = {
        'catalog': {
            'hats': {'path': root, 'max_tiles': 5},
            'objid_field': 'ID', 'lon_field': 'RA', 'lat_field': 'DEC',
            'mag_1_field': 'RA', 'mag_err_1_field': 'DEC',
            'mag_2_field': 'RA', 'mag_err_2_field': 'DEC',
            'selection': "(self.data['FLAG'] == 0)",
        },
        'coords': {'nside_catalog': 16},
    }
    cols = hc.default_columns(config)
    assert cols == ['ID', 'RA', 'DEC', 'FLAG']
    r1 = hc.get_reader(config)
    r2 = hc.get_reader(config)
    assert r1 is r2
    assert r1.columns == cols and r1.max_tiles == 5
    data = hc.load_catalog_pixels(config, np.arange(hp.nside2npix(16)))
    assert np.all(data['FLAG'] == 0)


FIXTURE = os.path.join(os.path.dirname(__file__), 'data', 'desy6', 'hats', 'desy6_fixture')


def test_add_derived():
    rec = np.zeros(3, dtype=[('A', 'f4'), ('B', 'f8')])
    rec['A'] = [1, 2, 3]
    rec['B'] = [0.5, 0.5, 1.0]
    out = hc.add_derived(rec, {'C': 'A - 2 * B', 'D': 'np.where(C > 0, C, np.nan)', 'A': 'A + 1'})
    assert out.dtype.names == ('B', 'C', 'D', 'A')
    np.testing.assert_allclose(out['C'], [0, 1, 1])
    np.testing.assert_allclose(out['D'], [np.nan, 1, 1])
    np.testing.assert_allclose(out['A'], [2, 3, 4])
    assert hc.add_derived(rec, None) is rec
    assert hc.derived_inputs({'C': 'A - 2 * B', 'D': 'np.abs(C)'}) == ['A', 'B', 'np', 'abs', 'C']


def test_table_to_recarray_null_bool():
    tab = pa.table({'F': pa.array([True, None, False], type=pa.bool_()),
                    'X': pa.array([1.0, None, 3.0])})
    rec = hc.table_to_recarray(tab)
    assert rec['F'].dtype == np.bool_
    np.testing.assert_array_equal(rec['F'], [True, False, False])
    assert np.isnan(rec['X'][1])


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason='DES Y6 HATS fixture not available')
def test_derived_columns_on_fixture():
    """A derived (dereddened-style) magnitude on the DES Y6 HATS fixture, usable
    as a mag field and in the selection string."""
    config = {
        'catalog': {
            'hats': {'path': FIXTURE, 'max_tiles': 16,
                     'derived': {'G_DERED': 'PSF_MAG_APER_8_G_CORRECTED - 0.1 * EXT_XGB',
                                 'GR': 'G_DERED - PSF_MAG_APER_8_R_CORRECTED'}},
            'objid_field': 'COADD_OBJECT_ID', 'lon_field': 'RA', 'lat_field': 'DEC',
            'mag_1_field': 'G_DERED', 'mag_err_1_field': 'PSF_MAG_ERR_APER_8_G',
            'mag_2_field': 'PSF_MAG_APER_8_R_CORRECTED', 'mag_err_2_field': 'PSF_MAG_ERR_APER_8_R',
            'selection': "(self.data['GR'] < 1.0) & (self.data['EXT_XGB'] >= 0)",
        },
        'coords': {'nside_catalog': 32},
    }
    cols = hc.default_columns(config)
    assert 'G_DERED' not in cols and 'GR' not in cols
    assert 'PSF_MAG_APER_8_G_CORRECTED' in cols and 'EXT_XGB' in cols
    reader = hc.get_reader(config)
    orders, pixels = hc.read_partition_info(FIXTURE)
    data = reader.get_tile((orders[0], pixels[0]))
    assert len(data) > 0
    np.testing.assert_allclose(data['G_DERED'],
                               data['PSF_MAG_APER_8_G_CORRECTED'] - 0.1 * data['EXT_XGB'], rtol=1e-6)
    assert np.all(data['GR'] < 1.0) and np.all(data['EXT_XGB'] >= 0)
    # the selection was applied after deriving: compare with a plain read
    raw = HatsCatalogReader(FIXTURE, columns=reader.columns).get_tile((orders[0], pixels[0]))
    gr = raw['PSF_MAG_APER_8_G_CORRECTED'] - 0.1 * raw['EXT_XGB'] - raw['PSF_MAG_APER_8_R_CORRECTED']
    assert len(data) == int(np.sum((gr < 1.0) & (raw['EXT_XGB'] >= 0)))
