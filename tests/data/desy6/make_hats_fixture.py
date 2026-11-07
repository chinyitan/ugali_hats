#!/usr/bin/env python
"""
Build the HATS-format copy of the DES Y6 fixture catalog used by
test_fixture_scan_hats in tests/test_des_scan.py.

The three trimmed nside=32 FITS catalog files in tests/data/desy6/healpix are
concatenated and partitioned with lsdb into a tiny HATS catalog at
tests/data/desy6/hats/desy6_fixture. The partitioning (lowest_order=3,
highest_order=7, partition_rows=8000) splits the large pixel (~62k rows)
to a deeper order than the sparse ones, so the catalog has partitions at
mixed orders and the mixed-order lookup in ugali.observation.hats_catalog
is exercised. lsdb's output is kept as written (a valid HATS collection,
~3 MB); ugali only reads partition_info.csv and the dataset parquet files.

Run from the repository root (needs lsdb, hats, pyarrow, fitsio):

  python tests/data/desy6/make_hats_fixture.py
"""
import os
import glob
import shutil

import numpy as np
import pandas as pd
import fitsio
import lsdb

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTION = os.path.join(HERE, 'hats')
OUT = os.path.join(COLLECTION, 'desy6_fixture')


def main():
    frames = []
    for f in sorted(glob.glob(os.path.join(HERE, 'healpix', 'y6_gold_2_2_*.fits'))):
        d = fitsio.read(f)
        # FITS is big-endian; convert to native byte order
        frames.append(pd.DataFrame({n: d[n].astype(d[n].dtype.newbyteorder('='))
                                    for n in d.dtype.names}))
    df = pd.concat(frames, ignore_index=True)

    cat = lsdb.from_dataframe(df, ra_column='RA', dec_column='DEC',
                              lowest_order=3, highest_order=7, partition_rows=8000,
                              margin_threshold=None, should_generate_moc=False,
                              use_pyarrow_types=False,
                              catalog_name='desy6_fixture')
    if os.path.exists(COLLECTION):
        shutil.rmtree(COLLECTION)
    # lsdb writes a HATS collection: <COLLECTION>/collection.properties and
    # the catalog itself in <COLLECTION>/desy6_fixture (= OUT)
    cat.to_hats(COLLECTION, catalog_name='desy6_fixture')


if __name__ == '__main__':
    main()
