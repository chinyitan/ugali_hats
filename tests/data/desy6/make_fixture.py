#!/usr/bin/env python
"""
Build the small self-contained DES Y6 likelihood-scan fixture used by
tests/test_des_scan.py from Chin-Yi's full DES Y6 run.

For a single nside=256 likelihood pixel, this keeps only:
  * the catalog rows inside the ROI that pass the config selection, and
    only the columns the scan reads (~4 MB instead of ~500 MB);
  * the g/r magnitude-limit masks for the catalog pixels touched;
  * the four Bressan2012 isochrone files the config asks for;
  * a config.yaml with relative paths and a reduced distance-modulus /
    extension grid, and the reference scan output for it.

Trimming the catalog is lossless for the scan: ugali drops rows outside
the ROI and failing the selection when it loads the catalog. The script
checks this by running the scan with the *full* grid on the fixture and
comparing it to the reference run.

Pixel 639648 sits at the edge of the catalog files in the reference run:
only 3 of the 10 nside=32 catalog pixels its 2 deg ROI touches exist, so
~22% of the background annulus has catalog data (58,875 of the 65,640
fixture rows are in the annulus, 6,765 in the interior). This is how
the production scan saw it, and the output matches it bit-for-bit. A
fully-covered pixel (e.g. 668973) would need ~19 MB of catalog.

Only needed to (re)build the fixture; requires access to the full run.

  UGALIDIR=tests/data/desy6 python tests/data/desy6/make_fixture.py
"""
import os, sys, re, shutil, subprocess, time
import numpy as np
import fitsio
import healpy as hp

import ugali.utils.config
import ugali.observation.roi
import ugali.analysis.scan

PIX, NSIDE = 639648, 256
REFDIR = os.environ.get('UGALI_DES_REFDIR', '/astro/users/chinyi/des_ugali')
ISODIR = os.path.expanduser('~/.ugali/isochrones/des/bressan2012')
HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, '..', '..', '..', 'artifacts', 'des_ugali_validation', 'fixture_check')

# Reduced grid for the fast test (the full run uses 17 moduli x 3 extensions)
DISTANCE_MODULI = [17.0, 19.0, 21.0]
EXTENSIONS = [0.02, 0.07, 0.15]

CAT_COLUMNS = ['COADD_OBJECT_ID', 'RA', 'DEC',
               'PSF_MAG_APER_8_G_CORRECTED', 'PSF_MAG_ERR_APER_8_G',
               'PSF_MAG_APER_8_R_CORRECTED', 'PSF_MAG_ERR_APER_8_R',
               'FLAGS_GOLD', 'EXT_XGB']


def write_config(src, dst, moduli, extensions):
    txt = open(src).read()
    txt = re.sub(r'(dirname:\s*)\S+/healpix', r'\1./healpix', txt)
    txt = re.sub(r'(dirname\s*:\s*)\S+/mask', r'\1./mask', txt)
    txt = re.sub(r'distance_modulus_array:.*', 'distance_modulus_array: %s' % moduli, txt)
    txt = re.sub(r'^(\s*)extension_array:.*', r'\1extension_array: %s' % extensions, txt, count=1, flags=re.M)
    header = ("# DES Y6 single-pixel test fixture (see make_fixture.py).\n"
              "# Derived from the scan-v0 config of Chin-Yi Tan's DES Y6 run;\n"
              "# only the sections used by ugali/analysis/scan.py are meaningful.\n")
    open(dst, 'w').write(header + txt)


def run_scan(configdir, outfile, log):
    env = dict(os.environ, UGALIDIR=HERE, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    cmd = [sys.executable, ugali.analysis.scan.__file__, 'config.yaml', outfile, '--hpx', str(NSIDE), str(PIX)]
    t0 = time.time()
    with open(log, 'w') as f:
        subprocess.check_call(cmd, cwd=configdir, env=env, stdout=f, stderr=subprocess.STDOUT)
    return time.time() - t0


def identical(a, b):
    fa, fb = fitsio.FITS(a), fitsio.FITS(b)
    for i in range(1, len(fa)):
        da, db = fa[i].read(), fb[i].read()
        for n in da.dtype.names:
            if not np.array_equal(da[n], db[n], equal_nan=True):
                return False
    return True


if __name__ == '__main__':
    srccfg = os.path.join(REFDIR, 'scan-v0', 'config.yaml')
    reffile = os.path.join(REFDIR, 'scan-v0', 'scan_%08i_cel.fits' % PIX)
    for d in ['healpix', 'mask', 'isochrones/des/bressan2012']:
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    os.makedirs(ARTIFACTS, exist_ok=True)

    cfg = ugali.utils.config.Config(srccfg)
    lon, lat = hp.pix2ang(NSIDE, PIX, lonlat=True)
    roi = ugali.observation.roi.ROI(cfg, lon, lat)
    selection = cfg['catalog']['selection']
    catpix = np.unique(hp.ang2pix(cfg['coords']['nside_catalog'],
                                  *hp.pix2ang(cfg['coords']['nside_pixel'], roi.pixels, lonlat=True), lonlat=True))

    class _Sel(object):  # evaluate the config selection string on the data
        def __init__(self, data): self.data = data

    for p in catpix:
        src = os.path.join(cfg['catalog']['dirname'], cfg['catalog']['basename'] % p)
        if not os.path.exists(src):
            print("catalog pixel %i not in reference run; skipping" % p); continue
        data = fitsio.read(src, columns=CAT_COLUMNS)
        inroi = np.isin(hp.ang2pix(cfg['coords']['nside_pixel'], data['RA'], data['DEC'], lonlat=True), roi.pixels)
        sel = eval(selection, {}, {'self': _Sel(data)})
        data = data[inroi & sel]
        dst = os.path.join(HERE, 'healpix', cfg['catalog']['basename'] % p)
        fitsio.write(dst, data, clobber=True)
        print("wrote %s: %i rows" % (dst, len(data)))
        for key in ['basename_1', 'basename_2']:
            m = cfg['mask'][key] % p
            shutil.copy(os.path.join(cfg['mask']['dirname'], m), os.path.join(HERE, 'mask', m))

    for iso in cfg['scan']['source']['isochrone']['isochrones']:
        f = 'iso_a%04.1f_z%.5f.dat' % (iso['age'], iso['z'])
        shutil.copy(os.path.join(ISODIR, f), os.path.join(HERE, 'isochrones', 'des', 'bressan2012', f))
        print("copied isochrone", f)

    # 1. Lossless check: full grid on the trimmed data must reproduce the reference run
    full = cfg['scan']['distance_modulus_array']
    fullext = cfg['scan']['extension_array']
    write_config(srccfg, os.path.join(ARTIFACTS, 'config.yaml'), full, fullext)
    for d in ['healpix', 'mask']:
        if os.path.lexists(os.path.join(ARTIFACTS, d)): os.remove(os.path.join(ARTIFACTS, d))
        os.symlink(os.path.join(HERE, d), os.path.join(ARTIFACTS, d))
    out = os.path.join(ARTIFACTS, 'scan_%08i_cel.fits' % PIX)
    dt = run_scan(ARTIFACTS, out, os.path.join(ARTIFACTS, 'scan_full.log'))
    ok = identical(reffile, out)
    print("full-grid scan on fixture: %.0fs, identical to reference: %s" % (dt, ok))
    if not ok:
        sys.exit("Fixture is NOT lossless; aborting")

    # 2. Reduced grid: the config and reference used by the unit test
    write_config(srccfg, os.path.join(HERE, 'config.yaml'), DISTANCE_MODULI, EXTENSIONS)
    out = os.path.join(HERE, 'reference_scan_%08i_cel.fits' % PIX)
    dt = run_scan(HERE, out, os.path.join(ARTIFACTS, 'scan_reduced.log'))
    print("reduced-grid scan: %.1fs -> %s" % (dt, out))
    shutil.rmtree(os.path.join(HERE, 'scan-v0'), ignore_errors=True)
    print("fixture size: %.1f MB" % (sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(HERE) for f in fs) / 1e6))
