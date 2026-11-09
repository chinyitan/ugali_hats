#!/usr/bin/env python
"""
Regression tests for the DES Y6 likelihood scan (ugali/analysis/scan.py).

test_fixture_scan
    Self-contained and fast (~10 s): runs the scan on the single-pixel
    fixture in tests/data/desy6 (trimmed DES Y6 catalog, masks and
    isochrones; see make_fixture.py there) and compares the output
    bit-for-bit with the stored reference.

test_reference_scan
    Full-fidelity check against Chin-Yi Tan's DES Y6 scan-v0 run. Needs
    that run and the matching isochrones, so it is skipped unless the
    reference directory exists (UGALI_DES_REFDIR; UGALI_DES_PIXELS
    selects the nside=256 pixels, default 639648; ~45 s per pixel).

Note on exactness: the fixture reference was produced with numpy 2.5
(python 3.14) by the original (pre-optimization) scan code, and the
fixture tests require a bit-for-bit match. numpy 1.x gives results that
differ from numpy 2.x by up to the richness-fit tolerance (|dlogL| <~
0.05 in flat likelihoods, 1e-5 typically; |dTS| < 1e-4 where TS > 10),
identically for the FITS and HATS paths. Chin-Yi's reference run was made
with numpy 1.26, so test_reference_scan compares exactly under numpy 1
and within that tolerance under numpy 2.
"""
import os
import sys
import shutil
import subprocess

import numpy as np
import pytest

import ugali.analysis.scan

NSIDE = 256
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'desy6')
FIXTURE_PIX = 639648

REFDIR = os.environ.get('UGALI_DES_REFDIR', '/astro/users/chinyi/des_ugali')
REF_SCANDIR = os.path.join(REFDIR, 'scan-v0')
REF_PIXELS = [int(p) for p in os.environ.get('UGALI_DES_PIXELS', '639648').split(',')]

# Header keys that may legitimately differ between runs
IGNORE_KEYS = {'DATE', 'CHECKSUM', 'DATASUM', 'COMMENT', 'HISTORY'}


def run_scan(workdir, pix, outfile, ugalidir=None, config='config.yaml'):
    """Run scan.py in `workdir` (which must hold config.yaml) as a subprocess.

    The isochrone directory is read from UGALIDIR when ugali is
    imported, so it has to be set in the child's environment.
    """
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if ugalidir is not None:
        env['UGALIDIR'] = ugalidir
    cmd = [sys.executable, ugali.analysis.scan.__file__, config, str(outfile),
           '--hpx', str(NSIDE), str(pix)]
    subprocess.check_call(cmd, cwd=str(workdir), env=env)


def compare_fits(ref, new):
    """Return a list of human-readable differences (empty if identical)."""
    import fitsio
    diffs = []
    fr, fn = fitsio.FITS(ref), fitsio.FITS(new)
    if len(fr) != len(fn):
        return ["HDU count %i != %i" % (len(fr), len(fn))]
    for i in range(len(fr)):
        hr, hn = fr[i].read_header(), fn[i].read_header()
        kr = {k: hr[k] for k in hr.keys() if k not in IGNORE_KEYS}
        kn = {k: hn[k] for k in hn.keys() if k not in IGNORE_KEYS}
        for k in sorted(set(kr) | set(kn)):
            if kr.get(k) != kn.get(k):
                diffs.append("HDU%i header %s: %r != %r" % (i, k, kr.get(k), kn.get(k)))
        if fr[i].get_exttype() == 'IMAGE_HDU':
            continue
        dr, dn = fr[i].read(), fn[i].read()
        if dr.dtype != dn.dtype or dr.shape != dn.shape:
            diffs.append("HDU%i dtype/shape %s%s != %s%s" % (i, dr.dtype, dr.shape, dn.dtype, dn.shape))
            continue
        for name in dr.dtype.names:
            a, b = dr[name], dn[name]
            same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b)
            if not same:
                msg = "HDU%i column %s differs" % (i, name)
                if a.dtype.kind == 'f':
                    d = np.abs(a.astype(float) - b.astype(float))
                    msg += " (n=%i/%i, max|diff|=%.3g)" % (np.sum(d > 0), d.size, np.nanmax(d))
                diffs.append(msg)
    return diffs


def _fixture_scan(tmp_path, config):
    """Scan the DES Y6 fixture pixel with `config` and compare with the stored reference."""
    # Work on a copy so the scan's output directories never land in the repo
    workdir = tmp_path / 'desy6'
    shutil.copytree(FIXTURE, workdir)
    reffile = workdir / ('reference_scan_%08i_cel.fits' % FIXTURE_PIX)
    outfile = workdir / 'out' / ('scan_%08i_cel.fits' % FIXTURE_PIX)
    outfile.parent.mkdir()

    run_scan(workdir, FIXTURE_PIX, outfile, ugalidir=str(workdir), config=config)
    assert outfile.exists()

    diffs = compare_fits(str(reffile), str(outfile))
    assert not diffs, "fixture scan differs from reference:\n  " + "\n  ".join(diffs)


def test_fixture_scan(tmp_path):
    """Scan the DES Y6 fixture pixel (FITS catalog) and compare with the stored reference."""
    _fixture_scan(tmp_path, 'config.yaml')


def test_fixture_scan_hats(tmp_path):
    """Same scan reading the catalog from the HATS copy (make_hats_fixture.py)."""
    pytest.importorskip('pyarrow')
    _fixture_scan(tmp_path, 'config_hats.yaml')


@pytest.mark.skipif(not os.path.exists(os.path.join(REF_SCANDIR, 'config.yaml')),
                    reason="DES Y6 reference run not found: %s" % REF_SCANDIR)
@pytest.mark.parametrize('pix', REF_PIXELS)
def test_reference_scan(pix, tmp_path):
    """Regenerate a pixel of the full DES Y6 reference run and compare."""
    reffile = os.path.join(REF_SCANDIR, 'scan_%08i_cel.fits' % pix)
    assert os.path.exists(reffile), "Missing reference file: %s" % reffile

    workdir = tmp_path / 'des_scan'
    workdir.mkdir()
    shutil.copy(os.path.join(REF_SCANDIR, 'config.yaml'), workdir / 'config.yaml')
    outfile = workdir / ('scan_%08i_cel.fits' % pix)

    run_scan(workdir, pix, outfile)
    assert outfile.exists()

    if int(np.__version__.split('.')[0]) < 2:
        diffs = compare_fits(reffile, str(outfile))
        assert not diffs, "scan_%08i_cel.fits differs from reference:\n  " % pix + "\n  ".join(diffs)
    else:
        # numpy 2 vs the numpy-1 reference: same candidates, not the same bits
        import fitsio
        ref, new = fitsio.read(reffile, ext=1), fitsio.read(str(outfile), ext=1)
        assert np.array_equal(ref['PIXEL'], new['PIXEL'])
        ts_ref, ts_new = 2.*ref['LOG_LIKELIHOOD'].astype(float), 2.*new['LOG_LIKELIHOOD'].astype(float)
        dts = np.abs(ts_ref - ts_new)
        assert dts.max() < 0.2, "max |dTS| = %.3g" % dts.max()
        strong = ts_ref > 10
        assert dts[strong].max(initial=0.) < 1e-3, "max |dTS| at TS>10 = %.3g" % dts[strong].max()
        assert np.array_equal(ref['FRACTION_OBSERVABLE'], new['FRACTION_OBSERVABLE'])
