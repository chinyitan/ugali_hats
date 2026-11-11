#!/usr/bin/env python
"""
Define some useful constants.
"""
import numpy as np

# http://www.adsabs.harvard.edu/abs/2002AJ....123..485S
# Determination of magnitude limits is rather complicated.
# The technique applied here is to derive the magnitude at
# which the 10sigma signal-to-noise threshold is reached.
# For SDSS, these values are (Table 21):
# u=22.12,g=22.60,r=22.29,i=21.85,z=20.35 
# However, the quoted 95% completeness limits are (Table 2):
# u=22.0,g=22.2,r=22.2,i=21.3,z=20.5
# What is responsible for this disconnect? Well, I think 
# the completeness was estimated by comparing with COSMOS
# on nights that weren't all that good seeing.
# The second set of numbers agree with those listed here:
# http://www.sdss3.org/dr10/scope.php

# http://des-docdb.fnal.gov:8080/cgi-bin/ShowDocument?docid=20
# DES simple magnitude limits come from the science 
# requirements document. These are 'requirements'
# are somewhat pessimistic and the document also
# conatains 'goals' for the magnitude limit:
# g=25.4, r=24.9, i=24.6, z=23.8, y=21.7
# Of course all of this is yet to be verified with
# data...

MAGLIMS = dict(
    ### ADW: Need to make sure everything matches
    # 95% completeness from ERD
    sdss = {
        'u': 22.0,
        'g': 22.2,
        'r': 22.2,
        'i': 21.3,
        'z': 20.5
    },
    # 10 sigma mag-limit
    dr10 = {
        'u': 22.12,
        'g': 22.60,
        'r': 22.29,
        'i': 21.85,
        'z': 20.35
    },
    des = {
        'g': 24.6,
        'r': 24.1,
        'i': 24.3,
        'z': 23.8,
        'Y': 21.5
    },
    sva1_gold = {
        'g': 22.5,
        'r': 22.5,
        'i': 22.5,
        'z': 22.0,
        'Y': 21.0
    },
    y1a1 = {
        #'g': 23.7, # Probably closer to 23.9 (magerr) or 24.2 (mangle)
        #'r': 23.7, # Probably closer to 23.9 (mangle)
        'g': 23.0, # Truncate at 23.0 (star-galaxy separation)
        'r': 23.0, # Truncate at 23.0
        },
    y2n = {
        'g': 23.0,
        'r': 23.0,
        },
    y2u1 = {
        'g': 24.0,
        'r': 24.0,
        },
    y2q1 = {
        'g': 23.5,
        'r': 23.5,
        },
    y17v1 = {
        'g': 23.5,
        'r': 23.5,
        },
    y3a2 = {
        'g': 24.0, 
        'r': 24.0,
        },
    ps1_dr1 = {
        'g': 23.0,
        'r': 23.0,
        },
)


# These values were derived by fitting the median of the 
# mag vs magerr distribution to the functional form:
# magerr = exp(a*(maglim-mag)+b) + c
# The fit values are band dependent, but not by very much.
# g: [a,b,c] = [ 0.97767842, -2.32235592,  0.01548962]
# r: [a,b,c] = [ 0.9796397 , -2.19844877,  0.01332762]
# i: [a,b,c] = [ 0.94384445, -2.22649283,  0.01351053]
# A useful generalization for SDSS DR10 is:
# [a,b,c] = [ 0.97767842, -2.32235592,  0.01548962]

MAGERR_PARAMS = dict(
    ### ADW: Need to rectify these numbers!!!
    dr10 = {
        'u': [np.nan, np.nan, np.nan],
        'g': [-0.98441923, -2.33286997, 0.01612018],
        'r': [-0.98045415, -2.2032579 , 0.0135355 ],
        'i': [-0.94704984, -2.23306942, 0.01381268],
        'z': [np.nan, np.nan, np.nan]
    },

    y1a1 = {
        'g': [ -8.71852748e-01,  -2.52245237e+00,   5.12942930e-04], 
        'r': [ -8.86883525e-01,  -2.38329071e+00,   4.56457328e-04], 
        'i': [ -0.7722337363302921,  -2.213972432551957,   0.001150015422301175],               
        },

    # This is bogus (copy from y1a1)
    y2n = {
        'g': [ -8.71852748e-01,  -2.52245237e+00,   5.12942930e-04], 
        'r': [ -8.86883525e-01,  -2.38329071e+00,   4.56457328e-04], 
        'i': [ -0.7722337363302921,  -2.213972432551957,   0.001150015422301175], 
        },

    # This is bogus (copy from y1a1)
    y2u1 = {
        'g': [ -8.71852748e-01,  -2.52245237e+00,   5.12942930e-04], 
        'r': [ -8.86883525e-01,  -2.38329071e+00,   4.56457328e-04], 
        'i': [ -0.7722337363302921,  -2.213972432551957,   0.001150015422301175], 
        },

    # This is bogus (copy from y1a1)
    y2q1 = {
        'g': [ -8.71852748e-01,  -2.52245237e+00,   5.12942930e-04], 
        'r': [ -8.86883525e-01,  -2.38329071e+00,   4.56457328e-04], 
        'i': [ -0.7722337363302921,  -2.213972432551957,   0.001150015422301175], 
        },

    # This is bogus (copy from y1a1)
    y17v1 = {
        'g': [ -8.71852748e-01,  -2.52245237e+00,   5.12942930e-04], 
        'r': [ -8.86883525e-01,  -2.38329071e+00,   4.56457328e-04], 
        'i': [ -0.7722337363302921,  -2.213972432551957,   0.001150015422301175], 
        },

    # This is bogus (copy from y1a1)
    y3a2 = {
        'g': [ -8.71852748e-01,  -2.52245237e+00,   5.12942930e-04], 
        'r': [ -8.86883525e-01,  -2.38329071e+00,   4.56457328e-04], 
        'i': [ -0.7722337363302921,  -2.213972432551957,   0.001150015422301175], 
        },

    # DES Y6 (Chin-Yi Tan)
    desy6 = {
        'g': [ -0.9133, -2.0625, 0.0003], 
        'r': [ -0.9127, -2.1588, 0.0005], 
        'i': [ -0.9044, -2.2738, 0.0005], 
        },

    # DELVE DR3 gold: fit to the median PSF_MAG_ERR_APER_8 of stars
    # (FLAGS_GOLD==0, EXT_XGB<=1) in 0.1 mag bins of maglim-mag, from 40
    # random HATS partitions and the DR3 healsparse depth maps (2026-11)
    delve_dr3 = {
        'g': [ -0.9104, -1.8007, 0.0005],
        'r': [ -0.9074, -1.9474, 0.0005],
        },

    # Rubin DP2: fit to the median {b}_psfMagErr of point sources
    # (refExtendedness<0.5, band unflagged; 3% random subsample of the
    # object_lc HATS catalog) in 0.1 mag bins of maglim-mag, for delta>=0.
    # maglim is the DP2 psf_maglim (5 sigma) depth map minus 0.75 mag
    # (~10 sigma), which is how the DP2 stream search uses the maps (2026-11)
    lsst_dp2 = {
        'u': [ -0.8401, -2.3476, 0.0016],
        'g': [ -0.8990, -2.3652, 0.0008],
        'r': [ -0.9052, -2.3690, 0.0006],
        'i': [ -0.9117, -2.3741, 0.0005],
        },

    # Euclid Q1 MER final catalogue (HATS), VIS = FLUX_VIS_PSF, H = FLUX_H_TEMPLFIT
    # (+0.10 mag colour term), fitted to the median reported magerr per 0.1 mag of
    # delta = maglim - mag for the stream-search selection (point sources) against
    # the dereddened depth maps used by the search: VIS maglim = min(10 sigma,
    # I_E 23.5 classifier limit), H maglim = 10 sigma (noise-model method of
    # streamobs). See projects/euclid/ugali/q1/fit_magerr.py (2026-11).
    euclid_q1 = {
        'VIS': [ -0.5860, -4.1791, 0.0008],
        'H'  : [ -0.6733, -2.2117, 0.0030],
        },

    # This is fit from some curves that Keith made
    ps1_dr1 = {
        'g' : [-0.84967134, -2.27897518,  0.00304501],
        'r' : [-0.92218737, -2.34012887,  0.00451849],
        }

)

# Calibration uncertainty
CALIB_ERR = dict(
    sdss = 0.03,
    y1a1 = 0.015,
    y2n  = 0.015,
    y2u1 = 0.015,
    y2q1 = 0.015,
    y3a2 = 0.015, # Y3 FGCM should be closer to 0.007...
    ps1_dr1 = 0.03, # Made up
)
