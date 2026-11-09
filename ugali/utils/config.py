
"""
Class for storing and updating config dictionaries.
"""
import os,sys
from os.path import join, exists
import pprint
import copy
from collections import OrderedDict as odict
import glob

import numpy as np
import healpy as hp
import yaml

from ugali.utils.logger import logger
import ugali.utils.config # To recognize own type
from ugali.utils.mlab import isstring

class Config(dict):
    """
    Configuration object
    """

    def __init__(self, config, default=None):
        """
        Initialize a configuration object from a filename or a dictionary.
        Provides functionality to merge with a default configuration.

        Parameters:
          config:   filename, dict, or Config object (deep copied)
          default:  default configuration to merge
        
        Returns:
          config
        """
        self.update(self._load(default))
        self.update(self._load(config))

        self._formatFilepaths()

        # For back-compatibility...
        self.params = self

        # Run some basic validation
        # ADW: This should be run after creating filenames
        self._validate()

        # Filenames from this config (masked by existence)
        self._filenames = None

    @property
    def filenames(self):
        """
        Masked array of catalog and mask filenames for each pixel.
        Built on first access; raises IOError if the data directories
        do not exist.
        """
        # Only create the filenames once
        if self._filenames is None:
            self._filenames = self._createFilenames()
        return self._filenames

    @filenames.setter
    def filenames(self, value):
        self._filenames = value


    def __str__(self):
        return yaml.dump(self)

    def _load(self, config):
        """ Load this config from an existing config 

        Parameters:
        -----------
        config : filename, config object, or dict to load

        Returns:
        --------
        params : configuration parameters
        """
        if isstring(config):
            self.filename = config
            params = yaml.safe_load(open(config))
        elif isinstance(config, Config):
            # This is the copy constructor...
            self.filename = config.filename
            params = copy.deepcopy(config)
        elif isinstance(config, dict):
            params = copy.deepcopy(config)
        elif config is None:
            params = {}
        else:
            raise Exception('Unrecognized input')

        return params

    def _validate(self):
        """ Enforce some structure to the config file """
        # This could be done with a default config

        # Check that specific keys exist
        sections = odict([
                ('catalog',['dirname','basename',
                            'lon_field','lat_field','objid_field',
                            'mag_1_band', 'mag_1_field', 'mag_err_1_field',
                            'mag_2_band', 'mag_2_field', 'mag_err_2_field',
                            ]),
                ('mask',[]),
                ('coords',['nside_catalog','nside_mask','nside_likelihood',
                           'nside_pixel','roi_radius','roi_radius_annulus',
                           'roi_radius_interior','coordsys',
                           ]),
                ('likelihood',[]),
                ('output',[]),
                ('batch',[]),
                ])  

        # A HATS catalog replaces the per-pixel FITS files
        if self.get('catalog',{}).get('hats'):
            sections['catalog'] = [k for k in sections['catalog']
                                   if k not in ('dirname','basename')]
            if 'path' not in self['catalog']['hats']:
                raise Exception("Missing key in catalog.hats: 'path'")

        keys = np.array(list(sections.keys()))
        found = np.isin(keys,list(self.keys()))

        if not np.all(found):
            msg = 'Missing sections: '+str(keys[~found])
            raise Exception(msg)

        for section,keys in sections.items():
            keys = np.array(keys)
            found = np.isin(keys,list(self[section].keys()))
            if not np.all(found):
                msg = 'Missing keys in %s: '%(section)+str(keys[~found])
                raise Exception(msg)

        #if hasattr(self,'filenames'):
        #    len(self.filenames)
            
    def _formatFilepaths(self):
        """
        Join dirnames and filenames from config.
        """
        likedir=self['output']['likedir']
        self.likefile  = join(likedir,self['output']['likefile'])

        mergedir=self['output'].get('mergedir',likedir)
        self.mergefile = join(mergedir,self['output']['mergefile'])
        self.roifile   = join(mergedir,self['output']['roifile'])

        searchdir=self['output']['searchdir']
        self.labelfile  = join(searchdir,self['output']['labelfile'])
        self.objectfile = join(searchdir,self['output']['objectfile'])
        self.assocfile  = join(searchdir,self['output']['assocfile'])
        self.candfile   = join(searchdir,self['output']['candfile'])

        mcmcdir=self['output']['mcmcdir']
        self.mcmcfile   = join(mcmcdir,self['output']['mcmcfile'])

    def write(self, filename):
        """
        Write a copy of this config object.

        Parameters:
        -----------
        outfile : output filename
        
        Returns:
        --------
        None
        """
        ext = os.path.splitext(filename)[1]
        writer = open(filename, 'w')
        if ext == '.py':
            writer.write(pprint.pformat(self))
        elif ext == '.yaml':
            writer.write(yaml.dump(self))
        else:
            writer.close()
            raise Exception('Unrecognized config format: %s'%ext)
        writer.close()

    def _createFilenames(self):
        """
        Create a masked records array of all filenames for the given set of
        pixels and store the existence of those files in the mask values.

        Parameters:
        -----------
        None

        Returns:
        --------
        recarray : pixels and mask value
        """
        nside_catalog = self['coords']['nside_catalog']
        npix = hp.nside2npix(nside_catalog)
        pixels = np.arange(npix)

        hats = bool(self['catalog'].get('hats'))
        if hats:
            # Catalog comes from a HATS catalog (see
            # ugali.observation.hats_catalog); there are no per-pixel
            # catalog files, so every pixel is considered covered and
            # only the mask files decide which pixels exist.
            catalog_dir = None
            catalog_path = 'hats:%i'
        else:
            catalog_dir = os.path.expandvars(self['catalog']['dirname'])
            if not os.path.isdir(catalog_dir):
                msg = "Directory does not exist: %s"%catalog_dir
                raise IOError(msg)
            catalog_base = self['catalog']['basename']
            catalog_path = os.path.join(catalog_dir,catalog_base)

        healsparse = self['mask'].get('healsparse')
        if healsparse:
            # Masks come from HealSparse depth maps: a pixel exists when
            # the maps of both bands cover it (see healsparse_mask).
            mask_dir = None
            mask_path_1 = 'healsparse:%i'
            mask_path_2 = 'healsparse:%i'
        else:
            mask_dir    = os.path.expandvars(self['mask']['dirname'])
            if not os.path.isdir(mask_dir):
                msg = "Directory does not exist: %s"%mask_dir
                raise IOError(msg)
            mask_base_1 = self['mask']['basename_1']
            mask_base_2 = self['mask']['basename_2']
            mask_path_1 = os.path.join(mask_dir,mask_base_1)
            mask_path_2 = os.path.join(mask_dir,mask_base_2)

        data = np.ma.empty(npix,dtype=[('pix',int), ('catalog',object), 
                                       ('mask_1',object), ('mask_2',object)])
        mask = np.ma.empty(npix,dtype=[('pix',bool), ('catalog',bool), 
                                       ('mask_1',bool), ('mask_2',bool)])

        # Build the filenames
        data['pix']     = pixels
        data['catalog'] = np.char.mod(catalog_path,pixels)
        data['mask_1']  = np.char.mod(mask_path_1,pixels)
        data['mask_2']  = np.char.mod(mask_path_2,pixels)

        # Build the mask of existing files using glob
        if hats:
            mask['catalog'] = False
        else:
            mask['catalog'] = ~np.isin(data['catalog'],glob.glob(catalog_dir+'/*'))
        if healsparse:
            from ugali.observation.healsparse_mask import covered_pixels
            mask['mask_1'] = ~covered_pixels(healsparse['maglim_1'],nside_catalog)
            mask['mask_2'] = ~covered_pixels(healsparse['maglim_2'],nside_catalog)
        else:
            mask['mask_1']  = ~np.isin(data['mask_1'],glob.glob(mask_dir+'/*'))
            mask['mask_2']  = ~np.isin(data['mask_2'],glob.glob(mask_dir+'/*'))

        for name in ['catalog','mask_1','mask_2']:
            if np.all(mask[name]): logger.warning("All '%s' files masked"%name)

        # mask 'pix' if all files not present
        mask['pix'] = mask['catalog'] | mask['mask_1'] | mask['mask_2']

        if np.all(mask['pix']): logger.warning("All pixels masked")

        return np.ma.MaskedArray(data, mask, fill_value=[-1,'','',''])

    def getFilenames(self,pixels=None):
        """
        Return the requested filenames.

        Parameters:
        -----------
        pixels : requeseted pixels

        Returns:
        --------
        filenames : recarray
        """
        logger.debug("Getting filenames...")
        if pixels is None:
            return self.filenames
        else:
            return self.filenames[np.isin(self.filenames['pix'],pixels)]

    getCatalogFiles = getFilenames

        
############################################################
