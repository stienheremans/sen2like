#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Telespazio Germany GmbH.
#
# This file is part of Sen2Cor3.
# See NOTICE.txt for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import xarray as xr
import rioxarray
import glob
import shutil
from L2A_AuxImport_Landsat import AuxImport, is_dted
from L2A_BandIdentifier import Band
from PIL import Image
from L2A_Library import showSceneClassification, showImage, showImageEqualized

try:
    from osgeo import gdal, osr
    from osgeo.gdalconst import *

    gdal.TermProgress = gdal.TermProgress_nocb
except ImportError:
    import gdal, osr
    from gdalconst import *
#
# see: https://www.usgs.gov/core-science-systems/nli/landsat/using-usgs-landsat-level-1-data-product
# for conversion formulas

def test_refl_rad(band):
    print


def dn2refl(band, M_p, A_p, S_e):
#    refl = ((band * M_p) + A_p) / np.sin(np.radians(S_e))
    refl = band
    refl = xr.where(refl != 0,((refl * M_p) + A_p) / np.sin(np.radians(S_e)), np.int32(0))
    return refl

def dn2rad(band, M_l, A_l):
#    rad = ((band * M_l) + A_l) * 0.1
    rad = band
    rad= xr.where(rad !=0, ((rad * M_l) + A_l) * 0.1, np.int32(0))
    return rad

def surfrefl2dn(band, M_p, A_p):
#    dn = ((band / M_p) - A_p) * np.sin(np.radians(S_e))
    dn = band
    # dn = xr.where(dn !=0, (((dn* np.sin(np.radians(S_e))) - A_p) / M_p), np.uint16(0))
    dn = xr.where(dn != 0, ((dn - A_p) / M_p), np.uint16(0))
    return dn

def refl2rad(band):
    # CoastalAerosol           Blue          Green            Red            NIR
    #        1895.33        2004.57        1820.75        1549.49         951.76
    #         Cirrus          SWIR1          SWIR2            Pan
    #         366.97         247.55          85.46        1723.88
    # return array * self.config.e0[index] * cos(radians(self.config.solze)) / pi
    E_sun = 1549.49 / 10.0
    S_aa = 150.85190143
    S_ea = 52.75523492
    S_za = 90 - S_ea
    rad = band * E_sun * np.cos(np.radians(S_za)) / np.pi
    return rad


def refl2rad_esun(xarr, index, S_e):
    # Landsat 8/9
    # CoastalAerosol           Blue          Green            Red            NIR
    #        1895.33        2004.57        1820.75        1549.49         951.76
    #         Cirrus          SWIR1          SWIR2            Pan
    #         366.97         247.55          85.46        1723.88
    # return array * self.config.e0[index] * cos(radians(self.config.solze)) / pi
    E_sun = np.array([1895.33, 2004.57, 1820.75, 1549.49, 951.76, 247.55, 85.46, 1723.88, 366.97]) * 0.1
    rad = xarr
    rad = xr.where(rad != 0, rad * E_sun[index] * np.sin(np.radians(S_e)) / np.pi, np.int32(0))
    return rad


def rad2dn(band, M_l, A_l):
    dn = band / M_l - A_l
    return dn.astype.np.uint16

class BandIO:
    def __init__(self, config):
        prefix = os.path.join(config.input_dir, config.l1c_tile_id)
        self._L1C_bandDir = config.input_dir #for SC_EVO
        # always get image dimensions from metadata and make globally accessible:
        if config.collection_number == 1: #coll_1
            config.panc_nrows = config.metadata['L1_METADATA_FILE']['PRODUCT_METADATA']['PANCHROMATIC_LINES']
            config.panc_ncols = config.metadata['L1_METADATA_FILE']['PRODUCT_METADATA']['PANCHROMATIC_SAMPLES']
            config.refl_nrows = config.metadata['L1_METADATA_FILE']['PRODUCT_METADATA']['REFLECTIVE_LINES']
            config.refl_ncols = config.metadata['L1_METADATA_FILE']['PRODUCT_METADATA']['REFLECTIVE_SAMPLES']
        else:  # coll_2
            config.panc_nrows = config.metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['PANCHROMATIC_LINES']
            config.panc_ncols = config.metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['PANCHROMATIC_SAMPLES']
            config.refl_nrows = config.metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['REFLECTIVE_LINES']
            config.refl_ncols = config.metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['REFLECTIVE_SAMPLES']

        self.name_file_reference=(prefix + '_B2.TIF')
        dataset_reference = gdal.Open(self.name_file_reference)
        self.projection_reference = dataset_reference.GetProjection()
        self.geotransform_reference = dataset_reference.GetGeoTransform()
        dataset_reference = None

        if config.ROI == 'OFF' or config.ROI == 'AUTO': # if not config.TESTMODE:
            self.dataset = xr.Dataset({
                'coastal_aerosol': rioxarray.open_rasterio(prefix + '_B1.TIF')[0, :, :].drop('band'),
                'blue': rioxarray.open_rasterio(prefix + '_B2.TIF')[0, :, :].drop('band'),
                'green': rioxarray.open_rasterio(prefix + '_B3.TIF')[0, :, :].drop('band'),
                'red': rioxarray.open_rasterio(prefix + '_B4.TIF')[0, :, :].drop('band'),
                'near_infrared': rioxarray.open_rasterio(prefix + '_B5.TIF')[0, :, :].drop('band'),
                'short_wave_infrared_1': rioxarray.open_rasterio(prefix + '_B6.TIF')[0, :, :].drop('band'),
                'short_wave_infrared_2': rioxarray.open_rasterio(prefix + '_B7.TIF')[0, :, :].drop('band'),
                'panchromatic': rioxarray.open_rasterio(prefix + '_B8.TIF')[0, :, :].drop('band'). \
                    rename({'x': 'x_panchromatic', 'y': 'y_panchromatic'}),
                'cirrus': rioxarray.open_rasterio(prefix + '_B9.TIF')[0, :, :].drop('band'),
                'thermal_infrared_1': rioxarray.open_rasterio(prefix + '_B10.TIF')[0, :, :].drop('band'),
                'thermal_infrared_2': rioxarray.open_rasterio(prefix + '_B11.TIF')[0, :, :].drop('band'),
                # 'quality_assessment': rioxarray.open_rasterio(prefix + '_BQA.TIF')[0, :, :].drop('band')
            })
            config.calc_region_of_interest(self.dataset)
            if config.ROI == "AUTO":
                refl_rmin, refl_cmin, refl_rmax, refl_cmax = config.get_region_of_interest()
                self.auto_rmin, self.auto_cmin, self.auto_rmax, self.auto_cmax = refl_rmin, refl_cmin, refl_rmax, refl_cmax
                panc_rmin, panc_cmin, panc_rmax, panc_cmax = [x * 2 for x in (refl_rmin, refl_cmin, refl_rmax, refl_cmax)]
                self.dataset = xr.Dataset({ #There has to be a way to modify this dataset object thing inplace instead of reopening everything a second time...
                    'coastal_aerosol': rioxarray.open_rasterio(prefix + '_B1.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    'blue': rioxarray.open_rasterio(prefix + '_B2.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    'green': rioxarray.open_rasterio(prefix + '_B3.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    'red': rioxarray.open_rasterio(prefix + '_B4.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    'near_infrared': rioxarray.open_rasterio(prefix + '_B5.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    'short_wave_infrared_1': rioxarray.open_rasterio(prefix + '_B6.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    'short_wave_infrared_2': rioxarray.open_rasterio(prefix + '_B7.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    'panchromatic': rioxarray.open_rasterio(prefix + '_B8.TIF')[0, :, :].drop('band') \
                        [panc_rmin:panc_rmax, panc_cmin:panc_cmax].rename({'x': 'x_panchromatic', 'y': 'y_panchromatic'}),
                    'cirrus': rioxarray.open_rasterio(prefix + '_B9.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    'thermal_infrared_1': rioxarray.open_rasterio(prefix + '_B10.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    'thermal_infrared_2': rioxarray.open_rasterio(prefix + '_B11.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                    # 'quality_assessment': rioxarray.open_rasterio(prefix + '_BQA.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax,refl_cmin:refl_cmax]
                })
                config.ROI = 'MANUAL'
                # Not sure how to proceed from here. Is this probably correct thus far? Do you need to set ROI to MANUAL now?
        else:
            # get sample size from config:
            row_offset = int(round(config.nrow_win) * 0.5)
            col_offset = int(round(config.ncol_win) * 0.5)
            refl_rmin = config.row0 - row_offset
            refl_rmax = config.row0 + row_offset
            refl_cmin = config.col0 - col_offset
            refl_cmax = config.col0 + col_offset
            panc_rmin = config.row0 * 2 - row_offset * 2
            panc_rmax = config.row0 * 2 + row_offset * 2
            panc_cmin = config.col0 * 2 - col_offset * 2
            panc_cmax = config.col0 * 2 + col_offset * 2

            # clip the sample window:
            if refl_rmin < 0: refl_rmin = 0
            if refl_rmax > config.refl_nrows: refl_rmax = config.refl_nrows
            if refl_cmin < 0: refl_cmin = 0
            if refl_cmax > config.refl_ncols: refl_cmax = config.refl_ncols

            if panc_rmin < 0: panc_rmin = 0
            if panc_rmax > config.panc_nrows: panc_rmax = config.panc_nrows
            if panc_cmin < 0: panc_cmin = 0
            if panc_cmax > config.panc_ncols: panc_cmax = config.panc_ncols

            # convert corner coordinates back into original system
            config.row0 = (refl_rmax + refl_rmin) / 2
            config.col0 = (refl_cmax + refl_cmin) / 2
            config.nrow_win = int(refl_rmax - refl_rmin)
            config.ncol_win = int(refl_cmax - refl_cmin)

            self.dataset = xr.Dataset({
                'coastal_aerosol': rioxarray.open_rasterio(prefix + '_B1.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                'blue': rioxarray.open_rasterio(prefix + '_B2.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                'green': rioxarray.open_rasterio(prefix + '_B3.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                'red': rioxarray.open_rasterio(prefix + '_B4.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                'near_infrared': rioxarray.open_rasterio(prefix + '_B5.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                'short_wave_infrared_1': rioxarray.open_rasterio(prefix + '_B6.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                'short_wave_infrared_2': rioxarray.open_rasterio(prefix + '_B7.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                'panchromatic': rioxarray.open_rasterio(prefix + '_B8.TIF')[0, :, :].drop('band') \
                    [panc_rmin:panc_rmax, panc_cmin:panc_cmax].rename({'x': 'x_panchromatic', 'y': 'y_panchromatic'}),
                'cirrus': rioxarray.open_rasterio(prefix + '_B9.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                'thermal_infrared_1': rioxarray.open_rasterio(prefix + '_B10.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                'thermal_infrared_2': rioxarray.open_rasterio(prefix + '_B11.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax, refl_cmin:refl_cmax],
                # 'quality_assessment': rioxarray.open_rasterio(prefix + '_BQA.TIF')[0, :, :].drop('band')[refl_rmin:refl_rmax,refl_cmin:refl_cmax]
            })
            config.calc_region_of_interest(self.dataset)
        self.ds_tmp = xr.Dataset()
        if config.collection_number == 1: #coll_1
            S_e = config.metadata['L1_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
        else: #coll_2
            S_e = config.metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
        for i, band in enumerate(self.dataset):
            # skip panchromatic and thermal_infrared bands
            if band == 'panchromatic' or band.startswith('thermal_infrared_'):
                continue
            # set zero values at the border to NaN
            # caution: if there are zero values in the image itself, they will also set to NaN
            # current assumption: zero values only exist at the border
#            self.dataset[band] = self.dataset[band].where(self.dataset[band] > 0) #FCP otherwhise all the output are 0

        self.bands = [Band.COASTAL_AEROSOL, Band.BLUE, Band.GREEN, Band.RED, Band.NEAR_INFRARED,
                      Band.SHORT_WAVE_INFRARED_1, Band.SHORT_WAVE_INFRARED_2,  # Band.PANCHROMATIC,
                      Band.CIRRUS, Band.THERMAL_INFRARED_1, Band.THERMAL_INFRARED_2]  # , Band.QUALITY_ASSESSMENT]
        config.nrows, config.ncols = self.dataset.coastal_aerosol.shape
        self.config = config
        self.logger = config.logger


    def import_aux_band_list(self):
        aux_import = AuxImport(self.config)
        self.lonMin_for_snowmap = aux_import.lonMin
        self.lonMax_for_snowmap = aux_import.lonMax
        self.latMin_for_snowmap = aux_import.latMin
        self.latMax_for_snowmap = aux_import.latMax
        demDir = self.config.demDirectory

        demDirectory = os.path.join(self.config.home, demDir)
        if not os.path.exists(demDirectory):
            self.config.timestamp("L2A_BAND_IO: dem directory does not exist, it will be created")
            os.mkdir(demDirectory)

        failed = False
        aux_import.dem_error_type = 'False' #update from Sen2Cor 2.10
        aux_import._input_dem_resolution = 0.0 #update from Sen2Cor 2.10

        if self.config.midLatitude == 'AUTO':
            aux_import.setMidLatitude()

        if self.config.ozoneSetpoint == 0:
            try:
                ozone_file =  aux_import.gdalCAMS_daily('gtco3')
                ozoneSetpoint = rioxarray.open_rasterio(ozone_file)[0, :, :].drop('band').mean().astype(int)
                self.config.timestamp("L2A_BAND_IO: ozone value retrieved from CAMS daily")
                self.config.ozoneSource = 'CAMS'
            except:
                self.logger.warning('no ozone values found in input data, default (331) will be used')
                ozoneSetpoint = 331
                self.config.ozoneSource = 'OTHER'

            self.config.setOzoneContentFromMetadata(ozoneSetpoint)

        elif self.config.aerosolType != 'AUTO':
            self.config.createAtmDataFilename()

        if demDir == 'NONE':
            self.logger.info('DEM directory not specified, flat surface is used, and DEM will not be exported')
            aux_import.dem_error_type = 'False'
            self.config.demOutput = False
            self.logger.info('DEM directory not specified, ESA CCI auxiliary data will not be loaded')
            self.config.demType = 'NONE'
            self.config.dem_terrain_correction = False
        else:
            # check if DEM is a DTED type, these files must exist in the given directory:
            if is_dted(self.config):
                # yes it is, run dem preparation for DTED:
                try:
                    demfile, dem_band = aux_import.gdalDEM_dted()
                    self.importBand(Band.DIGITAL_ELEVATION_MAP, dem_band)
                    self._input_dem_resolution = aux_import._input_dem_resolution
                    self.config.timestamp('L2A_BANDIO: band ' + Band.DIGITAL_ELEVATION_MAP.name + ' imported')
                    if self._input_dem_resolution == 0.00027778:
                        self.config.demType = 'DTED_30'
                    else:
                        self.config.demType = 'DTED'
                except:
                    if self.config.demError == True and aux_import.dem_error_type == True:
                        self.logger.stream('Application will terminate')
                        return False
                    self.config.demDirectory = 'NONE'
                    self.config.demType = 'NONE'
                    self.config.demOutput = False
                    self.config.timestamp('L2A_BAND_IO: Application will continue with a flat terrain')
            else:
                try:
                    if 'srtm' in (self.config.demReference).lower(): # run DEM preparation for SRTM:
                        demfile, dem_band = aux_import.gdalDEM_srtm()
                        self.importBand(Band.DIGITAL_ELEVATION_MAP, dem_band)
                        self._input_dem_resolution = aux_import._input_dem_resolution
                        self.config.timestamp('L2A_BAND_IO: band ' + Band.DIGITAL_ELEVATION_MAP.name + ' imported')
                        self.config.demType = 'SRTM'
                    elif 'aws' in (self.config.demReference).lower(): # run DEM preparation for AWS
                        demfile, dem_band = aux_import.gdalDEM_aws()
                        self.importBand(Band.DIGITAL_ELEVATION_MAP, dem_band)
                        self._input_dem_resolution = aux_import._input_dem_resolution
                        self.config.timestamp('L2A_BAND_IO: band ' + Band.DIGITAL_ELEVATION_MAP.name + ' imported')
                        if self._input_dem_resolution == 0.00027778:
                            self.config.demType = 'AWS_COPERNICUS_30'
                        else:
                            self.config.demType = 'AWS_COPERNICUS_90'
                    else: # run DEM preparation for Copernicus
                        demfile, dem_band = aux_import.gdalDEM_copernicus()
                        self.importBand(Band.DIGITAL_ELEVATION_MAP, dem_band)
                        self._input_dem_resolution = aux_import._input_dem_resolution
                        self.config.timestamp('L2A_BAND_IO: band ' + Band.DIGITAL_ELEVATION_MAP.name + ' imported')
                        if self._input_dem_resolution == 0.00027778:
                            self.config.demType = 'COPERNICUS_30'
                        else:
                            self.config.demType = 'COPERNICUS_90'
                except:
                    if self.config.demError == True and aux_import.dem_error_type == True:
                        self.logger.stream('Application will terminate')
                        return False
                    self.config.demDirectory = 'NONE'
                    self.config.demType = 'NONE'
                    self.config.demOutput = False
                    self.config.timestamp('L2A_BAND_IO: Application will continue with a flat terrain')

            if self.config.demType != 'NONE':
                # generate hill shadow, slope and aspect using DEM:
                # sdwfile = aux_import.gdalDEM_Shade(demfile)
                sdwfile = self.gdalDEM_Shade_Cast(demfile)
                try:
                    self.importBand(Band.SHADOW_MAP, sdwfile)
                    self.config.timestamp('L2A_BAND_IO: band ' + Band.SHADOW_MAP.name + ' imported')
                except:
                    self.logger.fatal('error generating DEM shadow')
                    failed = True

                slpfile = aux_import.gdalDEM_Slope(demfile)
                try:
                    self.importBand(Band.SLOPE, slpfile)
                    self.config.timestamp('L2A_BAND_IO: band ' + Band.SLOPE.name + ' imported')
                except:
                    self.logger.fatal('error generating DEM slope')
                    failed = True

                if self.config.resolution > 10:
                    aspfile = aux_import.gdalDEM_Aspect(demfile)
                    try:
                        self.importBand(Band.ASPECT, aspfile)
                        self.config.timestamp('L2A_BAND_IO: band ' + Band.ASPECT.name + ' imported')
                    except:
                        self.logger.fatal('error generating DEM aspect')
                        failed = True

            if failed == True:
                return False

        # else continue with import of other aux data:
        if self.config.resolution > 10:
            if self.config.demDirectory != 'NONE':
                auxfile = aux_import.gdalCCI_wb()
                try:
                    self.importBand(Band.WATER_BODY_INDEX, auxfile)
                    self.config.timestamp('L2A_BAND_IO: band ' + Band.WATER_BODY_INDEX.name + ' imported')
                except:
                    # Continue without ESA CCI Water Bodies (150m) a priori information
                    pass

                auxfile = aux_import.gdalCCI_lccs()
                try:
                    self.importBand(Band.LAND_COVERAGE_MAP, auxfile)
                    self.config.timestamp('L2A_BAND_IO: band ' + Band.LAND_COVERAGE_MAP.name + ' imported')
                except:
                    # Continue without ESA CCI Land Cover (300m) a priori information (urban)
                    pass

                auxfile = aux_import.gdalCCI_snowc()
                try:
                    self.importBand(Band.SNOW_CONDITION_MAP, auxfile)
                    self.config.timestamp('L2A_BAND_IO: band ' + Band.SNOW_CONDITION_MAP.name + ' imported')
                except:
                    # Continue without ESA CCI Snow Condition (500m) a priori information
                    pass

#           implementation of daily cams (aod550)
            vis_output = 0
            try:
                aux_aod550 = aux_import.gdalCAMS_daily('aod550')
                geopotential = aux_import.import_geopotential('geopotential')
                auxfile = aux_import.compute_visibility_landsat(aux_aod550, geopotential)
                vis_output = 1
            except:
                try:
                    aux_aod550 = aux_import.gdalCAMS_monthly('aod550')
                    geopotential = aux_import.import_geopotential('geopotential')
                    auxfile = aux_import.compute_visibility_landsat(aux_aod550, geopotential)
                    vis_output = 1
                except:
                    vis_output = 0

            if vis_output == 0:
                auxfile = aux_import.gdalCAMS_aod550()
                vis_output = 0

            try:
                if vis_output == 1 :
                    self.setBand(Band.VISIBILITY_INDEX_MAP, auxfile)  # importband
                    self.config.timestamp('L2A_BAND_IO: band ' + Band.VISIBILITY_INDEX_MAP.name + ' imported')
                elif  vis_output == 0 :
                    self.importBand(Band.VISIBILITY_INDEX_MAP, auxfile)
                    self.config.timestamp('L2A_BAND_IO: band ' + Band.VISIBILITY_INDEX_MAP.name + ' imported from local')
            except:
                # Continue without CAMS ECMWF aerosol forecast information
                self.config.timestamp("aux_import: continue without CAMS ECMWF aerosol forecast information")

            # implementation of daily cams (water vapour)
            # get the mask for the no data values
            mask = self.get_no_data_map()
            auxfile_name = False
            try:
                auxfile_name = aux_import.gdalCAMS_daily('tcwv') # try with daily CAMS
            except:
                auxfile_name = False
            if auxfile_name == False:
                try:
                    auxfile_name = aux_import.gdalCAMS_monthly('tcwv') # if not available try to find monthly CAMS
                except:
                    auxfile_name = False
            if auxfile_name == False: # if also not available use historical CAMS instead
                try:
                    auxfile = aux_import.noCAMS_waterVapour_fallback(mask)
                    self.setBand(Band.WATER_VAPOUR, auxfile.data)
                    self.config.timestamp('L2A_BAND_IO: band ' + Band.WATER_VAPOUR.name + ' imported')
                except:
                    self.logger.fatal('Error in generating water vapour band')
                    return False
            else: # handling daily and monthly CAMS
                try:
                    # dataset = gdal.Open(auxfile_name, gdal.GA_Update)
                    # band_data = dataset.GetRasterBand(1).ReadAsArray()
                    # band_data[mask == 0] = 0
                    # dataset.GetRasterBand(1).WriteArray(band_data)
                    # dataset = None
                    # self.importBand(Band.WATER_VAPOUR, auxfile_name)
                    auxfile = rioxarray.open_rasterio(auxfile_name)[0, :, :].drop('band')
                    auxfile = auxfile.where(mask != 0, 0) # no data values should be 0 instead of what cams data proposes
                    self.setBand(Band.WATER_VAPOUR, auxfile.data)
                    self.config.timestamp('L2A_BAND_IO: band ' + Band.WATER_VAPOUR.name + ' imported')
                except:
                    self.logger.fatal('Error in generating water vapour band')
                    return False
        try:
            shutil.rmtree(aux_import.tmpdir)
            del aux_import
        except:
            pass

        return True


    def getBand(self, identifier, radiance=False):
        if not identifier.is_msi():
            return self.dataset[identifier.name.lower()].values
        for i, band in enumerate(self.dataset):
            if identifier.name == 'PANCHROMATIC' or identifier.name.startswith('THERMAL_INFRARED'):
                # skip panchromatic and thermal infrared band:
                return self.dataset[band].values
            # skip all other bands except selected one:
            if not identifier.name.lower() in band:
                continue
            if not radiance:
                # convert to reflectance:
                if self.config.collection_number == 1: #coll_1
                    M_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
                        'REFLECTANCE_MULT_BAND_' + str(i + 1)]
                    A_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
                        'REFLECTANCE_ADD_BAND_' + str(i + 1)]
                    S_e = self.config.metadata['L1_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
                    dataset = dn2refl(self.dataset[band], M_p, A_p, S_e)
                else: #coll_2
                    M_p = self.config.metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][
                        'REFLECTANCE_MULT_BAND_' + str(i + 1)]
                    A_p = self.config.metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][
                        'REFLECTANCE_ADD_BAND_' + str(i + 1)]
                    S_e = self.config.metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
                    dataset = dn2refl(self.dataset[band], M_p, A_p, S_e)
            else:
                if self.config.collection_number == 1: #coll_1
                    # first convert to reflectance:
                    M_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
                        'REFLECTANCE_MULT_BAND_' + str(i + 1)]
                    A_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
                        'REFLECTANCE_ADD_BAND_' + str(i + 1)]
                    S_e = self.config.metadata['L1_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
                    dataset = dn2refl(self.dataset[band], M_p, A_p, S_e)
                    # then convert to radiance:
                    dataset = refl2rad_esun(dataset, i, S_e)
                else: #coll_2
                    M_p = self.config.metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][
                        'REFLECTANCE_MULT_BAND_' + str(i + 1)]
                    A_p = self.config.metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][
                        'REFLECTANCE_ADD_BAND_' + str(i + 1)]
                    S_e = self.config.metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
                    dataset = dn2refl(self.dataset[band], M_p, A_p, S_e)
                    # then convert to radiance:
                    dataset = refl2rad_esun(dataset, i, S_e)

            break
        return dataset.values


    # def getBand(self, identifier, radiance=False):
    #     if not identifier.is_msi():
    #         return self.dataset[identifier.name.lower()].values
    #     for i, band in enumerate(self.dataset):
    #         if identifier.name == 'PANCHROMATIC' or identifier.name.startswith('THERMAL_INFRARED'):
    #             # skip panchromatic and thermal infrared band:
    #             return self.dataset[band].values
    #         # skip all other bands except selected one:
    #         if not identifier.name.lower() in band:
    #             continue
    #         if not radiance:
    #             # convert to reflectance:
    #             M_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
    #                 'REFLECTANCE_MULT_BAND_' + str(i + 1)]
    #             A_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
    #                 'REFLECTANCE_ADD_BAND_' + str(i + 1)]
    #             S_e = self.config.metadata['L1_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
    #             dataset = dn2refl(self.dataset[band], M_p, A_p, S_e)
    #         else:
    #             # convert to radiance:
    #             M_l = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
    #                 'RADIANCE_MULT_BAND_' + str(i + 1)]
    #             A_l = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
    #                 'RADIANCE_ADD_BAND_' + str(i + 1)]
    #             dataset = dn2rad(self.dataset[band], M_l, A_l)
    #         break
    #     return dataset.values


    def testReflRad(self, identifier):
        if not identifier.is_msi():
            return self.dataset[identifier.name.lower()].values
        for i, band in enumerate(self.dataset):
            if identifier.name == 'PANCHROMATIC' or identifier.name.startswith('THERMAL_INFRARED'):
                # skip panchromatic and thermal infrared band:
                return self.dataset[band].values
            # skip all other bands except selected one:
            if not identifier.name.lower() in band:
                continue
            # convert to reflectance:
            M_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
                'REFLECTANCE_MULT_BAND_' + str(i + 1)]
            A_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
                'REFLECTANCE_ADD_BAND_' + str(i + 1)]
            S_e = self.config.metadata['L1_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
            data = dn2refl(self.dataset[band], M_p, A_p, S_e).data
            print(data.min(), data.max(), data.mean())
            refl = data
            # convert to radiance:
            M_l = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
                'RADIANCE_MULT_BAND_' + str(i + 1)]
            A_l = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING'][
                'RADIANCE_ADD_BAND_' + str(i + 1)]
            data = dn2rad(self.dataset[band], M_l, A_l).data
            print(data.min(), data.max(), data.mean())
            data = refl2rad(refl)
            print(data.min(), data.max(), data.mean())
            break
        return


    def hasBand(self, identifier):
        return identifier.name.lower() in self.dataset

    def setBand(self, identifier, band):
        self.dataset.update({identifier.name.lower(): (['y', 'x'], band)})

    def importBand(self, identifier, filename):
        self.dataset.update({identifier.name.lower(): (['y', 'x'], rioxarray.open_rasterio(filename)[0, :, :].drop('band').data)})


    def getTmpBand(self, identifier):
        return self.ds_tmp[identifier.name.lower()].values

    def setTmpBand(self, identifier, band):
        self.ds_tmp.update({identifier.name.lower(): (['y', 'x'], band)})

    def sceneCouldHaveSnow(self):
#        aux_import = AuxImport(self.config)
        globalSnowMapFn = self.config.snowMapReference #aux_import.snowMapReference
        globalSnowMapFn = os.path.join(self.config.aux_dir, globalSnowMapFn)
        if ((os.path.isfile(globalSnowMapFn)) == False):
            self.logger.error('global snow map not present, snow detection will be performed')
            return True

        img = Image.open(globalSnowMapFn)
        globalSnowMap = np.array(img)
        lonMin= self.lonMin_for_snowmap
        lonMax= self.lonMax_for_snowmap
        latMin= self.latMin_for_snowmap
        latMax= self.latMax_for_snowmap

        if lonMin > 180.0:
            lonMin = lonMin - 360.0
        if lonMax > 180.0:
            lonMax = lonMax - 360.0

        # Fix for SCOR-28: International Date Line handling
        # Snow map should have a dimension of 7200 x 3600, 20 pixels per degree:
        lonMinId = int(np.rint((lonMin + 180.0) * 20.0))
        lonMaxId = int(np.rint((lonMax + 180.0) * 20.0))
        latMinId = 3600 - int(np.rint((latMax + 90.0) * 20.0))  # Inverted by intention
        latMaxId = 3600 - int(np.rint((latMin + 90.0) * 20.0))  # Inverted by intention

        if latMinId == latMaxId:
            if lonMinId == lonMaxId:
                return globalSnowMap[latMinId,lonMinId] > 0
            elif (lonMaxId - lonMinId) < 180:
            # elif lonMinId < lonMaxId:
                aoi = globalSnowMap[latMinId, lonMinId:lonMaxId]
            else: # lonMinId > lonMaxId, date line crossed:
                self.logger.info(
                    'This tile is crossing the international date line, a particular processing is performed')
                aoi_east = globalSnowMap[latMinId, 0:lonMinId]
                aoi_west = globalSnowMap[latMinId, lonMaxId:]
                aoi = np.concatenate((aoi_east, aoi_west), axis=1)
        else:
            if lonMinId == lonMaxId:
                aoi = globalSnowMap[latMinId:latMaxId, lonMinId]
            elif (lonMaxId - lonMinId) < 180:
            # elif lonMinId < lonMaxId:
                aoi = globalSnowMap[latMinId:latMaxId, lonMinId:lonMaxId]
            else: # lonMinId > latMaxId, date line crossed:
                self.logger.info(
                    'This tile is crossing the international date line, a particular processing is performed')
                aoi_east = globalSnowMap[latMinId:latMaxId, 0:lonMinId]
                aoi_west = globalSnowMap[latMinId:latMaxId, lonMaxId:]
                aoi = np.concatenate((aoi_east, aoi_west), axis=1)

        if aoi.max() > 0:
            return True

        return False

    def put_ROI_into_full_image(self, band):
        full_array = np.zeros((self.config.refl_nrows, self.config.refl_ncols), dtype=band.dtype)
        try: #manual
            x1, x2 = int(self.config.row0-self.config.nrow_win/2), int(self.config.row0+self.config.nrow_win/2)
            y1, y2 = int(self.config.col0-self.config.ncol_win/2), int(self.config.col0+self.config.ncol_win/2)
            full_array[x1:x2, y1:y2] = band
        except: #auto
            full_array[self.auto_rmin:self.auto_rmax, self.auto_cmin:self.auto_cmax] = band
        return full_array

    def export_bands(self):
        for idx, band_identifier in enumerate \
                    ([Band.COASTAL_AEROSOL, Band.BLUE, Band.GREEN, Band.RED,
                      Band.NEAR_INFRARED, Band.SHORT_WAVE_INFRARED_1, Band.SHORT_WAVE_INFRARED_2]): #Band.CIRRUS
            if not self.hasBand(band_identifier):
                continue
            try:
                self.export_msi_bands(band_identifier)
                self.config.logger.info('exporting band %s' % band_identifier.name)
                self.config.timestamp('exporting band %s' % band_identifier.name)
            except:
                self.config.logger.error('error in exporting band %s' % band_identifier.name)

        for idx, band_identifier in enumerate \
                    ([Band.SCENE_CLASSIFICATION, Band.SNOW_MAP, Band.CLOUD_MAP, Band.DARK_DENSE_VEGETATION]):
            if band_identifier == Band.DARK_DENSE_VEGETATION and self.config.ddvOutput == False:
                continue
            if not self.hasBand(band_identifier):
                continue
            try:
                self.export_quality_bands(band_identifier)
                self.config.logger.info('exporting band %s' % band_identifier.name)
                self.config.timestamp('exporting band %s' % band_identifier.name)
            except:
                self.config.logger.error('error in exporting band %s' % band_identifier.name)

        for idx, band_identifier in enumerate ([Band.AEROSOL_OPTICAL_THICKNESS, Band.WATER_VAPOUR]):
            if not self.hasBand(band_identifier):
                continue
            try:
                self.export_ac_bands(band_identifier)
                self.config.logger.info('exporting band %s' % band_identifier.name)
                self.config.timestamp('exporting band %s' % band_identifier.name)
            except:
                self.config.logger.error('error in exporting band %s' % band_identifier.name)

        if self.hasBand(Band.DIGITAL_ELEVATION_MAP): # special case just for dem due to offset
            if self.config.demOutput == True:
                try:
                    self.export_dem(Band.DIGITAL_ELEVATION_MAP)
                    self.config.logger.info('exporting band %s' % Band.DIGITAL_ELEVATION_MAP.name)
                    self.config.timestamp('exporting band %s' % Band.DIGITAL_ELEVATION_MAP.name)
                except:
                    self.config.logger.error('error in exporting band %s' % Band.DIGITAL_ELEVATION_MAP.name)

        self.export_tci()
        return True

    def export_msi_bands(self, band_identifier):
        if self.config.scOnly:
            prefix = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID, self.config.file_prefix)
        else:
            prefix = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID, self.config.L2A_TILE_ID)
        ImgFn = prefix + '_B' + str(self.reindex(band_identifier)) + '.TIF'
        if self.config.collection_number == 1: #coll_1
            M_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_' +
                                                                                    str(self.reindex(band_identifier))]
            A_p = self.config.metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_' +
                                                                                    str(self.reindex(band_identifier))]
            S_e = self.config.metadata['L1_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
        else: #coll_2
            M_p = self.config.metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][
                'REFLECTANCE_MULT_BAND_' + str(self.reindex(band_identifier))]
            A_p = self.config.metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][
                'REFLECTANCE_ADD_BAND_' + str(self.reindex(band_identifier))]
            S_e = self.config.metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']

        if self.config.scOnly:
            band = self.getBand(band_identifier)
        else:
            band = self.getTmpBand(band_identifier)
            # rho_low = (1.0 / M_p - A_p) * np.sin(np.radians(S_e))
#            band = np.clip(band, rho_low, band.max())
            mask = self.get_no_data_map()
            band[mask == 0] = 0
        if self.config.ROI == 'MANUAL':
            band = self.put_ROI_into_full_image(band)

        band = surfrefl2dn(band, M_p, A_p)
        # band = band * 10000
        band[band < 0] = 1
        band = band.astype(np.uint16)

        (h, w) = band.shape
        driver = gdal.GetDriverByName('GTiff')
        creation_options = ["COMPRESS=LZW"]
        ds = driver.Create(ImgFn, w, h, 1, gdal.GDT_UInt16, options=creation_options)
        ds.SetGeoTransform(self.geotransform_reference)
        ds.SetProjection(self.projection_reference)
        outBand = ds.GetRasterBand(1)
        outBand.WriteArray(band)
        outBand.SetScale(float(M_p))
        outBand.SetOffset(float(A_p))
        outBand.SetNoDataValue(0)
        outBand.FlushCache()
        ds = None
        return

    def export_quality_bands(self, band_identifier):
        prefix = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID, self.config.L2A_TILE_ID)
        ImgFn = prefix + '_' + str(band_identifier.value) + '.TIF'
        band = self.getBand(band_identifier)
        if self.config.ROI == 'MANUAL':
            band = self.put_ROI_into_full_image(band)
        (h, w) = band.shape
        driver = gdal.GetDriverByName('GTiff')
        creation_options = ["COMPRESS=LZW"]
        ds = driver.Create(ImgFn, w, h, 1, gdal.GDT_Byte, options=creation_options)
        ds.SetGeoTransform(self.geotransform_reference)
        ds.SetProjection(self.projection_reference)
        outBand = ds.GetRasterBand(1)
        outBand.WriteArray(band)
        #outBand.SetNoDataValue(0) # not applicable for Cloud and Snow probabilities
        outBand.FlushCache()
        ds = None

        if band_identifier == Band.SCENE_CLASSIFICATION and self.config.scCog:
            filename_cog = os.path.splitext(ImgFn)[0] + '.TIF'
            # self.glymurWrapper(filename, band)
            srcfile = gdal.Open(ImgFn)
            MEM_driver = gdal.GetDriverByName("MEM")
            MEM_dataset = MEM_driver.CreateCopy("", srcfile)
            MEM_dataset.BuildOverviews(resampling="MODE", overviewlist=(2, 4, 12))
            # kwargs = "-co TILED=YES -co COPY_SRC_OVERVIEWS=YES -co GDAL_TIFF_OVR_BLOCKSIZE=1024 -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co COMPRESS=LZW"
            kwargs = "-co TILED=YES -co COPY_SRC_OVERVIEWS=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co COMPRESS=LZW"
            os.environ["GDAL_TIFF_OVR_BLOCKSIZE"] = "1024"
            os.remove(ImgFn)
            gdal.Translate(filename_cog, MEM_dataset, options=kwargs)

        return

    def export_ac_bands(self, band_identifier):
        prefix = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID, self.config.L2A_TILE_ID)
        ImgFn = prefix + '_' + str(band_identifier.value) + '.TIF'
        band = self.getBand(band_identifier)
        if self.config.ROI == 'MANUAL':
            band = self.put_ROI_into_full_image(band)
        band = band.astype(np.uint16)
        (h, w) = band.shape
        driver = gdal.GetDriverByName('Gtiff')
        creation_options = ["COMPRESS=LZW"]
        ds = driver.Create(ImgFn, w, h, 1, gdal.GDT_UInt16, options=creation_options)
        ds.SetGeoTransform(self.geotransform_reference)
        ds.SetProjection(self.projection_reference)
        outBand = ds.GetRasterBand(1)
        outBand.WriteArray(band)
        scale = 1e-03
        offset = 0.0
        outBand.SetScale(scale)
        outBand.SetOffset(offset)
        outBand.SetNoDataValue(0)
        outBand.FlushCache()
        ds = None
        return

    def export_dem(self, band_identifier):
        prefix = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID, self.config.L2A_TILE_ID)
        ImgFn = prefix + '_' + str(band_identifier.value) + '.TIF'
        band = self.getBand(band_identifier)
        band[self.get_no_data_map() == 1] += 10000 # apply offset to handle negative values only where there is actual data
        if self.config.ROI == 'MANUAL':
            band = self.put_ROI_into_full_image(band)
        band = band.astype(np.uint16)
        (h, w) = band.shape
        driver = gdal.GetDriverByName('Gtiff')
        creation_options = ["COMPRESS=LZW"]
        ds = driver.Create(ImgFn, w, h, 1, gdal.GDT_UInt16, options=creation_options)
        ds.SetGeoTransform(self.geotransform_reference)
        ds.SetProjection(self.projection_reference)
        outBand = ds.GetRasterBand(1)
        outBand.WriteArray(band)
        outBand.FlushCache()
        ds = None
        return

    def export_tci(self):
        prefix = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID, self.config.L2A_TILE_ID)
        ImgFn = prefix + '_TCI.TIF'

        r = self.scaleTci(Band.RED)
        g = self.scaleTci(Band.GREEN)
        b = self.scaleTci(Band.BLUE)

        data_cube = np.dstack((r, g, b))
        (h, w, num_bands) = data_cube.shape
        driver = gdal.GetDriverByName('Gtiff')
        creation_options = ["COMPRESS=LZW"]
        ds = driver.Create(ImgFn, w, h, num_bands, gdal.GDT_Byte, options=creation_options)
        ds.SetGeoTransform(self.geotransform_reference)
        ds.SetProjection(self.projection_reference)
        for b in range(0, num_bands):
            outBand = ds.GetRasterBand(b + 1)
            outBand.WriteArray(data_cube[:, :, b].astype(np.uint8))
            outBand.SetNoDataValue(0)
        outBand.FlushCache()
        ds = None
        return

    def scaleTci(self, band_identifier):
        # band is already reflectance:
        if self.config.scOnly:
            band = self.getBand(band_identifier)
        else:
            band = self.getTmpBand(band_identifier)
        if self.config.ROI == 'MANUAL':
            band = self.put_ROI_into_full_image(band)

        min_ = np.float32(0.0)
        max_ = np.float32(0.25)
        scale = np.float32(254.0)
        offset = np.float32(1.0)

        # scaling in line with L1C TCI with 0 reserved for No_Data
        scaledArr = np.uint8(np.clip(band, min_, max_) * scale / max_ + offset)
        scaledArr[band == 0.0] = 0
        return scaledArr

    def reindex(self, band_index):
        new_index = \
            ['NA', # NA
             1,   # COASTAL_AEROSOL\
             2,   # BLUE\
             3,   # GREEN\
             4,   # RED\
             8,   # PANCHROMATIC\
            'NA', # VEGETATION_1\
            'NA', # VEGETATION_2\
            'NA', # VEGETATION_3\
             5,   # NEAR_INFRARED\
            'NA', # VEGETATION_4\
            'NA', # WATER_VAPOUR_INPUT\
            9,    # CIRRUS\
            6,    # SHORT_WAVE_INFRARED_1\
            7,    # SHORT_WAVE_INFRARED_2 \
            10,   # THERMAL_INFRARED_1 \
            11,   # THERMAL_INFRARED_2 \
           'NA',  # LONG_WAVE_INFRARED_1\
           'NA',  # LONG_WAVE_INFRARED_2\
           'NA']  # QUALITY_ASSESSMENT

        if new_index[band_index.value] == 'NA':
            return False
        return new_index[band_index.value]

    def get_no_data_map(self):
        band = self.getBand(Band.RED)
        no_data = np.zeros_like(band, dtype=np.uint8)
        no_data[band > 0] = 1
        # exclude negative pixels from no_data <=> include saturated-overflow-negative landsat in "data":
        no_data[band < 0] = 1
        return no_data

    def update_landsat_metadata(self, output_mtd_file):

        result_update_mtd = False

        if self.config.collection_number == 1: #coll_1 dictionary
            Fields = {
                '    DATA_TYPE = "L1TP"\n': '    DATA_TYPE = "L2TP"\n'
            }
        else: #coll_2 dictionary
            Fields = {
                '    PROCESSING_LEVEL = "L1TP"\n': '    PROCESSING_LEVEL = "L2TP"\n'
            }

        try:
            import fileinput
            import sys
            for line in fileinput.input(output_mtd_file, inplace=1):
                for field in Fields:
                    if field in line:
                        line = line.replace(field, Fields[field])
                    sys.stdout.write(line)
            result_update_mtd = True
        except:
            self.config.timestamp('Warning: Error in updating metadata from L1')
            result_update_mtd = False

        return result_update_mtd


    def export_landsat_metadata(self):

        response_meta = False

        #copying the _ANG.txt
        ang_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_ANG.txt')
        #keep _ANG.txt L1 filename identical as in input L1, consistent with MTL.txt content
        destination = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID, self.config.file_prefix)

        ang_output = destination + '_ANG.txt'
        if os.path.exists(ang_output):
            os.remove(ang_output)
        shutil.copy(ang_input, ang_output)

        #copying the _MTL.txt
        mtl_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_MTL.txt')
        if self.config.scOnly:
            destination = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID, self.config.file_prefix)
        else:
            destination = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID,self.config.L2A_TILE_ID)

        mtl_output = destination + '_MTL.txt'
        if os.path.exists(mtl_output):
            os.remove(mtl_output)
        shutil.copy(mtl_input, mtl_output)

        #updating the _MTL.txt
        if self.config.scOnly == False:
            response_meta = self.update_landsat_metadata(mtl_output)
        else:
            response_meta = True

        return response_meta

    def export_not_used_bands(self):
        response_copy_bands = True

        B8_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_B8.TIF')
        B10_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_B10.TIF')
        B11_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_B11.TIF')

        if self.config.collection_number == 1:
            BQA_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_BQA.TIF')
            list_bands = [B8_input, B10_input, B11_input, BQA_input]
        else:
            #extra bands here for coll_2
            BQA_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_QA_PIXEL.TIF')
            SAA_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_SAA.TIF')
            SZA_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_SZA.TIF')
            VAA_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_VAA.TIF')
            VZA_input = os.path.join(self.config.input_dir, self.config.file_prefix + '_VZA.TIF')
            list_bands = [B8_input, B10_input, B11_input, BQA_input, SAA_input, SZA_input, VAA_input, VZA_input]

        destination = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID)

        for file in list_bands:
            full_destination = os.path.join(self.config.output_dir, self.config.L2A_TILE_ID, file.split(os.sep)[-1])
            if os.path.exists(full_destination):
                try:
                    os.remove(full_destination)
                except PermissionError:
                    print("Permission error")
        try:
            for file in list_bands:
                if os.path.exists(file):
                    shutil.copy2(file, destination)
                else:
                    self.config.timestamp(f'Warning: {file} does not exist. It cannot be copied to L2 folder')
        except:
            self.config.timestamp('Warning: Error in copying unprocessed L1 Bands to L2 folder')
            response_copy_bands = False
        return response_copy_bands

    def gdalDEM_Shade_Cast(self, demfile):
        sdwfile = demfile.replace('_dem', '_sdw')

        # step 1 compute hillshade for the computation of local illumination angle cbeta
        altitude = 90.0 - np.float32(self.config._solze_noclip)
        azimuth = np.float32(np.mean(self.config.solaz_arr))
        kwargs = '-compute_edges -az ' + str(azimuth) + ' -alt ' + str(altitude)
        try:
            gdal.DEMProcessing(sdwfile, demfile, 'hillshade', options=kwargs)
        except:
            self.logger.warning('error using gdal dem processing option hillshade, trying again')
            try:
                sleep(5)  # Sleep for 3 seconds
                gdal.DEMProcessing(sdwfile, demfile, 'hillshade', options=kwargs)
                self.config.timestamp('L2A_Tables: option hillshade correctly executed')
            except Exception as e:
                self.logger.fatal(e, exc_info=True)
                self.logger.fatal('error using gdal dem processing option hillshade')
                return False

        # step 2 compute casted shadows for the scene classification (SCL == 2)
        # only if at least one pixel is not illluminated by the sun (cbeta_sdw == 1)
        #from skimage import io
        #cbeta_sdw = io.imread(sdwfile)
        sdw_gdal = gdal.Open(sdwfile)
        geotransform = sdw_gdal.GetGeoTransform()
        projection = sdw_gdal.GetProjection()

        cbeta_sdw = np.array(sdw_gdal.GetRasterBand(1).ReadAsArray())
        sdw_gdal = None

        topo_shadows_gdal = (cbeta_sdw == 1)

        if 'LANDSAT' in self.config.spacecraftName:
            solze_check= self.config.solze
        else:
            solze_check = self.config._solze
        if solze_check == 70.0: #if self.config._solze == 70.0:
            # step 1 compute hillshade for the computation of local illumination angle cbeta
            altitude = 90.0 - solze_check #self.config._solze
            azimuth = np.float32(np.mean(self.config.solaz_arr))
            kwargs = '-compute_edges -az ' + str(azimuth) + ' -alt ' + str(altitude)
            try:
                gdal.DEMProcessing(sdwfile, demfile, 'hillshade', options=kwargs)
            except Exception as e:
                self.logger.fatal(e, exc_info=True)
                self.logger.fatal('error using gdal dem processing option hillshade')
                return False

        if topo_shadows_gdal.sum() > 0:  # compute casted shadow only if at least one pixel is not illluminated by the sun

            try:
                dem = self.getBand(Band.DIGITAL_ELEVATION_MAP) #self.getBand(self.DEM)
            except Exception as e:
                self.logger.fatal(e, exc_info=True)
                self.logger.fatal('execution error reading imported DEM for cast shadow computation')
                return False

            rows, cols = dem.shape

            # casting shadow algorithm is performed at sen2cor processing resolution or 20 m in case of 10 m processing
            if self.config.resolution > 10:
                resolution = self.config.resolution
            else:
                resolution = 20
                dem = (skit_resize(dem.astype(int16), ([rows/2, cols/2]), order=1) * 32767.).round().astype(int16)

            sza = np.float64(self.config._solze_noclip) # noclip version of solze needs to be used here to compute real casted shadows
            saa = np.float64(np.mean(self.config.solaz_arr))

            # casted shadow function here:
            import topographicshadows_cython_03
            from scipy.ndimage.filters import median_filter
            from skimage.transform import resize as skit_resize
            sdw = topographicshadows_cython_03.project_shadows(dem, np.array([sza, saa]), np.float64(resolution), np.float64(resolution))

            # apply median filter to remove some horizontal stripes
            sdw = median_filter(sdw, 3)

            # order=1 is for bi-linear interpolation (casted shadow upsampling from 60 m to 20 m):
            # order=0 is for nearest interpolation (casted shadow upsampling from 60 m to 20 m):
            if self.config.resolution > 10:
                sdw = sdw.astype(np.uint8)
            else:
                sdw = (skit_resize(sdw.astype(np.uint8), ([rows, cols]), order=0) * 255.).round().astype(uint8)
                sdw = median_filter(sdw, 3).astype(np.uint8)

            # Merge (hillshade + casted shadow) information in a single layer: SDW
            T_Shadow = 0

            # reload cbeta_sdw in case solar zenith angles were clipped to 70.0 deg
            if self.config.solze == 70.0:
                cbeta_sdw = None
                #cbeta_sdw = io.imread(sdwfile)
                sdw_gdal = None
                sdw_gdal = gdal.Open(sdwfile)
                cbeta_sdw = np.array(sdw_gdal.GetRasterBand(1).ReadAsArray())
                cbeta_sdw[topo_shadows_gdal] = 1  # set shadows to 1 using first gdal hillshade output when sza > 70

            cbeta_sdw[sdw == T_Shadow] = 1  # set shadows to 1 like in the gdal hillshade output (O is for no data)
            # Question: Should all shadows be set to 0 like no data? To be tested.
            # Comment: in L2A_AtmCorr dtm_array (), it seems that 0 or even < 0 is used to detect shadows from shd file
            # it may be an heritage from when shadows where computed using another algorithm than gdaldem --hillshade

            # save Merge information in sdwfile
            #io.imsave(sdwfile, cbeta_sdw)
            sdw_gdal = None
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(sdwfile, cols, rows, 1, gdal.GDT_Byte)
            dataset.GetRasterBand(1).WriteArray(cbeta_sdw)
            dataset.SetGeoTransform(geotransform)
            dataset.SetProjection(projection)
            dataset = None

        return sdwfile
