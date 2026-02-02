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

import sys, os, logging, inspect, shutil
import configparser
from L2A_Library import chmod_recursive
import datetime as dt
from lxml import etree, objectify
#import multiprocessing
from multiprocessing import Lock
import numpy as np
from L2A_Logger import L2A_Logger, getLevel
from L2A_XmlParser import L2A_XmlParser
l = Lock()


def get_script_dir(follow_symlinks=True):
    if getattr(sys, 'frozen', False):  # py2exe, PyInstaller, cx_Freeze
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)


def read_metadata(filename):

    def read_array(file, array):
        if array.endswith(')'):
            array = array.replace('(', '').replace(')', '').strip()
            if '.' in array:
                array = np.fromstring(array, sep=',', dtype=np.float32)
            else:
                array = np.fromstring(array, sep=',', dtype=np.int32)
            return array
        else:
            array = array.replace('(', '').strip()
            for row in file:
                row = row.strip()
                if row.endswith(')'):
                    array += row.replace(')', '')
                    if '.' in array:
                        array = np.fromstring(array, sep=',', dtype=np.float32)
                    else:
                        array = np.fromstring(array, sep=',', dtype=np.int32)
                    return array
                else:
                    array += row

    def read_value(value):
        if value.startswith('"') and value.endswith('"'):
            value = value.replace('"', '')
        else:
            try:
                value = np.int32(value)
            except ValueError:
                try:
                    value = np.float32(value)
                except ValueError:
                    return value
        return value

    def read_group(file, current_group_name):
        group = dict()
        for line in file:
            line = line.strip()
            if line.startswith('END_GROUP'):
                if not line.endswith(current_group_name):
                    group_name = line.split(' = ')[1]
                    raise AttributeError('An unexpected group was closed. Expected to close {}, but {} was closed.'
                                         .format(current_group_name, group_name))
                return group
            if line.startswith('GROUP'):
                group_name = line.split(' = ')[1]
                group.update({group_name: read_group(file, group_name)})
            elif line.split(' = ')[1].startswith('('):
                array_name, array = line.split(' = ')
                group.update({array_name: read_array(file, array)})
            else:
                key, value = line.split(' = ')
                group.update({key: read_value(value)})
        return group

    metadata = dict()
    file = open(filename)
    for line in file:
        line = line.strip()
        if line.startswith('GROUP'):
            group_name = line.split(' = ')[1]
            metadata.update({group_name: read_group(file, group_name)})
        elif line == 'END':
            return metadata
        else:
            raise AttributeError('GROUP or END expected, but {} was given.'.format(line))

    raise ValueError('Unexpected end of metadata file before END was specified.')


class L2A_Config(object):
    def __init__(self, logger, input_dir=False):
        self.processorName = 'Sen2Cor'
        self.processorMode = 'Landsat 8-9'
        self._processorVersion = '03.03.01'
        self._processorDate = '2025.04.10'
        self._productVersion = '15.0'
        self._ROI = None
        self.input_dir = input_dir
        self.work_dir = None
        self.output_dir = None
        self.config_dir = None
        self.log_dir = None
        self.l1c_tile_id = None
        self.L2A_TILE_ID = None
        self.ncols = None
        self.nrows = None
        self.logger = logger
        self.log_level = 'INFO'
        self.fnLog = None
        self.configFn = None
        self.configSC = None
        self.configAC = None
        self.sc_lp_blu = 1.0
        self.intpol760 = 1  # always 1 for Sentinel2
        self.intpol725_825 = 1  # always 1 for Sentinel2
        self.intpol1400 = 1  # always 1 for Sentinel2
        self.intpol940_1130 = 1  # always 1 for Sentinel2
        self.smooth_wvmap = 100.0
        self.processingStatusFn = None
        self.processing_estimation_fn = None
        self.home = None
        self.current_timestamp = dt.datetime.utcnow()
        self.processing_start_timestamp = dt.datetime.utcnow()
        self.file_prefix = os.path.basename(self.input_dir)
        self.scOnly = False
        self.t_est_30_L = 1.0
        
        self.collection_number = int(1)

        self.metadata = read_metadata(os.path.join(self.input_dir, self.file_prefix + '_MTL.txt'))
        self.angle_coefficient = read_metadata(os.path.join(self.input_dir, self.file_prefix + '_ANG.txt'))

        try:
            self.collection_number = int(self.metadata['L1_METADATA_FILE']['METADATA_FILE_INFO']['COLLECTION_NUMBER']) #coll_1
        except:
            self.collection_number = int(self.metadata['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['COLLECTION_NUMBER']) #coll_2

        if self.collection_number == 1:
            self.acquisitionDate = self.metadata['L1_METADATA_FILE']['PRODUCT_METADATA']['DATE_ACQUIRED'] #coll_1
        else:
            self.acquisitionDate = self.metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['DATE_ACQUIRED'] #coll_2

        if self.collection_number == 1:
            self.acquisitionTime= self.metadata['L1_METADATA_FILE']['PRODUCT_METADATA']['SCENE_CENTER_TIME'] #coll_1
        else:
            self.acquisitionTime= self.metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SCENE_CENTER_TIME'] #coll_2
        # configuration parameter taken from landsat metadata:
        # configuration parameter taken from landsat metadata:
        if self.collection_number == 1:
            self.cellsize = self.metadata['L1_METADATA_FILE']['PROJECTION_PARAMETERS']['GRID_CELL_SIZE_REFLECTIVE'] #coll_1
        else:
            self.cellsize = self.metadata['LANDSAT_METADATA_FILE']['LEVEL1_PROJECTION_PARAMETERS']['GRID_CELL_SIZE_REFLECTIVE'] #coll_2
        self.resolution = self.cellsize
        self.pixelsize = self.cellsize

        if self.collection_number == 1:
            self.solaz = self.metadata['L1_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_AZIMUTH'] #coll_1
            self.solze = 90 - self.metadata['L1_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
        else:
            self.solaz = self.metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_AZIMUTH'] #coll_2
            self.solze = 90 - self.metadata['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
            
        self.solaz_arr = np.ones([10,10], dtype=np.float32) * self.solaz
        self.solze_arr = np.ones([10,10], dtype=np.float32) * self.solze

        self._solze_noclip = np.absolute(self.solze) #from L2A_config.py 5226
        
        self.spacecraftName = self.angle_coefficient['FILE_HEADER']['SPACECRAFT_ID']
        self.operationMode = self.spacecraftName
        self.nrBands = self.angle_coefficient['FILE_HEADER']['NUMBER_OF_BANDS']
        # ToDo: this must be adapted:
        self.bandIndex = self.angle_coefficient['FILE_HEADER']['BAND_LIST'] #[ 1  2  3  4  5  6  7  8  9 10 11] 
        self.bandIndex = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

        # ToDo: this is inherited from Sentinel config. Are there Landsat Metadata available?
        self.vaa = 1.0
        self.vza = 1.0
        vaa_arr = np.ones([10,10], dtype=np.float32) * self.vaa
        vza_arr = np.ones([10,10], dtype=np.float32) * self.vza
        self.vaa_arr = vaa_arr[self.vaa > 0].mean()
        self.vza_arr = vza_arr[self.vza > 0].mean()
        
        self.thv = 0.0  # tilt view angle, tilt azimuth angle
        self.phiv = 90.0  # sensor view zenith and azimuth angle
        # configuration and processing parameter taken from L2A_GIPP.xml:
        self.gippScheme2a = None
        self.gippSchemeSc = None
        self.gippSchemeAc = None
        self.medianFilter = 0
        self.aerosolType = None
        self.midLatitude = None
        self.ozoneSetpoint = None
        self.Generate_TCI_Output = None
        self.cirrus_correction = False
        # ToDo: check this for Landsat:
        if self.collection_number == 1:
            self.dnScale = self.metadata['L1_METADATA_FILE']['MIN_MAX_PIXEL_VALUE']['QUANTIZE_CAL_MAX_BAND_1'] #coll_1
        else:
            self.dnScale = self.metadata['LANDSAT_METADATA_FILE']['LEVEL1_MIN_MAX_PIXEL_VALUE']['QUANTIZE_CAL_MAX_BAND_1'] #coll_2

        # only dummy values to satisfy config check of AtmCorr
        self.e0 = np.ones([self.nrBands], dtype=np.uint8)
        #esd = self.metadata['L1_METADATA_FILE']['IMAGE_ATTRIBUTES']['EARTH_SUN_DISTANCE']
        #self.d2 = esd * esd
        self.d2 = 1.0  # this is a constant like for Sentinel-2

# for the new L2A_Quality report
        self.visibility = 40
        self.aot_retieval_accuracy = 0
        self.aot_retrieval_method = 'None'
        self.ddv_pixel_percentage = 0
        self.ddv_reflectance_range = 9999
        self.final_visibility = 9999
        self.ground_elevation_above_3 = 'False'
        self.radiative_transfer_accuracy = 0
        self.visibility_from_ddv = 9999
        self.water_vapour_retrieval_accuracy = 0
        self.water_vapour_retrieval_method = 'CAMS'
        self.nr_px_clouds_over_land = 0
        self.aux_data_filelist = []
        self.lut_data_filelist = []

# from L2A_Config down
        self._AC_Min_Ddv_Area = None
        self._rho_retrieval_step2 = None
        self._scaling_disabler = None
        self._scaling_limiter = None
        self._ch940 = np.array([8, 8, 9, 9, 0, 0])
        self._min_sc_blu = 0.9
        self._max_sc_blu = 1.1
        self._c0 = None
        self._c1 = None
        self._wvlsen = None
        self._fwhm = None
        self._db_compression_level = 0
        self._L2A_BOA_NEGATIVE_VALUES_LIST = np.zeros([13], np.uint32)
        self._logLevel = 'INFO'
        self._iwaterwv = 1

# from L2A_Config for Cirrus correction
#         self.wl940a = np.array([0.895, 1.000])  # range of moderate wv absorption region around  940 nm
#         self.wl1130a = np.array([1.079, 1.180])  # range of moderate wv absorption region around 1130 nm
#         self.wl1400a = np.array([1.330, 1.490])  # range for interpolation
#         self.wl1900a = np.array([1.780, 1.970])  # range for interpolation
#         self.wv_thr_cirrus = 0.60        
        

    #ACL_Prio_1
    def get_ac_min_ddv_area(self):
        return self._AC_Min_Ddv_Area

    def set_ac_min_ddv_area(self, value):
        pass

    def del_ac_min_ddv_area(self):
        del self._AC_Min_Ddv_Area

    def get_ac_swir_refl_lower_th(self):
        return self._AC_Swir_Refl_Lower_Th

    def set_ac_swir_refl_lower_th(self, value):
        self._AC_Swir_Refl_Lower_Th = value

    def del_ac_swir_refl_lower_th(self):
        del self._AC_Swir_Refl_Lower_Th

    def get_ac_cut_off_aot_iter_vegetation(self):
        return self._AC_Cut_Off_Aot_Iter_Vegetation

    def set_ac_cut_off_aot_iter_vegetation(self, value):
        self._AC_Cut_Off_Aot_Iter_Vegetation = value

    def del_ac_cut_off_aot_iter_vegetation(self):
        del self._AC_Cut_Off_Aot_Iter_Vegetation

    def get_ac_cut_off_aot_iter_water(self):
        return self._AC_Cut_Off_Aot_Iter_Water

    def set_ac_cut_off_aot_iter_water(self, value):
        self._AC_Cut_Off_Aot_Iter_Water = value

    def del_ac_cut_off_aot_iter_water(self):
        del self._AC_Cut_Off_Aot_Iter_Water

    def get_ac_aerosol_type_ratio_th(self):
        return self._AC_Aerosol_Type_Ratio_Th

    def set_ac_aerosol_type_ratio_th(self, value):
        self._AC_Aerosol_Type_Ratio_Th = value

    def del_ac_aerosol_type_ratio_th(self):
        del self._AC_Aerosol_Type_Ratio_Th

    def get_ac_swir_22um_red_refl_ratio(self):
        return self._AC_Swir_22um_Red_Refl_Ratio

    def set_ac_swir_22um_red_refl_ratio(self, value):
        self._AC_Swir_22um_Red_Refl_Ratio = value

    def del_ac_swir_22um_red_refl_ratio(self):
        del self._AC_Swir_22um_Red_Refl_Ratio

    def get_ac_red_blue_refl_ratio(self):
        return self._AC_Red_Blue_Refl_Ratio

    def set_ac_red_blue_refl_ratio(self, value):
        self._AC_Red_Blue_Refl_Ratio = value

    def del_ac_red_blue_refl_ratio(self):
        del self._AC_Red_Blue_Refl_Ratio

    def get_ac_cut_off_aot_iter_vegetation(self):
        return self._AC_Cut_Off_Aot_Iter_Vegetation

    def set_ac_cut_off_aot_iter_vegetation(self, value):
        self._AC_Cut_Off_Aot_Iter_Vegetation = value

    def del_ac_cut_off_aot_iter_vegetation(self):
        del self._AC_Cut_Off_Aot_Iter_Vegetation

    def get_ac_cut_off_aot_iter_water(self):
        return self._AC_Cut_Off_Aot_Iter_Water

    def set_ac_cut_off_aot_iter_water(self, value):
        self._AC_Cut_Off_Aot_Iter_Water = value

    def del_ac_cut_off_aot_iter_water(self):
        del self._AC_Cut_Off_Aot_Iter_Water

    def get_ac_aerosol_type_ratio_th(self):
        return self._AC_Aerosol_Type_Ratio_Th

    def set_ac_aerosol_type_ratio_th(self, value):
        self._AC_Aerosol_Type_Ratio_Th = value

    def del_ac_aerosol_type_ratio_th(self):
        del self._AC_Aerosol_Type_Ratio_Th

    def get_ac_topo_corr_th(self):
        return self._AC_Topo_Corr_Th

    def set_ac_topo_corr_th(self, value):
        self._AC_Topo_Corr_Th = value

    def del_ac_topo_corr_th(self):
        del self._AC_Topo_Corr_Th

    def get_ac_slope_th(self):
        return self._AC_Slope_Th

    def set_ac_slope_th(self, value):
        self._AC_Slope_Th = value

    def del_ac_slope_th(self):
        del self._AC_Slope_Th

    def get_ac_dem_p2p_val(self):
        return self._AC_Dem_P2p_Val

    def set_ac_dem_p2p_val(self, value):
        self._AC_Dem_P2p_Val = value

    def del_ac_dem_p2p_val(self):
        del self._AC_Dem_P2p_Val

    #ACL_Prio_2
    def get_ac_swir_refl_ndvi_th(self):
        return self._AC_Swir_Refl_Ndvi_Th

    def set_ac_swir_refl_ndvi_th(self, value):
        self._AC_Swir_Refl_Ndvi_Th = value

    def del_ac_swir_refl_ndvi_th(self):
        del self._AC_Swir_Refl_Ndvi_Th

    def get_ac_ddv_swir_refl_th_1(self):
        return self._AC_Ddv_Swir_Refl_Th1

    def set_ac_ddv_swir_refl_th_1(self, value):
        self._AC_Ddv_Swir_Refl_Th1 = value

    def del_ac_ddv_swir_refl_th_1(self):
        del self._AC_Ddv_Swir_Refl_Th1

    def get_ac_ddv_swir_refl_th_2(self):
        return self._AC_Ddv_Swir_Refl_Th2

    def set_ac_ddv_swir_refl_th_2(self, value):
        self._AC_Ddv_Swir_Refl_Th2 = value

    def del_ac_ddv_swir_refl_th_2(self):
        del self._AC_Ddv_Swir_Refl_Th2

    def get_ac_ddv_swir_refl_th_3(self):
        return self._AC_Ddv_Swir_Refl_Th3

    def set_ac_ddv_swir_refl_th_3(self, value):
        self._AC_Ddv_Swir_Refl_Th3 = value

    def del_ac_ddv_swir_refl_th_3(self):
        del self._AC_Ddv_Swir_Refl_Th3

    def get_ac_ddv_16um_refl_th_1(self):
        return self._AC_Ddv_16um_Refl_Th1

    def set_ac_ddv_16um_refl_th_1(self, value):
        self._AC_Ddv_16um_Refl_Th1 = value

    def del_ac_ddv_16um_refl_th_1(self):
        del self._AC_Ddv_16um_Refl_Th1

    def get_ac_ddv_16um_refl_th_2(self):
        return self._AC_Ddv_16um_Refl_Th2

    def set_ac_ddv_16um_refl_th_2(self, value):
        self._AC_Ddv_16um_Refl_Th2 = value

    def del_ac_ddv_16um_refl_th_2(self):
        del self._AC_Ddv_16um_Refl_Th2

    def get_ac_ddv_16um_refl_th_3(self):
        return self._AC_Ddv_16um_Refl_Th3

    def set_ac_ddv_16um_refl_th_3(self, value):
        self._AC_Ddv_16um_Refl_Th3 = value

    def del_ac_ddv_16um_refl_th_3(self):
        del self._AC_Ddv_16um_Refl_Th3

    def get_ac_dbv_nir_refl_th(self):
        return self._AC_Dbv_Nir_Refl_Th

    def set_ac_dbv_nir_refl_th(self, value):
        self._AC_Dbv_Nir_Refl_Th = value

    def del_ac_dbv_nir_refl_th(self):
        del self._AC_Dbv_Nir_Refl_Th

    def get_ac_dbv_ndvi_th(self):
        return self._AC_Dbv_Ndvi_Th

    def set_ac_dbv_ndvi_th(self, value):
        self._AC_Dbv_Ndvi_Th = value

    def del_ac_dbv_ndvi_th(self):
        del self._AC_Dbv_Ndvi_Th

    def get_ac_red_ref_refl_th(self):
        return self._AC_Red_Ref_Refl_Th

    def set_ac_red_ref_refl_th(self, value):
        self._AC_Red_Ref_Refl_Th = value

    def del_ac_red_ref_refl_th(self):
        del self._AC_Red_Ref_Refl_Th

    def get_ac_dbv_red_veget_tst_ndvi_th(self):
        return self._AC_Dbv_Red_Veget_Tst_Ndvi_Th

    def set_ac_dbv_red_veget_tst_ndvi_th(self, value):
        self._AC_Dbv_Red_Veget_Tst_Ndvi_Th = value

    def del_ac_dbv_red_veget_tst_ndvi_th(self):
        del self._AC_Dbv_Red_Veget_Tst_Ndvi_Th

    def get_ac_dbv_red_veget_refl_th(self):
        return self._AC_Dbv_Red_Veget_Refl_Th

    def set_ac_dbv_red_veget_refl_th(self, value):
        self._AC_Dbv_Red_Veget_Refl_Th = value

    def del_ac_dbv_red_veget_refl_th(self):
        del self._AC_Dbv_Red_Veget_Refl_Th

    def get_ac_wv_iter_start_summer(self):
        return self._AC_Wv_Iter_Start_Summer

    def set_ac_wv_iter_start_summer(self, value):
        self._AC_Wv_Iter_Start_Summer = value

    def del_ac_wv_iter_start_summer(self):
        del self._AC_Wv_Iter_Start_Summer

    def get_ac_wv_iter_start_winter(self):
        return self._AC_Wv_Iter_Start_Winter

    def set_ac_wv_iter_start_winter(self, value):
        self._AC_Wv_Iter_Start_Winter = value

    def del_ac_wv_iter_start_winter(self):
        del self._AC_Wv_Iter_Start_Winter

    def get_ac_rng_nbhd_terrain_corr(self):
        return self._AC_Rng_Nbhd_Terrain_Corr

    def set_ac_rng_nbhd_terrain_corr(self, value):
        self._AC_Rng_Nbhd_Terrain_Corr = value

    def del_ac_rng_nbhd_terrain_corr(self):
        del self._AC_Rng_Nbhd_Terrain_Corr

    def get_ac_max_nr_topo_iter(self):
        return self._AC_Max_Nr_Topo_Iter

    def set_ac_max_nr_topo_iter(self, value):
        self._AC_Max_Nr_Topo_Iter = value

    def del_ac_max_nr_topo_iter(self):
        del self._AC_Max_Nr_Topo_Iter

    def get_ac_topo_corr_cutoff(self):
        return self._AC_Topo_Corr_Cutoff

    def set_ac_topo_corr_cutoff(self, value):
        self._AC_Topo_Corr_Cutoff = value

    def del_ac_topo_corr_cutoff(self):
        del self._AC_Topo_Corr_Cutoff

    def get_ac_vegetation_index_th(self):
        return self._AC_Vegetation_Index_Th

    def set_ac_vegetation_index_th(self, value):
        self._AC_Vegetation_Index_Th = value

    def del_ac_vegetation_index_th(self):
        del self._AC_Vegetation_Index_Th

    #ACL_PRIO_3
    def get_ac_limit_area_path_rad_scale(self):
        return self._AC_Limit_Area_Path_Rad_Scale

    def set_ac_limit_area_path_rad_scale(self, value):
        self._AC_Limit_Area_Path_Rad_Scale = value

    def del_ac_limit_area_path_rad_scale(self):
        del self._AC_Limit_Area_Path_Rad_Scale

    def get_ac_ddv_smooting_window(self):
        return self._AC_Ddv_Smooting_Window

    def set_ac_ddv_smooting_window(self, value):
        self._AC_Ddv_Smooting_Window = value

    def del_ac_ddv_smooting_window(self):
        del self._AC_Ddv_Smooting_Window

    def get_ac_terrain_refl_start(self):
        return self._AC_Terrain_Refl_Start

    def set_ac_terrain_refl_start(self, value):
        self._AC_Terrain_Refl_Start = value

    def del_ac_terrain_refl_start(self):
        del self._AC_Terrain_Refl_Start

    def get_ac_spr_refl_percentage(self):
        return self._AC_Spr_Refl_Percentage

    def set_ac_spr_refl_percentage(self, value):
        self._AC_Spr_Refl_Percentage = value

    def del_ac_spr_refl_percentage(self):
        del self._AC_Spr_Refl_Percentage

    def get_ac_spr_refl_promille(self):
        return self._AC_Spr_Refl_Promille

    def set_ac_spr_refl_promille(self, value):
        self._AC_Spr_Refl_Promille = value

    def del_ac_spr_refl_promille(self):
        del self._AC_Spr_Refl_Promille
    #Other Parameters
    def get_rho_retrieval_step2(self):
        return self._rho_retrieval_step2

    def set_rho_retrieval_step2(self, value):
        self._rho_retrieval_step2 = value

    def del_rho_retrieval_step2(self):
        del self._rho_retrieval_step2

    def get_scaling_disabler(self):
        return self._scaling_disabler

    def set_scaling_disabler(self, value):
        self._scaling_disabler = value

    def del_scaling_disabler(self):
        del self._scaling_disabler

    def get_scaling_limiter(self):
        return self._scaling_limiter

    def set_scaling_limiter(self, value):
        self._scaling_limiter = value

    def del_scaling_limiter(self):
        del self._scaling_limiter

    def get_ch_940(self):
        return self._ch940

    def set_ch_940(self, value):
        self._ch940 = value

    def del_ch_940(self):
        del self._ch940

    def get_min_sc_blu(self):
        return self._min_sc_blu

    def set_min_sc_blu(self, value):
        self._min_sc_blu = value

    def del_min_sc_blu(self):
        del self._min_sc_blu

    def get_max_sc_blu(self):
        return self._max_sc_blu

    def set_max_sc_blu(self, value):
        self._max_sc_blu = value

    def del_max_sc_blu(self):
        del self._max_sc_blu

    def get_c_0(self):
        return self._c0

    def set_c_0(self, value):
        self._c0 = value

    def del_c_0(self):
        del self._c0

    def get_c_1(self):
        return self._c1

    def set_c_1(self, value):
        self._c1 = value

    def del_c_1(self):
        del self._c1

    def get_wvlsen(self):
        return self._wvlsen

    def set_wvlsen(self, value):
        self._wvlsen = value

    def del_wvlsen(self):
        del self._wvlsen

    def get_fwhm(self):
        return self._fwhm

    def set_fwhm(self, value):
        self._fwhm = value

    def del_fwhm(self):
        del self._fwhm

    def get_db_compression_level(self):
        return self._db_compression_level

    def set_db_compression_level(self, value):
        self._db_compression_level = value

    def del_db_compression_level(self):
        del self._db_compression_level

    #SC_GIPP
    def get_snow_map_reference(self):
        return self._snowMapReference

    def set_snow_map_reference(self, value):
        self._snowMapReference = value

    def del_snow_map_reference(self):
        del self._snowMapReference

    def get_esacci_wb_map_reference(self):
        return self._esacciWaterBodiesReference

    def set_esacci_wb_map_reference(self, value):
        self._esacciWaterBodiesReference = value

    def del_esacci_wb_map_reference(self):
        del self._esacciWaterBodiesReference

    def get_esacci_lccs_map_reference(self):
        return self._esacciLandCoverReference

    def set_esacci_lccs_map_reference(self, value):
        self._esacciLandCoverReference = value

    def del_esacci_lccs_map_reference(self):
        del self._esacciLandCoverReference

    def get_esacci_snowc_map_reference_directory(self):
        return self._esacciSnowConditionReferenceDir

    def set_esacci_snowc_map_reference_directory(self, value):
        self._esacciSnowConditionReferenceDir = value

    def del_esacci_snowc_map_reference_directory(self):
        del self._esacciSnowConditionReferenceDir

    def get_no_data(self):
        return self._noData

    def set_no_data(self, value):
        self._noData = value

    def del_no_data(self):
        del self._no_data

    def get_saturated_defective(self):
        return self._saturatedDefective

    def set_saturated_defective(self, value):
        self._saturatedDefective = value

    def del_saturated_defective(self):
        del self._saturatedDefective

    def get_dark_features(self):
        return self._darkFeatures

    def set_dark_features(self, value):
        self._darkFeatures = value

    def del_dark_features(self):
        del self._darkFeatures

    def get_cloud_shadows(self):
        return self._cloudShadows

    def set_cloud_shadows(self, value):
        self._cloudShadows = value

    def del_cloud_shadows(self):
        del self._cloudShadows

    def get_vegetation(self):
        return self._vegetation

    def set_vegetation(self, value):
        self._vegetation = value

    def del_vegetation(self):
        del self._vegetation

    def get_bare_soils(self):
        return self._bareSoils

    def set_bare_soils(self, value):
        self._bareSoils = value

    def del_bare_soils(self):
        del self._bareSoils

    def get_water(self):
        return self._water

    def set_water(self, value):
        self._water = value

    def del_water(self):
        del self._water

    def get_low_proba_clouds(self):
        return self._lowProbaClouds

    def set_low_proba_clouds(self, value):
        self._lowProbaClouds = value

    def del_low_proba_clouds(self):
        del self._lowProbaClouds

    def get_med_proba_clouds(self):
        return self._medProbaClouds

    def set_med_proba_clouds(self, value):
        self._medProbaClouds = value

    def del_med_proba_clouds(self):
        del self._medProbaClouds

    def get_high_proba_clouds(self):
        return self._highProbaClouds

    def set_high_proba_clouds(self, value):
        self._highProbaClouds = value

    def del_high_proba_clouds(self):
        del self._highProbaClouds

    def get_thin_cirrus(self):
        return self._thinCirrus

    def set_thin_cirrus(self, value):
        self._thinCirrus = value

    def del_thin_cirrus(self):
        del self._thinCirrus

    def get_snow_ice(self):
        return self._snowIce

    def set_snow_ice(self, value):
        self._snowIce = value

    def del_snow_ice(self):
        del self._snowIce
    #tresholds
    def get_t_1_b_04(self):
        return self._T1_B04

    def set_t_1_b_04(self, value):
        self._T1_B04 = value

    def del_t_1_b_04(self):
        del self._T1_B04

    def get_t_2_b_04(self):
        return self._T2_B04

    def set_t_2_b_04(self, value):
        self._T2_B04 = value

    def del_t_2_b_04(self):
        del self._T2_B04

    def get_t_1_ndsi_cld(self):
        return self._T1_NDSI_CLD

    def set_t_1_ndsi_cld(self, value):
        self._T1_NDSI_CLD = value

    def del_t_1_ndsi_cld(self):
        del self._T1_NDSI_CLD

    def get_t_2_ndsi_cld(self):
        return self._T2_NDSI_CLD

    def set_t_2_ndsi_cld(self, value):
        self._T2_NDSI_CLD = value

    def del_t_2_ndsi_cld(self):
        del self._T2_NDSI_CLD

    def get_t_1_ndsi_snw(self):
        return self._T1_NDSI_SNW

    def set_t_1_ndsi_snw(self, value):
        self._T1_NDSI_SNW = value

    def del_t_1_ndsi_snw(self):
        del self._T1_NDSI_SNW

    def get_t_2_ndsi_snw(self):
        return self._T2_NDSI_SNW

    def set_t_2_ndsi_snw(self, value):
        self._T2_NDSI_SNW = value

    def del_t_2_ndsi_snw(self):
        del self._T2_NDSI_SNW

    def get_t_1_b_02(self):
        return self._T1_B02

    def set_t_1_b_02(self, value):
        self._T1_B02 = value

    def del_t_1_b_02(self):
        del self._T1_B02

    def get_t_2_b_02(self):
        return self._T2_B02

    def set_t_2_b_02(self, value):
        self._T2_B02 = value

    def del_t_2_b_02(self):
        del self._T2_B02

    def get_t_1_b_8_a(self):
        return self._T1_B8A

    def set_t_1_b_8_a(self, value):
        self._T1_B8A = value

    def del_t_1_b_8_a(self):
        del self._T1_B8A

    def get_t_2_b_8_a(self):
        return self._T2_B8A

    def set_t_2_b_8_a(self, value):
        self._T2_B8A = value

    def del_t_2_b_8_a(self):
        del self._T2_B8A

    def get_t_2_b_10(self):
        return self._T2_B10

    def set_t_2_b_10(self, value):
        self._T2_B10 = value

    def del_t_2_b_10(self):
        del self._T2_B10

    def get_t_1_b_12(self):
        return self._T1_B12

    def set_t_1_b_12(self, value):
        self._T1_B12 = value

    def del_t_1_b_12(self):
        del self._T1_B12

    def get_t_2_b_12(self):
        return self._T2_B12

    def set_t_2_b_12(self, value):
        self._T2_B12 = value

    def del_t_2_b_12(self):
        del self._T2_B12

    def get_t_1_r_b_02_b_04(self):
        return self._T1_R_B02_B04

    def set_t_1_r_b_02_b_04(self, value):
        self._T1_R_B02_B04 = value

    def del_t_1_r_b_02_b_04(self):
        del self._T1_R_B02_B04

    def get_t_2_r_b_02_b_04(self):
        return self._T2_R_B02_B04

    def set_t_2_r_b_02_b_04(self, value):
        self._T2_R_B02_B04 = value

    def del_t_2_r_b_02_b_04(self):
        del self._T2_R_B02_B04

    def get_t_1_r_b_02_b_04(self):
        return self._T1_R_B02_B04

    def set_t_1_r_b_02_b_04(self, value):
        self._T1_R_B02_B04 = value

    def del_t_1_r_b_02_b_04(self):
        del self._T1_R_B02_B04

    def get_t_2_r_b_02_b_04(self):
        return self._T2_R_B02_B04

    def set_t_2_r_b_02_b_04(self, value):
        self._T2_R_B02_B04 = value

    def del_t_2_r_b_02_b_04(self):
        del self._T2_R_B02_B04


    def get_t_1_r_b_8_a_b_03(self):
        return self._T1_R_B8A_B03

    def set_t_1_r_b_8_a_b_03(self, value):
        self._T1_R_B8A_B03 = value

    def del_t_1_r_b_8_a_b_03(self):
        del self._T1_R_B8A_B03

    def get_t_2_r_b_8_a_b_03(self):
        return self._T2_R_B8A_B03

    def set_t_2_r_b_8_a_b_03(self, value):
        self._T2_R_B8A_B03 = value

    def del_t_2_r_b_8_a_b_03(self):
        del self._T2_R_B8A_B03

    def get_t_1_r_b_8_a_b_11(self):
        return self._T1_R_B8A_B11

    def set_t_1_r_b_8_a_b_11(self, value):
        self._T1_R_B8A_B11 = value

    def del_t_1_r_b_8_a_b_11(self):
        del self._T1_R_B8A_B11

    def get_t_2_r_b_8_a_b_11(self):
        return self._T2_R_B8A_B11

    def set_t_2_r_b_8_a_b_11(self, value):
        self._T2_R_B8A_B11 = value

    def del_t_2_r_b_8_a_b_11(self):
        del self._T2_R_B8A_B11

    def get_t_1_snow(self):
        return self._T1_SNOW

    def set_t_1_snow(self, value):
        self._T1_SNOW = value

    def del_t_1_snow(self):
        del self._T1_SNOW

    def get_t_2_snow(self):
        return self._T2_SNOW

    def set_t_2_snow(self, value):
        self._T2_SNOW = value

    def del_t_2_snow(self):
        del self._T2_SNOW

    def get_t_1_ndvi(self):
        return self._T1_NDVI

    def set_t_1_ndvi(self, value):
        self._T1_NDVI = value

    def del_t_1_ndvi(self):
        del self._T1_NDVI

    def get_t_2_ndvi(self):
        return self._T2_NDVI

    def set_t_2_ndvi(self, value):
        self._T2_NDVI = value

    def del_t_2_ndvi(self):
        del self._T2_NDVI

    def get_t_11_b_02(self):
        return self._T11_B02

    def set_t_11_b_02(self, value):
        self._T11_B02 = value

    def del_t_11_b_02(self):
        del self._T11_B02

    def get_t_12_b_02(self):
        return self._T12_B02

    def set_t_12_b_02(self, value):
        self._T12_B02 = value

    def del_t_12_b_02(self):
        del self._T12_B02

    def get_t_11_r_b_02_b_11(self):
        return self._T11_R_B02_B11

    def set_t_11_r_b_02_b_11(self, value):
        self._T11_R_B02_B11 = value

    def del_t_11_r_b_02_b_11(self):
        del self._T11_R_B02_B11

    def get_t_12_r_b_02_b_11(self):
        return self._T12_R_B02_B11

    def set_t_12_r_b_02_b_11(self, value):
        self._T12_R_B02_B11 = value

    def del_t_12_r_b_02_b_11(self):
        del self._T12_R_B02_B11

    def get_t_21_b_12(self):
        return self._T21_B12

    def set_t_21_b_12(self, value):
        self._T21_B12 = value

    def del_t_21_b_12(self):
        del self._T21_B12

    def get_t_22_b_12(self):
        return self._T22_B12

    def set_t_22_b_12(self, value):
        self._T22_B12 = value

    def del_t_22_b_12(self):
        del self._T22_B12

    def get_t_21_r_b_02_b_11(self):
        return self._T21_R_B02_B11

    def set_t_21_r_b_02_b_11(self, value):
        self._T21_R_B02_B11 = value

    def del_t_21_r_b_02_b_11(self):
        del self._T21_R_B02_B11

    def get_t_22_r_b_02_b_11(self):
        return self._T22_R_B02_B11

    def set_t_22_r_b_02_b_11(self, value):
        self._T22_R_B02_B11 = value

    def del_t_22_r_b_02_b_11(self):
        del self._T22_R_B02_B11

    def get_sc_classic(self):
        return self._scClassic

    def set_sc_classic(self, value):
        self._scClassic = value

    def del_sc_classic(self):
        del self._scClassic

    def get_dem_error(self):
        return self._demError

    def set_dem_error(self, value):
        self._demError = value

    def del_dem_error(self):
        del self._demError

    def get_sc_lp_blu(self):
        return self._sc_lp_blu

    def set_sc_lp_blu(self, value):
        self._sc_lp_blu = value

    def del_sc_lp_blu(self):
        del self._sc_lp_blu

    def get_l2a_boa_negative_values_list(self):
        return self._L2A_BOA_NEGATIVE_VALUES_LIST

    def set_l2a_boa_negative_values_list(self, value):
        self._L2A_BOA_NEGATIVE_VALUES_LIST = value

    def del_l2a_boa_negative_values_list(self):
        del self._L2A_BOA_NEGATIVE_VALUES_LIST

    def get_log_level(self):
        return self._logLevel

    def set_log_level(self, value):
        self._logLevel = value

    def del_log_level(self):
        del self._logLevel

    def get_iwaterwv(self):
        return self._iwaterwv

    def set_iwaterwv(self, value):
        self._iwaterwv = value

    def del_iwaterwv(self):
        del self._iwaterwv

    #ACL_Prio_1
    AC_Min_Ddv_Area = property(get_ac_min_ddv_area, set_ac_min_ddv_area, del_ac_min_ddv_area,
                               "AC_Min_Ddv_Area's docstring")
    AC_Swir_Refl_Lower_Th = property(get_ac_swir_refl_lower_th, set_ac_swir_refl_lower_th, del_ac_swir_refl_lower_th,
                                     "AC_Swir_Refl_Lower_Th's docstring")
    AC_Cut_Off_Aot_Iter_Vegetation = property(get_ac_cut_off_aot_iter_vegetation, set_ac_cut_off_aot_iter_vegetation,
                                              del_ac_cut_off_aot_iter_vegetation,
                                              "AC_Cut_Off_Aot_Iter_Vegetation's docstring")
    AC_Cut_Off_Aot_Iter_Water = property(get_ac_cut_off_aot_iter_water, set_ac_cut_off_aot_iter_water,
                                         del_ac_cut_off_aot_iter_water, "AC_Cut_Off_Aot_Iter_Water's docstring")
    AC_Aerosol_Type_Ratio_Th = property(get_ac_aerosol_type_ratio_th, set_ac_aerosol_type_ratio_th,
                                        del_ac_aerosol_type_ratio_th, "AC_Aerosol_Type_Ratio_Th's docstring")
    AC_Swir_22um_Red_Refl_Ratio = property(get_ac_swir_22um_red_refl_ratio, set_ac_swir_22um_red_refl_ratio,
                                           del_ac_swir_22um_red_refl_ratio, "AC_Swir_22um_Red_Refl_Ratio's docstring")
    AC_Red_Blue_Refl_Ratio = property(get_ac_red_blue_refl_ratio, set_ac_red_blue_refl_ratio,
                                      del_ac_red_blue_refl_ratio, "AC_Red_Blue_Refl_Ratio's docstring")
    AC_Cut_Off_Aot_Iter_Vegetation = property(get_ac_cut_off_aot_iter_vegetation, set_ac_cut_off_aot_iter_vegetation,
                                              del_ac_cut_off_aot_iter_vegetation,
                                              "AC_Cut_Off_Aot_Iter_Vegetation's docstring")
    AC_Cut_Off_Aot_Iter_Water = property(get_ac_cut_off_aot_iter_water, set_ac_cut_off_aot_iter_water,
                                         del_ac_cut_off_aot_iter_water, "AC_Cut_Off_Aot_Iter_Water's docstring")
    AC_Aerosol_Type_Ratio_Th = property(get_ac_aerosol_type_ratio_th, set_ac_aerosol_type_ratio_th,
                                        del_ac_aerosol_type_ratio_th, "AC_Aerosol_Type_Ratio_Th's docstring")
    AC_Topo_Corr_Th = property(get_ac_topo_corr_th, set_ac_topo_corr_th, del_ac_topo_corr_th,
                               "AC_Topo_Corr_Th's docstring")
    AC_Slope_Th = property(get_ac_slope_th, set_ac_slope_th, del_ac_slope_th, "AC_Slope_Th's docstring")
    AC_Dem_P2p_Val = property(get_ac_dem_p2p_val, set_ac_dem_p2p_val, del_ac_dem_p2p_val, "AC_Dem_P2p_Val's docstring")
    #ACL_Prio_2
    AC_Swir_Refl_Ndvi_Th = property(get_ac_swir_refl_ndvi_th, set_ac_swir_refl_ndvi_th, del_ac_swir_refl_ndvi_th,
                                    "AC_Swir_Refl_Ndvi_Th's docstring")
    AC_Ddv_Swir_Refl_Th1 = property(get_ac_ddv_swir_refl_th_1, set_ac_ddv_swir_refl_th_1, del_ac_ddv_swir_refl_th_1,
                                    "AC_Ddv_Swir_Refl_Th1's docstring")
    AC_Ddv_Swir_Refl_Th2 = property(get_ac_ddv_swir_refl_th_2, set_ac_ddv_swir_refl_th_2, del_ac_ddv_swir_refl_th_2,
                                    "AC_Ddv_Swir_Refl_Th2's docstring")
    AC_Ddv_Swir_Refl_Th3 = property(get_ac_ddv_swir_refl_th_3, set_ac_ddv_swir_refl_th_3, del_ac_ddv_swir_refl_th_3,
                                    "AC_Ddv_Swir_Refl_Th3's docstring")
    AC_Ddv_16um_Refl_Th1 = property(get_ac_ddv_16um_refl_th_1, set_ac_ddv_16um_refl_th_1, del_ac_ddv_16um_refl_th_1,
                                    "AC_Ddv_16um_Refl_Th1's docstring")
    AC_Ddv_16um_Refl_Th2 = property(get_ac_ddv_16um_refl_th_2, set_ac_ddv_16um_refl_th_2, del_ac_ddv_16um_refl_th_2,
                                    "AC_Ddv_16um_Refl_Th2's docstring")
    AC_Ddv_16um_Refl_Th3 = property(get_ac_ddv_16um_refl_th_3, set_ac_ddv_16um_refl_th_3, del_ac_ddv_16um_refl_th_3,
                                    "AC_Ddv_16um_Refl_Th3's docstring")
    AC_Dbv_Nir_Refl_Th = property(get_ac_dbv_nir_refl_th, set_ac_dbv_nir_refl_th, del_ac_dbv_nir_refl_th,
                                  "AC_Dbv_Nir_Refl_Th's docstring")
    AC_Dbv_Ndvi_Th = property(get_ac_dbv_ndvi_th, set_ac_dbv_ndvi_th, del_ac_dbv_ndvi_th, "AC_Dbv_Ndvi_Th's docstring")
    AC_Red_Ref_Refl_Th = property(get_ac_red_ref_refl_th, set_ac_red_ref_refl_th, del_ac_red_ref_refl_th,
                                  "AC_Red_Ref_Refl_Th's docstring")
    AC_Dbv_Red_Veget_Tst_Ndvi_Th = property(get_ac_dbv_red_veget_tst_ndvi_th, set_ac_dbv_red_veget_tst_ndvi_th,
                                            del_ac_dbv_red_veget_tst_ndvi_th,
                                            "AC_Dbv_Red_Veget_Tst_Ndvi_Th's docstring")
    AC_Dbv_Red_Veget_Refl_Th = property(get_ac_dbv_red_veget_refl_th, set_ac_dbv_red_veget_refl_th,
                                        del_ac_dbv_red_veget_refl_th, "AC_Dbv_Red_Veget_Refl_Th's docstring")
    AC_Wv_Iter_Start_Summer = property(get_ac_wv_iter_start_summer, set_ac_wv_iter_start_summer,
                                       del_ac_wv_iter_start_summer, "AC_Wv_Iter_Start_Summer's docstring")
    AC_Wv_Iter_Start_Winter = property(get_ac_wv_iter_start_winter, set_ac_wv_iter_start_winter,
                                       del_ac_wv_iter_start_winter, "AC_Wv_Iter_Start_Winter's docstring")
    AC_Rng_Nbhd_Terrain_Corr = property(get_ac_rng_nbhd_terrain_corr, set_ac_rng_nbhd_terrain_corr,
                                        del_ac_rng_nbhd_terrain_corr, "AC_Rng_Nbhd_Terrain_Corr's docstring")
    AC_Max_Nr_Topo_Iter = property(get_ac_max_nr_topo_iter, set_ac_max_nr_topo_iter, del_ac_max_nr_topo_iter,
                                   "AC_Max_Nr_Topo_Iter's docstring")
    AC_Topo_Corr_Cutoff = property(get_ac_topo_corr_cutoff, set_ac_topo_corr_cutoff, del_ac_topo_corr_cutoff,
                                   "AC_Topo_Corr_Cutoff's docstring")
    AC_Vegetation_Index_Th = property(get_ac_vegetation_index_th, set_ac_vegetation_index_th,
                                      del_ac_vegetation_index_th, "AC_Vegetation_Index_Th's docstring")
    #ACL_Prio_3
    AC_Limit_Area_Path_Rad_Scale = property(get_ac_limit_area_path_rad_scale, set_ac_limit_area_path_rad_scale,
                                            del_ac_limit_area_path_rad_scale,
                                            "AC_Limit_Area_Path_Rad_Scale's docstring")
    AC_Ddv_Smooting_Window = property(get_ac_ddv_smooting_window, set_ac_ddv_smooting_window,
                                      del_ac_ddv_smooting_window, "AC_Ddv_Smooting_Window's docstring")
    AC_Terrain_Refl_Start = property(get_ac_terrain_refl_start, set_ac_terrain_refl_start, del_ac_terrain_refl_start,
                                     "AC_Terrain_Refl_Start's docstring")
    AC_Spr_Refl_Percentage = property(get_ac_spr_refl_percentage, set_ac_spr_refl_percentage,
                                      del_ac_spr_refl_percentage, "AC_Spr_Refl_Percentage's docstring")
    AC_Spr_Refl_Promille = property(get_ac_spr_refl_promille, set_ac_spr_refl_promille, del_ac_spr_refl_promille,
                                    "AC_Spr_Refl_Promille's docstring")
    #Other Parameters
    rho_retrieval_step2 = property(get_rho_retrieval_step2, set_rho_retrieval_step2, del_rho_retrieval_step2,
                                   "rho_retrieval_step2's docstring")
    scaling_disabler = property(get_scaling_disabler, set_scaling_disabler, del_scaling_disabler,
                                "scaling_disabler's docstring")
    scaling_limiter = property(get_scaling_limiter, set_scaling_limiter, del_scaling_limiter,
                               "scaling_limiter's docstring")
    ch940 = property(get_ch_940, set_ch_940, del_ch_940, "ch940's docstring")
    min_sc_blu = property(get_min_sc_blu, set_min_sc_blu, del_min_sc_blu, "min_sc_blu's docstring")
    max_sc_blu = property(get_max_sc_blu, set_max_sc_blu, del_max_sc_blu, "max_sc_blu's docstring")
    c0 = property(get_c_0, set_c_0, del_c_0, "c0's docstring")
    c1 = property(get_c_1, set_c_1, del_c_1, "c1's docstring")
    wvlsen = property(get_wvlsen, set_wvlsen, del_wvlsen, "wvlsen's docstring")
    fwhm = property(get_fwhm, set_fwhm, del_fwhm, "fwhm's docstring")
    db_compression_level = property(get_db_compression_level, set_db_compression_level, del_db_compression_level,
                                    "db_compression_level's docstring")
    #SC_GIPP
    snowMapReference = property(get_snow_map_reference, set_snow_map_reference, del_snow_map_reference,
                                "snowMapReference's docstring")
    esacciWaterBodiesReference = property(get_esacci_wb_map_reference, set_esacci_wb_map_reference,
                                          del_esacci_wb_map_reference,
                                          "esacciWaterBodiesReference's docstring")
    esacciLandCoverReference = property(get_esacci_lccs_map_reference, set_esacci_lccs_map_reference,
                                        del_esacci_lccs_map_reference,
                                        "esacciLandCoverReference's docstring")
    esacciSnowConditionDirReference = property(get_esacci_snowc_map_reference_directory,
                                               set_esacci_snowc_map_reference_directory,
                                               del_esacci_snowc_map_reference_directory,
                                               "esacciSnowConditionDirReference's docstring")
    noData = property(get_no_data, set_no_data, del_no_data, "noData's docstring")
    saturatedDefective = property(get_saturated_defective, set_saturated_defective, del_saturated_defective,
                                  "saturatedDefective's docstring")
    darkFeatures = property(get_dark_features, set_dark_features, del_dark_features, "darkFeatures's docstring")
    cloudShadows = property(get_cloud_shadows, set_cloud_shadows, del_cloud_shadows, "cloudShadows's docstring")
    vegetation = property(get_vegetation, set_vegetation, del_vegetation, "vegetation's docstring")
    bareSoils = property(get_bare_soils, set_bare_soils, del_bare_soils, "bareSoils's docstring")
    water = property(get_water, set_water, del_water, "water's docstring")
    lowProbaClouds = property(get_low_proba_clouds, set_low_proba_clouds, del_low_proba_clouds,
                              "lowProbaClouds's docstring")
    medProbaClouds = property(get_med_proba_clouds, set_med_proba_clouds, del_med_proba_clouds,
                              "medProbaClouds's docstring")
    highProbaClouds = property(get_high_proba_clouds, set_high_proba_clouds, del_high_proba_clouds,
                               "highProbaClouds's docstring")
    thinCirrus = property(get_thin_cirrus, set_thin_cirrus, del_thin_cirrus, "thinCirrus's docstring")
    snowIce = property(get_snow_ice, set_snow_ice, del_snow_ice, "snowIce's docstring")
    #tresholds
    T1_B04 = property(get_t_1_b_04, set_t_1_b_04, del_t_1_b_04, "T1_B04's docstring")
    T2_B04 = property(get_t_2_b_04, set_t_2_b_04, del_t_2_b_04, "T2_B04's docstring")
    T1_NDSI_CLD = property(get_t_1_ndsi_cld, set_t_1_ndsi_cld, del_t_1_ndsi_cld, "T1_NDSI_CLD's docstring")
    T2_NDSI_CLD = property(get_t_2_ndsi_cld, set_t_2_ndsi_cld, del_t_2_ndsi_cld, "T2_NDSI_CLD's docstring")
    T1_NDSI_SNW = property(get_t_1_ndsi_snw, set_t_1_ndsi_snw, del_t_1_ndsi_snw, "T1_NDSI_SNW's docstring")
    T2_NDSI_SNW = property(get_t_2_ndsi_snw, set_t_2_ndsi_snw, del_t_2_ndsi_snw, "T2_NDSI_SNW's docstring")
    T1_B02 = property(get_t_1_b_02, set_t_1_b_02, del_t_1_b_02, "T1_B02's docstring")
    T2_B02 = property(get_t_2_b_02, set_t_2_b_02, del_t_2_b_02, "T2_B02's docstring")
    T1_B8A = property(get_t_1_b_8_a, set_t_1_b_8_a, del_t_1_b_8_a, "T1_B8A's docstring")
    T2_B8A = property(get_t_2_b_8_a, set_t_2_b_8_a, del_t_2_b_8_a, "T2_B8A's docstring")
    T2_B10 = property(get_t_2_b_10, set_t_2_b_10, del_t_2_b_10, "T2_B10's docstring")
    T1_B12 = property(get_t_1_b_12, set_t_1_b_12, del_t_1_b_12, "T1_B12's docstring")
    T2_B12 = property(get_t_2_b_12, set_t_2_b_12, del_t_2_b_12, "T2_B12's docstring")
    T1_R_B02_B04 = property(get_t_1_r_b_02_b_04, set_t_1_r_b_02_b_04, del_t_1_r_b_02_b_04, "T1_R_B02_B04's docstring")
    T2_R_B02_B04 = property(get_t_2_r_b_02_b_04, set_t_2_r_b_02_b_04, del_t_2_r_b_02_b_04, "T2_R_B02_B04's docstring")
    T1_R_B02_B04 = property(get_t_1_r_b_02_b_04, set_t_1_r_b_02_b_04, del_t_1_r_b_02_b_04, "T1_R_B02_B04's docstring")
    T2_R_B02_B04 = property(get_t_2_r_b_02_b_04, set_t_2_r_b_02_b_04, del_t_2_r_b_02_b_04, "T2_R_B02_B04's docstring")
    T1_R_B8A_B03 = property(get_t_1_r_b_8_a_b_03, set_t_1_r_b_8_a_b_03, del_t_1_r_b_8_a_b_03,
                            "T1_R_B8A_B03's docstring")
    T2_R_B8A_B03 = property(get_t_2_r_b_8_a_b_03, set_t_2_r_b_8_a_b_03, del_t_2_r_b_8_a_b_03,
                            "T2_R_B8A_B03's docstring")
    T1_R_B8A_B11 = property(get_t_1_r_b_8_a_b_11, set_t_1_r_b_8_a_b_11, del_t_1_r_b_8_a_b_11,
                            "T1_R_B8A_B11's docstring")
    T2_R_B8A_B11 = property(get_t_2_r_b_8_a_b_11, set_t_2_r_b_8_a_b_11, del_t_2_r_b_8_a_b_11,
                            "T2_R_B8A_B11's docstring")
    T1_SNOW = property(get_t_1_snow, set_t_1_snow, del_t_1_snow, "T1_SNOW's docstring")
    T2_SNOW = property(get_t_2_snow, set_t_2_snow, del_t_2_snow, "T2_SNOW's docstring")
    T1_NDVI = property(get_t_1_ndvi, set_t_1_ndvi, del_t_1_ndvi, "T1_NDVI's docstring")
    T2_NDVI = property(get_t_2_ndvi, set_t_2_ndvi, del_t_2_ndvi, "T2_NDVI's docstring")
    T11_B02 = property(get_t_11_b_02, set_t_11_b_02, del_t_11_b_02, "T11_B02's docstring")
    T12_B02 = property(get_t_12_b_02, set_t_12_b_02, del_t_12_b_02, "T12_B02's docstring")
    T11_R_B02_B11 = property(get_t_11_r_b_02_b_11, set_t_11_r_b_02_b_11, del_t_11_r_b_02_b_11,
                             "T11_R_B02_B11's docstring")
    T12_R_B02_B11 = property(get_t_12_r_b_02_b_11, set_t_12_r_b_02_b_11, del_t_12_r_b_02_b_11,
                             "T12_R_B02_B11's docstring")
    T21_B12 = property(get_t_21_b_12, set_t_21_b_12, del_t_21_b_12, "T21_B12's docstring")
    T22_B12 = property(get_t_22_b_12, set_t_22_b_12, del_t_22_b_12, "T22_B12's docstring")
    T21_R_B02_B11 = property(get_t_21_r_b_02_b_11, set_t_21_r_b_02_b_11, del_t_21_r_b_02_b_11,
                             "T21_R_B02_B11's docstring")
    T22_R_B02_B11 = property(get_t_22_r_b_02_b_11, set_t_22_r_b_02_b_11, del_t_22_r_b_02_b_11,
                             "T22_R_B02_B11's docstring")
    #other
    scClassic = property(get_sc_classic, set_sc_classic, del_sc_classic, "scClassic's docstring")
    demError = property(get_dem_error, set_dem_error, del_dem_error, "dem_error's docstring")
    sc_lp_blu =  property(get_sc_lp_blu, set_sc_lp_blu, del_sc_lp_blu, "sc_lp_blu's docstring")
    L2A_BOA_NEGATIVE_VALUES_LIST = property(get_l2a_boa_negative_values_list, set_l2a_boa_negative_values_list,
                                            del_l2a_boa_negative_values_list,
                                            "L2A_BOA_NEGATIVE_VALUES_LIST's docstring")
    logLevel = property(get_log_level, set_log_level, del_log_level, "logLevel's docstring")
    iwaterwv = property(get_iwaterwv, set_iwaterwv, del_iwaterwv, "iwaterwv's docstring")
    # from L2A_Config up
    def init_home_directory(self):
        try:
            self.home = os.environ['SEN2COR_BIN']
        except:
            self.home = get_script_dir()
        self.libDir = os.path.join(self.home, 'lib_L08')
        self.aux_dir = os.path.join(self.home, 'aux_data')
        self.config_dir = os.path.join(self.home, 'cfg')
        if not os.path.exists(self.config_dir):
            os.mkdir(self.config_dir)

        if self.work_dir:
            self.log_dir = self.work_dir
        else:
            self.log_dir = os.path.join(self.home, 'log')
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

        self.configFn = os.path.join(self.home, 'cfg', 'L2A_GIPP.xml')
        self.configSC = os.path.join(self.config_dir, 'L2A_CAL_SC_GIPP.xml')
        self.configAC = os.path.join(self.config_dir, 'L2A_CAL_AC_GIPP.xml')

        self.processingStatusFn = os.path.join(self.log_dir, '.progress.Landsat')
        self.processing_estimation_fn = os.path.join(self.log_dir, '.estimation.Landsat')

        if not os.path.isfile(self.processing_estimation_fn):
            # init processing estimation file:
            config = configparser.RawConfigParser()
            config.add_section('time estimation')
            config.set('time estimation', 't_est_30_L', self.t_est_30_L)
            configfile = open(self.processing_estimation_fn, 'w')
            config.write(configfile)
            configfile.close()

    def create_l2a_tile(self, date):
        L2A_TILE_ID = self.l1c_tile_id.replace('L1','L2')
        tile_id_list = L2A_TILE_ID.split('_')
        tile_id_list[4] = date
        tile_id_str = '_'.join(tile_id_list)
        l2a_target_dir = os.path.join(self.output_dir, tile_id_str)
        self.L2A_TILE_ID = tile_id_str
        self.L2A_QI_REPORT_XML_LANDSAT=os.path.join(l2a_target_dir,'L2A_QUALITY.xml')
        self.qiReportScheme2a_LANDSAT = os.path.join(self.config_dir, 'L2A_Quality.xsd')

        #Either abort if file already exists
        '''
        if os.path.exists(l2a_target_dir):
            self.logger.fatal(f'Tile with name {tile_id_str} already exists. Aborting.')
            return False

        os.mkdir(l2a_target_dir)
        chmod_recursive(l2a_target_dir, 0o755)
        '''
        #or proceed (as before) but inform user, that the old file will be overwritten
        if not os.path.exists(l2a_target_dir):
            os.mkdir(l2a_target_dir)
            chmod_recursive(l2a_target_dir, 0o755)
        else:
            self.timestamp(f'Tile with name {tile_id_str} already exists. Previous Export will be overwritten')

            for item in filter(lambda x: os.path.isdir(os.path.join(l2a_target_dir, x)), os.listdir(l2a_target_dir)):
                item_path = os.path.join(l2a_target_dir, item)
                try:
                    shutil.rmtree(item_path)
                    self.timestamp(f'Temporary Directory Deleted: {item_path}')
                except Exception as e:
                    continue

        #or add a more precise timestamp (or similar identifier) to the folder name, similar to sentinel exports. Would that be allowed?
        #Would need to be implemented if chosen.
        self.work_dir = l2a_target_dir
        return True

    def init_logger(self):
        self.fnLog = os.path.join(self.log_dir, self.L2A_TILE_ID + '_report.xml')
        self.logger = L2A_Logger(self.spacecraftName, fnLog=self.fnLog \
                             , logLevel=self.log_level \
                             , operation_mode=self.operationMode)
        f = open(self.fnLog, 'w')
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<Sen2Cor_Level-2A_Report_File>\n')
        f.flush()
        f.close()

    def timestamp(self, procedure):
        try:
            t_now = dt.datetime.utcnow()
            t_delta = t_now - self.current_timestamp
            t_total_delta = t_now - self.processing_start_timestamp
            self.current_timestamp = t_now
            if self.logger.getEffectiveLevel() != logging.NOTSET:
                self.logger.info('Procedure: ' + procedure + ', elapsed time[s]: %0.3f, total: %s' % (
                t_delta.total_seconds(), t_total_delta))

            f = open(self.processingStatusFn, 'r')
            t_total = float(f.readline()) * 0.01
            f.close()
            increment = t_delta.total_seconds() / self.t_est_30_L
            t_total += increment
            if t_total > 1.0:
                t_weighted = 100.0 - np.exp(-t_total)
            elif t_total > 0.98:
                t_weighted = t_total * 100.0 - np.exp(-t_total)
            else:
                t_weighted = t_total * 100.0
            self.logger.stream('Progress[%%]: %03.2f : %s, elapsed time[s]: %0.3f, total: %s'
                % (t_weighted, procedure, t_delta.total_seconds(), t_total_delta))

            f = open(self.processingStatusFn, 'w')
            f.write(str(t_weighted) + '\n')
            f.close()
        except:
            f = open(self.processingStatusFn, 'w')
            f.write('0.0\n')
            f.close()
        return

    def setTimeEstimation(self):
        config = configparser.RawConfigParser(allow_no_value=True)
        try:
            config.read(self.processing_estimation_fn)
            self.t_est_30_L = config.getfloat('time estimation', 't_est_30_L')
        except:
            self.writeTimeEstimation(1.0)
        return

    def writeTimeEstimation(self, tMeasure):
        config = configparser.RawConfigParser()
        config.read(self.processing_estimation_fn)
        tEst = config.getfloat('time estimation', 't_est_30_L')
        tMeasure=tMeasure.total_seconds()
        tMeasureAsString = str((tEst + tMeasure) / 2.0)
        config.set('time estimation', 't_est_30_L', tMeasureAsString)
        configFile = open(self.processing_estimation_fn, 'w')
        config.write(configFile)
        configFile.close()
        return

    def setSchemes(self):
        try:
            doc = objectify.parse(self.configFn)
            root = doc.getroot()
            self.gippScheme2a = root.Common_Section.GIPP_Scheme.text + '.xsd'
            self.gippSchemeSc = root.Common_Section.SC_Scheme.text + '.xsd'
            self.gippSchemeAc = root.Common_Section.AC_Scheme.text + '.xsd'
        except:

            if not self.logger:
                from L2A_Logger import L2A_Logger
                self.logger = L2A_Logger('sen2cor')
            self.logger.fatal('wrong identifier for xml structure.')

        return

    def parNotFound(self, parameter):
        basename = os.path.basename(self.configFn)
        self.logger.fatal('Configuration parameter <%s> not found in %s' % (parameter, basename))
        return False

    def readPreferences(self):

        ###READING FROM L2A_CAL_SC_GIPP.xml
        ### Scaling:
        xp = L2A_XmlParser(self, 'SC_GIPP')
        xp.validate()

        ### Snow_map_reference
        node = xp.getTree('Scene_Classification', 'References')

        par = node.Snow_Map
        if par is None: self.parNotFound(node)
        self.snowMapReference = par.text

        ### ESA_CCI_WaterBodies_map_reference
        par = node.ESACCI_WaterBodies_Map
        if par is None: self.parNotFound(node)
        self.esacciWaterBodiesReference = par.text

        ### ESA_CCI_LandCover_map_reference
        par = node.ESACCI_LandCover_Map
        if par is None: self.parNotFound(node)
        self.esacciLandCoverReference = par.text

        ### ESA_CCI_SnowCondition_map_directory_reference
        par = node.ESACCI_SnowCondition_Map_Dir
        if par is None: self.parNotFound(node)
        self.esacciSnowConditionDirReference = par.text

        node = xp.getTree('Scene_Classification', 'Classificators')

        par = node.NO_DATA
        if par is None: self.parNotFound(node)
        self.noData = np.int32(par.pyval)

        par = node.SATURATED_DEFECTIVE
        if par is None: self.parNotFound(node)
        self.saturatedDefective = np.int32(par.pyval)

        par = node.DARK_FEATURES
        if par is None: self.parNotFound(node)
        self.darkFeatures = np.int32(par.pyval)

        par = node.CLOUD_SHADOWS
        if par is None: self.parNotFound(node)
        self.cloudShadows = np.int32(par.pyval)

        par = node.VEGETATION
        if par is None: self.parNotFound(node)
        self.vegetation = np.int32(par.pyval)

        par = node.NOT_VEGETATED
        if par is None: self.parNotFound(node)
        self.bareSoils = np.int32(par.pyval)

        par = node.WATER
        if par is None: self.parNotFound(node)
        self.water = np.int32(par.pyval)

        par = node.UNCLASSIFIED
        if par is None: self.parNotFound(node)
        self.lowProbaClouds = np.int32(par.pyval)

        par = node.MEDIUM_PROBA_CLOUDS
        if par is None: self.parNotFound(node)
        self.medProbaClouds = np.int32(par.pyval)

        par = node.HIGH_PROBA_CLOUDS
        if par is None: self.parNotFound(node)
        self.highProbaClouds = np.int32(par.pyval)

        par = node.THIN_CIRRUS
        if par is None: self.parNotFound(node)
        self.thinCirrus = np.int32(par.pyval)

        par = node.SNOW_ICE
        if par is None: self.parNotFound(node)
        self.snowIce = np.int32(par.pyval)

        ### Thresholds
        node = xp.getTree('Scene_Classification', 'Thresholds')

        par = node.T1_B02
        if par is None: self.parNotFound(node)
        self.T1_B02 = np.float32(par.pyval)

        par = node.T2_B02
        if par is None: self.parNotFound(node)
        self.T2_B02 = np.float32(par.pyval)

        par = node.T1_B04
        if par is None: self.parNotFound(node)
        self.T1_B04 = np.float32(par.pyval)

        par = node.T2_B04
        if par is None: self.parNotFound(node)
        self.T2_B04 = np.float32(par.pyval)

        par = node.T1_B8A
        if par is None: self.parNotFound(node)
        self.T1_B8A = np.float32(par.pyval)

        par = node.T2_B8A
        if par is None: self.parNotFound(node)
        self.T2_B8A = np.float32(par.pyval)

        par = node.T1_B10
        if par is None: self.parNotFound(node)
        self.T1_B10 = np.float32(par.pyval)

        par = node.T2_B10
        if par is None: self.parNotFound(node)
        self.T2_B10 = np.float32(par.pyval)

        par = node.T1_B12
        if par is None: self.parNotFound(node)
        self.T1_B12 = np.float32(par.pyval)

        par = node.T2_B12
        if par is None: self.parNotFound(node)
        self.T2_B12 = np.float32(par.pyval)

        par = node.T1_NDSI_CLD
        if par is None: self.parNotFound(node)
        self.T1_NDSI_CLD = np.float32(par.pyval)

        par = node.T2_NDSI_CLD
        if par is None: self.parNotFound(node)
        self.T2_NDSI_CLD = np.float32(par.pyval)

        par = node.T1_NDSI_SNW
        if par is None: self.parNotFound(node)
        self.T1_NDSI_SNW = np.float32(par.pyval)

        par = node.T2_NDSI_SNW
        if par is None: self.parNotFound(node)
        self.T2_NDSI_SNW = np.float32(par.pyval)

        par = node.T1_R_B02_B04
        if par is None: self.parNotFound(node)
        self.T1_R_B02_B04 = np.float32(par.pyval)

        par = node.T2_R_B02_B04
        if par is None: self.parNotFound(node)
        self.T2_R_B02_B04 = np.float32(par.pyval)

        par = node.T1_R_B8A_B03
        if par is None: self.parNotFound(node)
        self.T1_R_B8A_B03 = np.float32(par.pyval)

        par = node.T2_R_B8A_B03
        if par is None: self.parNotFound(node)
        self.T2_R_B8A_B03 = np.float32(par.pyval)

        par = node.T1_R_B8A_B11
        if par is None: self.parNotFound(node)
        self.T1_R_B8A_B11 = np.float32(par.pyval)

        par = node.T2_R_B8A_B11
        if par is None: self.parNotFound(node)
        self.T2_R_B8A_B11 = np.float32(par.pyval)

        par = node.T1_SNOW
        if par is None: self.parNotFound(node)
        self.T1_SNOW = np.float32(par.pyval)

        par = node.T2_SNOW
        if par is None: self.parNotFound(node)
        self.T2_SNOW = np.float32(par.pyval)

        par = node.T1_NDVI
        if par is None: self.parNotFound(node)
        self.T1_NDVI = np.float32(par.pyval)

        par = node.T2_NDVI
        if par is None: self.parNotFound(node)
        self.T2_NDVI = np.float32(par.pyval)

        par = node.T1_R_B8A_B03
        if par is None: self.parNotFound(node)
        self.T1_R_B8A_B03 = np.float32(par.pyval)

        par = node.T2_R_B8A_B03
        if par is None: self.parNotFound(node)
        self.T2_R_B8A_B03 = np.float32(par.pyval)

        par = node.T11_B02
        if par is None: self.parNotFound(node)
        self.T11_B02 = np.float32(par.pyval)

        par = node.T12_B02
        if par is None: self.parNotFound(node)
        self.T12_B02 = np.float32(par.pyval)

        par = node.T11_R_B02_B11
        if par is None: self.parNotFound(node)
        self.T11_R_B02_B11 = np.float32(par.pyval)

        par = node.T12_R_B02_B11
        if par is None: self.parNotFound(node)
        self.T12_R_B02_B11 = np.float32(par.pyval)

        par = node.T21_B12
        if par is None: self.parNotFound(node)
        self.T21_B12 = np.float32(par.pyval)

        par = node.T22_B12
        if par is None: self.parNotFound(node)
        self.T22_B12 = np.float32(par.pyval)

        par = node.T21_R_B02_B11
        if par is None: self.parNotFound(node)
        self.T21_R_B02_B11 = np.float32(par.pyval)

        par = node.T22_R_B02_B11
        if par is None: self.parNotFound(node)
        self.T22_R_B02_B11 = np.float32(par.pyval)

        par = node.T_CLOUD_LP
        if par is None: self.parNotFound(node)
        self.T_CLOUD_LP = np.float32(par.pyval)

        par = node.T_CLOUD_MP
        if par is None: self.parNotFound(node)
        self.T_CLOUD_MP = np.float32(par.pyval)

        par = node.T_CLOUD_HP
        if par is None: self.parNotFound(node)
        self.T_CLOUD_HP = np.float32(par.pyval)

        par = node.T1_B10
        if par is None: self.parNotFound(node)
        self.T1_B10 = np.float32(par.pyval)

        par = node.T2_B10
        if par is None: self.parNotFound(node)
        self.T2_B10 = np.float32(par.pyval)

        par = node.T_SDW
        if par is None: self.parNotFound(node)
        self.T_SDW = np.float32(par.pyval)

        par = node.T_B02_B12
        if par is None: self.parNotFound(node)
        self.T_B02_B12 = np.float32(par.pyval)



        ###READING FROM L2A_CAL_AC_GIPP.xml
        ### Scaling:
        xp = L2A_XmlParser(self, 'AC_GIPP')
        xp.validate()

        #from L2A_Config
        node = xp.getNode('Flags')
        try:
            par = node.Scaling_Disabler
            self._scaling_disabler = node.Scaling_Disabler.pyval
            par = node.Scaling_Limiter
            self._scaling_limiter = node.Scaling_Limiter.pyval
            # implementation of SIIMPC-557, UMW:
            par = node.Rho_Retrieval_Step2.pyval
            self.rho_retrieval_step2 = node.Rho_Retrieval_Step2.pyval
            # end implementation of SIIMPC-557
        except:
            self.parNotFound(par)

        node = xp.getNode('References')
        par = node.Lib_Dir.pyval
        if par is None: self.parNotFound(par)
        if self.resolution == 10:
            bandIndex = [1, 2, 3, 7]
            self._ch940 = [0, 0, 0, 0, 0, 0]
        elif self.resolution == 30:
            # TBD, must be changed afterwards
            # bandIndex = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12] #francesco: are those values correct for landsat?
            bandIndex = [0, 1, 2, 3, 8, 11, 12] ##COASTAL_AEROSOL,BLUE, GREEN, RED, NI, SWI_1, SWI_2
            self._ch940 = [5, 5, 0, 0, 0, 0] #[5, 5, 0, 0, 0, 0]
        else:
            bandIndex = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12] 
            self._ch940 = [8, 8, 9, 9, 0, 0] 

        sensor = xp.getNode('Sensor')
        try:
            par = sensor.Calibration.min_sc_blu
            self.min_sc_blu = par.pyval
            par = sensor.Calibration.max_sc_blu
            self.max_sc_blu = par.pyval
        except:
            self.parNotFound(par)

        wavelength = sensor.Calibration.Band_List.wavelength #from here to... #francesco
        i = 0
        self._c0 = np.zeros(np.size(bandIndex), dtype=np.float32)
        self._c1 = np.zeros(np.size(bandIndex), dtype=np.float32)
        self._wvlsen = np.zeros(np.size(bandIndex), dtype=np.float32)
        for index in bandIndex:
            self._c0[i] = np.float32(wavelength[index].attrib['c0'])
            self._c1[i] = np.float32(wavelength[index].attrib['c1'])
            self._wvlsen[i] = np.float32(wavelength[index].text)
            i += 1

        i = 0
        self._fwhm = np.zeros(np.size(bandIndex), dtype=np.float32)
        for index in bandIndex:
            par = sensor.Resolution.Band_List.fwhm[index]
            if par is None: self.parNotFound(par)
            self._fwhm[i] = np.float32(par.pyval)
            i += 1
            #...here to be checked #francesco

            ###
            ### New parameters for SEN2COR 2.3:
            ###
        node = xp.getNode('ACL_Prio_1')
        try:
            self._AC_Min_Ddv_Area = node.AC_Min_Ddv_Area.pyval  # OK
            self._AC_Swir_Refl_Lower_Th = node.AC_Swir_Refl_Lower_Th.pyval
            self._AC_Swir_22um_Red_Refl_Ratio = node.AC_Swir_22um_Red_Refl_Ratio.pyval  # OK
            self._AC_Red_Blue_Refl_Ratio = node.AC_Red_Blue_Refl_Ratio.pyval  # OK
            self._AC_Cut_Off_Aot_Iter_Vegetation = node.AC_Cut_Off_Aot_Iter_Vegetation.pyval
            self._AC_Cut_Off_Aot_Iter_Water = node.AC_Cut_Off_Aot_Iter_Water.pyval
            self._AC_Aerosol_Type_Ratio_Th = node.AC_Aerosol_Type_Ratio_Th.pyval
            self._AC_Topo_Corr_Th = node.AC_Topo_Corr_Th.pyval
            self._AC_Slope_Th = node.AC_Slope_Th.pyval
            self._AC_Dem_P2p_Val = node.AC_Dem_P2p_Val.pyval
        except:
            self.parNotFound(node)

        node = xp.getNode('ACL_Prio_2')
        try:
            self._AC_Swir_Refl_Ndvi_Th = node.AC_Swir_Refl_Ndvi_Th.pyval
            self._AC_Ddv_Swir_Refl_Th1 = node.AC_Ddv_Swir_Refl_Th1.pyval
            self._AC_Ddv_Swir_Refl_Th2 = node.AC_Ddv_Swir_Refl_Th2.pyval
            self._AC_Ddv_Swir_Refl_Th3 = node.AC_Ddv_Swir_Refl_Th3.pyval
            self._AC_Ddv_16um_Refl_Th1 = node.AC_Ddv_16um_Refl_Th1.pyval
            self._AC_Ddv_16um_Refl_Th2 = node.AC_Ddv_16um_Refl_Th2.pyval
            self._AC_Ddv_16um_Refl_Th3 = node.AC_Ddv_16um_Refl_Th3.pyval
            self._AC_Dbv_Nir_Refl_Th = node.AC_Dbv_Nir_Refl_Th.pyval
            self._AC_Dbv_Ndvi_Th = node.AC_Dbv_Ndvi_Th.pyval
            self._AC_Red_Ref_Refl_Th = node.AC_Red_Ref_Refl_Th.pyval
            self._AC_Dbv_Red_Veget_Tst_Ndvi_Th = node.AC_Dbv_Red_Veget_Tst_Ndvi_Th.pyval
            self._AC_Dbv_Red_Veget_Refl_Th = node.AC_Dbv_Red_Veget_Refl_Th.pyval
            self._AC_Wv_Iter_Start_Summer = node.AC_Wv_Iter_Start_Summer.pyval
            self._AC_Wv_Iter_Start_Winter = node.AC_Wv_Iter_Start_Winter.pyval
            self._AC_Rng_Nbhd_Terrain_Corr = node.AC_Rng_Nbhd_Terrain_Corr.pyval
            self._AC_Max_Nr_Topo_Iter = node.AC_Max_Nr_Topo_Iter.pyval
            self._AC_Topo_Corr_Cutoff = node.AC_Topo_Corr_Cutoff.pyval
            self._AC_Vegetation_Index_Th = node.AC_Vegetation_Index_Th.pyval
        except:
            self.parNotFound(node)

        node = xp.getNode('ACL_Prio_3')
        try:
            self._AC_Limit_Area_Path_Rad_Scale = node.AC_Limit_Area_Path_Rad_Scale.pyval
            self._AC_Ddv_Smooting_Window = node.AC_Ddv_Smooting_Window.pyval
            self._AC_Terrain_Refl_Start = node.AC_Terrain_Refl_Start.pyval
            self._AC_Spr_Refl_Percentage = node.AC_Spr_Refl_Percentage.pyval
            self._AC_Spr_Refl_Promille = node.AC_Spr_Refl_Promille.pyval
        except:
            self.parNotFound(node)

        #####READING FROM L2A_GIPP.xml
        xp = L2A_XmlParser(self, 'GIPP')
        # fix for SIIMPC-599, UMW
        # xp.export()
        xp.validate()

        ### Common_Section:
        node = xp.getNode('Common_Section')
        if node is None: self.parNotFound(node)

        par = node.Log_Level
        if par is None: self.parNotFound(par)
        self._logLevel = par.text

        subnode = node.Region_Of_Interest
        try:
            self.row0 = subnode.row0.pyval
            self.col0 = subnode.col0.pyval
            self.nrow_win = subnode.nrow_win.pyval
            self.ncol_win = subnode.ncol_win.pyval
            if self.row0 == 'OFF':
                self.ROI = 'OFF'
            elif self.row0 == 'AUTO':
                self.ROI = 'AUTO'
            else:
                self.ROI = 'MANUAL'
        except:
            self.parNotFound(subnode)

        par = node.Nr_Threads
        if par is None: self.parNotFound(par)
        self.nrThreads = par.pyval

        par = node.DEM_Directory
        if par is None: self.parNotFound(par)
        self.demDirectory = par.text

        par = node.DEM_Reference
        if par is None: self.parNotFound(par)
        self.demReference = par.text

        par = node.Generate_DEM_Output
        if par is None:
            self.parNotFound(par)
        elif par == 'TRUE':
            self.demOutput = True
        else:
            self.demOutput = False

        par = node.Force_Exit_On_DEM_Error
        if par is None:
            self.parNotFound(par)
        elif par == 'TRUE':
            self.demError = True
        else:
            self.demError = False

        par = node.Generate_TCI_Output
        if par is None:
            self.parNotFound(par)
        elif par == 'TRUE':
            self.tciOutput = True
        else:
            self.tciOutput = False

        par = node.Generate_DDV_Output
        if par is None:
            self.parNotFound(par)
        elif par == 'TRUE':
            self.ddvOutput = True
        else:
            self.ddvOutput = False

            ### Scene Classification:
            ### Filters:
        node = xp.getTree('Scene_Classification', 'Filters')
        if node is None: self.parNotFound(node)

        par = node.Median_Filter
        if par is None: self.parNotFound(node)
        self.medianFilter = int(par.pyval)



        ### Atmospheric Correction:
        ### References:
        node = xp.getTree('Atmospheric_Correction', 'Look_Up_Tables')
        if node is None: self.parNotFound(node)

        if self.aerosolType is None:
            par = node.Aerosol_Type
            if par is None: self.parNotFound(par)
            self.aerosolType = par.text

        if self.midLatitude is None:
            par = node.Mid_Latitude
            if par is None: self.parNotFound(par)
            self.midLatitude = par.text

        if self.ozoneSetpoint is None:
            par = node.Ozone_Content
            if par is None: self.parNotFound(par)
            self.ozoneSetpoint = np.float32(par.pyval)
            if self.ozoneSetpoint == np.float32(0):
                self.ozoneSource = 'CAMS'
            else:
                self.ozoneSource = 'CONFIG'
            

            ### Flags:
        node = xp.getTree('Atmospheric_Correction', 'Flags')
        if node is None: self.parNotFound(node)

        # SIIMPC-1019, enabled for release 2.6.7
        par = node.DEM_Terrain_Correction
        if par is None: self.parNotFound(par)
        value = par.pyval
        if value == 'TRUE':
            self.dem_terrain_correction = True
        else:
            self.dem_terrain_correction = False

        par = node.BRDF_Correction
        if par is None: self.parNotFound(par)
        self.ibrdf = np.int32(par.pyval)

        par = node.BRDF_Lower_Bound
        if par is None: self.parNotFound(par)
        self.thr_g = np.float32(par.pyval)

        #par = node.WV_Correction
        #if par is None: self.parNotFound(par)
        # self.iwaterwv = 1 # HERE WATER VAPOUR MISSING

        par = node.WV_Correction
        if par is None: self.parNotFound(par)
        self.iwaterwv = np.int32(par.pyval)
        self.preserve_iwaterwv = self.iwaterwv

        par = node.VIS_Update_Mode
        if par is None: self.parNotFound(par)
        self.npref = np.int32(par.pyval)

        par = node.WV_Watermask
        if par is None: self.parNotFound(par)
        self.iwv_watermask = np.int32(par.pyval)

        par = node.Cirrus_Correction
        if par is None: self.parNotFound(par)
        value = par.pyval
        if value == 'TRUE':
            self.cirrus_correction = True
            self.timestamp('Warning: Cirrus correction is still under implementation: automatically set to False')
            self.logger.warning('Warning: Cirrus correction is still under implementation: switch to False')
            self.cirrus_correction = False
        else:
            self.cirrus_correction = False

        ### Calibration:
        node = xp.getTree('Atmospheric_Correction', 'Calibration')
        if node is None: self.parNotFound(node)

        par = node.Adj_Km
        if par is None: self.parNotFound(par)
        self.adj_km = np.float32(par.pyval)

        par = node.Visibility
        if par is None: self.parNotFound(par)
        self.visibility = np.float32(par.pyval)

        par = node.Altitude
        if par is None: self.parNotFound(par)
        self.altit = np.float32(par.pyval)

        par = node.Smooth_WV_Map
        if par is None: self.parNotFound(par)
        self.smooth_wvmap = np.float32(par.pyval)
        if (self.smooth_wvmap < 0.0): self.smooth_wvmap = 0.0

        par = node.WV_Threshold_Cirrus # From L2A_Config
        if par is None: self.parNotFound(par)
        self.wv_thr_cirrus = np.clip(np.float32(par.pyval), 0.1, 1.0)

        par = node.Database_Compression_Level #from L2A_Config
        if par is None: self.parNotFound(par)
        self.db_compression_level = np.int32(par.pyval)
        return True


    def createAtmDataFilename(self):
        extension = '.atm'
        height = '99000_'
        if self.midLatitude == 'SUMMER':
            waterVapour = 'wv20_'
        elif self.midLatitude == 'WINTER':
            waterVapour = 'wv04_'
        elif self.midLatitude == 'AUTO':
            waterVapour = 'wv00_'
        else:
            waterVapour = 'wv20_'  # default

        par = self.aerosolType
        if par == 'RURAL':
            aerosolType = 'rura'
        elif par == 'MARITIME':
            aerosolType = 'mari'
        elif par == 'AUTO':
            aerosolType = 'auto'
        else:
            aerosolType = 'rura'  # default

        delta = self.assignOzoneContent()
        self.logger.info(
            'Ozone_Content is set to %s with %f least difference to input value' % (self.ozoneContent, delta))

        atmDataFn = self.ozoneContent + height + waterVapour + aerosolType + extension
        self.logger.info('generated file name for look up tables is: ' + atmDataFn)
        self.atmDataFn = os.path.join(self.libDir, atmDataFn)
        # implementation of SIIMPC-889-2, UMW: automatic detection and switch of LUT:
        try:
            os.stat(self.atmDataFn)
            self.logger.info('look up table for %s found and used' % self.spacecraftName)
            self.lut_data_filelist.append(atmDataFn)
            return
        except:
            self.logger.warning('no specific look up table for S2B found, default one will be used instead')
            self.atmDataFn = self.atmDataFn.replace('S2B', 'S2A')
        try:
            os.stat(self.atmDataFn)
            self.logger.info('look up table for S2A found and used')
            self.lut_data_filelist.append(atmDataFn)
            return
        except:
            self.logger.fatal('look up table not found: ' + self.atmDataFn)
            return
            # end of implementation SIIMPC-889-2

    # implementation of SIIMPC-828, UMW: moved from L2A_Tables:
    def assignOzoneContent(self):
        # get the ozone value from metadata:
        columns = None
        ozoneSetpoint = self.ozoneSetpoint
        if self.midLatitude == 'SUMMER':
            columns = {
                "f": abs(250 - ozoneSetpoint),
                "g": abs(290 - ozoneSetpoint),
                "h": abs(331 - ozoneSetpoint),
                "i": abs(370 - ozoneSetpoint),
                "j": abs(410 - ozoneSetpoint),
                "k": abs(450 - ozoneSetpoint)
            }
        elif self.midLatitude == 'WINTER':
            columns = {
                "t": abs(250 - ozoneSetpoint),
                "u": abs(290 - ozoneSetpoint),
                "v": abs(330 - ozoneSetpoint),
                "w": abs(377 - ozoneSetpoint),
                "x": abs(420 - ozoneSetpoint),
                "y": abs(460 - ozoneSetpoint)
            }

        self.ozoneContent = min(columns, key=columns.get)
        delta = columns[self.ozoneContent]
        return delta

    # implementation of SIIMPC-828, UMW: moved from L2A_Tables:
    def setOzoneContentFromMetadata(self, ozoneSetpoint):
        if ozoneSetpoint:
            self.ozoneSetpoint = ozoneSetpoint
            try:
                self.logger.info('ozone mean value is: ' + str(self.ozoneSetpoint.values))
            except:
                self.logger.info('ozone mean value is: ' + str(self.ozoneSetpoint))
        else:
            if self.midLatitude == 'SUMMER':
                self.logger.info('no ozone data present, standard mid summer will be used')
                self.ozoneContent = 'h'
            elif self.midLatitude == 'WINTER':
                self.logger.info('no ozone data present, standard mid winter will be used')
                self.ozoneContent = 'w'
            else:
                self.logger.info(
                    'no ozone data present and no mid latitude configured, standard mid summer will be used')
                self.ozoneContent = 'h'

        if self.aerosolType != 'AUTO':
            self.createAtmDataFilename()
        return

    def calc_region_of_interest(self, dataset):
        bands = dataset.data_vars.variables.mapping
        if self.row0 == 'OFF':
            self._rowTop = bands["blue"].shape[0]
            self._colLeft = bands["blue"].shape[1]
            self._rowBottom = 0
            self._colRight = 0

        elif self.row0 == 'AUTO':
            sys.stdout.write('Attempting AUTO ROI detection')
    
            self._rowTop = bands["blue"].shape[0]
            self._colLeft = bands["blue"].shape[1]
            self._rowBottom = 0
            self._colRight = 0
    
            for e in bands:
                if e == "panchromatic":
                    scale = 2
                else:
                    scale = 1
    
                roi = np.where(bands[e].values[:] > 0)
                y1, x1 = (np.amin(roi, axis=1) / scale).astype(int)
                y2, x2 = np.ceil(np.amax(roi, axis=1) / scale).astype(int)
    
                if self._rowTop > y1:
                    self._rowTop = y1
                if self._colLeft > x1:
                    self._colLeft = x1
                if self._rowBottom < y2:
                    self._rowBottom = y2
                if self._colRight < x2:
                    self._colRight = x2

        else:
            self._rowTop = int(max(self.row0 - self.nrow_win/2, 0))
            self._colLeft = int(max(self.col0 - self.ncol_win/2, 0)) 
            self._rowBottom = int(min(self.row0 + self.nrow_win/2, self.refl_nrows))
            self._colRight = int(min(self.col0 + self.ncol_win/2, self.refl_ncols))

        return

    def getRegionOfInterest(self):
        rowTop = self._rowTop
        colLeft = self._colLeft
        nrows = self.nrows
        ncols = self.ncols
        return rowTop, colLeft, nrows, ncols

    def get_region_of_interest(self, indataset=False):
        if indataset:
            src_nrows = indataset[0].shape[0]
            scale = float32(src_nrows / 1830.0)
            rowTop = int16(rint(self._rowTop * scale))
            colLeft = int16(rint(self._colLeft * scale))
            rowBottom = int16(rint(self._rowBottom * scale))
            colRight = int16(rint(self._colRight * scale))
        else:
            rowTop = self._rowTop
            colLeft = self._colLeft
            rowBottom = self._rowBottom
            colRight = self._colRight
        return rowTop, colLeft, rowBottom, colRight

