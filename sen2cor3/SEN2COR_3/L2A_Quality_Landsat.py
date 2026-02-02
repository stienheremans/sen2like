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

import os, sys
from lxml import etree, objectify
from time import strftime
from datetime import datetime
import numpy as np
import shutil
from L2A_XmlParser import L2A_XmlParser
from L2A_BandIdentifier import Band


class L2A_Quality_Landsat(object):
    def __init__(self, config, tables):
        self._config = config
        self._tables = tables

        self.initialize_qi_file()

        self._mission_id = 'LANDSAT'
        self._aot_retieval_accuracy = config.aot_retieval_accuracy
        self._blue_path_radiance_rescaling_factor = config.get_sc_lp_blu()
        self._ddv_pixel_percentage = config.ddv_pixel_percentage
        self._ddv_reflectance_range = config.ddv_reflectance_range
        self._final_visibility = config.final_visibility
        self._ground_elevation_above_3 = config.ground_elevation_above_3
        self._water_vapour_retrieval_accuracy = config.water_vapour_retrieval_accuracy
        self._ozone_setpoint = config.ozoneSetpoint
        self._ozone_source = config.ozoneSource
        self._start_visibility = config.visibility
        self._visibility_from_ddv = config.visibility_from_ddv
        self._ac_enabled = 'True'
        if config.scOnly == False:
            self._aot_retrieval_method = config.aot_retrieval_method
            self._water_vapour_retrieval_method = config.water_vapour_retrieval_method
            self._ac_enabled = 'True'
        else:
            self._aot_retrieval_method = 'DEFAULT'
            self._water_vapour_retrieval_method = 'NA'
            self._ac_enabled = 'False'

        # to retrieve the scene classification parameters:
        self._nr_px_full_tile = (config.refl_nrows * config.refl_ncols)
        sc = tables.getBand(Band.SCENE_CLASSIFICATION)  #tables.getBand(tables.SCL)
        self._nr_px_total_scl = sc.size
        self._nr_px_no_data_scl = sc[sc==config.noData].size
        self._nr_px_saturated_defective = sc[sc==config.saturatedDefective].size

        self._nr_px_total_nodata_pixel = (self._nr_px_full_tile - self._nr_px_total_scl) + self._nr_px_no_data_scl

        self._nr_px_valid = self._nr_px_full_tile - self._nr_px_total_nodata_pixel - self._nr_px_saturated_defective

        self._nr_px_dark_features = sc[sc==config.darkFeatures].size
        self._nr_px_cloud_shadows = sc[sc==config.cloudShadows].size
        self._nr_px_vegetated = sc[sc==config.vegetation].size
        self._nr_px_not_vegetated = sc[sc==config.bareSoils].size
        self._nr_px_water = sc[sc==config.water].size
        self._nr_px_unclassified = sc[sc==config.lowProbaClouds].size
        self._nr_px_med_proba_clouds = sc[sc==config.medProbaClouds].size
        self._nr_px_hi_proba_clouds = sc[sc==config.highProbaClouds].size
        self._nr_px_thin_cirrus = sc[sc==config.thinCirrus].size
        self._nr_px_snow_ice = sc[sc==config.snowIce].size
        self._nr_px_clouds_over_land = config.nr_px_clouds_over_land
        self._nr_cloud_coverage = self._nr_px_med_proba_clouds + self._nr_px_hi_proba_clouds + self._nr_px_thin_cirrus
        self._all_clouds = 12
        self._clouds_over_land = 13
        self._negative_boa_pixels = 14
        self._degraded_msi_data = 15
        sc = None



        # these parameters can be retrieved via the tables module:
        if config.scOnly == False:
            aot = tables.getBand(Band.AEROSOL_OPTICAL_THICKNESS)  # tables.getBand(tables.AOT)
            self._granule_mean_aot = np.float32(aot[aot > 0].mean() * 0.001)
            if self._granule_mean_aot > 1.0:
                self._aot_above_1 = 'True'
            else:
                self._aot_above_1 = 'False'
            aot = None
        else:
            self._granule_mean_aot = -999
            self._aot_above_1= 'NA'

        if config.scOnly == False:
            if config.preserve_iwaterwv > 0:
                wvp = tables.getBand(Band.WATER_VAPOUR)  # tables.getBand(tables.WVP)
                all_zero = not wvp.any()
                if not all_zero:
                    self._granule_mean_wv = np.float32(wvp[wvp > 0].mean() * 0.001)
                else:
                    self._granule_mean_wv = 0.0
                if self._granule_mean_wv > 5.0:
                    self._wv_above_5 = 'True'
                else:
                    self._wv_above_5 = 'False'
                wvp = None
            else:
                self._granule_mean_wv = -999
                self._wv_above_5 = 'NA'
        else:
            self._granule_mean_wv = -999
            self._wv_above_5 = 'NA'

        if config.scOnly == False:
            vis = tables.getBand(Band.VISIBILITY)  # tables.getBand(tables.VIS)
            self._granule_mean_vis = vis[vis > 0].mean()
            vis = None
            if self._granule_mean_vis < 5.0:
                self._vis_less_5 = 'True'
            else:
                self._vis_less_5 = 'False'
        else:
            self._granule_mean_vis = -999
            self._vis_less_5  = 'NA'

        self._solar_zenith_angle = config._solze_noclip #config.get_solze()
        if self._solar_zenith_angle > 70.0:
            self._sza_above_70 = 'True'
        else:
            self._sza_above_70 = 'False'

        if config.demType == 'SRTM':
            self._dem_type = 'SRTM_90'
        elif config.demType == 'DTED':
            self._dem_type = 'DTED_90'
        elif config.demType == 'DTED_30':
            self._dem_type = 'DTED_30'
        elif config.demType == 'COPERNICUS_90':
            self._dem_type = 'COPERNICUS_90'
        elif config.demType == 'COPERNICUS_30':
            self._dem_type = 'COPERNICUS_30'
        elif config.demType == 'AWS_COPERNICUS_90':
            self._dem_type = 'AWS_COPERNICUS_90'
        elif config.demType == 'AWS_COPERNICUS_30':
            self._dem_type = 'AWS_COPERNICUS_30'
        else:
            self._dem_type = 'NA'

        if self._dem_type == 'NA':
            self._dem_mean_alt = np.float64(self._config.altit*1000.) #'NA'
            self._dem_mean_slope = np.float64(0.0) #'NA'
            self._average_dem_profile = 'FLAT'
            self._average_dem_slope = 'FLAT'
        else:
            dem = tables.getBand(Band.DIGITAL_ELEVATION_MAP) #tables.getBand(tables.DEM)
            all_zero_dem = not dem.any()
            if not all_zero_dem:
                self._dem_mean_alt = dem[dem > 0].mean()
            else:
                self._dem_mean_alt = np.float64(self._config.altit * 1000.)
            if self._dem_mean_alt < 100:
                self._average_dem_profile = 'FLAT'
            elif self._dem_mean_alt < 610:
                self._average_dem_profile = 'MIDDLE'
            else:
                self._average_dem_profile = 'MOUNTAINOUS'
            dem = None

            slp = tables.getBand(Band.SLOPE) #tables.getBand(tables.SLP)
            self._dem_mean_slope = slp[slp > 0].mean()
            if self._dem_mean_slope < 1.0:
                self._average_dem_slope = 'FLAT'
            elif self._dem_mean_slope < 5.0:
                self._average_dem_slope = 'GENTLE'
            else:
                self._average_dem_slope = 'STEEP'
            slp = None


    def update_qi_landsat(self):
        switch_case = {
            'CLOUDY_PIXEL_PERCENTAGE': format('%f' % self.get_percentage \
                (self._all_clouds, self._nr_cloud_coverage)),
            'CLOUDY_PIXEL_OVER_LAND_PERCENTAGE': format('%f' % self.get_cloud_over_land_percentage()),
            # 'DEGRADED_MSI_DATA_PERCENTAGE': format('%f' % self._nr_px_degraded_msi_data),
            'NODATA_PIXEL_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.noData, self._nr_px_total_nodata_pixel)),
            'SATURATED_DEFECTIVE_PIXEL_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.saturatedDefective, self._nr_px_saturated_defective)),
            'DARK_FEATURES_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.darkFeatures, self._nr_px_dark_features)),
            'CLOUD_SHADOW_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.cloudShadows, self._nr_px_cloud_shadows)),
            'VEGETATION_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.vegetation, self._nr_px_vegetated)),
            'NOT_VEGETATED_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.bareSoils, self._nr_px_not_vegetated)),
            'WATER_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.water, self._nr_px_water)),
            'UNCLASSIFIED_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.lowProbaClouds, self._nr_px_unclassified)),
            'MEDIUM_PROBA_CLOUDS_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.medProbaClouds, self._nr_px_med_proba_clouds)),
            'HIGH_PROBA_CLOUDS_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.highProbaClouds, self._nr_px_hi_proba_clouds)),
            'THIN_CIRRUS_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.thinCirrus, self._nr_px_thin_cirrus)),
            'SNOW_ICE_PERCENTAGE': format('%f' % self.get_percentage \
                (self._config.snowIce, self._nr_px_snow_ice))
        }
        if not self.update_qi_report_landsat(switch_case, 'SCENE_CLASS_QUALITY'):
            return False

        switch_case = {
            'AOT_RETRIEVAL_ACCURACY': format('%f' % self._aot_retieval_accuracy),
            'GRANULE_MEAN_AOT': format('%f' % self._granule_mean_aot),
            'WV_RETRIEVAL_ACCURACY': format('%f' % self._water_vapour_retrieval_accuracy),
            'GRANULE_MEAN_WV': format('%f' % self._granule_mean_wv),
            'OZONE_VALUE': format('%f' % self._ozone_setpoint),
            'START_VISIBILITY_KM': format('%f' % self._start_visibility),
            'VISIBILITY_FROM_DDV_KM': format('%f' % self._visibility_from_ddv),
            'FINAL_VISIBILITY_KM': format('%f' % self._final_visibility),
            'AVERAGE_SOLAR_ZENITH_ANGLE': format('%f' % self._solar_zenith_angle),
            'DDV_PIXEL_PERCENTAGE': format('%f' % self._ddv_pixel_percentage),
            'DDV_REFLECTANCE_RANGE': format('%f' % self._ddv_reflectance_range),
            'BLUE_PATH_RADIANCE_RESCALING_FACTOR': format('%f' % self._blue_path_radiance_rescaling_factor),
            'AC_ENABLED': format('%s' % self._ac_enabled),
            'WV_RETRIEVAL_METHOD': format('%s' % self._water_vapour_retrieval_method),
            'AOT_RETRIEVAL_METHOD': format('%s' % self._aot_retrieval_method),
            'VISIBILITY_LESS_THAN_5_KM': format('%s' % self._vis_less_5),
            'AOT_ABOVE_1': format('%s' % self._aot_above_1),
            'GRANULE_WV_ABOVE_5_CM': format('%s' % self._wv_above_5),
            'B01': format('%f' % self.get_percentage(self._negative_boa_pixels, self._config.L2A_BOA_NEGATIVE_VALUES_LIST[0])),
            'B02': format('%f' % self.get_percentage(self._negative_boa_pixels, self._config.L2A_BOA_NEGATIVE_VALUES_LIST[1])),
            'B03': format('%f' % self.get_percentage(self._negative_boa_pixels, self._config.L2A_BOA_NEGATIVE_VALUES_LIST[2])),
            'B04': format('%f' % self.get_percentage(self._negative_boa_pixels, self._config.L2A_BOA_NEGATIVE_VALUES_LIST[3])),
            'B05': format('%f' % self.get_percentage(self._negative_boa_pixels, self._config.L2A_BOA_NEGATIVE_VALUES_LIST[4])),
            'B06': format('%f' % self.get_percentage(self._negative_boa_pixels, self._config.L2A_BOA_NEGATIVE_VALUES_LIST[5])),
            'B07': format('%f' % self.get_percentage(self._negative_boa_pixels, self._config.L2A_BOA_NEGATIVE_VALUES_LIST[6])),
            # 'B08': format('%s' % 'N/A'), # self._config.L2A_BOA_NEGATIVE_VALUES_LIST[7])
            # 'B09': format('%s' % 'N/A'), # self._config.L2A_BOA_NEGATIVE_VALUES_LIST[8])
            # 'B10': format('%s' % 'N/A'), # self._config.L2A_BOA_NEGATIVE_VALUES_LIST[11])
            # 'B11': format('%s' % 'N/A'), # self._config.L2A_BOA_NEGATIVE_VALUES_LIST[12])
            "LUT_DATA_FILES": self._config.lut_data_filelist
        }
        if not self.update_qi_report_landsat(switch_case, 'ATMOSPHERIC_CORRECTION_QUALITY'):
            return False

        switch_case = {
            "DEM_TYPE": self._dem_type,
            "DEM_MEAN_ALTITUDE_KM": str((self._dem_mean_alt / 1000.).astype(np.float32)),
            "DEM_MEAN_SLOPE": str(self._dem_mean_slope),
            'GROUND_ELEVATION_ABOVE_3_KM': self._ground_elevation_above_3,
            #            "AVERAGE_DEM_PROFILE": self._average_dem_profile,
            #            "AVERAGE_DEM_SLOPE": self._average_dem_slope,
            'SOLAR_ZENITH_ANGLE_ABOVE_70_DEG': self._sza_above_70,
            'OZONE_SOURCE': self._ozone_source,
            "AUX_DATA_FILES": self._config.aux_data_filelist
        }
        if not self.update_qi_report_landsat(switch_case, 'AUX_DATA_QUALITY'):
            return False

        return True

    def update_qi_report_landsat(self, switch_case, qi):
        xp = L2A_XmlParser(self._config, 'QIL')
        qh = xp.getNode('L2A_Quality_Header')

        qh.Fixed_Header.File_Class._setText('USER')
        dir = os.path.join(self._config.output_dir, self._config.L2A_TILE_ID)
        creation_date = 'UTC=' + strftime('%Y-%m-%dT%H:%M:%S', datetime.utcnow().timetuple())
        qh.Fixed_Header.Source.Creation_Date._setText(creation_date)
        qh.Fixed_Header.Mission._setText(self._mission_id)
        # url_path = os.path.join(dir, 'GRANULE', self._config.L2A_TILE_ID)
        url_path = 'N/A' #dir here?
        breaker = False
        try:
            db = xp.getNode('Data_Block')
            db.report.attrib['gippVersion'] = self._config._processorVersion
            db.report.attrib['date'] = self._config._processorDate.replace('.', '-') + 'T00:00:00Z'
            for checklist in db.report.checkList:
                checklist.item.attrib['name'] = self._config.L2A_TILE_ID
                if qi == 'AUX_DATA_QUALITY':
                    # checklist.item.attrib['url'] = os.path.join(url_path, 'AUX_DATA')
                    checklist.item.attrib['url'] = 'N/A'
                else:
                    # checklist.item.attrib['url'] = os.path.join(url_path, 'IMG_DATA', 'R20')
                    checklist.item.attrib['url'] = 'N/A'
                if checklist.name == qi:
                    breaker = True
                    for check in checklist.check:
                        for value in check.extraValues.value:
                            #print value.attrib['name']
                            value_text = str(switch_case.get(value.attrib['name'], 'NA'))
                            value._setText(value_text)
                            #print value_text
                if breaker: break
        except:
            self._config.logger.error("Error in modifying Quality Report data")
        xp.export()
        if xp.validate():
            return True
        else:
            return False

    def initialize_qi_file(self):
        src_qi_xml_file = os.path.join(self._config.config_dir, 'L2A_QUALITY_Landsat.xml')
        destination_qi_report = os.path.join(self._config.output_dir, self._config.L2A_TILE_ID)
        dst_qi_xml_file = os.path.join(destination_qi_report, 'L2A_QUALITY.xml')
        shutil.copy(src_qi_xml_file, dst_qi_xml_file)
        return

    def get_percentage(self, classificator, value):
        nr_px_classified = value
        if classificator == self._config.noData:
            nr_px_total = self._nr_px_full_tile
        elif classificator == self._config.saturatedDefective:
            nr_px_total = self._nr_px_full_tile - self._nr_px_total_nodata_pixel
        else:
            nr_px_total = self._nr_px_valid
        if nr_px_total > 0:
            fraction = np.float32(nr_px_classified) / np.float32(nr_px_total)
        else:
            fraction = 0
        percentage = np.clip(fraction * 100.0, 0, 100)
        return percentage

    def get_cloud_over_land_percentage(self):
        if (self._tables.hasBand(Band.WATER_BODY_INDEX) == True): #self._tables.WBI
            WBI = self._tables.getBand(Band.WATER_BODY_INDEX) #self._tables.WBI
            SCL = self._tables.getBand(Band.SCENE_CLASSIFICATION) #self._tables.SCL
            land = ((WBI != 2) & (SCL > self._config.saturatedDefective))
            nr_land_pixels = land.sum()
            if nr_land_pixels > 0:
                SCL_Clouds = (SCL == self._config._medProbaClouds) | (SCL == self._config._highProbaClouds) | (SCL == self._config._thinCirrus)
                nr_cloud_over_land = (SCL_Clouds & land).sum()
                fraction = np.float32(nr_cloud_over_land) / np.float32(nr_land_pixels)
            else:
                fraction = 0
            percentage = np.clip(fraction * 100.0, 0, 100)
            return percentage
        else:
            fraction = 0
            percentage = np.clip(fraction * 100.0, 0, 100)
            return percentage