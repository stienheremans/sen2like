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

from enum import Enum, unique, auto

@unique
class Band(Enum):
    def is_msi(self):
        if self == Band.COASTAL_AEROSOL\
                or self == Band.BLUE\
                or self == Band.GREEN\
                or self == Band.RED\
                or self == Band.PANCHROMATIC\
                or self == Band.VEGETATION_1\
                or self == Band.VEGETATION_2\
                or self == Band.VEGETATION_3\
                or self == Band.NEAR_INFRARED\
                or self == Band.VEGETATION_4\
                or self == Band.WATER_VAPOUR_INPUT\
                or self == Band.CIRRUS\
                or self == Band.SHORT_WAVE_INFRARED_1\
                or self == Band.SHORT_WAVE_INFRARED_2 \
                or self == Band.THERMAL_INFRARED_1 \
                or self == Band.THERMAL_INFRARED_2 \
                or self == Band.LONG_WAVE_INFRARED_1\
                or self == Band.LONG_WAVE_INFRARED_2\
                or self == Band.QUALITY_ASSESSMENT:
            return True
        else:
            return False

    # bands of the MSI on satellite
    COASTAL_AEROSOL = auto()
    BLUE = auto()
    GREEN = auto()
    RED = auto()
    PANCHROMATIC = auto()
    VEGETATION_1 = auto()
    VEGETATION_2 = auto()
    VEGETATION_3 = auto()
    NEAR_INFRARED = auto()
    VEGETATION_4 = auto()
    WATER_VAPOUR_INPUT = auto()
    CIRRUS = auto()
    SHORT_WAVE_INFRARED_1 = auto()
    SHORT_WAVE_INFRARED_2 = auto()
    THERMAL_INFRARED_1 = auto()
    THERMAL_INFRARED_2 = auto()
    LONG_WAVE_INFRARED_1 = auto()
    LONG_WAVE_INFRARED_2 = auto()
    QUALITY_ASSESSMENT = auto()
    # calculated bands
    DIGITAL_ELEVATION_MAP = 'DEM'
    SCENE_CLASSIFICATION = 'SCL'
    SNOW_MAP = 'SNW'
    CLOUD_MAP = 'CLD'
    AEROSOL_OPTICAL_THICKNESS = 'AOT'
    WATER_VAPOUR = 'WVP'
    VISIBILITY = 'VIS'
    PREVIEW = 'PRV'
    ILLUMINATION = 'ILU'
    SLOPE = 'SLP'
    ASPECT = 'ASP'
    HAZE = 'HAZ'
    SHADOW_MAP = 'SDW'
    DARK_DENSE_VEGETATION = 'DDV'
    HAZE_CLOUD_WATER = 'HCW'
    ELEVATION = 'ELE'
    PRECIPITABLE_WATER_CONTENT = 'PWC'
    MEAN_SEA_LEVEL = 'MSL'
    OZONE = 'OZO'
    TRUE_COLOR_IMAGE = 'TCI'
    WATER_BODY_INDEX = 'WBI'
    LAND_COVERAGE_MAP = 'LCM'
    SNOW_CONDITION_MAP = 'SNC'
    VISIBILITY_INDEX_MAP = 'VIM'
    QUALITY_MASK = 'QMS'
    # resampled bands
    R_DIGITAL_ELEVATION_MAP = 'RDEM'
    R_SCENE_CLASSIFICATION = 'RSCL'
    R_SNOW_MAP = 'RSNW'
    R_CLOUD_MAP = 'RCLD'
    R_AEROSOL_OPTICAL_THICKNESS = 'RAOT'
    R_WATER_VAPOUR = 'RWVP'
    R_DARK_DENSE_VEGETATION = 'RDDV'
    R_TRUE_COLOR_IMAGE = 'RTCI'
