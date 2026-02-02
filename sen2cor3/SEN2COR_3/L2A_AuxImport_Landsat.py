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

import os, tempfile, fnmatch, shutil
import numpy as np
from L2A_XmlParser import L2A_XmlParser
from osgeo.gdal_array import BandReadAsArray
import datetime as dt

from multiprocessing import Lock

l = Lock()

try:
    from osgeo import gdal, osr
    from osgeo.gdalconst import *

    gdal.TermProgress = gdal.TermProgress_nocb
except ImportError:
    import gdal, osr
    from gdalconst import *


def getMonth(dateStr):
    ''' returns the month from string.

        :param date
        :type date : str
        :return: month
        :rtype: unsigned int

    '''
    from datetime import datetime

    try: # Landsat?
        month_landsat = datetime.strptime(dateStr, '%Y-%m-%d').timetuple().tm_mon
    except: # Sentinel:
        try:
            month_landsat = datetime.strptime(dateStr, '%Y-%m-%dT%H:%M:%SZ').timetuple().tm_mon
        except:
            month_landsat = datetime.strptime(dateStr, '%Y-%m-%dT%H:%M:%S.%fZ').timetuple().tm_mon

    return month_landsat

def get_extent(gt, cols, rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext = []
    xarr = [0, cols]
    yarr = [0, rows]

    for px in xarr:
        for py in yarr:
            x = gt[0] + (px * gt[1]) + (py * gt[2])
            y = gt[3] + (px * gt[4]) + (py * gt[5])
            ext.append([x, y])
        yarr.reverse()
    return ext

def transform_utm_to_wgs84(easting, northing, zone1, zone2):
    utm_coordinate_system = osr.SpatialReference()  # Create a new spatial reference object using a named parameter
    utm_coordinate_system.SetWellKnownGeogCS("WGS84")  # Set geographic coordinate system to handle lat/lon
    zone = zone1
    hemi = zone2
    if (hemi == 'N'):  # N is Northern Hemisphere
        utm_coordinate_system.SetUTM(zone, 1)  # call sets detailed projection transformation parameters
    else:
        utm_coordinate_system.SetUTM(zone, 0)
    wgs84_coordinate_system = utm_coordinate_system.CloneGeogCS()  # Clone ONLY the geographic coordinate system
    # create transform component
    utm_to_wgs84_geo_transform = osr.CoordinateTransformation(utm_coordinate_system, wgs84_coordinate_system)
    return utm_to_wgs84_geo_transform.TransformPoint(np.double(easting), np.double(northing), 0)  # returns lon, lat, altitude

def transform_wgs84_to_utm(lon, lat):
    utm_coordinate_system = osr.SpatialReference()
    utm_coordinate_system.SetWellKnownGeogCS("WGS84")  # Set geographic coordinate system to handle lat/lon
    utm_coordinate_system.SetUTM(get_utm_zone(lon), is_northern(lat))
    wgs84_coordinate_system = utm_coordinate_system.CloneGeogCS()  # Clone ONLY the geographic coordinate system
    # create transform component
    wgs84_to_utm_geo_transform = osr.CoordinateTransformation(wgs84_coordinate_system, utm_coordinate_system)
    return wgs84_to_utm_geo_transform.TransformPoint(np.double(lon), np.double(lat), 0)  # returns easting, northing, altitude

def get_utm_zone(longitude):
    return (int(1 + (longitude + 180.0) / 6.0))

def is_northern(latitude):  # Determines if given latitude is a northern for UTM
    if (latitude < 0.0):
        return 0
    else:
        return 1

def get_day_of_year(date_str):
    ''' returns the day of year from string.

        :return: day of year
        :rtype: unsigned int

    '''
    from datetime import datetime
    from calendar import isleap

    try: # Landsat?
        doy = datetime.strptime(date_str, '%Y-%m-%d').timetuple().tm_yday
        year = datetime.strptime(date_str, '%Y-%m-%d').timetuple().tm_year
    except: # Sentinel:
        try:
            doy = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ').timetuple().tm_yday
            year = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ').timetuple().tm_year
        except:
            doy = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').timetuple().tm_yday
            year = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').timetuple().tm_year
    return doy, isleap(year)

def get_resolution_index(resolution):
    res = resolution
    if res == 10:
        return 0
    elif res == 20:
        return 1
    elif res == 60:
        return 2
    else:
        return False

def get_geo_coordinates_landsat(config):
    if 'LANDSAT' in config.spacecraftName:
        if config.collection_number == 1: #coll_1
            ulx = int(config.metadata['L1_METADATA_FILE']['PRODUCT_METADATA']['CORNER_UL_PROJECTION_X_PRODUCT'])
            uly = int(config.metadata['L1_METADATA_FILE']['PRODUCT_METADATA']['CORNER_UL_PROJECTION_Y_PRODUCT'])
        else: #coll_2
            ulx = int(config.metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['CORNER_UL_PROJECTION_X_PRODUCT'])
            uly = int(config.metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['CORNER_UL_PROJECTION_Y_PRODUCT'])
    else:  # Sentinel
        # get the target resolution and metadata for the resampled bands below:
        xp = L2A_XmlParser(config, 'T2A')
        tg = xp.getTree('Geometric_Info', 'Tile_Geocoding')
        idx = get_resolution_index(config.resolution)
        ulx = tg.Geoposition[idx].ULX
        uly = tg.Geoposition[idx].ULY

    res = np.float32(config.resolution)
    geoTransformation = [ulx, res, 0.0, uly, 0.0, -res]

    if config.ROI != 'OFF': #reduntandt line down but fix later
        if (config.row0 == 'AUTO') and (config.col0 == 'AUTO'):
            # ROI was determined automatically:
            yTop, xLeft, nrows, ncols = config.getRegionOfInterest()
            if config.resolution == 20:
                yTop = int(round(yTop * 0.5))
                xLeft = int(round(xLeft * 0.5))
                nrows = int(round(nrows * 0.5))
                ncols = int(round(ncols * 0.5))
        else:  # ROI is determined by configuration:
            top = 0
            left = 0
            if config.resolution == 60:
                bottom = 1830
                right = 1830
                nrows = config.nrow_win
                ncols = config.ncol_win
                row0 = config.row0
                col0 = config.col0
            if config.resolution == 30:  # Landsat
                bottom = config.refl_ncols #7921 #should bottom and right be switched?
                right = config.refl_nrows #7811
                nrows = config.nrow_win
                ncols = config.ncol_win
                row0 = config.row0
                col0 = config.col0
            elif config.resolution == 20:
                bottom = 5490
                right = 5490
                nrows = int(round(config.nrow_win * 0.5))
                ncols = int(round(config.ncol_win * 0.5))
                row0 = int(round(config.row0 * 0.5))
                col0 = int(round(config.col0 * 0.5))
            elif config.resolution == 10:
                bottom = 10980
                right = 10980
                nrows = config.nrow_win
                ncols = config.ncol_win
                row0 = config.row0
                col0 = config.col0

            rowOffset = int(round(nrows * 0.5))
            colOffset = int(round(ncols * 0.5))
            xLeft = int(round(col0 - colOffset))
            yTop = int(round(row0 - rowOffset))

            if yTop < top:
                yTop = top
            if yTop + nrows > bottom:
                yTop = bottom - nrows
            if xLeft < left:
                xLeft = left
            if xLeft + ncols > right:
                xLeft = right - ncols

        extent = get_extent(geoTransformation, xLeft, yTop)
        # set the new x0,y0 in geo coordinates:
        xy = np.asarray(extent)
        ulx2 = int(xy[2, 0])
        uly2 = int(xy[2, 1])
        geoTransformation = [ulx2, res, 0.0, uly2, 0.0, -res]
    else:
        nrows = config.nrows
        ncols = config.ncols

    extent = get_extent(geoTransformation, ncols, nrows)
    return np.asarray(extent)

def get_lon_lat(config, xy):
    if 'LANDSAT' in config.spacecraftName:
        if config.collection_number == 1: #coll_1
            zone1 = int(config.metadata['L1_METADATA_FILE']['PROJECTION_PARAMETERS']['UTM_ZONE'])
            zone2 = (config.metadata['L1_METADATA_FILE']['PROJECTION_PARAMETERS']['ORIENTATION'])[0]
        else: #coll_2
            zone1 = int(config.metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['UTM_ZONE'])
            zone2 = (config.metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['ORIENTATION'])[0]
    else:  # Sentinel or hyper
        # get the target resolution and metadata for the resampled bands below:
        xp = L2A_XmlParser(config, 'T2A')
        tg = xp.getTree('Geometric_Info', 'Tile_Geocoding')
        hcsName = tg.HORIZONTAL_CS_NAME.text
        zone = hcsName.split()[-1]
        zone1 = int(zone[:-1])
        zone2 = zone[-1:].upper()

    # lonMin, latMin, dummy = transform_utm_to_wgs84(xy[1, 0], xy[1, 1], zone1, zone2) old
    # lonMax, latMax, dummy = transform_utm_to_wgs84(xy[3, 0], xy[3,1], zone1, zone2) old
    # latMin, lonMin, dummy = transform_utm_to_wgs84(xy[1, 0], xy[1, 1], zone1, zone2) new
    # latMax, lonMax, dummy = transform_utm_to_wgs84(xy[3, 0], xy[3, 1], zone1, zone2) new
    lat0, lon0, dummy = transform_utm_to_wgs84(xy[0, 0], xy[0, 1], zone1, zone2)
    lat1, lon1, dummy = transform_utm_to_wgs84(xy[1, 0], xy[1, 1], zone1, zone2)
    lat2, lon2, dummy = transform_utm_to_wgs84(xy[2, 0], xy[2, 1], zone1, zone2)
    lat3, lon3, dummy = transform_utm_to_wgs84(xy[3, 0], xy[3, 1], zone1, zone2)

    lonMin = min(lon0, lon1, lon2, lon3)
    latMin = min(lat0, lat1, lat2, lat3)
    lonMax = max(lon0, lon1, lon2, lon3)
    latMax = max(lat0, lat1, lat2, lat3)

    return lonMin, latMin, lonMax, latMax

    # fix for SIIMPC-550.2, UMW:
def is_dted(config):
    demDir = os.path.join(config.home, config.demDirectory)
    filemask = 'e*.dt*'
    if os.path.exists(demDir):
        files = sorted(os.listdir(demDir))
        for filename in files:
            if fnmatch.fnmatch(filename, filemask):
                return True
    return False

class AuxImport(object):
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.corner_coordinates = None
        self.tmpdir = ''


        # reading references from config:
        xp = L2A_XmlParser(config, 'SC_GIPP')
        xp.validate()

        ### Snow_map_reference
        node = xp.getTree('Scene_Classification', 'References')

        par = node.Snow_Map
        if par is None: config.parNotFound(node)
        self.snowMapReference = par.text

        ### ESA_CCI_WaterBodies_map_reference
        par = node.ESACCI_WaterBodies_Map
        if par is None: config.parNotFound(node)
        self.esacciWaterBodiesReference = par.text

        ### ESA_CCI_LandCover_map_reference
        par = node.ESACCI_LandCover_Map
        if par is None: config.parNotFound(node)
        self.esacciLandCoverReference = par.text

        ### ESA_CCI_SnowCondition_map_directory_reference
        par = node.ESACCI_SnowCondition_Map_Dir
        if par is None: config.parNotFound(node)
        self.esacciSnowConditionDirReference = par.text
        self.tmpdir = tempfile.mkdtemp(dir = config.work_dir)

        if 'LANDSAT' in config.spacecraftName:
            if config.collection_number == 1: #coll_1
                zone = int(config.metadata['L1_METADATA_FILE']['PROJECTION_PARAMETERS']['UTM_ZONE'])
                orientation = (config.metadata['L1_METADATA_FILE']['PROJECTION_PARAMETERS']['ORIENTATION'])[0]
            else: #coll_2
                zone = int(config.metadata['LANDSAT_METADATA_FILE']['LEVEL1_PROJECTION_PARAMETERS']['UTM_ZONE'])
                orientation = (config.metadata['LANDSAT_METADATA_FILE']['LEVEL1_PROJECTION_PARAMETERS']['ORIENTATION'])[0]
            epsg_code = 32600
            epsg_code += zone
            if orientation == 'S':
                epsg_code += 100
            self.hcsCode = 'EPSG:' + str(epsg_code)
        else: # Sentinel
            xp = L2A_XmlParser(config, 'T2A')
            tg = xp.getTree('Geometric_Info', 'Tile_Geocoding')
            self.hcsCode = tg.HORIZONTAL_CS_CODE.text

        if 'LANDSAT' in config.spacecraftName:
            xy = get_geo_coordinates_landsat(config)
        else: # Sentinel
            xy = self.config.get_geo_coordinates_sentinel()
        self.lonMin, self.latMin, self.lonMax, self.latMax = get_lon_lat(config, xy)
        self.xy = xy

    def __del__(self):
        try:
            shutil.rmtree(self.tmpdir)
        except:
            pass

    def get_auxdata_from_product(self, bandIndex):
        '''
        PWC (Precipitable Water Content), Grib Unit [kg/m^2]
        MSL (Mean Sea Level pressure),    Grib Unit [Pa]
        OZO (Ozone),                      Grib Unit [kg/m^2]

        calculation for Ozone according to R. Richter (20/1/2016):
        ----------------------------------------------------------
        GRIB_UNIT = [kg/m^2]
        standard ozone column is 300 DU (Dobson Units),
        equals to an air column of 3 mm at STP (standard temperature (0 degree C) and pressure of 1013 mbar).

        Thus, molecular weight of O3 (M = 48): 2.24 g (equals to 22.4 liter at STP)

        300 DU = 3 mm  (equals to (0.3*48 / 2.24) [g/m^2])
         = 6.428 [g/m^2] = 6.428 E-3 [kg/m^2]

        Example:

        ozone (GRIB) = 0.005738 (equals to DU = 300 * 0.005738/6.428 E-3)
        ozone (DU)   = 267.4 DU

        Thus, ozone GRIB will be weighted with factor 155.5694 (equals to 1/6.428 E-3)
        in order to receive ozone in DU
        '''
        auxBands = [self.PWC, self.MSL, self.OZO]
        if bandIndex in auxBands == False:
            self.logger.error('wrong band index for aux data')
            return False

        bandIndex -= 29  # bandIndex starts at 30
        ozoneFactor = 155.5694  # 1/6.428 E-3
        standardOzoneColumn = 300.0

        straux_src = os.path.join(self._L2A_AuxDataDir, self.aux_src)
        curdir = os.path.curdir
        head, tail = os.path.split(straux_src)
        l.acquire()
        os.chdir(head)
        arr = False
        while True:
            try:
                dataSet = gdal.Open(tail, GA_ReadOnly)
                band = dataSet.GetRasterBand(bandIndex)
                arr = BandReadAsArray(band)
                if bandIndex == 3:  # recalculate to 300 DU:
                    arr = arr * standardOzoneColumn * ozoneFactor
                break
            except:
                self.logger.error('error in reading ozone values from aux data')
            finally:
                os.chdir(curdir)
                l.release()
                return arr

    # implementation of SIIMPC-828, UMW: get mid latitude from geoposition and date:
    def setMidLatitude(self):
        if self.config.midLatitude in ['SUMMER', 'WINTER']:
            return
        # else:
        lat_cen = int((self.latMax + self.latMin) / 2)
        doy_acq, isLeap = get_day_of_year(self.config.acquisitionDate)
        if not isLeap:
            Apr1 = 91  # 1. April
            Oct1 = 274  # 1. October
        else:
            Apr1 = 92  # 1. April
            Oct1 = 275  # 1. October

        self.logger.info('center of latitude is: %d. Day of year is: %d' % (lat_cen, doy_acq))

        # for Tropical/Equatorial areas ( latitude [-30:30] ):
        if -30 <= lat_cen < 30:
            self.config.midLatitude = 'SUMMER'
        # for Northern Hemisphere( latitude [ 30:90] ):
        elif 30 <= lat_cen < 90:
            if Apr1 <= doy_acq < Oct1:
                self.config.midLatitude = 'SUMMER'
            else:
                self.config.midLatitude = 'WINTER'
        # for Southern Hemisphere( latitude [-90:-30]):
        elif -90 <= lat_cen < -30:
            if Apr1 <= doy_acq < Oct1:
                self.config.midLatitude = 'WINTER'
            else:
                self.config.midLatitude = 'SUMMER'
        self.logger.info('mid latitude set to %s according to area and date' % (self.config.midLatitude))
        return

    def gdalDEM_dted(self):
        import scipy.misc #update from L2A_tables
        demDir = self.config.demDirectory
        if demDir == 'NONE':
            self.logger.info('DEM directory not specified, flat surface is used, and DEM will not be exported')
            self.dem_error_type = 'False'
            self.config.demOutput = False
            return False , None

        self.logger.info('Start DEM alignment for tile')
        sourceDir = os.path.join(self.config.home, demDir)
        
        lonMin = int(np.rint(self.lonMin))
        lonMax = int(np.rint(self.lonMax))
        latMin = int(np.rint(self.latMin))
        latMax = int(np.rint(self.latMax))
        dtedf_src = ''

        filelist = sorted(os.listdir(sourceDir))
        found = False

        # Fix for SIIMPC-944 VD-JL - International Date Line handling for DEM mosaicking
        if (self.lonMax - self.lonMin) < 180:
        # if lonMin <= lonMax:
            lons = list(range(lonMin - 1, lonMax + 1))
        else:
            lons = list(range(-180, int(self.lonMin))) + list(
                range(int(self.lonMax), 180))  # gives [179, -180] for int(self.lonMin)=179 and int(self.lonMax)=-179
            self.logger.info('This tile is crossing the international date line, a particular processing is performed')

        infiles = []
        for lon in lons:
            for lat in range(latMin - 1, latMax + 1):
                if lon < 0:
                    lonMask = 'w'
                else:
                    lonMask = 'e'
                if lat < 0:
                    latMask = 's'
                else:
                    latMask = 'n'

                file_mask = '%s%03d_%s%02d.dt*' % (lonMask, abs(lon), latMask, abs(lat))
                # end fix for SIIMPC-573, 944
                for filename in filelist:
                    if (fnmatch.fnmatch(filename, file_mask) == True):
                        infiles.append(os.path.join(sourceDir, filename))
                        # if 'LANDSAT' not in self.config.spacecraftName: #update from Sen2Cor 2.10
                        self.config.aux_data_filelist.append(os.path.basename(filename)) #update from Sen2Cor 2.10
                        found = True
                        break
        if (self.lonMax - self.lonMin) < 180:
        # if lonMin <= lonMax:
            # Fix for SIIMPC-944 VD-JL - International Date Line handling for DEM mosaicking
            kwargs = ' -r bilinear '  # fix for SIIMPC-1006.2 UMW
        else:
            gdal.SetConfigOption('CENTER_LONG','180')
            kwargs = ' -r bilinear -t_srs EPSG:4326'
        kwargs += ' -ot Int16'
        # Fix for SIIMPC-1613:
        kwargs += ' -dstnodata -20000'

        tmpDir = self.tmpdir
        dtedf_dest = os.path.join(self.tmpdir, 'dted_' + self.config.L2A_TILE_ID + '_src.tif')

        if infiles == []:
            self.logger.stream('No DEM files found.')
            self.dem_error_type = 'False' #update from Sen2Cor 2.10
            return False , None #update from Sen2Cor 2.10

        try:
            ds = gdal.Warp(dtedf_dest, infiles, options=kwargs)
            # Fix for SIIMPC-1613:
            NODATA = -20000

            ds4demresolution = gdal.Open(infiles[0]) #update from Sen2Cor 2.10
            self._input_dem_resolution = round(abs(ds4demresolution.GetGeoTransform()[5]), 8) #update from Sen2Cor 2.10
            ds4demresolution = None #update from Sen2Cor 2.10

            dem_band = ds.GetRasterBand(1)
            dem_arr = dem_band.ReadAsArray()
            dem_arr[dem_arr == NODATA] = 0
            dem_band.WriteArray(dem_arr)
            dem_band.FlushCache()
            ds = None
            gdal.SetConfigOption('CENTER_LONG','0')
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('error using gdalwarp')
            self.dem_error_type = 'True' #update from Sen2Cor 2.10
            return False , None
            
        dtedf_src = dtedf_dest
        dtedf_dest = os.path.join(tmpDir, 'dted_' + self.config.L2A_TILE_ID + '_dem.tif')
        xy = self.xy
        kwargs = '-ot Float32'
        kwargs += ' -t_srs ' + self.hcsCode
        kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
        kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
        kwargs += ' -r cubicspline'
        # Fix for SIIMPC-1613:
        kwargs += ' -dstnodata -20000.0'

        try:
            ds = gdal.Warp(dtedf_dest, dtedf_src, options=kwargs)
            # Fix for SIIMPC-1613:
            NODATA = -20000.0
            dem_band = ds.GetRasterBand(1)
            dem_arr = dem_band.ReadAsArray()
            dem_arr[dem_arr == NODATA] = 0.0
            dem_band.WriteArray(dem_arr)
            dem_band.FlushCache()
            ds = None
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('error using gdalwarp')
            self.dem_error_type = 'True' #update from Sen2Cor 2.10
            os.remove(dtedf_src)
            return False , None

        if dem_arr.max() == 0.0: #update from Sen2Cor 2.10
            self.logger.info('DEM retrieved data contains no values, switching to flat surface')
            self.logger.warning('DEM retrieved data contains no values, switching to flat surface')
            os.remove(dtedf_src)
            # if 'LANDSAT' not in self.config.spacecraftName:  # update from Sen2Cor 2.10
            self.config.aux_data_filelist = []
            self.dem_error_type = 'True'
            return False , None #update from Sen2Cor 2.10

        # fix for SIIMPC-792, JL:
        kwargs = '-ot Int16'
        dtedf_dst_int16 = os.path.join(tmpDir, 'dted_' + self.config.L2A_TILE_ID + '_dem_int16.tif')
        try:
            gdal.Translate(dtedf_dst_int16, dtedf_dest, options=kwargs)
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('Error reading DEM, flat surface will be used')
            self.dem_error_type = 'True' #update from Sen2Cor 2.10
            os.remove(dtedf_dest)
            return False , None
        finally:
            os.remove(dtedf_src)
        # end of fix for SIIMPC-792
        self.logger.info('DEM received and prepared')
        return dtedf_dest, dtedf_dst_int16

    def gdalCCI_wb(self):
        esacciWaterBodies = self.esacciWaterBodiesReference
        esacciWaterBodies = os.path.join(self.config.aux_dir, esacciWaterBodies)
        if ((os.path.isfile(esacciWaterBodies)) == False):
            self.logger.warning(
                'ESA CCI Water Bodies map not present, water detection will be performed without a priori information')
            return True

        lonMin = int(np.rint(self.lonMin))
        lonMax = int(np.rint(self.lonMax))
        latMin = int(np.rint(self.latMin))
        latMax = int(np.rint(self.latMax))
        tmpDir = self.tmpdir

        # step 1: check if the S2 tile crosses the International Date Line:
        if (lonMax - lonMin) < 180:
        # if lonMin <= lonMax:
            kwargs = ''
        else:
            self.logger.warning('International Date Line is crossed, ESA CCI WBI map reframing is performed')

            ymin = np.clip(latMin - 0.5, -90.0, 90.0)
            ymax = np.clip(latMax + 0.5, -90.0, 90.0)

            cci_wb_dst_east = os.path.join(tmpDir, 'cci_wb' + self.config.L2A_TILE_ID + '_{0}m_east.tif'.format(
                self.config.resolution))
            cci_wb_dst_west = os.path.join(tmpDir, 'cci_wb' + self.config.L2A_TILE_ID + '_{0}m_west.tif'.format(
                self.config.resolution))
            cci_wb_dst_dateline = os.path.join(tmpDir, 'cci_wb' + self.config.L2A_TILE_ID + '_{0}m_dateline.tif'.format(
                self.config.resolution))

            kwargs = '-te 170.0 {0} 180.0 {1} '.format(ymin, ymax) #update from Sen2Cor 2.10
            # kwargs = '-te 178.5 {0} 180 {1} '.format(ymin, ymax)
            try:
                gdal.Warp(cci_wb_dst_east, esacciWaterBodies, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.warning(
                    'Cannot perform reframing step 1, no water bodies a priori information will be used')
                return False

            kwargs = '-te -180.0 {0} -170.0  {1} '.format(ymin, ymax) #update from Sen2Cor 2.10
            # kwargs = '-te -180 {0} -178.5 {1} '.format(ymin, ymax)
            try:
                gdal.Warp(cci_wb_dst_west, esacciWaterBodies, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.warning(
                    'Cannot perform reframing step 2, no water bodies a priori information will be used')
                return False

            gdal.SetConfigOption('CENTER_LONG','180')
            kwargs = '-t_srs EPSG:4326'
            infiles = [cci_wb_dst_west, cci_wb_dst_east]
            try:
                gdal.Warp(cci_wb_dst_dateline, infiles, options=kwargs)
                gdal.SetConfigOption('CENTER_LONG','0')
            except Exception as e:
                self.logger.error(e)
                self.logger.warning(
                    'Cannot perform reframing step 3, no water bodies a priori information will be used')
                return False

            esacciWaterBodies = cci_wb_dst_dateline

        # step 2: extraction and reprojection into S2 tile geometry:
        xy = self.xy
        kwargs = '-t_srs ' + self.hcsCode
        kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
        kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
        kwargs +=' -r cubicspline'

        cci_wb_dst = os.path.join(tmpDir, 'cci_wb' + self.config.L2A_TILE_ID + '_{0}m.tif'.format(self.config.resolution))
        try:
            gdal.Warp(cci_wb_dst, esacciWaterBodies, options=kwargs)
        except Exception as e:
            self.logger.error(e)
            self.logger.warning('Cannot read esa cci, no water bodies a priori information will be used')
            return False

        try:
            os.remove(cci_wb_dst_east)
            os.remove(cci_wb_dst_west)
            os.remove(cci_wb_dst_dateline)
        except:
            pass

        self.logger.info('ESA CCI Water Bodies received and prepared')
        return cci_wb_dst

    def gdalCCI_lccs(self):
        esacciLandCover = self.esacciLandCoverReference
        esacciLandCover = os.path.join(self.config.aux_dir, esacciLandCover)
        if ((os.path.isfile(esacciLandCover)) == False):
            self.logger.warning(
                'ESA CCI Land Cover map not present, cloud detection over urban areas will be performed without a priori information')
            return True

        lonMin = int(np.rint(self.lonMin))
        lonMax = int(np.rint(self.lonMax))
        latMin = int(np.rint(self.latMin))
        latMax = int(np.rint(self.latMax))
        tmpDir = self.tmpdir

        # step 1: check if the S2 tile crosses the International Date Line:
        if (lonMax - lonMin) < 180:
        # if lonMin <= lonMax:
            kwargs = ''
        else:
            self.logger.warning('International Date Line is crossed, ESA CCI LCM map reframing is performed')

            ymin = np.clip(latMin - 0.5, -90.0, 90.0)
            ymax = np.clip(latMax + 0.5, -90.0, 90.0)

            cci_lccs_dst_east = os.path.join(tmpDir, 'cci_lccs' + self.config.L2A_TILE_ID + '_{0}m_east.tif'.format(
                self.config.resolution))
            cci_lccs_dst_west = os.path.join(tmpDir, 'cci_lccs' + self.config.L2A_TILE_ID + '_{0}m_west.tif'.format(
                self.config.resolution))
            cci_lccs_dst_dateline = os.path.join(tmpDir,
                                                 'cci_lccs' + self.config.L2A_TILE_ID + '_{0}m_dateline.tif'.format(
                                                     self.config.resolution))

            kwargs = '-te 170.0 {0} 180.0 {1}'.format(ymin, ymax) #update from Sen2Cor 2.10
            # kwargs = '-te 178.5 {0} 180 {1}'.format(ymin, ymax)
            try:
                gdal.Warp(cci_lccs_dst_east, esacciLandCover, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.warning(
                    'Cannot perform reframing step 1, no land cover a priori information will be used')
                return False

            kwargs = '-te -180.0 {0} -170.0 {1}'.format(ymin, ymax) #update from Sen2Cor 2.10
            # kwargs = '-te -180 {0} -178.5 {1}'.format(ymin, ymax)
            try:
                gdal.Warp(cci_lccs_dst_west, esacciLandCover, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.warning(
                    'Cannot perform reframing step 2, no land cover a priori information will be used')
                return False
            gdal.SetConfigOption('CENTER_LONG','180')
            kwargs = ' -t_srs EPSG:4326'
            infiles = [cci_lccs_dst_west, cci_lccs_dst_east]
            try:
                gdal.Warp(cci_lccs_dst_dateline, infiles, options=kwargs)
                gdal.SetConfigOption('CENTER_LONG','0')
            except Exception as e:
                self.logger.error(e)
                self.logger.warning(
                    'Cannot perform reframing step 3, no land cover a priori information will be used')
                return False

            esacciLandCover = cci_lccs_dst_dateline
            kwargs = ''

        # step 2: extraction and reprojection into S2 tile geometry:
        xy = self.xy
        kwargs = '-t_srs ' + self.hcsCode
        kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
        kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
        kwargs += ' -r near'

        cci_lccs_dst = os.path.join(tmpDir,
                                    'cci_lccs_' + self.config.L2A_TILE_ID + '_{0}m.tif'.format(self.config.resolution))
        try:
            gdal.Warp(cci_lccs_dst, esacciLandCover, options=kwargs)
        except Exception as e:
            self.logger.error(e)
            self.logger.warning('Cannot read esa cci lccs, no land cover a priori information will be used')
            return False

        try:
            os.remove(cci_lccs_dst_east)
            os.remove(cci_lccs_dst_west)
            os.remove(cci_lccs_dst_dateline)
        except:
            pass

        self.logger.info('ESA CCI Land Cover map prepared')
        return cci_lccs_dst

    def gdalCCI_snowc(self):
        from datetime import datetime
        import glob
        month = getMonth(self.config.acquisitionDate)
        doy_acq, isLeap = get_day_of_year(self.config.acquisitionDate)
        esacciSnowConditionDir = self.esacciSnowConditionDirReference
        esacciSnowConditionDir = os.path.join(self.config.aux_dir, esacciSnowConditionDir)

        listSnowConditionFiles = glob.glob(
            os.path.join(esacciSnowConditionDir, 'ESACCI-LC-L4-Snow-Cond-AggOcc-500m-MONTHLY-2000-2012-2000*-2.4.tif'))
        if listSnowConditionFiles != []:
            listSnowConditionFiles.sort()
            esacciSnowCondition = listSnowConditionFiles[month - 1]
            # if (doy_acq > 363) | ((doy_acq > 362) & (not isLeap)):
            #     esacciSnowCondition = listSnowConditionFiles[-1]
            # else:
            #     # special handling for Day Of Year 61
            #     if doy_acq == 61:
            #         esacciSnowCondition = listSnowConditionFiles[8]  # corresponds to 9th weekly product of the year
            #
            #     datePattern = os.path.join(esacciSnowConditionDir,
            #                                'ESACCI-LC-L4-Snow-Cond-AggOcc-500m-P13Y7D-2000-2012-%Y%m%d-v2.4.tif')
            #     for SnowConditionFile in listSnowConditionFiles:
            #         doy_file = datetime.strptime(SnowConditionFile, datePattern).timetuple().tm_yday
            #         if abs(doy_acq - doy_file) < 4:
            #             esacciSnowCondition = SnowConditionFile
            #             break
        else:
            self.logger.warning('ESA CCI Snow Condition map not present, no snow map post-processing will be done')
            return True


        lonMin = int(np.rint(self.lonMin))
        lonMax = int(np.rint(self.lonMax))
        latMin = int(np.rint(self.latMin))
        latMax = int(np.rint(self.latMax))

        tmpDir = self.tmpdir
        # step 1: check if the S2 tile crosses the International Date Line:
        if (lonMax - lonMin) < 180:
        # if self.lonMin <= self.lonMax:
            kwargs = ''
        else:
            self.logger.warning('International Date Line is crossed, ESA CCI SNC map reframing is performed')

            ymin = np.clip(latMin - 0.5, -90.0, 90.0)
            ymax = np.clip(latMax + 0.5, -90.0, 90.0)

            cci_snowc_dst_east = os.path.join(tmpDir, 'cci_snowc' + self.config.L2A_TILE_ID + '_{0}m_east.tif'.format(
                self.config.resolution))
            cci_snowc_dst_west = os.path.join(tmpDir, 'cci_snowc' + self.config.L2A_TILE_ID + '_{0}m_west.tif'.format(
                self.config.resolution))
            cci_snowc_dst_dateline = os.path.join(tmpDir,
                                                  'cci_snowc' + self.config.L2A_TILE_ID + '_{0}m_dateline.tif'.format(
                                                      self.config.resolution))

            kwargs = '-te 170.0 {0} 180.0 {1}'.format(ymin, ymax) #update from Sen2Cor 2.10
            # kwargs = '-te 178.5 {0} 180 {1}'.format(ymin, ymax)
            try:
                gdal.Warp(cci_snowc_dst_east, esacciSnowCondition, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.warning(
                    'Cannot perform reframing step 1, no snow condition a priori information will be used')
                return False

            kwargs = '-te -180.0 {0} -170.0 {1}'.format(ymin, ymax) #update from Sen2Cor 2.10
            # kwargs = '-te -180 {0} -178.5 {1}'.format(ymin, ymax)
            try:
                gdal.Warp(cci_snowc_dst_west, esacciSnowCondition, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.warning(
                    'Cannot perform reframing step 2, no snow condition a priori information will be used')
                return False

            gdal.SetConfigOption('CENTER_LONG','180')
            kwargs = ' -t_srs EPSG:4326'
            infiles = [cci_snowc_dst_west, cci_snowc_dst_east]
            try:
                gdal.Warp(cci_snowc_dst_dateline, infiles, options=kwargs)
                gdal.SetConfigOption('CENTER_LONG','0')
            except Exception as e:
                self.logger.error(e)
                self.logger.warning(
                    'Cannot perform reframing step 3, no snow condition a priori information will be used')
                return False

            esacciSnowCondition = cci_snowc_dst_dateline
            kwargs = ''

        # step 2: extraction and reprojection into S2 tile geometry:
        xy = self.xy
        kwargs = '-t_srs ' + self.hcsCode
        kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
        kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
        kwargs += ' -r near'

        cci_snowc_dst = os.path.join(tmpDir,
                                     'cci_snowc_' + self.config.L2A_TILE_ID + '_{0}m.tif'.format(self.config.resolution))
        try:
            gdal.Warp(cci_snowc_dst, esacciSnowCondition, options=kwargs)
        except Exception as e:
            self.logger.error(e)
            self.logger.warning('Cannot read esa cci snowc, no snow condition a priori information will be used')
            return False
        try:
            os.remove(cci_snowc_dst_east)
            os.remove(cci_snowc_dst_west)
            os.remove(cci_snowc_dst_dateline)
        except:
            pass

        self.logger.info('ESA CCI Snow Condition map prepared')
        return cci_snowc_dst

    def getCAMS_singledate(self, hour1900, ncfile):
        """
        Retrieve the ECMWF CAMS aod550 for the given date,
         and convert it into visibilty
        :param hour1900:
        :param ncfile:
        :return: visibility
        """
        from netCDF4 import Dataset

        # read aod550 at time hour1900
        rootgrp = Dataset(ncfile, 'r')
        nctimes = rootgrp.variables['time'][:]
        if hour1900 not in nctimes:
            self.config.timestamp(
                'L2A_Tables: CAMS ncfile: TIME ERROR: {0} is not found in {1} '.format(ncfile, hour1900))
            return None
        aod550 = rootgrp.variables['aod550'][nctimes.tolist().index(hour1900), ...]

        # load Geopential altitude
        cams_z = os.path.join(self.camsDir, 'z_cams_c_ecmf_20180407000000_prod_an_sfc_000_z.nc')
        rootgrp_z = Dataset(cams_z, 'r')
        g = 9.80665
        z = rootgrp_z.variables['z'][0, ...] / g

        # convert to visibility
        """Return the visibility (VIS) for given AOT at 550nm and altitude (z)
                Source of relation: ATCOR manual
                Coefficients from Rolf Richter (personal communication)
                Parameters a and b had been determined by least squares fit
                of AOT as a function of visibilities. AOT was taken from MODTRAN
                transmission computed with different values of VIS.
                VIS = exp( (ln AOT - a) / b)
                AOT = exp( a + b*ln VIS)
                a = a (z) = -0,026235*z^2-0,182321*z+1,54641
                b = b (z) = 0,003096*z^2+0,011604*z-0,854022"""

        z = z / 1000.  # in km
        a = -0.026235 * (z ** 2) - 0.182321 * z + 1.54641
        b = 0.003096 * (z ** 2) + 0.011604 * z - 0.854022

        vis = np.exp((np.log(aod550) - a) / b)
        vis = np.clip(vis, 0, 650)

        return vis

    def getCAMS_findnc(self, hour1900):
        """
        find netcf file for the forecast date given
        :param hour1900: hours since 1900-01-01T00:00:00 (CAMS conventions)
        :return: path to ncdf file, and source
        """

        camsFtpDir = os.path.join(self.camsDir, 'CAMS_NREALTIME')
        camsServerDir = os.path.join(self.camsDir, 'ECMWFDataServer')

        # example of "Server format" : Macc_aot550_World_20160924_.nc
        # example of "FTP format" : z_cams_c_ecmf_20180411120000_prod_fc_sfc_002_aod550.nc


        date = dt.datetime(1900, 1, 1) + dt.timedelta(hours=hour1900)

        # Try ncdf from FTP (every hour)
        hour = date.hour
        if hour == 0:
            subdir = (date - dt.timedelta(days=1)).strftime('%Y%m%d')
            hour = 24
        else:
            subdir = date.strftime('%Y%m%d')
        if hour <= 12:
            subdir += '00'
            offset = hour
        else:
            subdir += '12'
            offset = hour - 12

        ncname = 'z_cams_c_ecmf_{0}0000_prod_fc_sfc_{1:03d}_aod550.nc'.format(subdir, offset)
        ncfile = os.path.join(camsFtpDir, subdir, ncname)

        if os.path.exists(ncfile):
            self.config.timestamp('L2A_Tables: CAMS ncfile (FTP):{}'.format(ncfile))
            return ncfile, 'FTP'

        # Try ncdf from SERVER (every 3 hours)
        if date.hour % 3 == 0:
            if date.hour < 3:
                # get day before
                datestr = (date - dt.timedelta(days=1)).strftime('%Y%m%d')
            else:
                datestr = date.strftime('%Y%m%d')

            ncname = 'Macc_aot550_World_{0}_.nc'.format(datestr)
            ncfile = os.path.join(camsServerDir, ncname)

            if os.path.exists(ncfile):
                self.config.timestamp('L2A_Tables: CAMS ncfile (SERVER):{}'.format(ncfile))
                return ncfile, 'SERVER'

        # all tries failed
        return None, None


    def noCAMS_waterVapour_fallback(self, wv):
        # initial "good" pixels value
        tcwv = 0.001
        doy = self.config.angle_coefficient['SOLAR_VECTOR']['SOLAR_EPOCH_DAY']
        lat_cen = int((self.latMax + self.latMin) * 0.5)

        # NH WINTER TIME, SH SUMMER TIME
        if (1 <= doy <= 92) or (275 <= doy <= 366):
            if 90 >= lat_cen >= 65:
                tcwv = 0.42
            if 65.0 > lat_cen >= 55.0:
                tcwv = 0.85
            if 55.0 > lat_cen >= 35.0:
                tcwv = 2.08
            if 35.0 > lat_cen >= 25.0:
                tcwv = 2.92
            if 25.0 > lat_cen >= -20.0:
                tcwv = 4.11
            if -20.0 > lat_cen >= -35.0:
                tcwv = 2.92
            if -35.0 > lat_cen >= -55.0:
                tcwv = 2.08
            if -55.0 > lat_cen >= -90.0:
                tcwv = 0.85
        # NH SUMMER TIME, SH WINTER TIME
        elif 93 <= doy <= 274:
            if 90 >= lat_cen >= 65:
                tcwv = 0.42
            if 65 > lat_cen >= 55:
                tcwv = 0.85
            if 55.0 > lat_cen >= 35.0:
                tcwv = 2.08
            if 35.0 > lat_cen >= 25.0:
                tcwv = 2.92
            if 25.0 > lat_cen >= -20.0:
                tcwv = 4.11
            if -20 > lat_cen >= -35:
                tcwv = 2.92
            if -35 > lat_cen >= -55:
                tcwv = 2.08
            if -55 > lat_cen >= -90:
                tcwv = 0.85

        # convert wv to np.float32 to support float assignment (wv is initially only the byte no_data mask)
        wv = wv.astype(np.float32)
        wv[wv != 0] = tcwv

        # return wv value as uint16 with the scaling factor 1000 applied
        return np.uint16(np.rint(wv * 1000.0))


    def compute_visibility_landsat(self,warped_aod,warped_geopotential):
        #
        #computing visibility form auxdata
        #
        read_aod=gdal.Open(warped_aod, gdal.GA_ReadOnly)
        band_aod = read_aod.GetRasterBand(1)
        aod_subset=band_aod.ReadAsArray()

        g = 9.80665
        read_geo=gdal.Open(warped_geopotential, gdal.GA_ReadOnly)
        band_geo = read_geo.GetRasterBand(1)
        geopotential_array=band_geo.ReadAsArray() / g

        # convert to visibility
        """Return the visibility (VIS) for given AOT at 550nm and altitude (z)
                Source of relation: ATCOR manual
                Coefficients from Rolf Richter (personal communication)
                Parameters a and b had been determined by least squares fit
                of AOT as a function of visibilities. AOT was taken from MODTRAN
                transmission computed with different values of VIS.
                VIS = exp( (ln AOT - a) / b)
                AOT = exp( a + b*ln VIS)
                a = a (z) = -0,026235*z^2-0,182321*z+1,54641
                b = b (z) = 0,003096*z^2+0,011604*z-0,854022"""

        geopotential_array = geopotential_array / 1000.  # in km
        a = -0.026235 * (geopotential_array ** 2) - 0.182321 * geopotential_array + 1.54641
        b = 0.003096 * (geopotential_array ** 2) + 0.011604 * geopotential_array - 0.854022

        vis_from_aux = np.exp((np.log(aod_subset) - a) / b)
        vis_from_aux = np.clip(vis_from_aux, 0, 650) #clipping to avoid integer overflow (To be investigated for lower bound visibility)
        vis_from_aux = np.uint16(vis_from_aux * 100 + 0.5)

        return vis_from_aux


    def import_geopotential(self, name_geo):
        from L2A_Library import rectBivariateSpline
        from netCDF4 import Dataset
        from skimage import io
        from datetime import datetime, timedelta

        input_name= name_geo
        #z_pattern ='z_cams_c_ecmf_20200507000000_prod_an_sfc_001_z.nc'
        z_pattern = 'z_cams_c_ecmf_20180407000000_prod_an_sfc_000_z.nc' # updated as no reference for z_cams_c_ecmf_20200507000000_prod_an_sfc_001_z.nc
        self.camsDir = os.path.join(self.config.aux_dir, 'ECMWF')
        selected_z = os.path.join(self.camsDir, z_pattern)
        if ((os.path.exists(selected_z)) == False):
            self.config.timestamp('L2A_Tables: [WARNING] Geopotential file not found. Try to use a fallback.')
            return False
        rootgrp_z = Dataset(selected_z)
        geopot_z = rootgrp_z.variables['z'][0, ...]

        cams_geop_z = os.path.join(self.tmpdir,'cams_geop_z.tif')
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(cams_geop_z, xsize=geopot_z.shape[1], ysize=geopot_z.shape[0], bands=1, eType=gdal.GDT_Float32)
        src_band = dataset.GetRasterBand(1)
        src_band.WriteArray(geopot_z)
        src_band.FlushCache()

        # setting the Geoinformation of the cams_geop_z file:
        dataset.SetGeoTransform([-0.2, 0.4, 0.0, 90.2, 0.0, -0.4])
        dataset = None

        # get the latitude longitude bounding box of the product
        lonMin, latMin, lonMax, latMax = get_lon_lat(self.config, self.xy)

        # apply specific UTM reprojection algorithm in case the product crosses the Greenwich meridian
        # crossing the greenwich meridian is identified by the condition (lonMin < 0) & (lonMax > 0)

        if (lonMin < 0) & (lonMax > 0):

            # 1st step is to mosaic CAMS geoid with greenwich meridian in the middle of the global world image
            # the reason behind this "mosaicking" or "shifting" the center of the global world image is to
            # simplify the gdal reprojection in the zone of the discontinuity at Greenwhich meridian zone
            # because original CAMS file have their center at the antimeridian, not at Greenwich meridian

            kwargs = '-ot Float32 '
            kwargs += '-s_srs EPSG:4326 '
            kwargs += '-t_srs EPSG:4326 '
            kwargs += ' -te %f %f %f %f' % (-180.2, -90.2, 179.8, 90.2)
            kwargs += ' -ts %d %d' % (900, 451)
            kwargs += ' -r nearest'
            gdal.SetConfigOption('CENTER_LONG', '180')

            geop_z_param_mosaic = os.path.join(self.tmpdir, 'cams_geop_z_mosaic.tif')

            try:
                gdal.Warp(geop_z_param_mosaic, cams_geop_z, options=kwargs)
                # gdal.SetConfigOption('CENTER_LONG', '0')
            except Exception as e:
                self.logger.error(e)
                self.logger.fatal('Error mosaicking geopotential')
                return False

            # 2nd step is to reproject geoid on MRGS tile

            xy = self.xy
            kwargs = '-ot Float32 '
            kwargs += '-s_srs EPSG:4326 '
            kwargs += '-t_srs ' + self.hcsCode
            kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
            kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
            kwargs += ' -r bilinear'
            gdal.SetConfigOption('CENTER_LONG', '0')

            geop_z_param_output = os.path.join(self.tmpdir, 'cams_geop_z_output.tif')

            try:
                gdal.Warp(geop_z_param_output, geop_z_param_mosaic, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.fatal('Error reading geopotential')
                return False

        else:
            xy = self.xy
            kwargs = '-ot Float32 '
            kwargs += '-s_srs EPSG:4326 '
            kwargs += '-t_srs ' + self.hcsCode
            kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
            kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
            kwargs += ' -r cubicspline'
            gdal.SetConfigOption('CENTER_LONG', '180')

            geop_z_param_output = os.path.join(self.tmpdir, 'cams_geop_z_output.tif')

            try:
                gdal.Warp(geop_z_param_output, cams_geop_z, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.fatal('Error reading geopotential')
                return False

        os.remove(cams_geop_z)
        self.logger.info('Geopotential from  ECMWF retrieved')

        return geop_z_param_output


    def assign_CAMS_Conventions(self,rootgrp):

        try:
            if rootgrp.variables['forecast_reference_time']:
                assigned_conventions = 'CF-1.7'
        except:
            if rootgrp.variables['time']:
                assigned_conventions = 'CF-1.6'

        return assigned_conventions

    def retrieveCAMS_parameter(self,nc_file_input,parameter,selected_day):
        from L2A_Library import rectBivariateSpline
        from netCDF4 import Dataset
        from skimage import io
        from datetime import datetime, timedelta

        rootgrp = Dataset(nc_file_input)

        try:
           CAMS_Conventions = rootgrp.Conventions
        except:
           CAMS_Conventions = self.assign_CAMS_Conventions(rootgrp)

        if CAMS_Conventions == 'CF-1.7':
            nctimes = rootgrp.variables['forecast_reference_time'][:]
            secs1970 = (selected_day - datetime(1970, 1, 1)).total_seconds()
        else:
            nctimes = rootgrp.variables['time'][:]
            hour1900 = (selected_day - datetime(1900, 1, 1)).total_seconds() / 60. / 60.

        self.config.timestamp('L2A_Tables: CAMS Conventions ' + str(rootgrp.Conventions))

        if CAMS_Conventions == 'CF-1.7':
            u = np.abs(nctimes - secs1970)
            # np.argpartition : used to retrieve the two closest values for statistiscs
            val = [np.sort(u), np.argsort(u)]
            # The two first index :
            index1 = val[1][0]
            v1 = datetime(1970, 1, 1) + timedelta(
                seconds=int(nctimes[index1]))  # VDE : failure for me if hours is type np.int32
            index2 = val[1][1]
            v2 = datetime(1970, 1, 1) + timedelta(seconds=int(nctimes[index2]))
        else:
            u = np.abs(nctimes - hour1900)
            # np.argpartition : used to retrieve the two closest values for statistiscs
            val = [np.sort(u), np.argsort(u)]
            # The two first index :
            index1 = val[1][0]
            v1 = datetime(1900, 1, 1) + timedelta(
                hours=int(nctimes[index1]))  # VDE : failure for me if hours is type np.int32
            index2 = val[1][1]
            v2 = datetime(1900, 1, 1) + timedelta(hours=int(nctimes[index2]))

        self.config.timestamp('L2A_Tables: Input observation date (hour) ' + str(selected_day))
        self.config.timestamp('L2A_Tables: v1 ' + str(v1))
        self.config.timestamp('L2A_Tables: v2 ' + str(v2))

        if CAMS_Conventions == 'CF-1.7':
            root_variable = np.squeeze(rootgrp.variables[parameter])
            var_1 = root_variable[index1, ...]
            var_2 = root_variable[index2, ...]
            interpolated_data = var_1 + (secs1970 - nctimes[index1]) * (var_2 - var_1) / (
                        nctimes[index2] - nctimes[index1])
        else:
            var_1 = rootgrp.variables[parameter][index1, ...]
            var_2 = rootgrp.variables[parameter][index2, ...]
            interpolated_data = var_1 + (hour1900 - nctimes[index1]) * (var_2 - var_1) / (
                        nctimes[index2] - nctimes[index1])

        # save to file and copy geo information
        cams_world = os.path.join(self.tmpdir, 'cams_aux_par_{0}_World_{1}m.tif'.format(selected_day.strftime('%H%M%S'), '60'))
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(cams_world, xsize=interpolated_data.shape[1], ysize=interpolated_data.shape[0], bands=1, eType=gdal.GDT_Float32)
        src_band = dataset.GetRasterBand(1)
        if parameter == 'tcwv':
            src_band.WriteArray(interpolated_data / 10 * 1000)  # *1000 because casted to int
        elif parameter == 'gtco3':
            ozoneFactor = 155.5694  # 1/6.428 E-3
            standardOzoneColumn = 300.0
            src_band.WriteArray(interpolated_data * standardOzoneColumn * ozoneFactor)  # ozone
        else:
            src_band.WriteArray(interpolated_data)
        src_band.FlushCache()

        # setting the Geoinformation of the cams_world file:
        dataset.SetGeoTransform([-0.2, 0.4, 0.0, 90.2, 0.0, -0.4])
        dataset = None

        # get the latitude longitude bounding box of the product
        lonMin, latMin, lonMax, latMax = get_lon_lat(self.config, self.xy)

        # apply specific UTM reprojection algorithm in case the product crosses the Greenwich meridian
        # crossing the greenwich meridian is identified by the condition (lonMin < 0) & (lonMax > 0)

        if (lonMin < 0) & (lonMax > 0):

            # 1st step is to mosaic CAMS geoid with greenwich meridian in the middle of the global world image
            # the reason behind this "mosaicking" or "shifting" the center of the global world image is to
            # simplify the gdal reprojection in the zone of the discontinuity at Greenwhich meridian zone
            # because original CAMS file have their center at the antimeridian, not at Greenwich meridian

            kwargs = '-ot Float32 '
            kwargs += '-s_srs EPSG:4326 '
            kwargs += '-t_srs EPSG:4326 '
            kwargs += ' -te %f %f %f %f' % (-180.2, -90.2, 179.8, 90.2)
            kwargs += ' -ts %d %d' % (900, 451)
            kwargs += ' -r nearest'
            gdal.SetConfigOption('CENTER_LONG', '180')

            cams_world_mosaic = os.path.join(self.tmpdir, 'cams_aux_par_{0}_World_mosaic_{1}m.tif'.format(
                selected_day.strftime('%Y%m%d'), '60'))

            try:
                gdal.Warp(cams_world_mosaic, cams_world, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.fatal('Error mosaicking CAMS param')
                return False

            # 2nd step is to reproject CAMS param on MRGS tile

            xy = self.xy
            if parameter == 'tcwv':
                kwargs = '-ot UInt16 '
            elif parameter == 'gtco3':
                kwargs = '-ot UInt16 '
            else:
                kwargs = '-ot Float32 '
            kwargs += '-s_srs EPSG:4326 '
            kwargs += '-t_srs ' + self.hcsCode
            kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
            kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
            kwargs += ' -r cubicspline'
            gdal.SetConfigOption('CENTER_LONG', '0')

            if parameter == 'aod550':
                cams_param = os.path.join(self.tmpdir,
                                          'cams_aod_{0}_{1}_{2}m.tif'.format(selected_day.strftime('%Y%m%d'), 'T50SLJ', 60))

            elif parameter == 'tcwv':
                cams_param = os.path.join(self.tmpdir,
                                          'cams_tcwv_{0}_{1}_{2}m.tif'.format(selected_day.strftime('%Y%m%d'), 'T50SLJ', 60))

            elif parameter == 'gtco3':
                cams_param = os.path.join(self.tmpdir,
                                          'cams_gtco3_{0}_{1}_{2}m.tif'.format(selected_day.strftime('%Y%m%d'), 'T50SLJ', 60))
            try:
                gdal.Warp(cams_param, cams_world_mosaic, options=kwargs)

            except Exception as e:
                self.logger.error(e)
                self.logger.fatal('Error reading cams' + parameter)
                return False

            os.remove(cams_world)
            os.remove(cams_world_mosaic)

        else:
            xy = self.xy

            if parameter == 'tcwv':
                kwargs = '-ot UInt16 '
            elif parameter == 'gtco3':
                kwargs = '-ot UInt16 '
            else:
                kwargs = '-ot Float32 '

            kwargs += '-s_srs EPSG:4326 '
            kwargs += '-t_srs ' + self.hcsCode
            kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
            kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
            kwargs += ' -r cubicspline'
            gdal.SetConfigOption('CENTER_LONG', '180')

            if parameter == 'aod550':
                cams_param = os.path.join(self.tmpdir,
                                          'cams_aod_{0}_{1}_{2}m.tif'.format(selected_day.strftime('%Y%m%d'), 'T50SLJ', 60))

            elif parameter == 'tcwv':
                cams_param = os.path.join(self.tmpdir,
                                          'cams_tcwv_{0}_{1}_{2}m.tif'.format(selected_day.strftime('%Y%m%d'), 'T50SLJ', 60))
            elif parameter == 'gtco3':
                cams_param = os.path.join(self.tmpdir,
                                          'cams_gtco3_{0}_{1}_{2}m.tif'.format(selected_day.strftime('%Y%m%d'), 'T50SLJ', 60))
            try:
                gdal.Warp(cams_param, cams_world, options=kwargs)
            except Exception as e:
                self.logger.error(e)
                self.logger.fatal('Error reading cams' + parameter)
                return False

            os.remove(cams_world)

        self.logger.info('CAMS from daily/monthly ECMWF '+ parameter + ' retrieved')

        return cams_param


    def gdalCAMS_daily(self, parameter_input):
        from L2A_Library import rectBivariateSpline
        from netCDF4 import Dataset
        from skimage import io
        from datetime import datetime, timedelta
        self.camsDir = os.path.join(self.config.aux_dir, 'ECMWF')

        if 'LANDSAT' in self.config.spacecraftName:
            acqdate = datetime.strptime(self.config.acquisitionDate, '%Y-%m-%d')
            try:
                h_m_n=datetime.strptime(self.config.acquisitionTime, '%H:%M:%S.%f0Z')
            except:
                self.config.acquisitionTime=self.config.acquisitionTime[:-2]
                h_m_n = datetime.strptime(self.config.acquisitionTime, '%H:%M:%S.%f')

            hour1900 = (acqdate - datetime(1900, 1, 1)).total_seconds() / 60. / 60.
            adding_sensing_time= (h_m_n - datetime(1900, 1, 1)).total_seconds() / 60. / 60.

            date = datetime(1900, 1, 1) + timedelta(hours=hour1900) + timedelta(hours=adding_sensing_time)

        else: #Sentinel_2

            try:
                acqdate = dt.datetime.strptime(self.config.acquisitionDate, '%Y-%m-%d')
            except:
                try:
                    acqdate = dt.datetime.strptime(self.config.acquisitionDate, '%Y-%m-%dT%H:%M:%SZ')
                except:
                    try:
                        acqdate = dt.datetime.strptime(self.config.acquisitionDate, '%Y-%m-%dT%H:%M:%S.%fZ')
                    except Exception as e:
                        self.logger.error(e)
                        return False

            h_sentinel = (acqdate - dt.datetime(1900, 1, 1)).total_seconds() / 60. / 60.
            date = datetime(1900, 1, 1) + timedelta(hours=h_sentinel)

        date_str = date.strftime('%Y%m%d')
        nc_name = f'CAMS_archive_aod550_tcwv_msl_gtco3_analysis_0H_6H_12H_18H_{date.year:4}-{date.month:02}-{date.day:02}.nc'
        nc_file = os.path.join(self.camsDir,'daily', date_str, nc_name)
        if os.path.exists(nc_file):
            output = self.retrieveCAMS_parameter(nc_file,parameter_input,date)
            self.config.timestamp(f'L2A_Tables: CAMS daily ncfile ({parameter_input}): FOUND')
        else:
            self.config.timestamp(f'L2A_Tables: CAMS daily ncfile ({parameter_input}): NOT FOUND')
            return False

        return output

    def gdalCAMS_monthly(self, parameter_input):
        from L2A_Library import rectBivariateSpline
        from netCDF4 import Dataset
        from skimage import io
        import glob
        from datetime import datetime, timedelta

        self.camsDir = os.path.join(self.config.aux_dir, 'ECMWF')

        if 'LANDSAT' in self.config.spacecraftName:
            acqdate = datetime.strptime(self.config.acquisitionDate, '%Y-%m-%d')
            try:
                h_m_n=datetime.strptime(self.config.acquisitionTime, '%H:%M:%S.%f0Z')
            except:
                self.config.acquisitionTime = self.config.acquisitionTime[:-2]
                h_m_n = datetime.strptime(self.config.acquisitionTime, '%H:%M:%S.%f')

            hour1900 = (acqdate - datetime(1900, 1, 1)).total_seconds() / 60. / 60.
            adding_sensing_time= (h_m_n - datetime(1900, 1, 1)).total_seconds() / 60. / 60.

            date = datetime(1900, 1, 1) + timedelta(hours=hour1900) + timedelta(hours=adding_sensing_time)
            date_str = date.strftime('%Y%m%d')

        else: #Sentinel_2

            try:
                acqdate = dt.datetime.strptime(self.config.acquisitionDate, '%Y-%m-%d')
            except:
                try:
                    acqdate = dt.datetime.strptime(self.config.acquisitionDate, '%Y-%m-%dT%H:%M:%SZ')
                except:
                    try:
                        acqdate = dt.datetime.strptime(self.config.acquisitionDate, '%Y-%m-%dT%H:%M:%S.%fZ')
                    except Exception as e:
                        self.logger.error(e)
                        return False

            h_sentinel = (acqdate - dt.datetime(1900, 1, 1)).total_seconds() / 60. / 60.
            date = datetime(1900, 1, 1) + timedelta(hours=h_sentinel)

        year_n = str(date.year)
        month_n = '{:02d}'.format(date.month)
        nc_file = glob.glob(os.path.join(self.camsDir,'monthly', year_n + month_n, '*.nc'), recursive = True)
        try:
            file_month = nc_file[0]
        except:
            self.config.timestamp('L2A_Tables: CAMS monthly ncfile: NOT FOUND')
            return False
        if os.path.exists(file_month):
            output = self.retrieveCAMS_parameter(file_month,parameter_input,date)
            self.config.timestamp('L2A_Tables: CAMS monthly ncfile: FOUND')
        else:
            self.config.timestamp('L2A_Tables: CAMS monthly ncfile: NOT FOUND')
            return False

        return output

    def gdalCAMS_aod550(self):
        # ecmwfDir = self.config.camsDirectory
        # self.camsDir = '/export/DATA/ECMWF'
        self.camsDir = os.path.join(self.config.aux_dir, 'ECMWF')

        try:
            acqdate = dt.datetime.strptime(self.config.acquisitionDate, '%Y-%m-%d')
        except:
            try:
                acqdate = dt.datetime.strptime(self.config.acquisitionDate, '%Y-%m-%dT%H:%M:%SZ')
            except:
                try:
                    acqdate = dt.datetime.strptime(self.config.acquisitionDate, '%Y-%m-%dT%H:%M:%S.%fZ')
                except Exception as e:
                    self.logger.error(e)
                    return False

        h = (acqdate - dt.datetime(1900, 1, 1)).total_seconds() / 60. / 60.

        # find h0
        for h0 in range(int(np.floor(h)), int(np.floor(h)) - 4, -1):
            (ncfile_h0, ncsource_h0) = self.getCAMS_findnc(h0)
            if ncfile_h0 is not None:
                break
        # find h1
        for h1 in range(int(np.ceil(h)), int(np.ceil(h)) + 4):
            (ncfile_h1, ncsource_h1) = self.getCAMS_findnc(h1)
            if h1 != h0 and ncfile_h1 is not None:
                break

        # no CAMS?
        if ncfile_h0 is None or ncfile_h1 is None:
            self.config.timestamp('L2A_Tables: CAMS ncfile: NOT FOUND.')
            return False

        # Mix of FTP and SERVER nc files?
        if ncsource_h0 != ncsource_h1:
            self.config.timestamp('L2A_Tables: CAMS ncfile: mixing from FTP and SERVER sources is not managed yet.')
            return False

        # get visibility for both dates
        print(dt.datetime(1900, 1, 1) + dt.timedelta(hours=h0))
        print(dt.datetime(1900, 1, 1) + dt.timedelta(hours=h1))
        vis_h0 = self.getCAMS_singledate(h0, ncfile_h0)
        vis_h1 = self.getCAMS_singledate(h1, ncfile_h1)
        if vis_h0 is None or vis_h1 is None:
            return False

        # interpolate to exact time and save to geotiff
        h = (acqdate - dt.datetime(1900, 1, 1)).total_seconds() / 60. / 60.
        vis = vis_h0 + (h - h0) * (vis_h1 - vis_h0) / (h1 - h0)

        # save to file and copy geo information
        cams_world = os.path.join(self.tmpdir, 'cams_vis_{0}_World_{1}m.tif'.format(acqdate.strftime('%H%M%S'), '60'))
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(cams_world, xsize=vis.shape[1], ysize=vis.shape[0], bands=1, eType=gdal.GDT_Float32)
        src_band = dataset.GetRasterBand(1)
        src_band.WriteArray(vis * 100)  # *100 because will be cast to int further on
        src_band.FlushCache()
        if ncsource_h0 == 'FTP':
            dataset.SetGeoTransform([-0.2000000033946138, 0.4000000067892276, 0.0, 90.2, 0.0, -0.4])
        elif ncsource_h0 == 'SERVER':
            dataset.SetGeoTransform([-180.200000003394621, 0.400000006789228, 0.0, 90.2, 0.0, -0.4])
        dataset = None

        xy = self.xy
        kwargs = '-ot UInt16 '
        kwargs += '-s_srs EPSG:4326 '
        kwargs += '-t_srs ' + self.hcsCode
        kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
        kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
        kwargs += ' -r cubicspline'
        if ncsource_h0 == 'FTP':
            gdal.SetConfigOption('CENTER_LONG','180')

        cams_vis = os.path.join(self.tmpdir,
                                'cams_vis_{0}_{1}_{2}m.tif'.format(acqdate.strftime('%Y%m%d'), 'T50SLJ', 60))
        try:
            gdal.Warp(cams_vis, cams_world, options=kwargs)
            if ncsource_h0 == 'FTP':
                gdal.SetConfigOption('CENTER_LONG','0')
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('Error reading cams aod550, no aerosol meteo forecast will be used.')
            return False

        os.remove(cams_world)
        self.logger.info('CAMS ECMWF aod550 visibility map prepared')
        return cams_vis

    def gdalDEM_srtm(self):
        import ssl
        import urllib.request, urllib.parse, urllib.error
        import zipfile
        isTemporary = False
        demDir = self.config.demDirectory
        if demDir == 'NONE':
            self.logger.info('DEM directory not specified, flat surface is used, and DEM will not be exported')
            self.dem_error_type = 'False'
            self.config.demOutput = False
            return False, None
        self.logger.info('Start DEM alignment for tile')
        sourceDir = os.path.join(self.config.home, demDir)
        tmpDir = self.tmpdir
        if (os.path.exists(sourceDir) == False):
            os.makedirs(sourceDir)

        lonMinId = int((-180 - self.lonMin) / -360.0 * 72.0 + 0.99)
        lonMaxId = int((-180 - self.lonMax) / -360.0 * 72.0 + 1.01)
        latMinId = int((60 - self.latMax) / 120.0 * 24.0 + 0.99)  # this is inverted by intention
        latMaxId = int((60 - self.latMin) / 120.0 * 24.0 + 1.01)  # this is inverted by intention
        # end fix SIIMPC-611

        if (lonMinId < 1) or (lonMaxId > 72) or (latMinId < 1) or (latMaxId > 24):
            self.logger.stream('no SRTM dataset available for this tile')
            return 'NOT_AVAILABLE', None

        # temporary fix for SIIMPC-944, JL:         # end temporary fix for SIIMPC-944
        if (self.lonMax - self.lonMin) < 180:
        # if (lonMinId <= lonMaxId):
            lons = list(range(lonMinId, lonMaxId + 1))
        else:
            lons = [1, 72]
            self.logger.info('This tile is crossing the international date line, a particular processing is performed')
        # end temporary fix for SIIMPC-944

        for i in lons:
            for j in range(latMinId, latMaxId + 1):
                tifFn = 'srtm_{:0>2d}_{:0>2d}.tif'.format(i, j)
                zipFn = 'srtm_{:0>2d}_{:0>2d}.zip'.format(i, j)

                try:  # does the tiff file already exist?
                    with open(os.path.join(sourceDir, tifFn)) as fp:
                        self.logger.info('Dem exists: %s', tifFn)
                        continue
                except IOError:
                    try:
                        # zipfile needs to be downloaded ...
                        self.logger.info('read zipfile: %s', zipFn)
                        prefix = self.config.demReference
                        self.logger.stream(
                            'Trying to retrieve DEM from URL %s this may take some time ...', prefix)
                        self.logger.info('Trying to retrieve DEM from URL: %s', prefix)
                        url = prefix + zipFn
                        ctx = ssl.create_default_context()
                        ctx.check_hostname = False
                        ctx.verify_mode = ssl.CERT_NONE
                        webFile = urllib.request.urlopen(url, context=ctx).read()
                        localFile = open(os.path.join(tmpDir, url.split('/')[-1]), 'wb')
                        localFile.write(webFile)
                        localFile.close()
                        self.logger.info('Zipfile downloaded: %s', zipFn)
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.stream('Zipfile download error for DEM: %s', zipFn)
                        self.dem_error_type = 'True'
                        return False , None
                    try:
                        zipf = zipfile.ZipFile(localFile.name, mode='r')
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.stream('DEM not available')
                        self.dem_error_type = 'True'
                        try:
                            os.remove(localFile.name)
                        except:
                            pass
                        return False, None
                    if (zipf.testzip() == None):
                        try:
                            zipf.extract(tifFn, sourceDir)
                            zipf.close()
                            self.logger.info('DEM unpacked and moved: %s', tifFn)
                            os.remove(localFile.name)
                            self.logger.info('Zipfile removed: %s', localFile.name)
                        except Exception as e:
                            self.logger.error(e)
                            self.logger.stream('Zipfile extraction error for DEM: %', localFile.name)
                            self.dem_error_type = 'True'
                            return False, None
                        # fix for SIIMPC-577, UMW:
                        self.SIIMPC_577(os.path.join(sourceDir, tifFn))
                        # end fix for SIIMPC-577
                        continue

        # step 1: performing mosaicking, if needed:
        if (self.lonMax - self.lonMin) < 180:
        # if self.lonMin <= self.lonMax:
            kwargs = ''
        else:
            gdal.SetConfigOption('CENTER_LONG', '180')
            kwargs = '-t_srs EPSG:4326 '
        kwargs += '-ot Int16'

        if (lonMinId == lonMaxId) & (latMinId == latMaxId):
            srtmf_src = os.path.join(sourceDir, 'srtm_{:0>2d}_{:0>2d}.tif'.format(i, j))
            # if 'LANDSAT' not in self.config.spacecraftName:  # update from Sen2Cor 2.10
            self.config.aux_data_filelist.append(os.path.basename(srtmf_src))
        else:
            # more than 1 DEM needs to be concatenated:
            infiles = []
            for i in lons:
                for j in range(latMinId, latMaxId + 1):
                    infile = os.path.join(sourceDir, 'srtm_{:0>2d}_{:0>2d}.tif'.format(i, j))
                    infiles.append(infile)
                    # if 'LANDSAT' not in self.config.spacecraftName:  # update from Sen2Cor 2.10
                    self.config.aux_data_filelist.append(os.path.basename(infile))

            if infiles == []:
                self.logger.stream('No DEM files found.')
                self.dem_error_type = 'False'
                return False , None

            srtmf_src = os.path.join(tmpDir, 'srtm_' + self.config.L2A_TILE_ID + '_src.tif')
            isTemporary = True
            try:
                gdal.Warp(srtmf_src, infiles, options=kwargs)
                gdal.SetConfigOption('CENTER_LONG', '0')
            except Exception as e:
                self.logger.fatal(e, exc_info=True)
                self.logger.fatal('error using gdalwarp')
                os.remove(srtmf_src)
                self.dem_error_type = 'True'
                return False, None

        # The following fix (fix for SIIMPC-550, UMW)
        # needs to be performed on original srtm tiff data
        # i.e. moved before reprojection and resizing (see Jira SIIMPC-550 discussion)
        # done here ...
        # fix for SIIMPC-550, UMW:
        src_ds = gdal.Open(srtmf_src, GA_Update)
        if src_ds is None:
            return False, None

        ds4demresolution = src_ds
        self._input_dem_resolution = round(abs(ds4demresolution.GetGeoTransform()[5]), 8)
        ds4demresolution = None

        src_band = src_ds.GetRasterBand(1)
        rows = src_ds.RasterYSize
        cols = src_ds.RasterXSize
        src_arr = src_band.ReadAsArray(0, 0, cols, rows)
        NODATA_DEM = -32768

        # Fix for SIIMPC-944 VD-JL - International Date Line handling for DEM mosaicking
        if (lonMinId > lonMaxId) and (cols == 12000):
            column_west = src_arr[:, 5999].astype(np.float32)
            column_east = src_arr[:, 6001].astype(np.float32)
            column_interp = src_arr[:, 6000]
            interp_valid = (column_west != NODATA_DEM) & (column_east != NODATA_DEM)
            column_interp[interp_valid] = ((column_west[interp_valid] + column_east[interp_valid]) / 2.).astype(np.int16)
            src_arr[:, 6000] = column_interp
        # end of fix for SIIMPC-944 VD-JL - International Date Line handling for DEM mosaicking

        src_arr[(src_arr == NODATA_DEM)] = 0
        src_band.WriteArray(src_arr, 0, 0)
        src_band.FlushCache()
        # end fix for SIIMPC-550

        # step 3: performing the resizing:
        xy = self.xy
        kwargs = '-t_srs ' + self.hcsCode
        kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
        kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
        kwargs += ' -r cubicspline'
        kwargs += ' -ot Float32'

        srtmf_dst = os.path.join(tmpDir, 'srtm_' + self.config.L2A_TILE_ID + '_dem.tif')
        try:
            gdal.Warp(srtmf_dst, srtmf_src, options=kwargs)
        except Exception as e:
            self.logger.fatal('Error using gdal.Warp, reason:')
            self.logger.fatal(e, exc_info=True)
            self.dem_error_type = 'True'
            return False , None

        test_nodata_dem=gdal.Open(srtmf_dst)
        nodata_dem_arr = test_nodata_dem.ReadAsArray()
        if nodata_dem_arr.max() == 0.0:
            test_nodata_dem = None
            self.logger.info('DEM retrieved data contains no values, switching to flat surface')
            self.logger.warning('DEM retrieved data contains no values, switching to flat surface')
            self.dem_error_type = 'True'
            # if 'LANDSAT' not in self.config.spacecraftName:  # update from Sen2Cor 2.10
            self.config.aux_data_filelist = []
            os.remove(srtmf_dst)
            return False , None

        kwargs = '-ot Int16 '
        srtmf_dst_int16 = os.path.join(tmpDir, 'srtm_' + self.config.L2A_TILE_ID + '_dem_int16.tif')
        try:
            gdal.Translate(srtmf_dst_int16, srtmf_dst, options=kwargs)
        except Exception as e:
            self.logger.fatal('Error using gdal.Translate, reason:')
            self.logger.fatal(e, exc_info=True)
            self.dem_error_type = 'True'
            os.remove(srtmf_dst)
            return False , None

        if isTemporary:
            os.remove(srtmf_src)

        self.logger.info('DEM received and prepared')
        return srtmf_dst , srtmf_dst_int16

    def gdalDEM_copernicus(self):
        import ssl
        import urllib.request, urllib.parse, urllib.error
        import tarfile
        demDir = self.config.demDirectory
        if demDir == 'NONE':
            self.logger.info('DEM directory not specified, flat surface is used, and DEM will not be exported')
            self.dem_error_type = 'False'
            self.config.demOutput = False
            return False , None

        self.logger.info('Start DEM alignment for tile')
        sourceDir = os.path.join(self.config.home, demDir)
        tmpDir = self.tmpdir
        dtedf_src = ''

        # Fix for SIIMPC-944 VD-JL - International Date Line handling for DEM mosaicking
        if (self.lonMax - self.lonMin) < 180:
        # if self.lonMin <= self.lonMax:
            if self.lonMax > 0:
                lonMax = int(self.lonMax)
            else:
                lonMax = int(self.lonMax) - 1

            if self.lonMin > 0:
                lonMin = int(self.lonMin)
            else:
                lonMin = int(self.lonMin) - 1

            lons = list(range(lonMin, lonMax + 1))

        else:
            lons = list(range(-180, int(self.lonMin))) + list(
                range(int(self.lonMax), 180))  # gives [179, -180] for int(self.lonMin)=179 and int(self.lonMax)=-179
            self.logger.info('This tile is crossing the international date line, a particular processing is performed')

        if self.latMax > 0:
            latMax = int(self.latMax)
        else:
            latMax = int(self.latMax) - 1

        if self.latMin > 0:
            latMin = int(self.latMin)
        else:
            latMin = int(self.latMin) - 1

        lats = list(range(latMin, latMax + 1))

        infiles = []

        if "GLO-30" in self.config.demReference:
            copernicus_list = 'CopernicusDEM_30_TIFF_missing_tiles.list'
        else:
            copernicus_list = 'CopernicusDEM_90_TIFF_missing_tiles.list'
        self.config.timestamp("L2A_DEM: reading list")
        try:  # does the list of missing tiles exist?
            with open(os.path.join(self.config.aux_dir, copernicus_list)) as missing_tiles:
                self.missing_tiles_list = [line[0:line.find('.tif') + 4] for line in missing_tiles]
                self.config.timestamp("L2A_DEM: list read")
                missing_tiles.close()
        except IOError:
            self.logger.info('Copernicus list of missing tiles is not present')
            self.dem_error_type = 'True'
            return False , None
        self.config.timestamp("L2A_DEM: loop in")
        for lon in lons:
            for lat in lats:
                if lon < 0:
                    lonMask = 'W'
                else:
                    lonMask = 'E'
                if lat < 0:
                    latMask = 'S'
                else:
                    latMask = 'N'

                if "GLO-30" in self.config.demReference:
                    tifFn = 'Copernicus_DSM_10_%s%02d_00_%s%03d_00_DEM.tif' % (latMask, abs(lat), lonMask, abs(lon))
                    tarFn = 'Copernicus_DSM_10_%s%02d_00_%s%03d_00.tar' % (latMask, abs(lat), lonMask, abs(lon))
                else:
                    tifFn = 'Copernicus_DSM_30_%s%02d_00_%s%03d_00_DEM.tif' % (latMask, abs(lat), lonMask, abs(lon))
                    tarFn = 'Copernicus_DSM_30_%s%02d_00_%s%03d_00.tar' % (latMask, abs(lat), lonMask, abs(lon))

                if tifFn in self.missing_tiles_list:
                    continue

                # ----------------------------------------------------------------------------------------------------------------------
                try:  # does the tiff file already exist?
                    with open(os.path.join(sourceDir, tifFn)) as fp:
                        self.logger.info('Dem exists: %s', tifFn)
                        infile = os.path.join(sourceDir, tifFn)
                        infiles.append(infile)
                        # if 'LANDSAT' not in self.config.spacecraftName: # update from Sen2Cor 2.10
                        self.config.aux_data_filelist.append(os.path.basename(infile))
                        continue
                except IOError:
                    try:
                        # tarfile needs to be downloaded ...
                        self.logger.info('read tarfile: %s', tarFn)
                        prefix = self.config.demReference
                        # prefix = "http://172.30.16.191/DEM/"
                        url = prefix + tarFn
                        self.logger.stream(
                            'Trying to retrieve DEM from URL %s this may take some time ...', url)
                        self.logger.info('Trying to retrieve DEM from URL: %s', url)
                        url = prefix + tarFn
                        ctx = ssl.create_default_context()
                        ctx.check_hostname = False
                        ctx.verify_mode = ssl.CERT_NONE
                        webFile = urllib.request.urlopen(url, context=ctx).read()
                        localFile = open(os.path.join(tmpDir, url.split('/')[-1]), 'wb')
                        localFile.write(webFile)
                        localFile.close()
                        self.logger.info('Tarfile downloaded: %s', tarFn)
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.stream('Tarfile download error for DEM: %s', tarFn)
                        self.dem_error_type = 'True'
                        return False , None
                    try:
                        tarf = tarfile.open(localFile.name, mode='r')
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.stream('DEM not available')
                        self.dem_error_type = 'True'
                        try:
                            os.remove(localFile.name)
                        except:
                            pass
                        return False , None
                    if tarfile.is_tarfile(localFile.name):
                        try:
                            def dem_file_from_tar(members):
                                for tarinfo in members:
                                    if 'DEM.tif' in tarinfo.name:
                                        tarinfo.name = os.path.basename(tarinfo.name)
                                        yield tarinfo

                            tarf.extractall(path=sourceDir, members=dem_file_from_tar(tarf))
                            tarf.close()
                            self.logger.info('DEM unpacked and moved: %s', tarFn)
                            os.remove(localFile.name)
                            self.logger.info('Tarfile removed: %s', localFile.name)
                        except Exception as e:
                            self.logger.error(e)
                            self.logger.stream('Tarfile extraction error for DEM: %', localFile.name)
                            self.dem_error_type = 'True'
                            return False , None
                        # fix for SIIMPC-577, UMW:
                        #self.SIIMPC_577(os.path.join(sourceDir, tifFn))
                        # end fix for SIIMPC-577
                        infile = os.path.join(sourceDir, tifFn)
                        infiles.append(infile)
                        # if 'LANDSAT' not in self.config.spacecraftName:  # update from Sen2Cor 2.10
                        self.config.aux_data_filelist.append(os.path.basename(infile))
                        continue
        self.config.timestamp("L2A_DEM: loop out")
        if (self.lonMax - self.lonMin) < 180:
        # if self.lonMin <= self.lonMax:
            # Fix for SIIMPC-944 VD-JL - International Date Line handling for DEM mosaicking
            kwargs = ' -r bilinear '  # fix for SIIMPC-1006.2 UMW
        else:
            gdal.SetConfigOption('CENTER_LONG', '180')
            kwargs = ' -r bilinear -t_srs EPSG:4326'
        kwargs += ' -ot Int16'
        # Fix for SIIMPC-1613:
        kwargs += ' -dstnodata -20000'

        tmpDir = self.tmpdir
        dtedf_dest = os.path.join(self.tmpdir, 'dted_' + self.config.L2A_TILE_ID + '_src.tif')

        if infiles == []:
            self.logger.stream('No DEM files found.')
            self.dem_error_type = 'False'
            return False , None

        try:
            ds = gdal.Warp(dtedf_dest, infiles, options=kwargs)
            # Fix for SIIMPC-1613:
            NODATA = -20000

            ds4demresolution = gdal.Open(infiles[0])
            self._input_dem_resolution = round(abs(ds4demresolution.GetGeoTransform()[5]), 8)
            ds4demresolution = None

            dem_band = ds.GetRasterBand(1)
            dem_arr = dem_band.ReadAsArray()
            dem_arr[dem_arr == NODATA] = 0
            dem_band.WriteArray(dem_arr)
            dem_band.FlushCache()
            ds = None
            gdal.SetConfigOption('CENTER_LONG', '0')
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('error using gdalwarp')
            self.dem_error_type = 'True'
            return False , None

        dtedf_src = dtedf_dest
        dtedf_dest = os.path.join(tmpDir, 'dted_' + self.config.L2A_TILE_ID + '_dem.tif')
        xy = self.xy
        kwargs = '-ot Float32'
        kwargs += ' -t_srs ' + self.hcsCode
        kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
        kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
        kwargs += ' -r cubicspline'
        # Fix for SIIMPC-1613:
        kwargs += ' -dstnodata -20000.0'

        try:
            ds = gdal.Warp(dtedf_dest, dtedf_src, options=kwargs)
            # Fix for SIIMPC-1613:
            NODATA = -20000.0
            dem_band = ds.GetRasterBand(1)
            dem_arr = dem_band.ReadAsArray()
            dem_arr[dem_arr == NODATA] = 0.0
            dem_band.WriteArray(dem_arr)
            dem_band.FlushCache()
            ds = None
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('error using gdalwarp')
            os.remove(dtedf_src)
            self.dem_error_type = 'True'
            return False , None

        if dem_arr.max() == 0.0:
            self.logger.info('DEM retrieved data contains no values, switching to flat surface')
            self.logger.warning('DEM retrieved data contains no values, switching to flat surface')
            os.remove(dtedf_src)
            # if 'LANDSAT' not in self.config.spacecraftName:  # update from Sen2Cor 2.10
            self.config.aux_data_filelist = []
            self.dem_error_type = 'True'
            return False , None

        # fix for SIIMPC-792, JL:
        kwargs = '-ot Int16'
        dtedf_dst_int16 = os.path.join(tmpDir, 'dted_' + self.config.L2A_TILE_ID + '_dem_int16.tif')
        try:
            gdal.Translate(dtedf_dst_int16, dtedf_dest, options=kwargs)
#            self.importBandRes(self.DEM, dtedf_dst_int16)  # fix for SIIMPC-792, JL
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('Error reading DEM, flat surface will be used')
            os.remove(dtedf_dest)
            self.dem_error_type = 'True'
            return False , None
        finally:
            os.remove(dtedf_src)
            #os.remove(dtedf_dst_int16)
        # end of fix for SIIMPC-792
        self.logger.info('DEM received and prepared')
        return dtedf_dest, dtedf_dst_int16

    def gdalDEM_aws(self):
        import ssl
        import urllib.request, urllib.parse, urllib.error
        import tarfile


        demDir = self.config.demDirectory
        if demDir == 'NONE':
            self.logger.info('DEM directory not specified, flat surface is used')
            self.dem_error_type = 'False'
            self.config.demOutput = False
            return False , None

        self.logger.info('Start DEM alignment for tile')
        sourceDir = os.path.join(self.config.home, demDir)
        tmpDir = self.tmpdir
        dtedf_src = ''

        # Fix for SIIMPC-944 VD-JL - International Date Line handling for DEM mosaicking
        if (self.lonMax - self.lonMin) < 180:
        # if self.lonMin <= self.lonMax:
            if self.lonMax > 0:
                lonMax = int(self.lonMax)
            else:
                lonMax = int(self.lonMax) - 1

            if self.lonMin > 0:
                lonMin = int(self.lonMin)
            else:
                lonMin = int(self.lonMin) - 1

            lons = list(range(lonMin, lonMax + 1))

        else:
            lons = list(range(-180, int(self.lonMin))) + list(
                range(int(self.lonMax), 180))  # gives [179, -180] for int(self.lonMin)=179 and int(self.lonMax)=-179
            self.logger.info(
                'This tile is crossing the international date line, a particular processing is performed')

        if self.latMax > 0:
            latMax = int(self.latMax)
        else:
            latMax = int(self.latMax) - 1

        if self.latMin > 0:
            latMin = int(self.latMin)
        else:
            latMin = int(self.latMin) - 1

        lats = list(range(latMin, latMax + 1))

        infiles = []

        if "dem-30m" in self.config.demReference:
            copernicus_list = 'CopernicusDEM_30_TIFF_missing_tiles.list'
        else:
            copernicus_list = 'CopernicusDEM_90_TIFF_missing_tiles.list'

        self.config.timestamp("L2A_DEM: reading list")
        try:  # does the list of missing tiles exist?
            with open(os.path.join(self.config.aux_dir, copernicus_list)) as missing_tiles:
                self.missing_tiles_list = [line[0:line.find('.tif') + 4] for line in missing_tiles]
                self.config.timestamp("L2A_DEM: list read")
                missing_tiles.close()
        except IOError:
            self.logger.info('Copernicus list of missing tiles is not present')
            self.dem_error_type = 'True'
            return False, None
        # self.config.timestamp("L2A_DEM: loop in")
        for lon in lons:
            for lat in lats:
                if lon < 0:
                    lonMask = 'W'
                else:
                    lonMask = 'E'
                if lat < 0:
                    latMask = 'S'
                else:
                    latMask = 'N'

                if "dem-30m" in self.config.demReference:
                    tifFn = 'Copernicus_DSM_10_%s%02d_00_%s%03d_00_DEM.tif' % (latMask, abs(lat), lonMask, abs(lon))
                    awsFn = 'Copernicus_DSM_COG_10_%s%02d_00_%s%03d_00_DEM.tif' % (latMask, abs(lat), lonMask, abs(lon))
                else:
                    tifFn = 'Copernicus_DSM_30_%s%02d_00_%s%03d_00_DEM.tif' % (latMask, abs(lat), lonMask, abs(lon))
                    awsFn = 'Copernicus_DSM_COG_30_%s%02d_00_%s%03d_00_DEM.tif' % ( latMask, abs(lat), lonMask, abs(lon))
                if tifFn in self.missing_tiles_list:
                    continue

                # ----------------------------------------------------------------------------------------------------------------------
                try:  # does the tiff file already exist?
                    with open(os.path.join(sourceDir, awsFn)) as fp:
                        self.logger.info('Dem exists: %s', awsFn)
                        infile = os.path.join(sourceDir, awsFn)
                        infiles.append(infile)
                        self.config.aux_data_filelist.append(os.path.basename(infile))
                        continue
                except IOError:
                    try:
                        # file needs to be downloaded ...
                        self.logger.info('read file: %s', awsFn)
                        prefix = self.config.demReference

                        parent_KEY = awsFn.split(".")[0]
                        path_KEY = os.path.join(parent_KEY,awsFn)
                        # url = os.path.join(prefix,path_KEY)
                        # url = os.path.join(prefix, parent_KEY,awsFn)
                        # Generate the full url
                        url = f"{prefix.rstrip('/')}/{parent_KEY}/{awsFn}"
                        destination_key = os.path.join(sourceDir, awsFn)

                        self.logger.stream('Trying to retrieve DEM from URL %s this may take some time ...', url)
                        self.logger.info('Trying to retrieve DEM from URL: %s', url)

                        ctx = ssl.create_default_context()
                        ctx.check_hostname = False
                        ctx.verify_mode = ssl.CERT_NONE
                        webFile = urllib.request.urlopen(url, context=ctx).read()
                        localFile = open(destination_key, 'wb')
                        localFile.write(webFile)
                        localFile.close()

                        infile = os.path.join(sourceDir, awsFn)
                        infiles.append(infile)
                        self.config.aux_data_filelist.append(os.path.basename(infile))
                        self.logger.info('File downloaded: %s', awsFn)
                        self.config.timestamp('AWS DEM {} retrieved and downloaded'.format(awsFn))
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.stream('file download error for DEM: %s', awsFn)
                        self.config.timestamp('the requested {}} could not be downloaded'.format(awsFn))
                        self.dem_error_type = 'True'
                        return False, None

        if (self.lonMax - self.lonMin) < 180:
        # if self.lonMin <= self.lonMax:
            # Fix for SIIMPC-944 VD-JL - International Date Line handling for DEM mosaicking
            kwargs = ' -r bilinear '  # fix for SIIMPC-1006.2 UMW
        else:
            gdal.SetConfigOption('CENTER_LONG', '180')
            kwargs = ' -r bilinear -t_srs EPSG:4326'
        # kwargs += ' -ot Int16'
        kwargs += ' -ot Float32'
        # Fix for SIIMPC-1613:
        kwargs += ' -dstnodata -20000'

        tmpDir = self.tmpdir
        dtedf_dest = os.path.join(self.tmpdir, 'dted_' + self.config.L2A_TILE_ID + '_src.tif')

        if infiles == []:
            self.logger.stream('No DEM files found.')
            self.dem_error_type = 'False'
            return False, None

        try:
            ds = gdal.Warp(dtedf_dest, infiles, options=kwargs)
            # Fix for SIIMPC-1613:
            NODATA = -20000

            ds4demresolution = gdal.Open(infiles[0])
            self._input_dem_resolution = round(abs(ds4demresolution.GetGeoTransform()[5]), 8)
            ds4demresolution = None

            dem_band = ds.GetRasterBand(1)
            dem_arr = dem_band.ReadAsArray()
            dem_arr[dem_arr == NODATA] = 0
            dem_band.WriteArray(dem_arr)
            dem_band.FlushCache()
            ds = None
            gdal.SetConfigOption('CENTER_LONG', '0')
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('error using gdalwarp')
            self.dem_error_type = 'True'
            return False, None

        dtedf_src = dtedf_dest
        dtedf_dest = os.path.join(tmpDir, 'dted_' + self.config.L2A_TILE_ID + '_dem.tif')
        xy = self.xy
        kwargs = '-ot Float32'
        kwargs += ' -t_srs ' + self.hcsCode
        kwargs += ' -te %f %f %f %f' % (xy[0, 0], xy[2, 1], xy[2, 0], xy[0, 1])
        kwargs += ' -ts %d %d' % (self.config.ncols, self.config.nrows)
        kwargs += ' -r cubicspline'
        # Fix for SIIMPC-1613:
        kwargs += ' -dstnodata -20000.0'

        try:
            ds = gdal.Warp(dtedf_dest, dtedf_src, options=kwargs)
            # Fix for SIIMPC-1613:
            NODATA = -20000.0
            dem_band = ds.GetRasterBand(1)
            dem_arr = dem_band.ReadAsArray()
            dem_arr[dem_arr == NODATA] = 0.0
            dem_band.WriteArray(dem_arr)
            dem_band.FlushCache()
            ds = None
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('error using gdalwarp')
            os.remove(dtedf_src)
            self.dem_error_type = 'True'
            return False, None

        if dem_arr.max() == 0.0:
            self.logger.info('DEM retrieved data contains no values, switching to flat surface')
            self.logger.warning('DEM retrieved data contains no values, switching to flat surface')
            os.remove(dtedf_src)
            self.config.aux_data_filelist = []
            self.dem_error_type = 'True'
            return False, None

        # fix for SIIMPC-792, JL:
        kwargs = '-ot Int16'
        dtedf_dst_int16 = os.path.join(tmpDir, 'dted_' + self.config.L2A_TILE_ID + '_dem_int16.tif')
        try:
            gdal.Translate(dtedf_dst_int16, dtedf_dest, options=kwargs)
            # self.importBandRes(self.DEM, dtedf_dst_int16)  # fix for SIIMPC-792, JL
        except Exception as e:
            self.logger.error(e)
            self.logger.fatal('Error reading DEM, flat surface will be used')
            os.remove(dtedf_dest)
            self.dem_error_type = 'True'
            return False, None
        finally:
            os.remove(dtedf_src)
            # os.remove(dtedf_dst_int16)
        # end of fix for SIIMPC-792
        self.logger.info('DEM received and prepared')
        return dtedf_dest, dtedf_dst_int16

    def SIIMPC_577(self, filename):
        # fix for SIIMPC-577, UMW:
        dataset = gdal.Open(filename, gdal.GA_Update)
        if dataset is None:
            return False

        # display current
        self.logger.info('Driver: %s / %s' % (dataset.GetDriver().ShortName, dataset.GetDriver().LongName))
        self.logger.info('Size is: %d x %d x %d' % (dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount))
        self.logger.info('Projection is: %s' % dataset.GetProjection())
        geotransform = dataset.GetGeoTransform()
        self.logger.info('Origin = (%f, %f)' % (geotransform[0], geotransform[3]))
        self.logger.info('Pixel Size = (%f, %f)' % (geotransform[1], geotransform[5]))
        dataset.SetGeoTransform(
            [geotransform[0] - geotransform[1] / 2, geotransform[1], geotransform[2],
             geotransform[3] + geotransform[5] / 2, geotransform[4], geotransform[5]])

        geotransform = dataset.GetGeoTransform()
        self.logger.info('Origin = (%f, %f)' % (geotransform[0], geotransform[3]))
        self.logger.info('Pixel Size = (%f, %f)' % (geotransform[1], geotransform[5]))
        dataset = None

        return True

    def gdalDEM_Shade(self, demfile):
        sdwfile = demfile.replace('_dem', '_sdw')

        altitude = 90.0 - np.float32(np.mean(self.config.solze_arr))
        azimuth = np.float32(np.mean(self.config.solaz_arr))
        kwargs = '-compute_edges -az ' + str(azimuth) + ' -alt ' + str(altitude)
        try:
            gdal.DEMProcessing(sdwfile, demfile, 'hillshade', options=kwargs)
        except Exception as e:
            self.logger.fatal(e, exc_info=True)
            self.logger.fatal('error using gdal dem processing option hillshade')
            return False

        return sdwfile

    def gdalDEM_Slope(self, demfile):
        slpfile = demfile.replace('_dem', '_slp')
        kwargs = '-compute_edges'
        try:
            gdal.DEMProcessing(slpfile, demfile, 'slope', options=kwargs)
        except Exception as e:
            self.logger.fatal(e, exc_info=True)
            self.logger.fatal('error using gdal dem processing option slope')
            return False

        return slpfile

    def gdalDEM_Aspect(self, demfile):
        aspfile = demfile.replace('_dem', '_asp')
        kwargs = '-compute_edges'
        try:
            gdal.DEMProcessing(aspfile, demfile, 'aspect', options=kwargs)
        except Exception as e:
            self.logger.fatal(e, exc_info=True)
            self.logger.fatal('error using gdal dem processing option aspect')
            return False

        return aspfile

