#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Important note (!):
    This is the source code of the derived DLR ATCOR(c) module of Sen2Cor.
    This derived ATCOR(c) code can be published by the European Space Agency (ESA) under the Sen2Cor
    license (Apache v.2.0) for the use of satellite data from the Sentinel-2 and Landsat series.

    Telespazio Germany GmbH, 2024.

'''

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

import os, time, datetime, sys, getopt, string, logging
from numpy import *
from scipy.signal import medfilt
from scipy.ndimage import map_coordinates
from scipy import interpolate as sp
from scipy.ndimage.filters import median_filter, uniform_filter, gaussian_filter
from scipy import ndimage

from L2A_Library import *
from L2A_XmlParser import L2A_XmlParser
from L2A_Config import L2A_Config
#from L2A_Tables import L2A_Tables
from scipy.ndimage.filters import median_filter
import pickle as pickle
from L2A_BandIdentifier import Band
from numpy import isin as in1d, ravel

set_printoptions(linewidth = 160, suppress = True, precision=7)

from multiprocessing import Lock
l = Lock()

# this is the library part of the derived ATCOR module:
def reverse(a): return a[::-1]


def interplin(vin, xin, uin):
    """
     NAME:
        interplin()

     PURPOSE:
        Perform 1-d linear interpolation.  Values outside the bounds are
        permitted unlike the scipy.interpolate.interp1d module. The are
        extrapolated from the line between the 0,1 or n-2,n-1 entries.
        This program is not as powerful as interp1d but it does provide
        this which makes it compatible with the IDL interpol() function.

     CALLING SEQUENCE:
        yint = interplin(y, x, u)

     INPUTS:
        y, x:  The y and x values of the data.
        u: The x-values to which will be interpolated.

     REVISION HISTORY:
        Created: 2006-10-24, Erin Sheldon, NYU, from:
        http://sdss.physics.nyu.edu/esheldon/python/code/astro_code-2009-05-15/es_util.py
    """
    # Make sure inputs are arrays.  Copy only made if they are not.
    v=array(vin, ndmin=1, copy=False)
    x=array(xin, ndmin=1, copy=False)
    u=array(uin, ndmin=1, copy=False)
    # Find closest indices
    xm = x.searchsorted(u) - 1

    # searchsorted returns size(array) when the input is larger than xmax
    # Also, we need the index to be less than the last since we interpolate
    # *between* points.
    w = where32(xm >= (x.size-1))
    if w.size > 0:
        xm[w] = x.size-2

    w = where32(xm < 0)
    if w.size > 0:
        xm[w] = 0

    xmp1 = xm+1
    return (u-x[xm])*(v[xmp1] - v[xm])/(x[xmp1] - x[xm]) + v[xm]


def interpol(vin, xin, uin):
    return interplin(vin, xin, uin)


def linear_interpolation(x, grid):
    ind = arange(x.size, dtype=int)
    lin1 = sp.interpolate.interp1d(ind, x)
    res = lin1(grid)
    return res


def interpol1d(x, num):
    ind = arange(x.size, dtype=int)
    lin1 = sp.interpolate.interp1d(ind, x)
    grid = linspace(0, x.size - 1, num)
    res = lin1(grid)
    return float32(res)


def interpol2d(lp, x, y):
    f = sp.interpolate.interp2d(arange(lp.shape[1], dtype=float32), arange(lp.shape[0], dtype=float32), lp)
    lph = f(x, y)
    return float32(lph)


def smooth(data, filt, edge_truncate=True):
    if edge_truncate:
        res = uniform_filter(data, size=filt, mode='nearest')
    else:
        res = uniform_filter(data, size=filt)
    return float32(res)


def smooth16(data, filt, edge_truncate=True):
    if edge_truncate:
        res = uniform_filter(data, size=filt, mode='nearest')
    else:
        res = uniform_filter(data, size=filt)
    return float16(res)


def where32(condition):
    return where(condition)[0].astype(int32)


def median_filter_2d(arr, width):
    if width == 1:
        medarray = arr
    else:
        medarray = medfilt(arr, kernel_size = width)
        #
        # medfilt doesn't handle edges the same way as IDL, so we
        # have to fix them up
        #
        istart = (width-1)//2
        iend = arr.shape[0] - (width+1)//2
        medarray[0:istart,:] = arr[0:istart,:]
        medarray[iend:,:] = arr[iend:,:]
        #
        iend = arr.shape[1] - (width+1)//2
        medarray[:, 0:istart] = arr[:, 0:istart]
        medarray[:, iend:] = arr[:, iend:]

    return float32(medarray)


def regress(x, y):
    A = array([x, ones(x.size)])
    w = linalg.lstsq(A.T, y, rcond=None)[0]
    return array([w[1], w[0]])


def fit_coeff(x, y):
    # Input:
    #     x = wv grid (usually 3 grid points, scaled wv values, e.g. [1000, 2000, 2900] )
    #     y = radiative transfer function (transmittance, Lpath, Edir etc)
    #
    #    Function type: y  = exp( a + b*sqrt(x) )
    #    Convert into linear regression:  log(y) = a + b*sqrt(x)
    #
    # Remarks:
    #     The inverse function to "fit_coeff" is provided by function "polx"
    #     an NaN usually happens in the 1.37-1.44 or 1.80-1.94 micron region of the LUT conversion
    #     The numpy warning is disabled, the nan is replaced by a nan_to_num

    seterr(invalid='ignore', divide='ignore')
    xr = sqrt(x)  # make 2-D array for function regress
    yrr = log(y)
    yr = nan_to_num(yrr)
    return regress(xr, yr)


def rebin(a, *args):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = a.shape
    lenShape = len(shape)
    factor = asarray(shape)//asarray(args)
    evList = ['a.reshape('] + \
                ['args[%d],factor[%d],'%(i, i) for i in range(lenShape)] + \
                [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
                ['/factor[%d]'%i for i in range(lenShape)]

    return eval(''.join(evList))


def check_if_finite(yfit, y0):
# IDL2PY_TBD; IDL2PY_TBD @onerror.inc

# Input:  yfit=fltarr(2) from regression
#            y0  = y for wv=0.4 cm (scalar, e.g. y=tdir, or y=e0t etc)
# Output: yfit is not altered if numeric data
#            yfit[0] = log(y0) if yfit contains NaN
#            yfit[1] =  1.0E-6                 "         "      "
#  Since  y(wv) = exp(yfit[0] + yfit[1]*sqrt(wv))  the NaN case provides
#            almost constant values of y independent of wv and y(wv=400) = y0
#            (wv is scaled water vapor [cm*1000])
# Called from "apdat_lut" routines

    lis = isfinite(yfit) # find equivalent for finite ...
    lix = where32(ravel(lis == 0))
    if (lix.size > 0):
        yfit[0] = log(y0)
        yfit[1] = 1.0e-6
    return yfit

def polx3(ju, uu, coeff):
# IDL2PY_TBD; IDL2PY_TBD @onerror.inc

# Input:
#     ju = start index for wv region based on scene average wv
#            coeff[ju:ju+1,*] is used (ju=0, 2, 4, or 6)
#            ju=0 : scaled wv region with grid points ( 400, 1000, 2000) is used
#            ju=2 :                                              (1000, 2000, 2900)
#            ju=4 :                                              (2000, 2900, 4000)
#            ju=6 :                                              (2900, 4000, 5000)
#     uu = scaled water vapor (vector or 2-D matrix)
#     coeff = fltarr(n,*) or fltarr(n,*,*)
# Output :
#     fct = exp( a + b*sqrt(uu))    (where a=coeff[0], b=coeff[1] )
# called from atcor3

    if(uu.ndim == 1):
        #fct = zeros([uu[0]], float32)
        fct = exp(coeff[:, ju] + coeff[:, ju + 1] * sqrt(uu)).astype(float32)
    else:
        #fct = zeros([uu[0], uu[1]], float32)
        fct = exp(coeff[:,:, ju] + coeff[:,:, ju + 1] * sqrt(uu)).astype(float32)

    return fct


def indexvis(visi1, nvisx, vis_reg):

# return the visibility index for given visibility visi1
# input : visi1
#            nvisx ; extended visib set (=55)
#            vis_reg : the nvisx vis. regions
# output: vis.index

    indvis = -1
    if (visi1 > vis_reg[0, 1]):
        indvis = 0
    if (visi1 <= vis_reg[nvisx - 2, 1]):
        indvis = nvisx - 1
    if (indvis >= 0):
        return indvis

    # now find visi1 in the vis_reg boundaries
    for j in arange(1, nvisx - 1):
        if (visi1 <= vis_reg[j, 0] and visi1 >= vis_reg[j, 1]):
            indvis = j
            break

    return indvis


#-------------------------------------------------------------------------
def extrapl(y):
# input : vector y[n]
# output: vector y[n+1], where last element is obtained with linear extrapolation
    n = y.size
    y = concatenate(array([y, array([y[n - 1] + y[n - 1] - y[n - 2]])],dtype=object))

    return y


#-----------------------------------------------------------------------
def adjacency_weight(nadj_regions, adj_km, pixelsize):

# set zone-dpendent weight factors for adjacency correction

# input : nadj_regions : # adjacency regions (1:5) , intarr(5)
#            adj_km         : adjacency range [km]
#            pixelsize     : pixel size [m]
# output:
#            nadj_pix(*)  = adj. range for each region (pixels), intarr(5)
#            wadj(*)        = weight factor for each region, fltarr(5)

    nadj_pix = 50
    wadj = 1.0

    if (int(adj_km * 1000 / pixelsize + 0.5) > 255):
        nadj_regions = 1    # max adj. range is 255 pixels from center
        return(nadj_pix, wadj)      # i.e. max adj. box is 510 pixels

    _expr = (nadj_regions)
    if _expr == 1:
        adj_rangem = array([1.0]) * adj_km * 1000.0
        wadj = array([1.0])
    elif _expr == 2:
        adj_rangem = array([0.70, 1.0]) * adj_km * 1000.0
        wadj = array([0.5355, 0.4645])        # weight factor
    elif _expr == 3:
        adj_rangem = array([0.65, 0.90, 1.0]) * adj_km * 1000.0
        wadj = array([0.4699, 0.3757, 0.1545])
    elif _expr == 4:
        adj_rangem = array([0.52, 0.75, 0.90, 1.0]) * adj_km * 1000.0
        wadj = array([0.3150, 0.3101, 0.2249, 0.1500])
    elif _expr == 5:
        adj_rangem = array([0.45, 0.65, 0.80, 0.90, 1.0]) * adj_km * 1000.0
        wadj = array([0.2384, 0.2455, 0.2169, 0.1501, 0.1491])
    else:
        pass

    nadj_pix = int(adj_rangem.item() / pixelsize.item() + 0.5)
    return(nadj_pix, wadj)

#-----------------------------------------------------------------------
def load_wv_tables_summer():
# Output:
#     suu1 = string array of wv values
#     nuu1 = n_elements(suu1)
#     nuu1, uu1, suu1 will later be updated according to actual number of atmospheric wv files (.atm)

    uu1 = array([400, 1002, 2001, 2904, 4000, 5007])    #  scaled wv=0.4, 1.0, 2.0, 2.9, 4.0, 5.0 cm
    suu1 = array(['04', '10', '20', '29', '40', '50'])    #  (refers to sea level)
    nuu1 = uu1.size

    # 41 altitudes for wv interpolation: 0(0.1)4.0 km
    # These indices (0:40) correspond to 100 m DEM elevation classes
    # Currently the range 0-3.5 km is used in ATCOR.

    uu1_altit = zeros([41, nuu1], float32)

    # MOD43 values:
    # wv=400, 1000, 2000, 2900 is generated with model=2 and H2OSTR=0.137, 0.343, 0.685, 0.994
    # wv=4000 is calculated with model=2 and H2OSTR=1.36878
    # wv=5000 is calculated with model=2 and H2OSTR=1.71360

    # scaled wv column 400, all 46 altitude entries
    uu1_altit[0:10, 0] = array([400, 381, 363, 346, 329, 313, 298, 283, 269, 256])
    uu1_altit[10:20, 0] = array([243, 230, 219, 207, 196, 186, 176, 166, 157, 149])
    uu1_altit[20:30, 0] = array([141, 133, 125, 118, 111, 105, 99, 94, 89, 84])
    uu1_altit[30:40, 0] = array([79, 74, 70, 66, 63, 59, 44, 33, 25, 19])
    uu1_altit[40, 0] = 17

    # scaled wv column 1000, all 46 altitude entries
    uu1_altit[0:10, 1] = array([1002, 955, 909, 866, 824, 785, 746, 710, 675, 641])
    uu1_altit[10:20, 1] = array([609, 578, 548, 519, 492, 466, 441, 417, 395, 373])
    uu1_altit[20:30, 1] = array([353, 333, 314, 297, 280, 264, 249, 236, 222, 210])
    uu1_altit[30:40, 1] = array([198, 187, 177, 167, 158, 149, 111, 83, 63, 48])
    uu1_altit[40, 1] = 43

    # scaled wv column 2000, all 46 altitude entries
    uu1_altit[0:10, 2] = array([2001, 1907, 1817, 1730, 1647, 1567, 1491, 1418, 1348, 1281])
    uu1_altit[10:20, 2] = array([1216, 1154, 1095, 1038, 983, 930, 881, 834, 789, 746])
    uu1_altit[20:30, 2] = array([705, 666, 628, 593, 559, 528, 499, 471, 445, 420])
    uu1_altit[30:40, 2] = array([396, 374, 353, 334, 315, 298, 223, 167, 126, 96])
    uu1_altit[40, 2] = 87

    # scaled wv column 2900, all 46 altitude entries
    uu1_altit[0:10, 3] = array([2904, 2768, 2637, 2511, 2390, 2274, 2164, 2057, 1956, 1858])
    uu1_altit[10:20, 3] = array([1765, 1675, 1589, 1506, 1426, 1350, 1278, 1210, 1145, 1083])
    uu1_altit[20:30, 3] = array([1023, 966, 912, 861, 812, 767, 724, 683, 645, 609])
    uu1_altit[30:40, 3] = array([575, 543, 513, 484, 457, 432, 323, 242, 183, 140])
    uu1_altit[40, 3] = 125

    # scaled wv column 4000, all 46 altitude entries
    uu1_altit[0:10, 4] = array([4000, 3811, 3631, 3458, 3291, 3132, 2980, 2833, 2693, 2559])
    uu1_altit[10:20, 4] = array([2431, 2307, 2189, 2074, 1964, 1859, 1761, 1666, 1577, 1491])
    uu1_altit[20:30, 4] = array([1409, 1331, 1256, 1185, 1118, 1056, 997, 941, 889, 839])
    uu1_altit[30:40, 4] = array([792, 748, 706, 667, 630, 595, 445, 334, 253, 193])
    uu1_altit[40, 4] = 152

    # scaled wv column 5000, all 46 altitude entries
    uu1_altit[0:10, 5] = array([5007, 4772, 4546, 4329, 4121, 3921, 3730, 3547, 3372, 3204])
    uu1_altit[10:20, 5] = array([3043, 2889, 2740, 2597, 2459, 2328, 2204, 2086, 1974, 1867])
    uu1_altit[20:30, 5] = array([1765, 1666, 1573, 1484, 1400, 1322, 1248, 1179, 1113, 1051])
    uu1_altit[30:40, 5] = array([992, 937, 884, 835, 789, 745, 558, 418, 317, 242])
    uu1_altit[40, 5] = 172

    return uu1, suu1, nuu1, uu1_altit

#-----------------------------------------------------------------------------------
def load_wv_tables_winter():
#
# Output:
#      uu1 = wv values (cm*1000)
#     suu1 = string array of wv values
#     nuu1 = n_elements(suu1)
#     nuu1, uu1, suu1 will later be updated according to actual number of atmospheric wv files (.atm or .bp7)

    uu1 = array([200, 400, 800, 1100])    #  scaled wv=0.2, 0.4, 0.8, 1.1 cm
    suu1 = array(['02', '04', '08', '11'])    #  (refers to sea level)
    nuu1 = uu1.size

    # 36 altitudes for wv interpolation: 0(0.1)3.5 km
    # These indices (0:35) correspond to 100 m DEM elevation classes

    uu1_altit = zeros([36, nuu1], float32)

    # MOD43 values:
    # wv= 200, 400, 800, 1100 is generated with model=3 and H2OSTR=0.23482, 0.46964,  0.93928, 1.29155
    # all are calulated with the same model to have the same trace gas columns for all wv levels

    # scaled wv column  200, all 36 altitude entries
    uu1_altit[0:10, 0] = array([199, 191, 184, 176, 169, 162, 155, 148, 142, 136])
    uu1_altit[10:20, 0] = array([130, 124, 119, 113, 108, 103, 98, 93, 89, 84])
    uu1_altit[20:30, 0] = array([80, 76, 72, 68, 65, 61, 58, 54, 51, 48])
    uu1_altit[30:36, 0] = array([45, 43, 40, 37, 35, 33])

    # scaled wv column  400, all 36 altitude entries
    uu1_altit[0:10, 1] = array([399, 383, 368, 353, 338, 324, 311, 297, 285, 272])
    uu1_altit[10:20, 1] = array([260, 249, 238, 227, 217, 207, 197, 187, 178, 169])
    uu1_altit[20:30, 1] = array([160, 152, 144, 137, 130, 122, 116, 109, 103, 97])
    uu1_altit[30:36, 1] = array([91, 86, 80, 75, 71, 66])

    # scaled wv column  800, all 36 altitude entries
    uu1_altit[0:10, 2] = array([799, 767, 736, 706, 677, 649, 622, 595, 570, 545])
    uu1_altit[10:20, 2] = array([521, 498, 476, 455, 434, 414, 394, 375, 356, 339])
    uu1_altit[20:30, 2] = array([321, 305, 289, 274, 260, 245, 232, 218, 206, 194])
    uu1_altit[30:36, 2] = array([182, 172, 161, 151, 142, 133])

    # scaled wv column 1100, all 36 altitude entries
    uu1_altit[0:10, 3] = array([1100, 1055, 1012, 971, 931, 892, 855, 819, 784, 750])
    uu1_altit[10:20, 3] = array([717, 685, 655, 625, 597, 569, 542, 515, 490, 466])
    uu1_altit[20:30, 3] = array([442, 420, 398, 377, 357, 337, 319, 301, 283, 267])
    uu1_altit[30:36, 3] = array([251, 236, 222, 208, 196, 184])

    return uu1, suu1, nuu1, uu1_altit

#------------------------------------------------------------------------------------------
def read_wv_trans945_1375(gamma, solze, h1_cirrus):

# Purpose: calculate the two-way (sun-cirrus-sensor) wv transmittance
#             for the 945 nm band of Sentinel2
#
# Input:
#             gamma = transmittance of the 1.38 micron band  (sun-cirrus-sensor)
#             solze = solar zenith angle (deg)
# Output:  trans945  = two-way (sun-cirrus-sensor) wv transmittance 945 nm
#             h1_cirrus = approximate height of cirrus (km)
#                             based on the US standard temperature/humidity profile

#  trans for 1375 nm and R=t(945)/t1375  for US standard 1976 atmosphere
#  R is independent of selected atmosphere (for h> 6 km)

# updated for final S2 filter functions (Sept. 2011)

    arr = array([0.28593, 3.00129, 0.28436, 3.01551, 0.27951, 3.06023, 0.27091, 3.14325, \
                             0.25765, 3.28119, 0.23796, 3.51201, 0.20821, 3.93628, 0.15891, 4.95467, \
                             0.44296, 2.05593, 0.44135, 2.06239, 0.43633, 2.08275, 0.42738, 2.12018, \
                             0.41336, 2.18185, 0.39207, 2.28328, 0.35866, 2.46481, 0.29931, 2.87851, \
                             0.61264, 1.55084, 0.61129, 1.55381, 0.60704, 1.56317, 0.59942, 1.58030, \
                             0.58731, 1.60836, 0.56860, 1.65383, 0.53835, 1.73362, 0.48151, 1.90849, \
                             0.75786, 1.28681, 0.75690, 1.28824, 0.75389, 1.29269, 0.74846, 1.30085, \
                             0.73982, 1.31400, 0.72627, 1.33520, 0.70395, 1.37168, 0.66037, 1.44940, \
                             0.84854, 1.16355, 0.84790, 1.16433, 0.84587, 1.16678, 0.84221, 1.17125, \
                             0.83635, 1.17843, 0.82717, 1.18983, 0.81193, 1.20921, 0.78172, 1.24949, \
                             0.93536, 1.06474, 0.93505, 1.06505, 0.93410, 1.06603, 0.93240, 1.06778, \
                             0.92966, 1.07060, 0.92537, 1.07502, 0.91829, 1.08231, 0.90424, 1.09694, \
                             0.96321, 1.03605, 0.96304, 1.03622, 0.96248, 1.03675, 0.96148, 1.03773, \
                             0.95990, 1.03927, 0.95743, 1.04167, 0.95337, 1.04562, 0.94541, 1.05326])

    arr = reshape(arr, (7, 8, 2)) # two columns: t(1375), R=t(945)/t(1375)
    hh = array([6., 7., 8., 9., 10., 12., 14.])
    nh = hh.size

    arr1375 = (arr[:,:, 0])
    ratio = arr[:,:, 1]
    x = arange(nh, dtype = float32)
    y = ones(nh, dtype = float32) * 0.1 * solze
    xy = array([x, y])

    tr1375 = ravel(map_coordinates(arr1375, xy, order=1, mode='nearest'))
    ratio1 = ravel(map_coordinates(ratio, xy, order=1, mode='nearest'))

    h1_cirrus = interpol(hh, tr1375, gamma)
    r1 = interpol(ratio1, hh, h1_cirrus)

    tr945 = (minimum(r1 * gamma, 1.0))
    return tr945, h1_cirrus

#-----------------------------------------------------------------------
def image_cells(nrow, ncol, cell_length_x):

# Input:
#     nrow                = rows of image     (y direction)
#     ncol                = columns of image (x direction)
#     cell_length_x = cell length in x direction (pixels)
#     Comment:     cell_length_y = 2*cell_length_x  as Delta(SZA) much smaller than Delta(VZA)
#                     i.e. the cell length in y is twice the size of the length in x direction
#                     Advantage: increase in speed and more DDV reference pixels can be found per cell
#
# Output:
#     nx_cell =  number of cells in x direction
#     ny_cell =     "          "    y    "
#     xcell    = lonarr(2,nx_cell) left/right pixel coordinates of all x cells
#     ycell    = lonarr(2,ny_cell) left/right pixel coordinates of all y cells

    cell_length_y = 2 * cell_length_x

    nx_cell = int(maximum((ncol // cell_length_x)+0.5, 1))
    ny_cell = int(maximum((nrow // (cell_length_y))+0.5, 1))
    nx = 0
    ny = 0

    if (cell_length_x < ncol):
        nx = ncol % cell_length_x
    if (cell_length_y < nrow):
        ny = nrow % cell_length_y

    if (nx > 0):
        nx_cell += 1
    if (ny > 0):
        ny_cell += 1

    xcell = zeros((nx_cell, 2), int)  # stores left/right pixel indices for each xcell (starting with 0)
    ycell = zeros((ny_cell, 2), int)  #      "                  "          "                ycell

    # calculate x cell pixel coordinates
    p1 = -cell_length_x
    p2 = -1

    for i in arange(0, (nx_cell)):
        p1 += cell_length_x
        p2 = minimum((p2 + cell_length_x), (ncol - 1))
        xcell[i, 0:2] = [p1, p2]

    # calculate y cell pixel coordinates
    p1 = -cell_length_y
    p2 = -1

    for i in arange(0, (ny_cell)):
        p1 += cell_length_y
        p2 = minimum((p2 + cell_length_y), (nrow - 1))
        ycell[i, 0:2] = [p1, p2]

    return ny_cell, nx_cell, ycell, xcell


def set_nadir_geometry(sza_arr, saa_arr, vza_arr, vaa_arr, itilt):
# Set nadir geometry or in case of tilt the nearest possible nadir region
# Input:
#     sza_arr  = fltarr(2,2) solar zenith angles (UL, UR, LL, LR)
#     saa_arr  =         "              azimuth  "
#     vza_arr  =         "        view zenith    "
#     vaa_arr  =         "        view azimuth  "
#     itilt     =0 no tilt, =1=tilt capability
# Output:
#      4-element vector [vza, vaa, sza, saa] with (close) nadir geometry

    vza = 0.0
    vaa = 0.0  # irrelevant for vza=0
    saa = 0.0  # irrelevant for vza=0
    try:
        sza = float32(mean(sza_arr))
    except:
        sza = sza_arr

    return array([vza, vaa, sza, saa], dtype=float32)


class L2A_AtmCorr(object):

    def chkBool(self, key, value):
        if((value >= 0 & value <= 1 & (value == int(value))) == False):
            self.logger.fatal('configuration: parameter '+ key + 'out of bounds [0|1]')
            

    def __init__(self, config, tables=None):
# classes for IO handling:
        self._config = config
        self._logger = config.logger
        self._tables = tables
# default initialisation as given by ATCOR:
        self._a_aot_fit = array([1.54641, 1.40593, 0.926640])
        self._altitude_grid = array([arange(36, dtype=float32) * 0.1, 4.0 + arange(10, dtype=float32) * 0.5],dtype=object) # (km)
        self._atc_version = '(Version 7.2.0, tailored to Sentinel-2) '
        self._b_aot_fit = array([-0.854022, -0.844382, -0.805663])
        self._blue = 1
        self._ch1130 = zeros(6)
        self._ch1130a1 = 0
        self._ch1130a2 = 0
        self._ch1130w1 = 0
        self._ch1130w2 = 0
        self._ch940a1 = 0
        self._ch940a2 = 0
        self._ch940w1 = 0
        self._ch940w2 = 0
        self._cntback = 0
        self._green = 2
        self._h_grid_aot = array([0, 36, 126]) # 20 m grid: elevation class index for h2=0.0, 0.7, 2.5 km
        self._h1_cirrus = 20.0 # default cirrus height
        self._iabs_region = zeros(5, dtype=uint16) #zeros(4)
        self._ibrdf = 0
        self._ibrdf_ini = 0
        self._iscale_path = 0
        self._itarget = 0
        self._cloud_refl_thr_blu = 25.0 # blue-green region (%) to define cloud mask
        self._water_refl_thr_nir = 5.0 # NIR (%) to define water mask
        self._water_refl_thr_swir1 = 3.0 # 1600 nm (%) to define water mask
        self._beta_thr = 0.0
        self._ratio_blu_red = self.config.AC_Red_Blue_Refl_Ratio
        self._ratio_red_swir = self.config.AC_Swir_22um_Red_Refl_Ratio
        self._refl_cutoff = self.config.AC_Topo_Corr_Cutoff #(%) cut-off limit -->
        self._iter_terrain = self.config.AC_Max_Nr_Topo_Iter
        self._itilt = self.config.AC_Limit_Area_Path_Rad_Scale
        self._mlistref = 0
        self._atmDataFn = None
        # fix for SIIMPC-672.1, UMW:
        self._aerosolTypeBest = 'RURAL'
        # end fix for SIIMPC-672.1
        self._aerosolDetection = 'STOPPED'

        if(self.config.resolution == 10):
            self._blue_band = 1
            self._green_band = 2
            self._red_band = 3
            self._nir_band = 4
            self._cirrus_band = 0
            self._snow_band = 0
            self._swir2_band = 0
            self._n_bands = 4
            self._n_bands_all = 4
            self._n_refl = 4
        else:
            self._blue_band = 1
            self._green_band = 2
            self._red_band = 3
            self._nir_band = 8
            self._cirrus_band = 10
            self._snow_band = 11
            self._swir2_band = 12
            self._n_bands = 12
            self._n_bands_all = 12
            self._n_refl = 12

        self._band_saturated = None

        self._band_index_wvdepend = None
        self._n_bands_min_interp = 20 # min number of bands to enable band interpolation (760, 940 nm etc)

        self._first_band = 0
        self._band_index = 0
        self._first_row = 0
        self._first_col = 0
#
        self._liback = 0
        self._meanvi = 10
        self._n_alt = 6 # 6 altitudes in database
        self._n_alti = (self._n_alt - 1) * 25 + 1 # # interpolated altitudes (126), 0.02 km grid
        self._n_alti1 = self._n_alti
        self._nadj = 33
        self._nadj_regions = 1

        self._neg_channels = 0 # total number of channels with neg.reflectance > 1% of scene (up to 30)
        self._neg_pixels_channels = zeros([30], uint8) # channel numbers with neg. refl. (counting from 1)
        self._neg_pixels_percent = zeros([30], float32) # percent of pixels with refl <= 0
        self._high_cloud_coverage = False
        self._np_cloud = 0
        self._np_cloudw = 0
        self._np_clshad = 0
        self._np_haze = 0
        self._np_hazew = 0
        self._np_shadow = 0
        self._np_snow = 0
        self._np_water = 0
        self._npref_dbv = 0
        self._npref = 1
        self._ddv_pixel_percentage = 0.0
        self._ddv_reflectance_range = 0.0
        self._nvis = 8
        self._nzen = 8
        self._radeg = 180.0 / pi
        self._dtor = pi / 180.0
        self._red = 3
        self._reflref = 1.7
        # fix for SIIMPC-1038, UMW: avoid floats as index, depecated for numpy > 1.11
        self._rel_saturation = 1 # DN(saturated) = 1 * DN(max): 8, 16 bit data
        self._relazi = 0.0
        self._sc_bet = 255.0 # scale factor cbeta
        self._scene_av_aot = 0.0
        self._thr_rho_red = 0.040 # upper threshold for rho(red)o
        self._visibility = self.config.visibility
        self._altit = self.config.altit
        self._adj_km = self.config.adj_km
        self._trwv945 = 1.0 # default for H2O transmittance (sun-cirrus-sensor) in 945nm band
        self._visarr = array([5.0, 7.0, 10.0, 15.0, 23.0, 40.0, 80.0, 120.0]) # basic vis. array
        self._visext = array([187.81, 165.90, 148.29, 133.86, 121.84, 111.67, 102.97, 95.45, 88.89, 83.12, \
                              78.01, 73.45, 69.36, 65.67, 62.33, 59.30, 56.52, 53.98, 51.64, 49.49, 47.49, \
                              45.64, 43.92, 42.31, 40.81, 39.41, 38.09, 36.85, 35.69, 34.59, 33.55, 32.57, \
                              31.64, 30.76, 29.92, 29.13, 28.37, 27.65, 26.96, 26.30, 25.67, 25.07, 24.50, \
                              23.95, 23.42, 22.91, 22.42, 21.95, 21.50, 21.07, 20.65, 20.25, 19.86, 19.48, \
                              19.12, 18.77, 18.44, 18.11, 17.79, 17.49, 17.19, 16.90, 16.62, 16.35, 16.09, \
                              15.84, 15.59, 15.35, 15.12, 14.89, 14.67, 14.45, 14.24, 14.04, 13.84, 13.65, \
                              13.46, 13.28, 13.10, 12.93, 12.76, 12.59, 12.43, 12.27, 12.12, 11.97, 11.82, \
                              11.67, 11.53, 11.39, 11.26, 11.13, 11.00, 10.87, 10.75, 10.63, 10.51, 10.39, \
                              10.28, 10.17, 10.06, 9.95, 9.85, 9.74, 9.64, 9.54, 9.45, 9.35, 9.26, 9.17, 9.08, \
                              8.99, 8.90, 8.82, 8.73, 8.65, 8.57, 8.49, 8.41, 8.33, 8.26, 8.18, 8.11, 8.04, \
                              7.97, 7.90, 7.83, 7.76, 7.70, 7.63, 7.57, 7.50, 7.44, 7.38, 7.32, 7.26, 7.20, \
                              7.14, 7.09, 7.03, 6.98, 6.92, 6.87, 6.81, 6.76, 6.71, 6.66, 6.61, 6.56, 6.51, \
                              6.46, 6.42, 6.37, 6.32, 6.28, 6.23, 6.19, 6.15, 6.10, 6.06, 6.02, 5.98, 5.94, \
                              5.90, 5.86, 5.82, 5.78, 5.74, 5.70, 5.66, 5.63, 5.59, 5.55, 5.52, 5.48, 5.45, \
                              5.41, 5.38, 5.35, 5.31, 5.28, 5.25, 5.00])

        self._nvisx = size(self._visext)
        self._vis_reg = zeros([self._nvisx, 2], float32)
        self._vis_reg[ 0,:] = [ 200.90, 176.22 ]
        self._vis_reg[ 1,:] = [ 176.22, 156.64 ]
        self._vis_reg[ 2,:] = [ 156.64, 140.74 ]
        self._vis_reg[ 3,:] = [ 140.74, 127.59 ]
        self._vis_reg[ 4,:] = [ 127.59, 116.55 ]
        self._vis_reg[ 5,:] = [ 116.55, 107.16 ]
        self._vis_reg[ 6,:] = [ 107.16, 99.08 ]
        self._vis_reg[ 7,:] = [ 99.08, 92.06 ]
        self._vis_reg[ 8,:] = [ 92.06, 85.92 ]
        self._vis_reg[ 9,:] = [ 85.92, 80.49 ]
        self._vis_reg[10,:] = [ 80.49, 75.66 ]
        self._vis_reg[11,:] = [ 75.66, 71.35 ]
        self._vis_reg[12,:] = [ 71.35, 67.47 ]
        self._vis_reg[13,:] = [ 67.47, 63.96 ]
        self._vis_reg[14,:] = [ 63.96, 60.78 ]
        self._vis_reg[15,:] = [ 60.78, 57.88 ]
        self._vis_reg[16,:] = [ 57.88, 55.22 ]
        self._vis_reg[17,:] = [ 55.22, 52.79 ]
        self._vis_reg[18,:] = [ 52.79, 50.54 ]
        self._vis_reg[19,:] = [ 50.54, 48.47 ]
        self._vis_reg[20,:] = [ 48.47, 46.55 ]
        self._vis_reg[21,:] = [ 46.55, 44.76 ]
        self._vis_reg[22,:] = [ 44.76, 43.10 ]
        self._vis_reg[23,:] = [ 43.10, 41.55 ]
        self._vis_reg[24,:] = [ 41.55, 40.10 ]
        self._vis_reg[25,:] = [ 40.10, 38.74 ]
        self._vis_reg[26,:] = [ 38.74, 37.46 ]
        self._vis_reg[27,:] = [ 37.46, 36.26 ]
        self._vis_reg[28,:] = [ 36.26, 35.13 ]
        self._vis_reg[29,:] = [ 35.13, 34.06 ]
        self._vis_reg[30,:] = [ 34.06, 33.05 ]
        self._vis_reg[31,:] = [ 33.05, 32.10 ]
        self._vis_reg[32,:] = [ 32.10, 31.19 ]
        self._vis_reg[33,:] = [ 31.19, 30.33 ]
        self._vis_reg[34,:] = [ 30.33, 29.52 ]
        self._vis_reg[35,:] = [ 29.52, 28.74 ]
        self._vis_reg[36,:] = [ 28.74, 28.00 ]
        self._vis_reg[37,:] = [ 28.00, 27.30 ]
        self._vis_reg[38,:] = [ 27.30, 26.63 ]
        self._vis_reg[39,:] = [ 26.63, 25.98 ]
        self._vis_reg[40,:] = [ 25.98, 25.37 ]
        self._vis_reg[41,:] = [ 25.37, 24.78 ]
        self._vis_reg[42,:] = [ 24.78, 24.22 ]
        self._vis_reg[43,:] = [ 24.22, 23.68 ]
        self._vis_reg[44,:] = [ 23.68, 23.16 ]
        self._vis_reg[45,:] = [ 23.16, 22.66 ]
        self._vis_reg[46,:] = [ 22.66, 22.19 ]
        self._vis_reg[47,:] = [ 22.19, 21.73 ]
        self._vis_reg[48,:] = [ 21.73, 21.28 ]
        self._vis_reg[49,:] = [ 21.28, 20.86 ]
        self._vis_reg[50,:] = [ 20.86, 20.45 ]
        self._vis_reg[51,:] = [ 20.45, 20.05 ]
        self._vis_reg[52,:] = [ 20.05, 19.67 ]
        self._vis_reg[53,:] = [ 19.67, 19.30 ]
        self._vis_reg[54,:] = [ 19.30, 18.95 ]
        self._vis_reg[55,:] = [ 18.95, 18.60 ]
        self._vis_reg[56,:] = [ 18.60, 18.27 ]
        self._vis_reg[57,:] = [ 18.27, 17.95 ]
        self._vis_reg[58,:] = [ 17.95, 17.64 ]
        self._vis_reg[59,:] = [ 17.64, 17.34 ]
        self._vis_reg[60,:] = [ 17.34, 17.04 ]
        self._vis_reg[61,:] = [ 17.04, 16.76 ]
        self._vis_reg[62,:] = [ 16.76, 16.49 ]
        self._vis_reg[63,:] = [ 16.49, 16.22 ]
        self._vis_reg[64,:] = [ 16.22, 15.96 ]
        self._vis_reg[65,:] = [ 15.96, 15.71 ]
        self._vis_reg[66,:] = [ 15.71, 15.47 ]
        self._vis_reg[67,:] = [ 15.47, 15.23 ]
        self._vis_reg[68,:] = [ 15.23, 15.00 ]
        self._vis_reg[69,:] = [ 15.00, 14.78 ]
        self._vis_reg[70,:] = [ 14.78, 14.56 ]
        self._vis_reg[71,:] = [ 14.56, 14.35 ]
        self._vis_reg[72,:] = [ 14.35, 14.14 ]
        self._vis_reg[73,:] = [ 14.14, 13.94 ]
        self._vis_reg[74,:] = [ 13.94, 13.75 ]
        self._vis_reg[75,:] = [ 13.75, 13.56 ]
        self._vis_reg[76,:] = [ 13.56, 13.37 ]
        self._vis_reg[77,:] = [ 13.37, 13.19 ]
        self._vis_reg[78,:] = [ 13.19, 13.01 ]
        self._vis_reg[79,:] = [ 13.01, 12.84 ]
        self._vis_reg[80,:] = [ 12.84, 12.67 ]
        self._vis_reg[81,:] = [ 12.67, 12.51 ]
        self._vis_reg[82,:] = [ 12.51, 12.35 ]
        self._vis_reg[83,:] = [ 12.35, 12.19 ]
        self._vis_reg[84,:] = [ 12.19, 12.04 ]
        self._vis_reg[85,:] = [ 12.04, 11.89 ]
        self._vis_reg[86,:] = [ 11.89, 11.75 ]
        self._vis_reg[87,:] = [ 11.75, 11.60 ]
        self._vis_reg[88,:] = [ 11.60, 11.46 ]
        self._vis_reg[89,:] = [ 11.46, 11.33 ]
        self._vis_reg[90,:] = [ 11.33, 11.19 ]
        self._vis_reg[91,:] = [ 11.19, 11.06 ]
        self._vis_reg[92,:] = [ 11.06, 10.94 ]
        self._vis_reg[93,:] = [ 10.94, 10.81 ]
        self._vis_reg[94,:] = [ 10.81, 10.69 ]
        self._vis_reg[95,:] = [ 10.69, 10.57 ]
        self._vis_reg[96,:] = [ 10.57, 10.45 ]
        self._vis_reg[97,:] = [ 10.45, 10.34 ]
        self._vis_reg[98,:] = [ 10.34, 10.22 ]
        self._vis_reg[99,:] = [ 10.22, 10.11 ]
        self._vis_reg[100,:] = [ 10.11, 10.01 ]
        self._vis_reg[101,:] = [ 10.01, 9.90 ]
        self._vis_reg[102,:] = [ 9.90, 9.80 ]
        self._vis_reg[103,:] = [ 9.80, 9.69 ]
        self._vis_reg[104,:] = [ 9.69, 9.59 ]
        self._vis_reg[105,:] = [ 9.59, 9.50 ]
        self._vis_reg[106,:] = [ 9.50, 9.40 ]
        self._vis_reg[107,:] = [ 9.40, 9.31 ]
        self._vis_reg[108,:] = [ 9.31, 9.21 ]
        self._vis_reg[109,:] = [ 9.21, 9.12 ]
        self._vis_reg[110,:] = [ 9.12, 9.03 ]
        self._vis_reg[111,:] = [ 9.03, 8.94 ]
        self._vis_reg[112,:] = [ 8.94, 8.86 ]
        self._vis_reg[113,:] = [ 8.86, 8.77 ]
        self._vis_reg[114,:] = [ 8.77, 8.69 ]
        self._vis_reg[115,:] = [ 8.69, 8.61 ]
        self._vis_reg[116,:] = [ 8.61, 8.53 ]
        self._vis_reg[117,:] = [ 8.53, 8.45 ]
        self._vis_reg[118,:] = [ 8.45, 8.37 ]
        self._vis_reg[119,:] = [ 8.37, 8.30 ]
        self._vis_reg[120,:] = [ 8.30, 8.22 ]
        self._vis_reg[121,:] = [ 8.22, 8.15 ]
        self._vis_reg[122,:] = [ 8.15, 8.07 ]
        self._vis_reg[123,:] = [ 8.07, 8.00 ]
        self._vis_reg[124,:] = [ 8.00, 7.93 ]
        self._vis_reg[125,:] = [ 7.93, 7.86 ]
        self._vis_reg[126,:] = [ 7.86, 7.80 ]
        self._vis_reg[127,:] = [ 7.80, 7.73 ]
        self._vis_reg[128,:] = [ 7.73, 7.66 ]
        self._vis_reg[129,:] = [ 7.66, 7.60 ]
        self._vis_reg[130,:] = [ 7.60, 7.53 ]
        self._vis_reg[131,:] = [ 7.53, 7.47 ]
        self._vis_reg[132,:] = [ 7.47, 7.41 ]
        self._vis_reg[133,:] = [ 7.41, 7.35 ]
        self._vis_reg[134,:] = [ 7.35, 7.29 ]
        self._vis_reg[135,:] = [ 7.29, 7.23 ]
        self._vis_reg[136,:] = [ 7.23, 7.17 ]
        self._vis_reg[137,:] = [ 7.17, 7.11 ]
        self._vis_reg[138,:] = [ 7.11, 7.06 ]
        self._vis_reg[139,:] = [ 7.06, 7.00 ]
        self._vis_reg[140,:] = [ 7.00, 6.95 ]
        self._vis_reg[141,:] = [ 6.95, 6.89 ]
        self._vis_reg[142,:] = [ 6.89, 6.84 ]
        self._vis_reg[143,:] = [ 6.84, 6.79 ]
        self._vis_reg[144,:] = [ 6.79, 6.74 ]
        self._vis_reg[145,:] = [ 6.74, 6.69 ]
        self._vis_reg[146,:] = [ 6.69, 6.63 ]
        self._vis_reg[147,:] = [ 6.63, 6.59 ]
        self._vis_reg[148,:] = [ 6.59, 6.54 ]
        self._vis_reg[149,:] = [ 6.54, 6.49 ]
        self._vis_reg[150,:] = [ 6.49, 6.44 ]
        self._vis_reg[151,:] = [ 6.44, 6.39 ]
        self._vis_reg[152,:] = [ 6.39, 6.35 ]
        self._vis_reg[153,:] = [ 6.35, 6.30 ]
        self._vis_reg[154,:] = [ 6.30, 6.26 ]
        self._vis_reg[155,:] = [ 6.26, 6.21 ]
        self._vis_reg[156,:] = [ 6.21, 6.17 ]
        self._vis_reg[157,:] = [ 6.17, 6.12 ]
        self._vis_reg[158,:] = [ 6.12, 6.08 ]
        self._vis_reg[159,:] = [ 6.08, 6.04 ]
        self._vis_reg[160,:] = [ 6.04, 6.00 ]
        self._vis_reg[161,:] = [ 6.00, 5.96 ]
        self._vis_reg[162,:] = [ 5.96, 5.92 ]
        self._vis_reg[163,:] = [ 5.92, 5.88 ]
        self._vis_reg[164,:] = [ 5.88, 5.84 ]
        self._vis_reg[165,:] = [ 5.84, 5.80 ]
        self._vis_reg[166,:] = [ 5.80, 5.76 ]
        self._vis_reg[167,:] = [ 5.76, 5.72 ]
        self._vis_reg[168,:] = [ 5.72, 5.68 ]
        self._vis_reg[169,:] = [ 5.68, 5.64 ]
        self._vis_reg[170,:] = [ 5.64, 5.61 ]
        self._vis_reg[171,:] = [ 5.61, 5.57 ]
        self._vis_reg[172,:] = [ 5.57, 5.54 ]
        self._vis_reg[173,:] = [ 5.54, 5.50 ]
        self._vis_reg[174,:] = [ 5.50, 5.47 ]
        self._vis_reg[175,:] = [ 5.47, 5.43 ]
        self._vis_reg[176,:] = [ 5.43, 5.40 ]
        self._vis_reg[177,:] = [ 5.40, 5.36 ]
        self._vis_reg[178,:] = [ 5.36, 5.33 ]
        self._vis_reg[179,:] = [ 5.33, 5.30 ]
        self._vis_reg[180,:] = [ 5.30, 5.26 ]
        self._vis_reg[181,:] = [ 5.26, 5.23 ]
        self._vis_reg[182,:] = [ 5.23, 5.00 ]

        self._depth = array([0.207020, 0.239582, 0.329422, 0.450916, 0.593042, 0.785337, 1.01929, 1.31589])
        self._nvisx = self._visext.size
        self._depthx = arange(self._nvisx, dtype=float32) * 0.006 + 0.182
        self._depthx[self._nvisx - 1] = 1.3116 # corresponds to vis=5 km
        self._visibility_ddv = self.config.visibility
        self._visib_save = self.config.visibility
        self._visib_vnir = self.config.visibility
        self._w995 = 0.995 # default for window channel right of 940 nm region
        self.config.wl725 = array([0.688, 0.740])
        self.config.wl760 = array([0.750, 0.772]) # wavelength limits for interpolation in 760 nm region
        self.config.wl825 = array([0.790, 0.840])
        self._wv_thr_cirrus = self.config.wv_thr_cirrus # cirrus algorithm disabled if wv(av) < threshold
        self._wvl_adj = 2.46 # [um], for wvl > wvl_adj the adjacency effect and spherical albedo are neglected
        self._dnScale = float32(self.config.dnScale)
#
# all others will be initialized with 0:
        self._lut_lp5 = 0
        self._lut_rt5 = 0
        self._n_rt5 = 0
        self._nwvl5 = 0
        self._xnodes5 = 0
        self._x_cell5 = 0
        self._lp = 0
        self._q = 0
        self._mlist_haze = 0
        self._mlist_cloud = 0
        self._mlist_cloudw = 0
        self._mlist_water = 0
        self._mlist_clear = 0
        self._mlist_shad = 0
        self._gamma_cir = 0
        self._wvl_gamma = 0
        self._arr_gamma = 0
        self._anz_mea = 0
        self._anz_ref = 0
        self._cnt_brdf = 0
        self._cnt_nonb = 0
        self._cbeta = 0
        self._dratio_aeros = None
        self._dratio_aeros_best = 999.0
        self._datyp = 0
        self._ediftx = 0
        self._ele_class = 0
        self._edifth_cell = 0
        self._edift_fit_cell = 0
        self._edifth = 0
        self._edifth1 = 0
        self._edifth2 = 0
        self._elmax = 0
        self._e0t_fit_cell = 0
        self._e0th = 0
        self._e0th1 = 0
        self._e0th2 = 0
        self._e0t_fit = 0
        self._e0t = 0
        self._e0t_all = 0
        self._e0th_cell = 0
        self._edift_all = 0
        self._edift_fit = 0
        self._edift = 0
        self._e0tx = 0
        self._eclass = 0
        self._es = None
        self._fwhm = None
        self._gap = 0
        self._hlevels = 0
        self._heights = 0
        self._iwin1_chan = 0
        self._iwin1_region = 0
        self._ivisrpix = 0
        self._iav_ele_ref = 0
        self._iabs_chan = 0
        self._iav_ele = 0
        self._icirrus = 0
        self._ihaze = 0
        self._lph = 0
        self._lp_red = 0
        self._lp_blu = 0
        self._lp_fit_cell = 0
        self._lutconv = 0
        self._lpx = 0
        self._lph_cell = 0
        self._litopo_shad = 0
        self._li_nonback = 0
        self._lp_fit = 0
        self._lp_all = 0
        self._measure_ch = 0
        self._measure_c = zeros(self._n_bands)
        self._meanvi_ddv = 0
        self._meanvi_check_neg = 0
        self._nkl_save = 0
        self._nx_cell = 0
        self._ny_cell = 0
        self._n_lines = 0
        self._n_pixels = 0
        self._n_fact = 0
        self._qh_cell = 0
        self._qx = 0
        self._qh = 0
        self._q_all = 0
        self._q_fit = 0
        self._reflref_vect = 0
        self._reflter = 0
        self._refl_red_av = 0
        self._refl_red_av = 0
        self._reflvis = 0
        self._rho_cir_app = 0
        self._refl_blu_av = 0
        self._reference_c = 0
        self._reference_ch = 0
        self._spha_fit = 0
        self._spha_all = 0
        self._spha = 0
        self._sphah = 0
        self._sphah_cell = 0
        self._tsun = 0
        self._thre_refl_swir = 0
        self._tsunh = 0
        self._tsunh_cell = 0
        self._tsunx = 0
        self._tsun_all = 0
        self._tsun_fit = 0
        self._tsun_fit_cell = 0
        self._tdifh = 0
        self._tdifx = 0
        self._tdir = 0
        self._tdif = 0
        self._tdif_all = 0
        self._tdir_all = 0
        self._tdirh = 0
        self._tdirx = 0
        self._uu1_altit = 0
        self._u500m = 0
        self._uu1 = 0
        self._vsky = 0
        self._vskyc = 0
        self._wv_av = 0
        self._wvlsen = None
        self._wvlarr = 0
        self._w_pos_it = 0
        self._xcell = 0
        self._ycell = 0
        self._nuu1 = 0
        self._solze = None
        self._logger.debug('Module L2A_AtmCorr initialized')
        # TODO reflectance_offset Issue: support for reflectance_add_offset / minimum value?
        self._band_offset_offset_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def get_aerosol_detection(self):
        return self._aerosolDetection


    def set_aerosol_detection(self, value):
        self._aerosolDetection = value


    def del_aerosol_detection(self):
        del self._aerosolDetection


    def get_logger(self):
        return self._logger


    def set_logger(self, value):
        self._logger = value


    def del_logger(self):
        del self._logger


    def get_tables(self):
        return self._tables


    def set_tables(self, value):
        self._tables = value


    def del_tables(self):
        del self._tables


    def __exit__(self):
        sys.exit(-1)


    def __del__(self):
        self.logger.debug('Module L2A_AtmCorr deleted')

    def get_config(self):
        return self._config


    def set_config(self, value):
        self._config = value


    def del_config(self):
        del self._config


    tables = property(get_tables, set_tables, del_tables, "tables's docstring")
    config = property(get_config, set_config, del_config, "config's docstring")
    logger = property(get_logger, set_logger, del_logger, "logger's docstring")
    aerosolDetection = property(get_aerosol_detection, set_aerosol_detection, del_aerosol_detection, "aerosolDetection's docstring")

    def checkConfiguration(self):
        self.logger.info('checking configuration')
        self.logger.debug('config filename: '+ self.config.configFn)

        # read input from config
        if (self.config.pixelsize <= 0.01):
            self.logger.fatal('configuration: pixelsize must be greater than 0.01')
            
        if (self.config.cellsize < 1.0):
            self.logger.fatal('configuration: cellsize must be greater or equal 1.0')
            

        # check lib directory:
        if((os.path.exists(self.config.libDir)) == False):
            self.logger.fatal('lib directory '+ self.config.libDir + ' not configured or present')
            

        self._adj_km = (maximum(self.config.adj_km, 3. * self.config.pixelsize * 0.001)) # min adjacency range is 3*config.pixelsize (km)
        if(self.config.visibility < 0):
            self.logger.fatal('configuration: visibility cannot be negative')
            

        if(self.config.altit < 0):
            self.logger.fatal('configuration: altitude cannot be negative')
            

        solze = self.config.solze
        if((solze < 0.0) | (solze > 70.0)):
            self.logger.fatal('configuration: solar zenith angle not in range 0-70 deg')
            

        solaz = self.config.solaz
        if((solaz < 0.0) | (solaz > 360.0)):
            self.logger.fatal('configuration: solar azimuth angle not in range 0-360 deg')
            

        # tilt view angle, tilt azimuth angle
        thv = self.config.thv
        if ((thv < 0.0) | (thv > 70.0)):
            self.logger.fatal('configuration: sensor tilt angle not in range 0-70 deg')
            

        phiv = self.config.phiv
        if((phiv < 0.0) | (phiv > 360.0)):
            self.logger.fatal('configuration: sensor view azimuth angle not in range 0-360 deg')
            

        self._npref = self.config.npref
        if(self._npref > 1):
            self.logger.info('configuration: npref, recommended default value is 1')
        '''
        self._ihaze = self.config.ihaze
        if((self._ihaze < 0) | (self._ihaze > 2)):
            self.logger.fatal('configuration: ihaze index out of bounds [0|2]')
            
        '''
        if(self.config.resolution == 10):
            self.config.iwaterwv = 0
            self.logger.info('water vapour retrieval is disabled for 10m resolution')
            self.logger.debug('water vapour retrieval is disabled for 10m resolution')
        else:
            self.chkBool('iwaterwv', self.config.iwaterwv)

        ibrdf = self.config.ibrdf
        # but not if ibrdf < 0 which forces brdf terrain correction
        # (Experience with ETM Swiss winter scenes shows that
        # brdf correction for snow usually yields bad results)
        # treat obsolete cases of ibrdf
        if(ibrdf < 0):
            self.logger.fatal('configuration: BRDF_Correction cannot be negative')
            
        elif (ibrdf == 3):
            ibrdf = 1 # cover indep, linear
        elif (ibrdf == 4):
            ibrdf = 2 # " " sqrt
        elif (ibrdf == 5):
            ibrdf = 11 # cover dep., linear
        elif (ibrdf == 6):
            ibrdf = 21 # " ", sqrt
        elif (ibrdf == 7):
            ibrdf = 11
        elif (ibrdf == 8):
            ibrdf = 21
        elif (ibrdf == 9):
            ibrdf = 12
        elif (ibrdf == 10):
            ibrdf = 22
        elif (ibrdf > 22):
                ibrdf = 0
        self._ibrdf = ibrdf

        # switch off rugged terrain BRDF correction if not a defined code number
        if((self._ibrdf > 0) & (self._beta_thr <= 0.0)):
            solze = self.config.solze
            beta_thr = self._beta_thr

            if (solze < 35.0):
                beta_thr = solze + 25.0
            elif ((solze >= 35.0) & (solze < 45.0)):
                beta_thr = solze + 20.0
            elif ((solze >= 45.0) & (solze < 50.0)):
                beta_thr = solze + 15.0
            elif ((solze >= 50.0) & (solze < 60.0)):
                beta_thr = solze + 10.0
            elif (solze >= 60.0):
                beta_thr = max([70.0, solze + 5.0])

            self._beta_thr = beta_thr

        if(self.config.thr_g < 0):
            self.logger.fatal('configuration: BRDF_Lower_Bound cannot be negative')
            

        self._ch940w1 = self.config.ch940[1]
        self._ch940a1 = self.config.ch940[2]
        self._ch940a2 = self.config.ch940[3]
        self._ch940w2 = self.config.ch940[4]

        self._ch1130w1 = self._ch1130[1]
        self._ch1130a1 = self._ch1130[2]
        self._ch1130a2 = self._ch1130[3]
        self._ch1130w2 = self._ch1130[4]

        if (self.config.ch940[0] == 0):
            self.config.ch940[0] = self.config.ch940[1]
        if (self._ch1130[0] == 0):
            self._ch1130[0] = self._ch1130[1]
        if (self.config.ch940[5] == 0):
            self.config.ch940[5] = self.config.ch940[4]
        if (self._ch1130[5] == 0):
            self._ch1130[5] = self._ch1130[4]

        self._ch940a2 = (maximum(self._ch940a2, self._ch940a1))
        self.config.ch940[2:4] = ([self._ch940a1, self._ch940a2])
        self._ch1130a2 = (maximum(self._ch1130a2, self._ch1130a1))
        self._ch1130[2:4] = ([self._ch1130a1, self._ch1130a2])

        self.cirrus_correction = self.config.cirrus_correction
        # additional check of water vapor bands is done later

        # 1. read config.solze_arr: config.solze at image corners
        # 2. read config.solaz_arr
        # 3. read config.vza_arr
        # 4. config.vaa_arr (view azimuth at corners)
        # these arrays are already defined (load_commons)
        try:
            solze_arr = self.config.solze_arr # sequence: UL, UR, LL, LR
            if solze_arr.min() < 0.0 or solze_arr.max() > 70:
                self.logger.fatal('configuration: solar zenith angle array out of bounds [0-70] deg')


            solaz_arr = self.config.solaz_arr
            if solaz_arr.min() < 0.0 or solaz_arr.max() > 360.0:
                self.logger.fatal('configuration: solar azimuth angle array out of bounds [0-360] deg')


            vza_arr = abs(self.config.vza_arr) # accept only positive values
            if vza_arr.max() > 40.0:
                self.logger.fatal('configuration: view zenith angle array vza above 40 deg')


            vaa_arr = self.config.vaa_arr
            if vaa_arr.min() < 0.0 or vaa_arr.max() > 360.0:
                self.logger.fatal('configuration: view azimuth angle array out of bounds [0-360] deg')
            
        except:  # LANDSAT:
            if self.config.solze < 0 or self.config.solze > 70.0:
                self.logger.fatal("configuration: solar zenith angle out of bounds [0-70] deg")
            if self.config.solaz < 0 or self.config.solaz > 360.0:
                self.logger.fatal("configuration: solar azimuth angle out of bounds [0-360] deg")

        self.load_sensor_a3_hs() # (hyperspectral and user-defined sensors)
        if(self._cirrus_band == 0):
            self.cirrus_correction = False #self._icirrus = 0

        ierr = False
        if(self.config.iwaterwv == 1):
            if ( self._ch940a1 == 0 | self._ch940a2 == 0 | self._ch940w1 == 0):
                ierr = True
        elif(self.config.iwaterwv == 2):
            if (self._ch1130a1 == 0 | self._ch1130a2 == 0 | self._ch1130w1 == 0):
                ierr = True
        elif(self.config.iwaterwv == 3):
            if ( self._ch940a1 == 0 | self._ch940a2 == 0 | self._ch940w1 == 0 | \
                    self._ch1130a1 == 0 | self._ch1130a2 == 0 | self._ch1130w1 == 0):
                ierr = True
        if(ierr):
            self.logger.fatal('configuration: water vapor channels not correctly specified')

        nadj_pix, wadj = adjacency_weight(self._nadj_regions, self._adj_km, self.config.pixelsize)

        # check availability of absorption & window channel to avoid cases such as MSS
        tmpList = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= 0.899, self._wvlsen[self._first_band:self._n_bands] <= 0.970)))
        cnt = tmpList.size
        if (cnt < 0):
            self._ch940a1 = 0
            self._ch940a2 = 0
            self._ch940w1 = 0
            self._ch940w2 = 0 # no absorption channels

        tmpList = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= 0.850, self._wvlsen[self._first_band:self._n_bands] < 0.899)))
        cnt = tmpList.size
        if (cnt == 0):
            tmpList = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= 0.820, self._wvlsen[self._first_band:self._n_bands] < 0.899)))
            cnt = tmpList.size
        if (cnt > 0 and self._ch940w2 > 0):
            if (self._wvlsen[self._ch940w2 - 1] < 0.990):
                self._ch940w2 = 0

        tmpList = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= 1.110, self._wvlsen[self._first_band:self._n_bands] <= 1.155)))
        cnt = tmpList.size
        if (cnt <= 0):
            self._ch1130a1 = 0 ; self._ch1130a2 = 0 ; self._ch1130w1 = 0 ; self._ch1130w2 = 0 # no absorption channels

        tmpList = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= 0.995, self._wvlsen[self._first_band:self._n_bands] <= 1.090)))
        cnt = tmpList.size
        if (cnt > 0):
            self._w995 = 0.995
        else:
            tmpList = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= 0.990, self._wvlsen[self._first_band:self._n_bands] <= 1.090)))
            cnt = tmpList.size
            if (cnt > 0):
                self._w995 = 0.990

        if 'LANDSAT' not in self.config.spacecraftName:
            self.wv_regions_940_1130_a3()

        if (self._npref == 1):

            # variable visib, possibly with variable water vapor
            # --------------------------------------------------
            if (self._ihaze == 0):
                self._np_haze = 1 # a single haze pixel, required to make the cloud mask
                self._mlist_haze = array([1], int)

        self.logger.debug('configuration check terminated successful')
        return


    def calc_rho_cir_app(self):

    # Purpose: calculate apparent reflectance of cirrus band (1380 nm)
    #
    # Input:
    # solze: solar zenith angle
    # wvlsen: wavelength array (micron)
    # c0, c1: offset, gain of radiometric calibration
    # es: solar irradiance array pertaining to selected sensor
    # first_col, n_cols: first, last column of selected image or sub-image
    # first_row, n_rows: first, last line of selected image or sub-image
    # d2 = dastr^2 square of earth-sun distance
    # Output:
    # rho_cir_app: zeros(n_rows,n_cols) reflectance unit (0-1) scaled with 10,000
    # The average of all cirrus bands in the 1363-1400 nm interval is taken
    # to improve SNR, because a single band in this region is always noisy.
        rho_cir_app = self.tables.getBand(Band.CIRRUS) #self.tables.B10 #warning check here
        rho_cir_app = median_filter_2d(rho_cir_app, 3)
        self._rho_cir_app = rho_cir_app
        return True

    def calc_gamma_cir(self):
        #
        #  R.Richter, DLR, July 2015 : improved gamma(cirrus) calculation
        #
        # Purpose: calculate the slope gamma of the cirrus band to subtract the cirrus effect
        #          from the other bands:
        #               rho_TOA(VNIR) = rho_TOA(VNIR) - rho_TOA(cir)/gamma
        #         The difference to the former 'calc_gamma_cir' is:
        #      Now the computation is performed for a range of gamma values (0.6,1.0)
        #      using:
        #       (1)  a cirrus mask rho(cir) > 2%, i.e. scaled rho_cir_app > 200, list li_cir
        #            li_noc = no cirrus, cirrus-free, i.e. rho(cir) < 0.8%, scaled rho_cir_app < 80
        #       (2)  subtract cirrus for 3 bands < 590 nm
        #       (3)  compare m1=mean_rho_TOA(li_cir, gamma) with m2=mean_rho_TOA(li_noc)
        #       (4)  select gamma(i) with closest match: abs(m1-m2) = Min
        #      The gamma range (0.6 - 1.0) is sufficient: it was checked with
        #      31 scenes of Landsat-8, S2, and AVIRIS-S2 resampled scenes.
        #
        # Input:
        #    solze     solar zenith angle (degrees)
        #    c0, c1    calibration offset, gain (per band)
        #    es        solar irradiance vector
        #    rho_cir_app cirrus apparent reflectance scaled with 1.0E4, integer array
        #
        # Output:
        #    gamma_land  gamma for land pixels
        #    gamma_water = gamma_land
        #    wvl_gamma   wavelength grid for arr_gamma
        #    arr_gamma   wavelength dependent gamma land
        #    arr_gamma2  wavelength dependent gamma water
        #
        gamma = 1.0
        self._arr_gamma  = array([  1.0,  1.00,  1.00,  1.0,  1.0 ], dtype=float32) * gamma
        self._arr_gamma2 = self._arr_gamma
        self._wvl_gamma  = array([ 0.40,  1.40,  1.45,  2.0, 2.55 ], dtype=float32)

        rho_cir_app = self.tables.getBand(Band.CIRRUS)
        rho_cir_app = median_filter_2d(rho_cir_app, 3) * self.config.dnScale
        # rho_cir_app = self._rho_cir_app * self.config.dnScale
        li_cir = where32(ravel(rho_cir_app > 200))
        cnt_cir = len(li_cir)
        if cnt_cir < 10000:
            li_cir = where32(ravel(rho_cir_app > 140))
            cnt_cir = len(li_cir)
        if cnt_cir < 10000:
            return

        # the ' GT 10' excludes background
        li_noc = where32(ravel(rho_cir_app[(rho_cir_app > 10) & (rho_cir_app < 80)]))
        cnt_noc = len(li_noc)
        if cnt_noc < 2000:
            li_noc = where32(ravel(rho_cir_app[(rho_cir_app > 10) & (rho_cir_app < 100)]))  # rho(cir)= 1% criterion
            cnt_noc = len(li_noc)
        if cnt_noc < 2000:
            li_noc = where32(ravel(rho_cir_app[(rho_cir_app > 10) & (rho_cir_app < 110)]))  # rho(cir)= 1.1% criterion
            cnt_noc = len(li_noc)
        if cnt_noc < 2000:
            return

         #---------------------------------------------------------------------
         # - Determine the number of gamma iterations that can be handled
         #   depending on available memory and the gamma range (0.6 - 1.0)
         # - The gamma range (0.6 - 1.0) is sufficient: it was checked with
         #   31 scenes of Landsat-8, S2, and AVIRIS-S2 resampled scenes.
         # - The gamma increment is adapted to memory threshold and image size
         #---------------------------------------------------------------------

        mem_thr = 2.1E9 # max 2.1 GB memory will be used
                        # this influences the number of gamma iterations
        np = cnt_cir + cnt_noc # number of pixels to be processed
        n_iter = int32( 0.1 + mem_thr / (4.0 * np ))  # max number of gamma iterations

        if (n_iter < 2): return # scene too big, use defaults
        elif (n_iter == 3): vector_gamma = 0.7 + arange(n_iter, dtype=float32) * 0.15
        elif (n_iter == 4): vector_gamma = 0.7 + arange(n_iter, dtype=float32) * 0.10
        elif (n_iter == 5): vector_gamma = 0.6 + arange(n_iter, dtype=float32) * 0.10
        elif (n_iter == 6): vector_gamma = 0.6 + arange(n_iter, dtype=float32) * 0.08
        elif (n_iter == 7): vector_gamma = 0.6 + arange(n_iter, dtype=float32) * 0.0666
        elif (n_iter == 8): vector_gamma = 0.6 + arange(n_iter, dtype=float32) * 0.0571
        elif (n_iter >= 9):
            n_iter = 9
            vector_gamma = 0.6 + arange(n_iter, dtype=float32) * 0.05

        n_gamma = len(vector_gamma)
        nb = 3 # use only up to 3 bands
        rho_cir = zeros([n_gamma, nb, cnt_cir], dtype=float32)
        s_cir = zeros([n_gamma, nb], dtype=float32)
        rho_noc = zeros([nb, cnt_noc], dtype=float32)
        s_noc = zeros([nb], dtype=float32)

        self.tables.acMode = False
        for j in arange(len(vector_gamma)):
            gamma = vector_gamma[j]
            for k, band in enumerate([Band.COASTAL_AEROSOL, Band.BLUE, Band.GREEN]):# for k in arange(nb):
              b = self.tables.getBand(band) #    b = self.tables.getBand(k)
              if j == 0:
                 rho_noc[k,:] = ravel(b)[li_noc]
                 s_noc[k] = rho_noc[k,:].mean()

              rho_cir[j, k,:] = ravel(rho_cir_app)[li_cir] / gamma
              s_cir[j, k] = rho_cir[j, k,:].mean()

        self.tables.acMode = True
        cr  = zeros(n_gamma, dtype=float32)
        for j in arange(len(vector_gamma)):
            cr[j] = abs((s_noc - s_cir[j,:]).sum())

        li1 = where32(cr == cr.min())
        js = li1[0]
        gamma_land  = vector_gamma[js]
        gamma_water = gamma_land
        self._gamma_cir   = gamma_land

        # avoid overcorrection for water:
        if gamma_water < 0.5: gamma_water *= 2.0
        if gamma_water < 0.7: gamma_water *= 1.2

        self._arr_gamma  = array([  1.0,  1.00,  1.00,  1.0,  1.0 ], dtype=float32) * gamma_land
        self._arr_gamma2 = array([  1.0,  1.00,  1.00,  1.0,  1.0 ], dtype=float32) * gamma_water
        self._wvl_gamma  = array([ 0.40,  1.40,  1.45,  2.0, 2.55 ], dtype=float32)
        return


    #@profile
    def process(self):
    #*******************************************************************************
    # MODEL ATCOR_S2: ATMOSPHERIC and TOPOGRAPHIC CORRECTION - SATELLITE VERSION
    # taylored to Sentinel-2
    #
    # Batch Version
    #
    #
    # Author: R. Richter
    # DLR - German Aerospace Center
    # D-82234 Wessling / Germany
    #
    # Last modified: December 2010
    #
    # documentation: 1. "Atmospheric/Topographic Correction for Satellite Imagery:
    # ATCOR-2/3 User Guide",
    # Report DLR - IB 564-01/10 (2010)
    # 2. Sentinel-2 MSI - Level 2A Products Algorithm Theoretical Basis
    # Document - Volume B (ATCOR), S2PAD-DLR-ATBD-0002 (2010)
    # S2PAD-DLR-ATBD-0002-1_6_L2A_ATBD_Volume_B_ATCOR.pdf
    # 3. Includes updates described in submitted paper (2 June 2010)
    # published: Int. J. Remote Sensing, Vol. 32, 2931-2941 (2011)
    # Richter et al. "Correction of cirrus effects in Sentinel-2 type
    # of imagery"
    #*******************************************************************************
        self.tables.acMode = True
        self._itarget = 1 # DDV pixels are searched

        for k, band in enumerate \
                    ([Band.COASTAL_AEROSOL, Band.BLUE, Band.GREEN, Band.RED, \
                      Band.VEGETATION_1, Band.VEGETATION_2, Band.VEGETATION_3, Band.NEAR_INFRARED, Band.VEGETATION_4, \
                      Band.WATER_VAPOUR, Band.CIRRUS, Band.SHORT_WAVE_INFRARED_1, Band.SHORT_WAVE_INFRARED_2]):
            if not self.tables.hasBand(band):
                continue
            if 'LANDSAT' in self.config.spacecraftName and band == Band.CIRRUS:
                continue
            # print(k, band) to keep for debugging

        self.initialize() # self.open_input3()
        self.checkConfiguration()
        self.checkDem()

        if(self.tables.hasBand(Band.DIGITAL_ELEVATION_MAP) == True): #self.tables.DEM
            ierror = self.dtm_array()
            # calculate corresponding topographic arrays
            if (ierror == 1):
                self.logger.fatal('Error with DEM')
            if (ierror == 2):
                self.logger.warning('DEM slope may have a lot of steps')
                self.logger.warning('Processing continues, but results may have artifacts')
            if (ierror == 3):
                self.logger.info('slopes below 6 degrees and max height difference below 300 m')
                self.dtm_flat()

        else:
            # flat terrain with average elevation specified
            # calculate required arrays to treat it as a special case of rugged terrain
            self.dtm_flat()


        if (self._liback.size == int(self.config.nrows * self.config.ncols)):
            self.logger.fatal('scene has only background pixels')
            
        self.config.timestamp('L2A_AtmCorr: end of calculation terrain maps')

        # set nadir geometry (view and sun angles) for the calculations of
        # (a) cirrus apparent reflectance
        # (b) masking of DDV reference pixels
        # However, the reflectance calculation of the DDV pixels is later performed for
        # geometry-dependent small cells and also the surface reflectance retrieval.

        if 'LANDSAT' in self.config.spacecraftName:
            self._thv = self.config.thv
            self._phiv = self.config.phiv
            self._solze = self.config.solze
            self._solaz = self.config.solaz
        else:
            x = set_nadir_geometry(self.config.solze_arr, self.config.solaz_arr, self.config.vza_arr, self.config.vaa_arr, self._itilt)
            self._thv = x[0]
            self._phiv = x[1]
            self._solze = x[2]
            self._solaz = x[3]

        # if (self.cirrus_correction):
        #     # calculate rho_cir_app (scaled with 1E4) is put in Common
        #     self.calc_rho_cir_app()
        if (self.cirrus_correction):
            # calculate gamma_cir
            self.calc_gamma_cir()
        if (self.cirrus_correction):
            self._trwv945, self._h1_cirrus = read_wv_trans945_1375(self._gamma_cir, self._solze, self._h1_cirrus)

        # f specifies the size in x direction (image columns),
        # twice this size is used in y direction (much smaller change in solar geometry)
        # " " 3 km x 6 km (Sentinel-2) config.pixelsize= 60 m
        #cell_length_x = int(self.config.cellsize * 1000 / self.config.pixelsize + 0.5)
        cell_length_x = 915 # resolution @ 60 m / 2
        #cell_length_x = int(3660/self.config.pixelsize + 0.5)
        #cell_length_x = int(54900/self.config.pixelsize + 0.5)
        #cell_length_x = 1830 # resolution @ 60 m / 2 Test for a single cell at 60m resolution for investigation
        self._ny_cell, self._nx_cell, self._ycell, self._xcell = image_cells(self.config.nrows, self.config.ncols, cell_length_x)

        # Check if visibility file exists:
        # (a) 60 m bands: creates the 'visibility'file
        # (b1) 20 m bands: may use this file, but it has to be resampled to 20m
        # and must be in the same directory as the (20m) input file.
        # OR
        # (b2) 20 m bands: calculate new 'vis'file (in the same folder as 20m input file)
        # The visibility/AOT processing is always needed in case of haze removal (ihaze=1)
        # because the "'_visindex.bsq'is calculated besides the '_haze_levels.bsq'and '_haze_removed.bsq'
        # (c) 10 m bands: need a resampled VIS band either from (a) or (b2), in the
        # directory of the 10m input image data.
        #
        # Generally: if the VIS band exists, the visibility/AOT calculation is skipped,
        # otherwise it is calculated. However, if it is missing for the 10 m bands the processing terminates,
        # because the vis/AOT calculation with those 4 bands is not accurate.
        # ----------------------------------------------------------------------

        if (self.config.resolution > 10):
            ic_fvis = False
            skip1 = False
        else: # resampling of AOT for 10 m bands:
            self.config.timestamp('L2A_AtmCorr: start of resampling visibility for %s m resolution' % self.config.resolution)
            ic_fvis = self.tables.hasBand(Band.VISIBILITY) #self.tables.VIS
            if (ic_fvis == False):
                self.logger.fatal('visibility index is required for 10 m processing, but not present')

            # open visindex file (required for 10m Sentinel-2 data)
            VIS = self.tables.getBand(Band.VISIBILITY) #self.tables.VIS
            SCL = self.tables.getBand(Band.SCENE_CLASSIFICATION) #self.tables.SCL
            self._meanvi = uint8(VIS[(SCL == self.config.vegetation)|(SCL == self.config.bareSoils)].mean() + 0.5)
            del VIS
            del SCL

            self._visibility = self._visext[self._meanvi]
            self._visib_save = self._visibility
            a_aot = interpol(self._a_aot_fit, self._h_grid_aot, arange(self._n_alti)) # vector of a_aot_fit(z)
            b_aot = interpol(self._b_aot_fit, self._h_grid_aot, arange(self._n_alti))
            self._scene_av_aot = float32(exp(a_aot[self._iav_ele] + b_aot[self._iav_ele] * log(self._visext[self._meanvi])))

            del a_aot
            del b_aot

            self.config.timestamp('L2A_AtmCorr: end of resampling visibility for %s m resolution' % self.config.resolution)
            if not self.config.demOutput:
                self.tables.removeBandRes(Band.DIGITAL_ELEVATION_MAP) #self.tables.DEM
            self.tables.removeBandRes(Band.SLOPE) #self.tables.SLP
            self.tables.removeBandRes(Band.SHADOW_MAP) #self.tables.SDW

            self.masking_a3()
            skip1 = True
        if(skip1 == False):
            self.config.timestamp('L2A_AtmCorr: start of AOT retrieval at 550nm')
            self._atmDataFn = self.config.atmDataFn
            self.altit3_atm()
            self.altit3v_atm()

            self._visib_save = self.config.visibility # visib is iterated in mask_veget_swir
            self.masking_a3() # calls mask_veget_swir_a3 if 1.6/2.2 um band
            self._visibility = self._visib_save
            self._visib_vnir = self._visibility # (if sensor has only VNR bands)
            self.config.timestamp('L2A_AtmCorr: end of internal classification')
            self._visib_save = self._visibility
            self.altit3v_atm() # yields lph(n_alti,nvisx,n_bands_all), lpx(n_alti,nvis,n_bands_all)
            self.config.timestamp('L2A_AtmCorr: end of interpolation LUTs')
            self._visibility = self._visib_save # restore visib

            if (self._n_bands > 4 and ic_fvis == False):
                iflag_haze = 0
                if (self._npref > 0):
                    self.ref_pixel_wfov(iflag_haze) # no haze removal for red band

            self.config.timestamp('L2A_AtmCorr: end retrieving reference pixels for dark areas')

            # save lph etc
            self._lph_save = self._lph.copy()
            self._e0th_save = self._e0th.copy()
            self._edifth_save = self._edifth.copy()
            self._tdirh_save = self._tdirh.copy()
            self._tdifh_save = self._tdifh.copy()
            self._tsunh_save = self._tsunh.copy()
            self._qh_save = self._qh.copy()
            self._sphah_save = self._sphah.copy()

            if (self._n_bands > 4 and ic_fvis == False):
                if (self.config.visibility > 0):
                    if (self._npref > 0):
                        # check of neg. reflectance pixels
                        self.check_negative_refl_wfov(Band.RED) # check for general vegetation, DDV + DBV #self._red_band

                        # self.check_negative_refl_wfov(Band.VEGETATION_4) # check for water pixels #self._nir_band
                        if 'LANDSAT' in self.config.spacecraftName:
                            self.check_negative_refl_wfov(Band.NEAR_INFRARED)
                        else:
                            self.check_negative_refl_wfov(Band.VEGETATION_4)
                        self._mlist_cloud_rd = 0 # free memory
                        self.config.timestamp('L2A_AtmCorr: end of check for negative reflectance pixels')
                        self.config.aot_retrieval_method = 'SEN2COR_DDV'
                    else:
                        # no DDV pixels
                        # stop here if in automated aerosol detection mode:
                        if self.aerosolDetection == 'STARTED':
                            return

                        self._mlistref = array(0)  # (might be a number < 1% of pixels, has to be reset to zero)

                        if self.tables.hasBand(Band.VISIBILITY_INDEX_MAP) & (self.config.npref > 0): #self.tables.VIM
                            # Use AUXDATA OR CAMS ECMWF data
                            vis = self.tables.getBand(Band.VISIBILITY_INDEX_MAP) / 100. # in km
                            # convert to the visibility index
                            bandvis = ones(vis.shape, int16)
                            for i in range(self._vis_reg.shape[0]):
                                bandvis = where(vis <= self._vis_reg[i, 0], i, bandvis)
                            #self.importBand(self.VIM, bandvis) # debug
                            scl = self.tables.getBand(Band.SCENE_CLASSIFICATION) #self.tables.SCL
                            noData = self.config.noData
                            satDef = self.config.saturatedDefective
                            self._meanvi = int(0.5 + mean(bandvis[(scl!=noData) & (scl!=satDef)])) # mean outside no data
                            self._meanvi_cams = self._meanvi
                            self._visibility = self._visext[self._meanvi]  # mean visibility in km
                            self._visib_save = self._visibility
                            self.config.timestamp(
                                'L2A_AtmCorr: end of reading AUX_DATA or ECMWF CAMS visibility to index')

                            self.check_negative_refl_wfov(Band.RED) #self._red_band
#                            self.check_negative_refl_wfov(Band.VEGETATION_4) #self._nir_band
                            if 'LANDSAT' in self.config.spacecraftName:
                                self.check_negative_refl_wfov(Band.NEAR_INFRARED)
                            else:
                                self.check_negative_refl_wfov(Band.VEGETATION_4)

                            # No DDV ref.pixels found for this image,
                            # but visib might have been updated in check_negative_refl (veget. red band; water, NIR band)
                            # Update meanvi:

                            # CAMS: add offset coming from check_neg
                            self.config.timestamp('L2A_AtmCorr: meanvis (CAMS): {0} km'.format(self._visext[self._meanvi_cams]))
                            self.config.timestamp('L2A_AtmCorr: meanvis (after check_neg): {0} km'.format(self._visext[self._meanvi]))
                            # values between meanvi and meanvi_cams -> to meanvi value
                            for val in range(self._meanvi + 1, self._meanvi_cams + 1):
                                bandvis[bandvis == val] = self._meanvi
                            # apply offset above meanvi_cams
                            bandvis[bandvis > self._meanvi_cams] = bandvis[bandvis > self._meanvi_cams] + self._meanvi - self._meanvi_cams

                            # clip negative values and convert to byte
                            bandvis = uint8(bandvis.clip(min=0))
                            self.config.aot_retrieval_method = 'CAMS'
                        else:
                            # no CAMS : mean visibility for all pixels (+ DBV + check negative refl. only if self.config.npref = 1)
                            if self.config.npref:
                                self.ref_pixel_dbv_wfov()  # check bright veg. pixels, red and NIR band
                                # -> also includes check_negative_refl_wfov
                                self.config.timestamp('L2A_AtmCorr: meanvis (after DBV/check_neg): {0} km'.format(self._visext[self._meanvi]))
                            self._meanvi = uint8(indexvis(abs(self._visibility), self._nvisx, self._vis_reg))
                            bandvis = zeros([self.config.nrows, self.config.ncols], dtype=uint8) + self._meanvi
                            self.config.aot_retrieval_method = 'DEFAULT'

                        self.config.timestamp(
                            'L2A_AtmCorr: end of check for dense bright vegetation pixels')
                else:
                    # visib_forced < 0, use visib=abs(visib_forced) as visibility
                    self._meanvi = uint(indexvis(self._visibility, self._nvisx, self._vis_reg))
                    bandvis = zeros([self.config.nrows, self.config.ncols], dtype=uint8) + self._meanvi

                # restore lph etc
                self._lph = self._lph_save
                self._e0th = self._e0th_save
                self._edifth = self._edifth_save
                self._tdirh = self._tdirh_save
                self._tdifh = self._tdifh_save
                self._tsunh = self._tsunh_save
                self._qh = self._qh_save
                self._sphah = self._sphah_save

                if (self._npref == 0):
                    self._ddv_pixel_percentage = 0 # percent of DDV pixels
                    self._ddv_reflectance_range = 0
            if (self._n_bands > 4 and ic_fvis == False):
                if (self._npref > 0):
                    self.ref_pixel_vi_map_wfov() # calculate final meanvi, visib, mlistref for ref. pixels
                    # DN of thin haze-over-land is subtracted from DDV if ihaze > 0
                    self.config.timestamp('L2A_AtmCorr: end of visibility index calculation')
                else:
                    # No DDV ref.pixels found for this image,
                    # but visib might have been updated in check_negative_refl (veget. red band; water, NIR band)
                    # Update meanvi:
                    self._meanvi = uint8(indexvis(abs(self._visibility), self._nvisx, self._vis_reg))
                    #bandvis = zeros([self.config.nrows, self.config.ncols], dtype=uint8) + self._meanvi
                    self.write_visindex_file_a3(bandvis)
                    self.altit3v_atm() # yields lph(n_alti,nvisx,n_bands_all), lpx(n_alti1,nvis,n_bands_all)
                    # (unlike for the flat terrain, visib is not used in altit3v_atm)
                    # no ddv ref.pixels, no scale_path_radiance(blue-red) # (of n_bands NE 4)
                    self.config.timestamp('L2A_AtmCorr: end of receiving atmospheric functions for all altitudes and visibilities')

            del self._edifth_save
            del self._e0th_save

        # stop here if in automated aerosol detection mode:
        # fix for SIIMPC-672.3, UMW:
        if self.aerosolDetection == 'STARTED':
            return
        #
        # -------------------------------------------
        # 2. loop over cells (water vapor retrieval): vis.index and AOT map for whole scene are known
        # -------------------------------------------
        #
        # skip1 statement from ATCOR code starts here
        self.config.timestamp('L2A_AtmCorr: end of AOT retrieval at 550nm')
        self.exluding_wv_dependency = 0
        if (self.config.iwaterwv > 0):
            self.config.timestamp('L2A_AtmCorr: start of water vapour retrieval')
            if 'LANDSAT' in self.config.spacecraftName: #if 'LANDSAT' not in self.config.spacecraftName: #if self.config.spacecraftName != 'LANDSAT_8':
                 wv = self.tables.getBand(Band.WATER_VAPOUR)  #/ 1000 #self.tables.WVP #note for all the wv
                 # no further wv_processing required
                 self.exluding_wv_dependency= self.config.iwaterwv
                 self.config.iwaterwv = 0
            else:
                nk1, nk2, nka, dn1w, dn2w, dnabs, dn0w, w1, w2 = self.prepare_wv_retrieval()
                self.config.timestamp('L2A_AtmCorr: end of water vapour retrieval preparation')
                self.wv_retrieval_apda1(nk1, nk2, nka, dn1w, dn2w, dnabs, dn0w, w1, w2)
                self.config.timestamp('L2A_AtmCorr: end of water vapour retrieval')
        '''
        # free memory, keep np_water for log file
        # mlist_cloud, mlist_hazew are still needed
        '''
        self.config.timestamp('L2A_AtmCorr: preparation of surface reflectance retrieval')
        # the next 3 routines replace the "vari_region_a3" of the narrow-FOV atcor3
        nw, ju, cnt_nonback, chan2win, li_stat, icloud, li_clear, b, cnt_vege, ga, ga_nir, nch_wvp = self.prepare_rho_retrieval()
        self.config.timestamp('L2A_AtmCorr: end of surface reflectance retrieval preparation')
        # band loop and cell loop, without terrain view factor, without adjacency correction
        self.config.timestamp('L2A_AtmCorr: start of surface reflectance retrieval')
        self.rho_retrieval_step1(nw, ju, li_stat, b, cnt_vege, ga, ga_nir)
        self.config.timestamp('L2A_AtmCorr: end of surface reflectance retrieval')

        del self._edift_fit_cell
        del li_stat
        # implementation of SIIMPC-557, UMW:
        if self.config.rho_retrieval_step2:
            # includes terrain view factor and adjacency correction
            del self._altitude_grid
            del self._altitude_grid_km
            del self._depthx
            del self._eclass_sub
            del self._edift
            del self._edift_all
            del self._edifth2
            del self._edifth1
            del self._litopo_shad
            del self._lp
            del self._lp_all
            del self._lpx
            del self._lut_lp5
            del self._lut_rt5
            del self._mlist_clear
            del self._mlist_clshad
            del self._mlist_water
            del self._q
            del self._q_all
            del self._q_fit
            del self._qx
            del self._spha
            del self._spha_all
            del self._spha_fit
            del self._tdifh
            del self._tdir
            del self._tdif
            del self._tdif_all
            del self._tdir_all
            del self._tdirh
            del self._tdirx
            del self._tdifx
            del self._x_cell5
            del self._xcell
            del self._visext
            del self._vis_reg
            del self._uu1_altit_save
            del self._uu1_altit
            del self._tsunx
            del self._tsunh_cell
            del self._tsun_all

            self.config.timestamp('L2A_AtmCorr: start of rho retrieval step 2')
            self.rho_retrieval_step2(nw, ju, cnt_nonback, chan2win, icloud, li_clear, b, cnt_vege, ga, ga_nir, nch_wvp)
            if self._high_cloud_coverage:
                self.config.timestamp(
                    'L2A_AtmCorr: adjacency correction has been disabled because of high cloud coverage')

            self.config.timestamp('L2A_AtmCorr: end of rho retrieval step 2')

        else:
            # log indicating the status of rho retrieval step 2 if image is too cloudy
            if self._high_cloud_coverage:
                self.config.timestamp(
                    'L2A_AtmCorr: rho retrieval step 2 has been disabled because of high cloud coverage')
            else:
                self.config.timestamp('L2A_AtmCorr: rho retrieval step 2 has been disabled by configuration')

        self.postprocess()
        return

    def calcDratioAerosol(self):
        self.config.createAtmDataFilename()
        self.config.timestamp('L2A_AtmCorr: testing mid latitude: %s, aerosol type: %s' % (self.config.midLatitude, self.config.aerosolType))
        self.process()
        if self._dratio_aeros == None:
            self.config.timestamp(
                'L2A_AtmCorr: not enough DDV pixel present, aerosol double ratio could not be determined')
            self.config.timestamp(
                'L2A_AtmCorr: standard model is: %s %s' % (self.config.midLatitude, self.config.aerosolType))
            self.aerosolDetection = 'STOPPED'
            return False

        self.logger.info('aerosol double ratio: %f' % self._dratio_aeros)
        dratio_aeros_abs_1 = abs(self._dratio_aeros - 1.0)
        dratio_aeros_abs_best_1 = abs(self._dratio_aeros_best - 1.0)

        if dratio_aeros_abs_1 < dratio_aeros_abs_best_1:
            self._dratio_aeros_best = self._dratio_aeros
            self.config.createAtmDataFilename()
            self._atmDataFn = self.config.atmDataFn
            self._aerosolTypeBest = self.config.aerosolType
        return True

    def postprocess(self):
        if (self.config.resolution > 10) and (self.config.ddvOutput == True):
            self.write_ddv_map()

        if self._elmax > 3000:
            self.config.ground_elevation_above_3 = 'True'
        self.config.visibility_from_ddv = self._visibility_ddv
        self.config.final_visibility = self._visibility
        self.config.ddv_pixel_percentage = self._ddv_pixel_percentage
        self.config.ddv_reflectance_range = self._ddv_reflectance_range

        if 'LANDSAT' in self.config.spacecraftName:
            return # no pickle exists for Landsat

        picFn = self.config.picFn
        self.config.logger = None
        l.acquire()
        try:
            f = open(picFn, 'wb')
            pickle.dump(self.config, f, 2)
            f.close()

            self.config.logger = self.logger
        except:
            self.config.logger = self.logger
            self.logger.fatal('cannot update configuration' % picFn)

        finally:
            l.release()
        return

    #*******************************************************************************
    # MODEL ATCOR_S2: ATMOSPHERIC and TOPOGRAPHIC CORRECTION - SATELLITE VERSION
    # taylored to Sentinel-2
    #
    # Author: R. Richter (November 2010)
    # DLR - German Aerospace Center
    # D-82234 Wessling / Germany
    #
    # documentation: 1. "Atmospheric/Topographic Correction for Satellite Imagery:
    # ATCOR-2/3 User Guide",
    # Report DLR - IB 565-01/10 (2010)
    # 2. Sentinel-2 MSI - Level 2A Products Algorithm Theoretical Basis
    # Document - Volume B (ATCOR), S2PAD-DLR-ATBD-0002 (2010)
    # 3. Includes updates described in submitted paper (2 June 2010)
    # published: Int. J. Remote Sensing, Vol. 32, 2931-2941 (2011)
    # Richter et al. "Correction of cirrus effects in Sentinel-2 type
    # of imagery"
    #
    # Last modified: December 2010
    #*******************************************************************************
    #
    # Note 1: criterion for quasi-flat DEM (subroutine dtm_array)
    # height difference < 300 m and slope < 6 degrees for less than 1% of pixels
    #
    # Note 2: the subroutine "load_commons_a3" contains the file name conventions for the
    # atmospheric LUTs (summer, winter, different ozone contents)
    #
    # Note 3: de-shadowing: (masking_cloud_shadow_init)
    # An external water map from filin+'_water_map.bsq'or filin+'_hcw.bsq'always overwrites
    # (internal) spectral water mask criteria.
    #
    # Note 4: The factor gamma=0.5 for the SWIR bands (see routine rho_retrieval_step1)

    def initialize(self):
        msg = "Reading initial image parameter"
        # res = self.config.resolution
        # if res == 10:
        #     self._band_index = [1, 2, 3, 7]
        #     self._bands = [Band.BLUE, Band.GREEN, Band.RED, Band.NEAR_INFRARED]
        # elif res == 20:
        #     self._band_index = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
        #     self._bands = [Band.COASTAL_AEROSOL, Band.BLUE, Band.GREEN, Band.RED, Band.VEGETATION_1,
        #                    Band.VEGETATION_2, Band.VEGETATION_3, Band.VEGETATION_4, Band.WATER_VAPOUR_INPUT,
        #                    Band.CIRRUS, Band.SHORT_WAVE_INFRARED_1, Band.SHORT_WAVE_INFRARED_2]
        # elif res == 60:
        #     self._band_index = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
        #     self._bands = [Band.COASTAL_AEROSOL, Band.BLUE, Band.GREEN, Band.RED, Band.VEGETATION_1,
        #                    Band.VEGETATION_2, Band.VEGETATION_3, Band.VEGETATION_4, Band.WATER_VAPOUR_INPUT,
        #                    Band.CIRRUS, Band.SHORT_WAVE_INFRARED_1, Band.SHORT_WAVE_INFRARED_2]
        # elif res == 30:
        #     self._band_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self._bands = self.tables.bands

        self.logger.debug(msg)
        self._first_row = 1
        self._first_col = 1

        if self.config.midLatitude == 'SUMMER':
            self._wv_av = self.config.AC_Wv_Iter_Start_Summer * 1000.0 # default wv=1.0 cm, scaled with 1000
            self._uu1, self._suu1, self._nuu1, self._uu1_altit = load_wv_tables_summer()
            msg = 'Load water vapour tables for summer period'
            self.logger.debug(msg)
        elif self.config.midLatitude == 'WINTER':
            self._wv_av = self.config.AC_Wv_Iter_Start_Winter * 1000.0  # default wv=0.4 cm, scaled with 1000
            self._uu1, self._suu1, self._nuu1, self._uu1_altit = load_wv_tables_winter()
            msg = 'Load water vapour tables for winter period'
            self.logger.debug(msg)

    def interpol_lut5(self, vtest):

    # Interpolate in a 5-D parameter space
    # Input: phi, vis, sza, ele, vza
    # and common block LUT5
    # returns the array (6, nwvl) with the interpolated RT quantities
    # Lp, Edift, E0t, Tdir, Tdif, spha
    # for the selected 5 values of phi, vis, sza, ele, vza
    # Algorithm: 1. Kraaijpoel, D., PhD thesis, Utrecht University, (2003)
    # "Seismic ray fields and ray field maps: theory and algorithms"
    # 2. Pluim, J.P.W., et al., IEEE Trans. Medical Imaging, Vol. 22, 986-1004,
    # "Mutual information based registration of medical images", (2003)

        #global lut_lp5, lut_rt5, n_rt5, nwvl5, xnodes5, n_nodes5, ndim5, lim5, lut_cell5, x_cell5

        # 5 element vector for the 5-D parameter space
        ndim5 = 5
        n_rt6 = 6

        # fix for SIIMPC-1038, UMW: avoid floats as index, depecated for numpy > 1.11
        lim5 = zeros([ndim5, 2], uint8)

        if self.config.resolution == 10:
            nrefl1 = 4
        elif self.config.resolution == 30:
            if self.config.Hyper:
                nrefl1 = 12
            else:
                nrefl1 = 9
        else:
            nrefl1 = 12

        lut_cell5 = zeros([nrefl1, self._nnodes5, n_rt6], float32)

        # hyper-cell vertices around interpolation point calculated
        for i in arange(0, ndim5):
            wh = where32(ravel(vtest[i] < self._xnodes5[:, i]))
            lim5[i, 0] = wh[0] - 1
            lim5[i, 1] = wh[0]

        # atmospheric parameters from each vertix are read
        cont = 0
        for i in arange(0, 2):
            for j in arange(0, 2):
                for k in arange(0, 2):
                    for ii in arange(0, 2):
                        for jj in arange(0, 2):
                        # index in xnodes: i,0 = phi j,1=vis k,2=sza ii,3=ele jj,4=vza
                            lut_cell5[:, cont, 0] = self._lut_lp5[lim5[4, jj], lim5[3, ii], lim5[2, k], lim5[1, j], lim5[0, i],:, 0]
                            for ind in arange(1, (self._n_rt5)):
                                lut_cell5[:, cont, ind] = self._lut_rt5[lim5[4, jj], lim5[3, ii], lim5[2, k], lim5[1, j], 0,:, ind - 1]
                            cont = cont + 1

        # input vector is scaled to [0., 1.] for each dimension
        for i in arange(0, ndim5):
            vtest[i] = (vtest[i] - self._xnodes5[lim5[i, 0], i]) / (self._xnodes5[lim5[i, 1], i] - self._xnodes5[lim5[i, 0], i])

        f_int = zeros([self._nwvl5, self._n_rt5], float32)
        for i in arange(0, (self._nnodes5)):
            weight = abs(prod(vtest - self._x_cell5[self._nnodes5 - 1 - i,:])) # product of all vector elements
            for ind in arange(0, (self._n_rt5)):
                f_int[:, ind] = f_int[:, ind] + (weight * lut_cell5[:, i, ind])

        return f_int


    def read_atm_hyper_a3(self):
        #global lut_lp5, lut_rt5, n_rt5, nwvl5, xnodes5, nnodes5, ndim5, lim5, lut_cell5, x_cell5

        # read atmospheric correction functions for all altitudes in database
        # Input: solze = solar zenith angle (deg)
        # solaz = solar azimuth " "
        # thv = current view zenith angle (deg)
        # phiv = current view amimuth angle "
        # visib = visibility (5 <= visib <= 120 km)
        # Output: lp, edift, e0t, tdir, tdif, q, spha, tsun (Common block)
        # lp = path radiance
        # edift = diffuse flux transmitted to sensor
        # edift = (tdir+tdif)*edif
        # edif = diffuse flux at ground
        # e0t = transmitted solar irradiance at sensor
        # e0t = (tdir+tdif)*tsun*e0
        # tsun = direct transmittance sun-to-ground
        # tdir = direct (beam) transmittance ground-to-sensor
        # tdif = diffuse transmittance ground-to-sensor
        # tsun = direct transmittance sun-to-ground
        # q = tdif/tdir for adjacency correction
        # spha = spherical albedo of atmosphere
        #
        # read data from .atm file
        # S2: nadir and 10 degr. off-nadir LUTs
        # structure: 6 RT fcts for nadir, then 4 RT + 7 Lp {rel.azimuth=0(30)180 deg} for the 10degr off-nadir angle,
        # then repeated for each of the 6 elevations 0(0.5)2.5 km.
        # Makes 6 + (4 + 7) = 6 + 11 = 17 terms, for each elevation.
        # The 4 off-nadir RT terms are Edift, E0t, Tdir, Tdif.

        import xdrlib

        # determine file structure (number of view angles)
        # -----------------------------------------------
        if self.config.resolution == 10:
            nrefl1 = 4
        elif self.config.resolution == 30:
            if self.config.Hyper:
                nrefl1 = 12
            else:
                nrefl1 = 9
        else:
            nrefl1 = 12

        n_rt6 = 6 # Lp, Edf, Edr, Tdr, Tdf, spha, 6 RT terms
        n_rt11 = 11 # Edf, Edr, Tdr, Tdf, Lp for 7 relazi angles spaced 30 deg

        # database size for nadir angle: factor 4 for float
        # the 2 accounts for vis, zen in the first 2 columns
        nbytes_nadir = (2 + asarray(n_rt6).astype(int32) * self._n_alt) * 4 * self._nzen * self._nvis * nrefl1

        # database size per off-nadir angle: factor 4 for float
        nbytes_offnad = asarray(n_rt11).astype(int32) * self._n_alt * 4 * self._nzen * self._nvis * nrefl1

        # read atm file, use xdrlib for unpacking:
        fn = self._atmDataFn
        fsize = os.path.getsize(fn)
        # fp = open(fn, 'rb')
        # try:
        #     l.acquire()
        l.acquire()
        try:
            fp = open(fn, "rb")
            xdr = xdrlib.Unpacker(fp.read())
            fp.close()
        finally:
            l.release()

        nview = 1 + int((fsize - nbytes_nadir) / nbytes_offnad)
        n1 = n_rt6 + (nview - 1) * n_rt11
        nitems = int(fsize/4)
        lut = xdr.unpack_farray(nitems, xdr.unpack_float)      
        arr = asarray(lut)
        arr = arr.reshape(nrefl1, self._nvis, self._nzen, 2 + n1 * self._n_alt)
        arr = arr[:,:,:, 2:(2 + n1 * self._n_alt - 1)+1] # remove vis, zen columns
        arr = arr.reshape(nrefl1, self._nvis, self._nzen, self._n_alt, n1)
        arr = transpose(arr) # fltarr(n_refl1, nvis, nzen, n_alt, n1) for faster processing
        self._lut_lp5 = zeros([nview, self._n_alt, self._nzen, self._nvis, 7, nrefl1, 1], float32) # 7 rel. azi Lp: 0(30)180 deg
        self._lut_rt5 = zeros([nview, self._n_alt, self._nzen, self._nvis, 1, nrefl1, 5], float32) # 5 RT fcts: Edift, E0t, Tdir, Tdif, s
        # assign nadir values (Edift, E0t, Tdir, Tdif, s)
        for j in arange(0, 5):
            self._lut_rt5[0,:,:,:, 0,:, j] = arr[j + 1,:,:,:,:] # arr index 0 is Lp, index 1: Edift etc

        # lut_lp for nadir (represents relazi=90 deg) is repeated 7 times
        # to enable proper interpolation for off-nadir tilt cases.
        for j in arange(0, 7):
            self._lut_lp5[0,:,:,:, j,:, 0] = arr[0,:,:,:,:] # arr index 0 is Lp

        # off_nadir
        for i in arange(1, (nview)):
            for j in arange(0, 4):
                self._lut_rt5[i,:,:,:, 0,:, j] = arr[6 + (i - 1) * n_rt11 + j,:,:,:,:] # Edift, E0t, Tdr, Tdf
            self._lut_rt5[i,:,:,:, 0,:, 4] = self._lut_rt5[0,:,:,:, 0,:, 4] # ; s_offnadir=s_nadir
            # ! spha for nadir

            for j in arange(0, 7):
                self._lut_lp5[i,:,:,:, j,:, 0] = arr[6 + (i - 1) * n_rt11 + 4 + j,:,:,:,:] # Lp 0(30)180 deg

        self._relazi = self._solaz - self._phiv
        # make range of relazi between 0 and 180 degree
        if (self._relazi < 0.0):
            self._relazi = 360.0 + self._relazi
        if (self._relazi > 180.0):
            self._relazi = 360.0 - self._relazi

        view_arr = arange(nview, dtype=float32) * 10.0 # tilt angles (deg
        sza_arr = arange(self._nzen, dtype=float32) * 10.0 # deg
        phi_arr = arange(7, dtype=float32) * 30.0 # relative azimuth angles (deg)

        dim_vza = nview
        dim_sza = self._nzen
        dim_phi = 7
        dim_ele = self._n_alt
        dim_vis = self._nvis

        # more than 1 view angle
        dim_lp = array([dim_phi, dim_vis, dim_sza, dim_ele, dim_vza])
        dim_rt = array([1, dim_vis, dim_sza, dim_ele, dim_vza])

        dim_max = max(dim_lp) # (= 8 = nzen)
        ndim5 = 5
        ndim = ndim5

        # lut_max_val is range for interpol_lut5
        # the minimum must be greater than the min database value for this parameter,
        # the maximum " " less " " max database value " " "
        lut_max_val = zeros([ndim, 2], float32)
        lut_max_val[0,:] = array([0.00, 179.999]) # relazi range (deg) for interpol_lut
        lut_max_val[1,:] = array([5.00, 119.999]) # vis (km)
        lut_max_val[2,:] = array([0.00, 69.999]) # sza (deg)
        lut_max_val[3,:] = array([0.00, 2.4999]) # elevation (km)
        lut_max_val[4,:] = array([0.00, view_arr[len(view_arr) - 1] - 0.001]) # vza (deg)

        phi1 = (minimum(self._relazi, lut_max_val[0, 1]))
        vis1 = (minimum(self._visibility, lut_max_val[1, 1]))
        sza1 = (minimum(self._solze, lut_max_val[2, 1]))
        ele1 = (minimum(self._altit, lut_max_val[3, 1]))
        vza1 = (minimum(self._thv, lut_max_val[4, 1]))

        xnodes1 = zeros([dim_max, ndim], float32) # for lp
        xnodes1[0:(dim_lp[0] - 1)+1, 0] = phi_arr
        xnodes1[0:(dim_lp[1] - 1)+1, 1] = self._visarr
        xnodes1[0:(dim_lp[2] - 1)+1, 2] = sza_arr
        xnodes1[0:(dim_lp[3] - 1)+1, 3] = arange(self._n_alt, dtype=float32) * 0.5
        xnodes1[0:(dim_lp[4] - 1)+1, 4] = view_arr

        dim_max = max(dim_rt) # = 8 = nzen
        xnodes2 = zeros([dim_max, ndim], float32) # for rt
        xnodes2[0:(dim_rt[0] - 1)+1, 0] = 1
        xnodes2[0:(dim_rt[1] - 1)+1, 1] = self._visarr
        xnodes2[0:(dim_rt[2] - 1)+1, 2] = sza_arr
        xnodes2[0:(dim_rt[3] - 1)+1, 3] = arange(self._n_alt, dtype=float32) * 0.5
        xnodes2[0:(dim_rt[4] - 1)+1, 4] = view_arr

        self._xnodes5 = zeros([dim_max, ndim], float32) # (ndim=5, max 8 grid points)
        self._xnodes5 = xnodes1
        self._nnodes5 = 2 ** ndim # (2^5 = 32)
        self._x_cell5 = zeros([self._nnodes5, ndim], float32)

        cont = 0
        for i in arange(0, 2):
            for j in arange(0, 2):
                for k in arange(0, 2):
                    for ii in arange(0, 2):
                        for jj in arange(0, 2):

                            self._x_cell5[cont, 0] = i
                            self._x_cell5[cont, 1] = j
                            self._x_cell5[cont, 2] = k
                            self._x_cell5[cont, 3] = ii
                            self._x_cell5[cont, 4] = jj

                            cont = cont + 1

        self._n_rt5 = 6 # 6 RT terms (Lp, Edift, E0t, Tdr, Tdf, s) for Common LUT5
        self._nwvl5 = nrefl1

        # calculate LUTs for fixed vza, phi, sza, and n_alt=6 elevations
        # --------------------------------------------------------------
        el_arr = arange(self._n_alt, dtype=float32) * 0.5
        rt = zeros([self._n_alt, nrefl1, 6], float32) # 6 RT functions

        for iele in arange(0, (self._n_alt)):
            ele1 = (minimum((maximum(el_arr[iele], lut_max_val[3, 0])), lut_max_val[3, 1]))
            vtest = array([phi1, vis1, sza1, ele1, vza1])
            rt[iele,:,:] = self.interpol_lut5(vtest)

        rt = transpose(rt) # fltarr(n_alt, nrefl1, 6)
        self._lp = zeros([self._n_bands_all, self._n_alt], float32)
        self._edift = zeros([self._n_bands_all, self._n_alt], float32)
        self._e0t = zeros([self._n_bands_all, self._n_alt], float32)
        self._tdir = zeros([self._n_bands_all, self._n_alt], float32)
        self._tdif = zeros([self._n_bands_all, self._n_alt], float32)
        self._spha = zeros([self._n_bands_all, self._n_alt], float32)

        self._lp[0:(nrefl1 - 1)+1,:] = rt[0,:,:]
        self._edift[0:(nrefl1 - 1)+1,:] = rt[1,:,:]
        self._e0t[0:(nrefl1 - 1)+1,:] = rt[2,:,:]
        self._tdir[0:(nrefl1 - 1)+1,:] = rt[3,:,:]
        self._tdif[0:(nrefl1 - 1)+1,:] = rt[4,:,:]
        self._spha[0:(nrefl1 - 1)+1,:] = rt[5,:,:]

        self._q = maximum(self._tdif / maximum(self._tdir, 0.01), 0.0)
        # remove possible spikes in q function around
        # 750-770 nm, 910-980nm. 1110-1180 nm, 1.4-1.5 um, 1.78-2.00 um, and lambda > 2.40
        wl1 = array([0.750, 0.910, 1.110, 1.400, 1.780])
        wl2 = array([0.770, 0.980, 1.180, 1.500, 2.000])

        for j in arange(0, len(wl1)):
            li = where32(ravel(bitwise_and(self._wvlsen >= wl1[j], self._wvlsen <= wl2[j])))
            if (li.size > 0):
                li1 = where32(ravel(self._wvlsen <= wl1[j] - 0.005)) ; cnt1 = li1.size
                li2 = where32(ravel(self._wvlsen >= wl2[j] + 0.005)) ; cnt2 = li2.size
                if (cnt1 > 0 and cnt2 > 0):
                    for k in arange(0, (self._n_alt)):
                        self._q[li, k] = interpol(array([self._q[li1[cnt1 - 1], k], self._q[li2[0], k]]), array([self._wvlsen[li1[cnt1 - 1]], self._wvlsen[li2[0]]]), self._wvlsen[li])

        li = where32(ravel(self._wvlsen > 2.35))
        if (li.size > 0):
            for k in arange(0, (self._n_alt)):
                self._q[li, k] = self._q[li[0], k] * exp(self._wvlsen[li[0]] - self._wvlsen[li])

        return


    def read3v_atm_hyper_a3(self):
        # global lut_lp5, lut_rt5, n_rt5, nwvl5, xnodes5, nnodes5, ndim5, lim5, lut_cell5, x_cell5

        # read atmospheric correction functions for all altitudes and all visibilites in database
        # input: solze = solar zenith angle (deg)
        # solaz = " azimuth " (deg)
        # thv = sensor view zenith angle (deg)
        # phiv = sensor view azimuth angle (deg)
        # visib = visibility (5 <= visib <= 120 km)
        # output: lp, edift, e0t, tdir, tdif, q, spha, tsun (Common block)
        # lp=fltarr(n_alt,nvis,n_bands_all)

        # read data from .atm file for hyperspectral sensors
        # (a) nadir view: 6 RT terms: (Lp, Edift, E0t, Tdir, Tdif, s) for n_alt=6 altitudes 0(0.5)2.5 km
        # (b) nadir plus one or more off-nadir view angles (itilt > 0): requires view angle interpolation
        # structure: 6 RT for nadir (Lp, Edift, E0t Tdir, Tdif, s), and
        # 4 RT (Edift, E0t Tdir, Tdif) plus 7 Lp (7 relazi angles) for each off-nadir angle.
        # Makes 6 + (4 + 7)*n_angles = 6 + 11*n_angles terms, for n_alt=6 elevations
        # s is independent of view angle, i.e., s = s(nadir)

        import xdrlib
        self._lp = zeros([self._n_bands_all, self._nvis, self._n_alt], float32)
        self._edift = zeros([self._n_bands_all, self._nvis, self._n_alt], float32)
        self._e0t = zeros([self._n_bands_all, self._nvis, self._n_alt], float32)
        self._tdir = zeros([self._n_bands_all, self._nvis, self._n_alt], float32)
        self._tdif = zeros([self._n_bands_all, self._nvis, self._n_alt], float32)
        self._q = zeros([self._n_bands_all, self._nvis, self._n_alt], float32)
        self._spha = zeros([self._n_bands_all, self._nvis, self._n_alt], float32)

        # determine file structure (number of view angles)
        # -----------------------------------------------
        if self.config.resolution == 10:
            nrefl1 = 4 #band sentinel2 10 m
        elif self.config.resolution == 30:
            if self.config.Hyper:
                nrefl1 = 12 #band Hyper 30 m with Sentinel 2 20 m
            else:
                nrefl1 = 9 #band Landsat 30 m
        else:
            nrefl1 = 12 #band sentinel2 20 m
        n_rt6 = 6 # Lp, Edf, Edr, Tdr, Tdf, spha, 6 RT terms
        n_rt11 = 11 # Edf, Edr, Tdr, Tdf, Lp for 7 relazi angles spaced 30 deg

        # database size for nadir angle: factor 4 for float
        # the 2 accounts for vis, zen in the first 2 columns
        nbytes_nadir = (2 + asarray(n_rt6).astype(int32) * self._n_alt) * 4 * self._nzen * self._nvis * nrefl1

        # database size per off-nadir angle: factor 4 for float
        nbytes_offnad = asarray(n_rt11).astype(int32) * self._n_alt * 4 * self._nzen * self._nvis * nrefl1

        # read atm file
        fn = self._atmDataFn
        fsize = os.path.getsize(fn)
        l.acquire()
        try:
            fp = open(fn, 'rb')
            xdr = xdrlib.Unpacker(fp.read())
            fp.close()
        finally:
            l.release()
                    
        nview = 1 + int((fsize - nbytes_nadir) / nbytes_offnad)
        n1 = n_rt6 + (nview - 1) * n_rt11
        nitems = int(fsize/4)
        lut = xdr.unpack_farray(nitems, xdr.unpack_float)
        arr = asarray(lut)
        arr = arr.reshape(nrefl1, self._nvis, self._nzen, 2 + n1 * self._n_alt)
        arr = arr[:,:,:, 2:(2 + n1 * self._n_alt - 1)+1] # remove vis, zen columns
        arr = arr.reshape(nrefl1, self._nvis, self._nzen, self._n_alt, n1)
        arr = transpose(arr) # fltarr(n_refl1, nvis, nzen, n_alt, n1) for faster processing
        self._lut_lp5 = zeros([nview, self._n_alt, self._nzen, self._nvis, 7, nrefl1, 1], float32) # 7 rel. azi Lp: 0(30)180 deg
        self._lut_rt5 = zeros([nview, self._n_alt, self._nzen, self._nvis, 1, nrefl1, 5], float32) # 5 RT fcts: Edift, E0t, Tdir, Tdif, s

        # assign nadir values (Edift, E0t, Tdir, Tdif, s)
        for j in arange(0, 5):
            self._lut_rt5[0,:,:,:, 0,:, j] = arr[j + 1,:,:,:,:] # arr index 0 is Lp, index 1: Edift etc

        # lut_lp for nadir (represents relazi=90 deg) is repeated 7 times
        # to enable proper interpolation for off-nadir tilt cases.
        for j in arange(0, 7):
            self._lut_lp5[0,:,:,:, j,:, 0] = arr[0,:,:,:,:] # arr index 0 is Lp

        # off_nadir
        for i in arange(1, (nview)):
            for j in arange(0, 4):
                self._lut_rt5[i,:,:,:, 0,:, j] = arr[6 + (i - 1) * n_rt11 + j,:,:,:,:] # Edift, E0t, Tdr, Tdf
            self._lut_rt5[i,:,:,:, 0,:, 4] = self._lut_rt5[0,:,:,:, 0,:, 4] # ; s_offnadir=s_nadir
            # ! spha for nadir

            for j in arange(0, 7):
                self._lut_lp5[i,:,:,:, j,:, 0] = arr[6 + (i - 1) * n_rt11 + 4 + j,:,:,:,:] # Lp 0(30)180 deg

        self._relazi = self._solaz - self._phiv
        # make range of relazi between 0 and 180 degree
        if (self._relazi < 0.0):
            self._relazi = 360.0 + self._relazi
        if (self._relazi > 180.0):
            self._relazi = 360.0 - self._relazi

        view_arr = arange(nview, dtype=float32) * 10 # tilt angles (deg
        sza_arr = arange(self._nzen, dtype=float32) * 10.0 # deg
        phi_arr = arange(7, dtype=float32) * 30.0 # relative azimuth angles (deg)

        dim_vza = nview
        dim_sza = self._nzen
        dim_phi = 7
        dim_ele = self._n_alt
        dim_vis = self._nvis

        # more than 1 view angle
        dim_lp = array([dim_phi, dim_vis, dim_sza, dim_ele, dim_vza])
        dim_rt = array([1, dim_vis, dim_sza, dim_ele, dim_vza])

        dim_max = max(dim_lp) # (= 8 = nzen)
        ndim5 = 5
        ndim = ndim5

        # lut_max_val is range for interpol_lut5
        # the minimum must be greater than the min database value for this parameter,
        # the maximum " " less " " max database value " " "
        lut_max_val = zeros([ndim, 2], float32)
        lut_max_val[0,:] = array([0.00, 179.999]) # relazi range (deg) for interpol_lut
        lut_max_val[1,:] = array([5.00, 119.999]) # vis (km)
        lut_max_val[2,:] = array([0.00, 69.999]) # sza (deg)
        lut_max_val[3,:] = array([0.00, 2.4999]) # elevation (km)
        lut_max_val[4,:] = array([0.00, view_arr[len(view_arr) - 1] - 0.001]) # vza (deg)

        phi1 = (minimum(self._relazi, lut_max_val[0, 1]))
        vis1 = (minimum(self._visibility, lut_max_val[1, 1]))
        sza1 = (minimum(self._solze, lut_max_val[2, 1]))
        ele1 = (minimum(self._altit, lut_max_val[3, 1]))
        vza1 = (minimum(self._thv, lut_max_val[4, 1]))

        xnodes1 = zeros([dim_max, ndim], float32) # for lp
        xnodes1[0:(dim_lp[0] - 1)+1, 0] = phi_arr
        xnodes1[0:(dim_lp[1] - 1)+1, 1] = self._visarr
        xnodes1[0:(dim_lp[2] - 1)+1, 2] = sza_arr
        xnodes1[0:(dim_lp[3] - 1)+1, 3] = arange(self._n_alt, dtype=float32) * 0.5
        xnodes1[0:(dim_lp[4] - 1)+1, 4] = view_arr

        dim_max = max(dim_rt) # = 8 = nzen
        xnodes2 = zeros([dim_max, ndim], float32) # for rt
        xnodes2[0:(dim_rt[0] - 1)+1, 0] = 1
        xnodes2[0:(dim_rt[1] - 1)+1, 1] = self._visarr
        xnodes2[0:(dim_rt[2] - 1)+1, 2] = sza_arr
        xnodes2[0:(dim_rt[3] - 1)+1, 3] = arange(self._n_alt, dtype=float32) * 0.5
        xnodes2[0:(dim_rt[4] - 1)+1, 4] = view_arr

        self._xnodes5 = zeros([dim_max, ndim], float32) # (ndim=5, max 8 grid points)
        self._xnodes5 = xnodes1
        self._nnodes5 = 2 ** ndim # (2^5 = 32)
        self._x_cell5 = zeros([self._nnodes5, ndim], float32)

        cont = 0
        for i in arange(0, 2):
            for j in arange(0, 2):
                for k in arange(0, 2):
                    for ii in arange(0, 2):
                        for jj in arange(0, 2):

                            self._x_cell5[cont, 0] = i
                            self._x_cell5[cont, 1] = j
                            self._x_cell5[cont, 2] = k
                            self._x_cell5[cont, 3] = ii
                            self._x_cell5[cont, 4] = jj

                            cont = cont + 1

        self._n_rt5 = 6 # 6 RT terms (Lp, Edift, E0t, Tdr, Tdf, s) for Common LUT5
        self._nwvl5 = nrefl1

        # calculate LUTs for fixed vza, phi, sza, and n_alt=6 elevations
        # --------------------------------------------------------------
        el_arr = arange(self._n_alt, dtype=float32) * 0.5
        rt = zeros([self._n_alt, self._nvis, nrefl1, 6], float32) # 6 RT functions

        for ivis in arange(0, (self._nvis)):
            vis1 = (minimum((maximum(self._visarr[ivis], lut_max_val[1, 0])), lut_max_val[1, 1]))
            for iele in arange(0, (self._n_alt)):
                ele1 = (minimum((maximum(el_arr[iele], lut_max_val[3, 0])), lut_max_val[3, 1]))
                vtest = array([phi1, vis1, sza1, ele1, vza1])
                rt[iele, ivis,:,:] = self.interpol_lut5(vtest)

        rt = transpose(rt) # make fltarr(n_alt, nvis, nrefl1, 6) ; 6 RT fct's
        self._lp[0:(nrefl1 - 1)+1,:,:] = rt[0,:,:,:]
        self._spha[0:(nrefl1 - 1)+1,:,:] = rt[5,:,:,:]
        self._edift[0:(nrefl1 - 1)+1,:,:] = rt[1,:,:,:]
        self._e0t[0:(nrefl1 - 1)+1,:,:] = rt[2,:,:,:]
        self._tdir[0:(nrefl1 - 1)+1,:,:] = rt[3,:,:,:]
        self._tdif[0:(nrefl1 - 1)+1,:,:] = rt[4,:,:,:]

        self._q = maximum(self._tdif / maximum(self._tdir, 0.01), 0.0)
        # remove possible spikes in q function around
        # 750-770 nm, 910-980nm. 1110-1180 nm, 1.4-1.5 um, 1.78-2.00 um, and lambda > 2.40
        wl1 = array([0.750, 0.910, 1.110, 1.400, 1.780])
        wl2 = array([0.770, 0.980, 1.180, 1.500, 2.000])
        for j in arange(0, wl1.size-1):
            li = where32(ravel(bitwise_and(self._wvlsen >= wl1[j], self._wvlsen <= wl2[j])))
            if (li.size > 0):
                li1 = where32(ravel(self._wvlsen <= wl1[j] - 0.05))
                li2 = where32(ravel(self._wvlsen >= wl2[j] + 0.05))
                if (li1.size > 0 and li2.size > 0):
                    for k in arange(0, (self._n_alt)):
                        for i in arange(0, (self._nvis)):
                            self._q[li, i, k] = interpol(array([self._q[li1[li1.size - 1], i, k], self._q[li2[0], i, k]]), array([self._wvlsen[li1[li1.size - 1]], self._wvlsen[li2[0]]]), self._wvlsen[li])
        li = where32(ravel(bitwise_and(self._wvlsen > 2.35, self._wvlsen < 2.55)))
        if (li.size > 0):
            for k in arange(0, (self._n_alt)):
                for i in arange(0, (self._nvis)):
                    self._q[li, i, k] = self._q[li[0], i, k] * exp(self._wvlsen[li[0]] - self._wvlsen[li])

        self._tsun = float32(exp(log(maximum(self._tdir, 0.01)) / cos(radians(self._solze))))

        return


    def altit3_atm(self):
    # generate altitude grid of atm. corr. functions (fixed visib and zenith)
    # 1. add altitude 3.0 km for atm. functions by extrapolation (if necessary)
    # 2. add altitude 3.5 km for atm. functions by extrapolation (if necessary)
    # 3. interpolate atm. fct's in 0-3.5 km region to 0.02 km grid
    #
    # Input (from read_atm_hyper_a3)
    # lp = fltarr(n_alt, n_bands_all), same for edift, e0t etc
    #
    # Output:
    # 1. arrays lph, edifth, e0th, tdirh, tdifh, tsunh, qh
    # lph = fltarr(n_alti, n_bands_all) (n_alti covers 0 - 3.5 km in 0.02 km steps)

        self.read_atm_hyper_a3() # (ihyper=1 for wide FOV)

        # interpolate on a 0.1 km grid: (n_alti-1)*5+1=n_alti1=26 range 0-2.5 km
        # arange(26*0.2, dtype=float32) = arange(26, dtype=float32)*0.1 * (5/2.5) ; index 5 at 2.5 km
        # n_alti1 = number of height layers from 0 to 2.5 km (100m or 20m grid)
        scf = 0.1 * (5 / 2.5) # 100 m grid
        if (self._n_alti1 > 100):
            scf = 0.02 * (5 / 2.5) # 20 m grid has min(n_alti1)=126

            x = arange(self._n_bands_all)
            y = arange(self._n_alti1, dtype=float32) * scf
            self._lph = rectBivariateSpline(x, y, self._lp)
            self._edifth = rectBivariateSpline(x, y, self._edift)
            self._e0th = rectBivariateSpline(x, y, self._e0t)
            self._tdirh = rectBivariateSpline(x, y, self._tdir)
            self._tdifh = rectBivariateSpline(x, y, self._tdif)
            self._qh = rectBivariateSpline(x, y, self._q)
            self._sphah = rectBivariateSpline(x, y, self._spha)

        # extrapolate if necessary (0.1 km grid or 0.02 km grid, up to 3.5 km)
        if (self._n_alti > self._n_alti1):
            n = self._n_alti - self._n_alti1

            # delta_h = 0.5 km in database, extrapolation from 2.5 km to 3.0 and 3.5 km is index 0.0 to 2.0
            if (self._n_alti1 < 100):
                dh1 = 0.1
            else:
                dh1 = 0.02 # (km)
            in0 = 2.0 / n
            hg0 = zeros([1, n], float)
            hg0[0,:] = in0 + arange(n, dtype=float32) * dh1 / 0.5
            hg = hg0
            for j in arange(1, (self._n_bands_all)):
                hg = concatenate((hg, hg0), axis=0) # repeat for all bands
            hg = hg.T
            
            self._lph1 = reshape(self._lp[:, self._n_alt - 2], -1)
            self._lph2 = reshape(self._lp[:, self._n_alt - 1], -1)
            a0 = zeros([1, self._lph1.size], float)
            a0[0,:] = self._lph1
            b0 = zeros([1, self._lph2.size], float)
            b0[0,:] = self._lph2
            a = a0
            b = b0
            for j in arange(1, (n)):
                a = concatenate((a, a0), axis=0)
            for j in arange(1, (n)):
                b = concatenate((b, b0), axis=0)
            x = b + (b - a) * hg
            self._lph = concatenate((self._lph, transpose(x)), axis=1) # concatenate 0-2.5 and > 2.5 km values

            self._edifth1 = reshape(self._edift[:, self._n_alt - 2], -1)
            self._edifth2 = reshape(self._edift[:, self._n_alt - 1], -1)
            a0[0,:] = self._edifth1
            b0[0,:] = self._edifth2
            a = a0
            b = b0
            for j in arange(1, (n)):
                a = concatenate((a, a0), axis=0)
            for j in arange(1, (n)):
                b = concatenate((b, b0), axis=0)
            x = b + (b - a) * hg
            self._edifth = concatenate((self._edifth, transpose(x)), axis=1) # concatenate 0-2.5 and > 2.5 km values

            self._e0th1 = reshape(self._e0t[:, self._n_alt - 2], -1)
            self._e0th2 = reshape(self._e0t[:, self._n_alt - 1], -1)
            a0[0,:] = self._e0th1
            b0[0,:] = self._e0th2
            a = a0
            b = b0
            for j in arange(1, (n)):
                a = concatenate((a, a0), axis=0)
            for j in arange(1, (n)):
                b = concatenate((b, b0), axis=0)
            x = b + (b - a) * hg
            self._e0th = concatenate((self._e0th, transpose(x)), axis=1) # concatenate 0-2.5 and > 2.5 km values

            self._tdirh1 = reshape(self._tdir[:, self._n_alt - 2], -1)
            self._tdirh2 = reshape(self._tdir[:, self._n_alt - 1], -1)
            a0[0,:] = self._tdirh1
            b0[0,:] = self._tdirh2
            a = a0
            b = b0
            for j in arange(1, (n)):
                a = concatenate((a, a0), axis=0)
            for j in arange(1, (n)):
                b = concatenate((b, b0), axis=0)
            x = b + (b - a) * hg
            self._tdirh = concatenate((self._tdirh, transpose(x)), axis=1) # concatenate 0-2.5 and > 2.5 km values

            self._tdifh1 = reshape(self._tdif[:, self._n_alt - 2], -1)
            self._tdifh2 = reshape(self._tdif[:, self._n_alt - 1], -1)
            a0[0,:] = self._tdifh1
            b0[0,:] = self._tdifh2
            a = a0
            b = b0
            for j in arange(1, (n)):
                a = concatenate((a, a0), axis=0)
            for j in arange(1, (n)):
                b = concatenate((b, b0), axis=0)
            x = b + (b - a) * hg
            self._tdifh = concatenate((self._tdifh, transpose(x)), axis=1) # concatenate 0-2.5 and > 2.5 km values
            
            self._sphah1 = reshape(self._spha[:, self._n_alt - 2], -1)
            self._sphah2 = reshape(self._spha[:, self._n_alt - 1], -1)
            a0[0,:] = self._sphah1
            b0[0,:] = self._sphah2
            a = a0
            b = b0
            for j in arange(1, (n)):
                a = concatenate((a, a0), axis=0)
            for j in arange(1, (n)):
                b = concatenate((b, b0), axis=0)
            x = b + (b - a) * hg
            self._sphah = concatenate((self._sphah, transpose(x)), axis=1) # concatenate 0-2.5 and > 2.5 km values
            
            self._qh1 = reshape(self._q[:, self._n_alt - 2], -1)
            self._qh2 = reshape(self._q[:, self._n_alt - 1], -1)
            a0[0,:] = self._qh1
            b0[0,:] = self._qh2
            a = a0
            b = b0
            for j in arange(1, (n)):
                a = concatenate((a, a0), axis=0)
            for j in arange(1, (n)):
                b = concatenate((b, b0), axis=0)
            x = b + (b - a) * hg
            self._qh = concatenate((self._qh, transpose(x)), axis=1) # concatenate 0-2.5 and > 2.5 km values
        # sun-to-ground beam transmittance
        self._tsunh = float32(exp(log(maximum(self._tdirh, 0.01)) / cos(radians(self._solze))))

        return


    def altit3v_atm(self):

    # get atm. functions for all altitudes, all visibilities
    # n_alt = 6 (6 altitudes in database 0, 0.5, 1, 1.5, 2, 2.5 km)
    # nvis = 8 (std vis. set: 5, 7, 10, 15, 23, 40, 80, 120 km)
    # n_bands = number of spectral bands
    # nvisx = extended set of visibilities (see array visext, main program)
    #
    # 1. read data from database: altitudes 0-2.5 km, std vis.set
    # --> dimension of arrays is arr(n_alt, nvis, n_bands)
    # 2. add elevations 3.0, 3.5 km --> arr(n_alt+2,nvis, n_bands) (if necessary)
    # 3. interpolate to 20 m grid --> arr(n_alti, nvis, n_bands)
    # 4. interpolate for visext --> arr(n_alti, nvisx,n_bands)
    # ------------------------------------------------------------
    #
    # Input (from read3v_atm)
    # lp= fltarr(n_alt,nvis,n_bands_all), also edift, e0t etc
    #
    # Output:
    # 1. arrays lph, edifth, e0th, tdirh, tdifh, tsunh, qh (for routine "rho_retrieval_step")
    # lph = fltarr(n_alti, nvisx, n_bands_all) (n_alti covers 0 - 3.5 km)
    # 2. arrays lpx, ediftx, e0tx, tdirx, tdifx, tsunx (for routine "ref_pixel3")
    # lpx = fltarr(n_alti, nvis, n_bands_all) (n_alti covers 0 - 3.5 km with 20 m grid)


    #; IF (ihyper EQ 0) THEN read3v_atm $ ; calls read_phasefct3 for ms sensors
    #; ELSE read3v_atm_hyper_a3 ; ihyper=1 for wide FOV

        self.read3v_atm_hyper_a3() # ihyper=1 for wide FOV
        # interpolate on a 0.1 km grid: (n_alt-1)*5+1=n_alt1=26 range 0-2.5 km
        # n_alti = 36 (max) range 0=3.5 km, 0.1 km grid
        # update: 20 m grid, i.e. n_alti1 = 126, n_alti up to 176

        self._lpx = zeros([self._n_bands_all, self._nvis, self._n_alti], float32)
        self._ediftx = zeros([self._n_bands_all, self._nvis, self._n_alti], float32)
        self._e0tx = zeros([self._n_bands_all, self._nvis, self._n_alti], float32)
        self._tdirx = zeros([self._n_bands_all, self._nvis, self._n_alti], float32)
        self._tdifx = zeros([self._n_bands_all, self._nvis, self._n_alti], float32)
        self._tsunx = zeros([self._n_bands_all, self._nvis, self._n_alti], float32)
        self._qx = zeros([self._n_bands_all, self._nvis, self._n_alti], float32)
        self._sphax = zeros([self._n_bands_all, self._nvis, self._n_alti], float32)

        self._lph = zeros([self._n_bands_all, self._nvisx, self._n_alti], float32)
        self._edifth = zeros([self._n_bands_all, self._nvisx, self._n_alti], float32)
        self._e0th = zeros([self._n_bands_all, self._nvisx, self._n_alti], float32)
        self._tdirh = zeros([self._n_bands_all, self._nvisx, self._n_alti], float32)
        self._tdifh = zeros([self._n_bands_all, self._nvisx, self._n_alti], float32)
        self._tsunh = zeros([self._n_bands_all, self._nvisx, self._n_alti], float32)
        self._qh = zeros([self._n_bands_all, self._nvisx, self._n_alti], float32)
        self._sphah = zeros([self._n_bands_all, self._nvisx, self._n_alti], float32)

        for ibnd in arange(0, (self._n_bands_all)):
            for j in arange(0, (self._nvis)):
                # step 1
                # ------
                yy = self._lp[ibnd, j,:]
                if (self._elmax > 2500):
                    yy = extrapl(yy) # 3.0 km
                if (self._elmax > 3000):
                    yy = extrapl(yy) # 3.5 km

                # step 2
                # ------
                res = interpol1d(yy, self._n_alti)
                self._lpx[ibnd, j,:] = res

                # same 2 steps for edift
                yy = self._edift[ibnd, j,:]
                if (self._elmax > 2500):
                    yy = extrapl(yy) # 3.0 km
                if (self._elmax > 3000):
                    yy = extrapl(yy) # 3.5 km
                res = interpol1d(yy, self._n_alti)
                self._ediftx[ibnd, j,:] = res

                # same 2 steps for e0t
                yy = self._e0t[ibnd, j,:]
                if (self._elmax > 2500):
                    yy = extrapl(yy) # 3.0 km
                if (self._elmax > 3000):
                    yy = extrapl(yy) # 3.5 km
                res = interpol1d(yy, self._n_alti)
                self._e0tx[ibnd, j,:] = res

                # same 2 steps for tdir
                yy = self._tdir[ibnd, j,:]
                if (self._elmax > 2500):
                    yy = extrapl(yy) # 3.0 km
                if (self._elmax > 3000):
                    yy = extrapl(yy) # 3.5 km
                res = interpol1d(yy, self._n_alti)
                self._tdirx[ibnd, j,:] = res

                # same 2 steps for tdif
                yy = self._tdif[ibnd, j,:]
                if (self._elmax > 2500):
                    yy = extrapl(yy) # 3.0 km
                if (self._elmax > 3000):
                    yy = extrapl(yy) # 3.5 km
                res = interpol1d(yy, self._n_alti)
                self._tdifx[ibnd, j,:] = res

                # same 2 steps for tsun
                yy = self._tsun[ibnd, j,:]
                if (self._elmax > 2500):
                    yy = extrapl(yy) # 3.0 km
                if (self._elmax > 3000):
                    yy = extrapl(yy) # 3.5 km
                res = interpol1d(yy, self._n_alti)
                self._tsunx[ibnd, j,:] = res

                # same 2 steps for q
                yy = self._q[ibnd, j,:]
                if (self._elmax > 2500):
                    yy = extrapl(yy) # 3.0 km
                if (self._elmax > 3000):
                    yy = extrapl(yy) # 3.5 km
                res = interpol1d(yy, self._n_alti)
                self._qx[ibnd, j,:] = res

                # same 2 steps for spha
                yy = self._spha[ibnd, j,:]
                if (self._elmax > 2500):
                    yy = extrapl(yy) # 3.0 km
                if (self._elmax > 3000):
                    yy = extrapl(yy) # 3.5 km
                res = interpol1d(yy, self._n_alti)
                self._sphax[ibnd, j,:] = res

            # step 3: interpolation for extended vis.set
            # -----------------------------------------------------
            for j in arange(0, (self._n_alti)):
                yy = self._lpx[ibnd,:, j]
                res = interpol(reverse(yy), self._depth, self._depthx)
                self._lph[ibnd,:, j] = res
                yy = self._ediftx[ibnd,:, j]
                res = interpol(reverse(yy), self._depth, self._depthx)
                self._edifth[ibnd,:, j] = res

                yy = self._e0tx[ibnd,:, j]
                res = interpol(reverse(yy), self._depth, self._depthx)
                self._e0th[ibnd,:, j] = res

                yy = self._tdirx[ibnd,:, j]
                res = interpol(reverse(yy), self._depth, self._depthx)
                self._tdirh[ibnd,:, j] = res

                yy = self._tdifx[ibnd,:, j]
                res = interpol(reverse(yy), self._depth, self._depthx)
                self._tdifh[ibnd,:, j] = res

                yy = self._qx[ibnd,:, j]
                res = interpol(reverse(yy), self._depth, self._depthx)
                self._qh[ibnd,:, j] = res

                yy = self._sphax[ibnd,:, j]
                res = interpol(reverse(yy), self._depth, self._depthx)
                self._sphah[ibnd,:, j] = res
        res = 0
        self._sphax = 0 # currently not used in ref_pixel !

        # beam transmittance sun-to-ground
        self._tsunh = float32(exp(log(maximum(self._tdirh, 0.01)) / cos(radians(self._solze))))

        return


    def load_sensor_a3_hs(self):

        self._iabs_region = zeros(5, dtype=uint16) #array(4, int)
        self._wvlsen = self.config.wvlsen
        self._fwhm = self.config.fwhm
        self._es = self.config.e0
        self._c0 = self.config.c0
        self._c1 = self.config.c1

        # the old array nch_wv is obsolete: if wv correction is set then all wavelength regions
        # have to be corrected for wv effects even the tiny absorptions at 590, 650, 1080 nm etc!
        self._reflter = zeros([self._n_bands_all], float32) + self.config.AC_Terrain_Refl_Start # default terrain reflectance (range 0-1)

        self._nadj = int(self._adj_km * 1000.0 / self.config.pixelsize + 0.5)
        if (self._nadj % 2 == 0):
            self._nadj = self._nadj + 1

        # the old array nch_wv is obsolete, replaced by band_index_wvdepend
        self._band_index_wvdepend = zeros(self._n_bands_all) # default: channel does not depend on water vapor
        # is updated in routine "wv_regions_940..."

        return


    def write_visindex_file_a3(self, bandvis):
        # write visindex map to file
        # final number of output bands
        scl = self.getClassificationMap()
        bandvis[scl <= self.config.saturatedDefective] = 0

        self.tables.setBand(Band.VISIBILITY, bandvis) #self.tables.VIS
        # write AOT (aerosol optical thickness) file (not for TIFF input file)
        # --------------------------------------------------------------------
        a_aot = interpol(self._a_aot_fit, self._h_grid_aot, arange(self._n_alti)) # vector of a_aot_fit(z)
        b_aot = interpol(self._b_aot_fit, self._h_grid_aot, arange(self._n_alti))

        # low pass filter only with 100m because of rugged terrain

        n1 = max([int(100./self.config.pixelsize), 3])
        n2 = min([21, self.config.ncols/2, self.config.nrows/2])
        nf = min([n1, n2])
        if (nf % 2 == 0):
            nf = nf + 1

        # smooth is performed because the vis.index of bandvis is in discrete integer steps
        # (has nothing to do with adjacency effect)
        # JL: smooth replaced by gaussian more adapted to large kernels
        bandvis[scl == 0] = bandvis[scl > 0].mean()
        aot = float32(gaussian_filter(exp(a_aot[self._eclass] + b_aot[self._eclass] * log(maximum(self._visext[bandvis], 1.0))), nf))
        mask = zeros_like(scl)
        mask[(scl == self.config.water ) | (scl == self.config.bareSoils) | (scl == self.config.vegetation)] = 1
        self._scene_av_aot = aot[mask>0].mean()

        aot_uint16 = asarray(1000.0 * aot + 0.5).astype(uint16)
        aot_uint16[scl <= self.config.saturatedDefective] = 0
        self.tables.setBand(Band.AEROSOL_OPTICAL_THICKNESS, aot_uint16) #self.tables.AOT
        return


    def mask_veget_swir_a3(self, cnt1):

    # Aerosol DDV algorithm: masking of DDV pixels and assignment of
    # surface reflectance relationship red/SWIR bands
    #
    # mask vegetation (DDV pixels) with 2.2 micron or 1.6 micron band
    # 2.2 micron: pixels with refl < 5 %
    # 1.6 micron: pixels with refl < 10 %
    # If number of ref. pixels less than th_percent=2% of image pixels then use
    # the thresholds: refl(2.2 um) < 10 % and refl(1.6 um) < 15 %
    # finally try: refl(2.2 um) < 12 % and refl(1.6 um) < 24 %
    #
    # Calculate reflectance in RED band:
    # refl(red) = reflref_vect = ratio_red_swir*refl(2.2 um) if 2.2 um band exists else
    # refl(red) = reflref_vect = ratio_red_swir*refl(1.6 um)
    # refl(blue)= ratio_blu_red*refl(red) = ratio_blu_red*reflref_vect
    # Default ratio_red_swir = 0.5 (2.2 um), ratio_red_swir=0.25 (1.6 um)
    # Water pixels are excluded with criterion NDVI > 0.1
    #
    # The relationship of blue-to-red band DDV pixels is treated in "scale_path_radiance"
    # refl(blue)= ratio_blu_red*refl(red) = ratio_blu_red*reflref_vect
    #
    # Input: cnt1 = number of non-background pixels
    # all others: from Common blocks

        th_percent = self.config.AC_Min_Ddv_Area # threshold for minimum required DDV scene pixels
        
        # fix for SIIMPC-1154 - UMW begin:
        # TBD: check for index!

        if 'LANDSAT' in self.config.spacecraftName:
            index = self.tables.reindex(Band.SHORT_WAVE_INFRARED_2) -1
        else :
            index = self.tables.reindex(Band.SHORT_WAVE_INFRARED_2)
        if (self._lph.ndim == 2):
            lpy = reshape(self._lph[index, self._eclass], (self.config.nrows, self.config.ncols))
            e0ty = reshape(self._e0th[index, self._eclass], (self.config.nrows, self.config.ncols))
            edifty = reshape(self._edifth[index, self._eclass], (self.config.nrows, self.config.ncols))
        else:
            # visindex corresponding to visib=23.0 km
            index2 = (minimum((maximum(indexvis(23., self._nvisx, self._vis_reg), 1)), (self._nvisx - 1)))
            lpy = reshape(self._lph[index, index2, self._eclass], (self.config.nrows, self.config.ncols))
            e0ty = reshape(self._e0th[index, index2, self._eclass], (self.config.nrows, self.config.ncols))
            edifty = reshape(self._edifth[index, index2, self._eclass], (self.config.nrows, self.config.ncols))

        bswir2_rad = self.tables.getBand(Band.SHORT_WAVE_INFRARED_2, radiance=True)
        ref = pi * (self.config.d2 * bswir2_rad - lpy) / (e0ty * cos(radians(self._solze)) + edifty)
        # fix for SIIMPC-1154 - UMW end.

        if (self._np_clshad > 0):
            ravel(ref)[self._mlist_clshad] = 0.0

        if (self._cntback > 0):
            ravel(ref)[self._liback] = 0.0

        if self.logger.level == logging.DEBUG: print((statistics(ref, 'S1: ref')))

        # exclude medium and high thickness cirrus pixels
        if (self.cirrus_correction):
            level = 1 # medium and thick cirrus pixels
            list_cirrus = self.read_hcw_file_get_cirrus(level)
            if (list_cirrus.size > 0):
                ravel(ref)[list_cirrus] = 0.0

        if (self._np_cloud > 1):
            # Exclude ref.pixels close to cloud. The following filter size (nf) and threshold (5)
            # remove ref.pixels closer than about 500m to cloud areas from the ref.pixel list.
            maskc = zeros([self.config.nrows, self.config.ncols], uint8)
            ravel(maskc)[self._mlist_cloud] = 255
            if (self._np_cloudw > 1):
                ravel(maskc)[self._mlist_cloudw] = 255
            #nf = min(int(array(1000. / self.config.pixelsize)), 500)
            nf = min([int(1000. / self.config.pixelsize), 500])  # correction for NumPy 2.1.3
            if (nf > (min([self.config.nrows / 2, self.config.ncols / 2]))):
                nf = min([self.config.nrows / 2, self.config.ncols / 2])
            if (nf % 2 == 0):
                nf = nf + 1

            maskc = smooth(maskc, nf, edge_truncate=True)
            list_cloud1 = where32(ravel(maskc > 5))
            if (list_cloud1.size > 0):
                ravel(ref)[list_cloud1] = 0.0

        if self.logger.level == logging.DEBUG: print((statistics(ref, 'S2: ref')))

        sc = self.config.dnScale
        sc = 1.0
        thre = self.config.AC_Ddv_Swir_Refl_Th1 * sc
        thr3 = self.config.AC_Ddv_16um_Refl_Th1 * sc
        thr4 = 0.38 * sc
        thr5 = self.config.AC_Swir_Refl_Lower_Th * sc

        #ref = int32(ref * sc + 0.5)

        if 'LANDSAT' in self.config.spacecraftName:
            index = self.tables.reindex(Band.NEAR_INFRARED) -1 #  different list
        else:
            index = self.tables.reindex(Band.VEGETATION_4) #  different list
        if (self._lph.ndim == 2):
            lpy = reshape(self._lph[index, self._eclass], (self.config.nrows, self.config.ncols))

            self._e0ty = reshape(self._e0th[index, self._eclass], (self.config.nrows, self.config.ncols))
            self._edifty = reshape(self._edifth[index, self._eclass], (self.config.nrows, self.config.ncols))

        else:
            index2 = (minimum((maximum(indexvis(23., self._nvisx, self._vis_reg), 1)), (self._nvisx - 1)))
            # visindex corresponding to visib=23.0 km
            lpy = reshape(self._lph[index, index2, self._eclass], (self.config.nrows, self.config.ncols))

            self._e0ty = reshape(self._e0th[index, index2, self._eclass], (self.config.nrows, self.config.ncols))
            self._edifty = reshape(self._edifth[index, index2, self._eclass], (self.config.nrows, self.config.ncols))

        if 'LANDSAT' in self.config.spacecraftName:
            band_nir_radiance = self.tables.getBand(Band.NEAR_INFRARED, radiance=True)
        else:
            band_nir_radiance = self.tables.getBand(Band.VEGETATION_4, radiance=True)
        rho4 = pi * (self.config.d2 * band_nir_radiance - lpy) / (self._e0ty * cos(radians(self._solze)) + self._edifty)
        del self._edifty
        del self._e0ty

        # remark: the topographic eq. with fluxpix, fluxm, fluxter does not yield better results for the DDV mask
        if self.logger.level == logging.DEBUG: print((statistics(rho4, 'S3: rho4')))
        sc1 = 1.0/float(self._sc_bet)

        # threshold tx for cbeta, exclude DDV pixels with cbeta < tx, or beta > solze_max
        cbeta_max = self._cbeta.max()
        solze_max = 75.0
        thr_c1 = cos(radians(solze_max))
        thr_c2 = fix(cos(radians(solze_max)) * self._sc_bet) # threshold for scaled byte ilu
        cbeta_ori = cos(radians(self.config.solze_arr.mean())) * self._sc_bet
        if cbeta_max > 1.0:
            tx = thr_c2
        else:
            tx = thr_c1 # byte_scaled or float case

        if self.logger.level == logging.DEBUG: print(('S4:  cbeta_max, tx =', cbeta_max, tx))  # No external water map:

        # The internal 'image_out_hcw.bsq'tends to include some water pixels as DDV,
        # therefore it is not used here (only if copied to 'image_hcw.bsq').
        # (examples: spot4_ruegen, etm_2000_07_30 Bordeaux).
        # Employ threshold NDVI > 0.1 to exclude water pixels (these have NDVI < 0.1).
        # The threshold ref3red < thr3=0.10 for the red band excludes snow for the case
        # of 1.6 um sensors where rho(1.6um,snow) < 5% is a dark target similar to
        # vegetation.

        band_red_reflectance = self.tables.getBand(Band.RED)
        if 'LANDSAT' in self.config.spacecraftName:
            band_nir_reflectance = self.tables.getBand(Band.NEAR_INFRARED)
        else:
            band_nir_reflectance = self.tables.getBand(Band.VEGETATION_4)

        ndvi = (band_nir_reflectance - band_red_reflectance) / \
               maximum((band_nir_reflectance + band_red_reflectance), 0.01)

        CM = self.getClassificationMap()
        vegetation = [CM == self.config.vegetation]
        CM = None

        # Selection of reference pixels for DDV algorithm
        self._mlistref = where32(
            ravel(
                (ref > thr5)
                & (ref < thre)
                & (band_red_reflectance <= thr3)
                & (ndvi > self.config.AC_Swir_Refl_Ndvi_Th)
                & (rho4 < thr4)
                & (cbeta_ori > tx)
                & vegetation
            )
        )

        if self.logger.level == logging.DEBUG:
            print((statistics(ref, 'S5: ref')))
            print((statistics(ndvi, 'S5: ndvi')))
            print((statistics(rho4, 'S5: rho4')))
            print('')
            print(('S5: cbeta_ori', cbeta_ori))
            print('')
            print((statistics(band_red_reflectance, 'S5: B04_refl')))

        # np_water = number of water pixels, calculated previously
        # fix for SIIMPC-880, UMW: mode for constant visibility selection, impLemented in 2.4.0
        if self._npref > 0:
            self._npref = self._mlistref.size
            if self.logger.level == logging.DEBUG:
                print(('S5: npref', self._npref))
        else: # modus for constant visibility:
            self.logger.info('constant visibility was selected, no iteration performed')
            return

        # at least th_percent % of image pixels (without water pixels) should be reference pixels

        if self.logger.level == logging.DEBUG:
            testarr = zeros([self.config.nrows, self.config.ncols], uint8)
            ravel(testarr)[self._mlistref] = 1
            showImage(testarr)

        # fix for SIIMPC-957, UMW: return if pure water scene
        #cnt2 = max(cnt1 - self._np_water, 1)
        cnt2 = max([cnt1 - self._np_water, 1])  # correction for NumPy 2.1.3
        self._ddv_pixel_percentage = self._npref * 100.0 / cnt2 # is put into "*.log"
        if self._np_water > 0.99 * cnt1:
            self.logger.info('more than 99% of the scene is water, thus no DDV iteration will be performed')
            self._ddv_pixel_percentage = 0
            self._npref = 0
            return

        if self.logger.level == logging.DEBUG:
            print('percentage of DDV at start of iteration %f: ' % (self._ddv_pixel_percentage))

        if self._ddv_pixel_percentage > th_percent:
            # reflref_vect is vector of reflectance of reference pixels in RED band (650 nm)
            self._reflref_vect = float32(self._ratio_red_swir * ravel(ref)[self._mlistref] * 100/sc) # percent unit
            if self.logger.level == logging.DEBUG:
                print('this is above threshold.')
        else:
            if self.logger.level == logging.DEBUG:
                print('increase threshold (0.10 / 0.15) to find more dark pixels ...')
            if Band.SHORT_WAVE_INFRARED_2:
                thre = self.config.AC_Ddv_Swir_Refl_Th2 * sc
            else:
                thre = self.config.AC_Ddv_16um_Refl_Th2 * sc

            self._mlistref = where32(
                ravel(
                    (ref > thr5)
                    & (ref < thre)
                    & (band_red_reflectance <= thr3)
                    & (ndvi > self.config.AC_Swir_Refl_Ndvi_Th)
                    & (rho4 < thr4)
                    & (cbeta_ori > tx)
                    & vegetation
                )
            )

            self._npref = self._mlistref.size
            try:
                self._ddv_pixel_percentage = self._npref * 100.0 / cnt1  # /cnt2 JL: ddv percentage on clear land pixels except snow
            except:
                self._ddv_pixel_percentage = 0
            if (self._ddv_pixel_percentage > th_percent):
                self._reflref_vect = float32(self._ratio_red_swir * ravel(ref)[self._mlistref] * 100) # percent unit
                self._thre_refl_swir = thre
                self._ddv_reflectance_range = self._thre_refl_swir
                if self.logger.level == logging.DEBUG:
                    testarr = zeros([self.config.nrows, self.config.ncols], uint8)
                    ravel(testarr)[self._mlistref] = 1
                    showImage(testarr)
            else:
                if self.logger.level == logging.DEBUG:
                    print('increase threshold (0.12 / 0.18) to find more dark pixels ...')
                if Band.SHORT_WAVE_INFRARED_2:
                    thre = self.config.AC_Ddv_Swir_Refl_Th3 * sc
                else:
                    thre = self.config.AC_Ddv_16um_Refl_Th3 * sc

                self._mlistref = where32(
                    ravel(
                        (ref > thr5)
                        & (ref < thre)
                        & (band_red_reflectance <= thr3)
                        & (ndvi > self.config.AC_Swir_Refl_Ndvi_Th)
                        & (rho4 < thr4)
                        & (cbeta_ori > tx)
                        & vegetation
                    )
                )

                self._npref = self._mlistref.size
                try:
                    self._ddv_pixel_percentage = self._npref * 100.0 / cnt1  # /cnt2 JL: ddv percentage on clear land pixels except snow
                except:
                    self._ddv_pixel_percentage = 0
                if (self._ddv_pixel_percentage > th_percent):
                    self._reflref_vect = self._ratio_red_swir * ravel(ref)[self._mlistref] * 100 # percent unit
                    self._thre_refl_swir = thre
                    self._ddv_reflectance_range = self._thre_refl_swir
                    if self.logger.level == logging.DEBUG:
                        testarr = zeros([self.config.nrows, self.config.ncols], uint8)
                        ravel(testarr)[self._mlistref] = 1
                        showImage(testarr)
                else:
                    self._npref = 0
                    self._ddv_reflectance_range = self._thre_refl_swir #0
                    if self.logger.level == logging.DEBUG:
                        print('not enough dark pixels found, constant visibility is used.')
                    # (end of 0.12 / 0.18 threshold)
                # (end of 0.10 / 0.15 threshold)
            # (end of 0.05 / 0.10 threshold)
        return

    def calcapda(self, reference_cube, measurement_cube):

    # PURPOSE:
    # Program to calculate atmospheric precorrected differential
    # absorption, using linear regression (APDA ratio)
    #
    # INPUTS:
    # reference_cube: cube of reference channels
    # measurement_cube: cube of measurement (absorption) channels
    #
    # OUTPUTS:
    # ratio array of apda
    #

    # calculation only where valid data is found
        index = where32(ravel(reference_cube[0,:,:] > 0))  #; exclude zeros from calculation (avoid divide by 0)

        apdaarr1d = zeros([self._n_lines * self._n_pixels], float32)
        reference_cube_1d = ravel(reference_cube)
        measurement_cube_1d = ravel(measurement_cube)

        if index.size > 0:
            apdaarr1d[index] = measurement_cube_1d[index] / reference_cube_1d[index]

        apdarr = reshape(apdaarr1d, array([self._n_lines, self._n_pixels]))
        return apdarr

    #-----------------------------------------------------------------------
    #+
    # NAME:
    # APDA_QUALUT
    #
    # PURPOSE:
    # returns the coefficients of an exponential interpolation through
    # the points of an array and chooses between several Interpolation
    # Functions
    #
    # INPUTS:
    # lut: LUT as created within APDA
    #
    # KEYWORD PARAMETERS:
    # group: the widget display is used (heritable);
    #
    # OUTPUTS:
    # result: array with the following values:
    # a(0): multiplicative constant
    # a(1): exponential constant
    # a(2): evtl. exponent of u
    #
    #
    # PROCEDURE:
    # This Function interpolates a minimum of 3 Quantification Values
    # for Trace gases following the formula given by Frouin,1990:
    #
    # R=a(0)*exp(-a(1)*sqrt(u) )
    #
    # or the one by Chris Borel:
    # R= exp(-a(1)*(u)^a(2)
    #
    # or one with three Parameters:
    # R=a(0)*exp(-a(1)*(u)^a(2)
    #
    # which means the measured ratio R to the trace gas amount u
    #
    #
    # EXAMPLE:
    # to interpolate the values of a lut
    # Result = QUAPOL(lut,interpol)
    # the variable interpol can get the values 'froexp', 'potexp',
    # or 'triexp'.
    #
    #
    # MODIFICATION HISTORY:
    # 18.6.2003: rewrite for operational APDA by Daniel Schlaepfer
    # July 2010: enhanced version, R. Richter
    #-
    #----------------------------------------------------------------------------

    # first the used interpolation function and their partial derivative
    # are defined to be used with the IDL-Curvefit-Function


    def apda_qualut(self, group=None, two=None, rng=None):
        from scipy.optimize import curve_fit

        def froexp(x, a, b):
            # interpolation function following Frouin
            # R= a*exp(-b u^(1/2))
            f = a * exp(-b * sqrt(x))
            #pder = array([array([exp(-b * sqrt(x))]), array([(a * (-sqrt(x) * exp(-b * sqrt(x))))])])
            return f

        def triexp(x, a, b, c):
            # interpolation function with unknown exponent
            # and 3 independent Variables
            f = a * exp(-b * (x ** c))
            #pder = array([array([exp(-b * ((x) ** c))]), array([-a * (x ** c) * exp(-b * x ** c)]), array([-a * b * (x ** c * log(x)) * exp(-b * (x ** c))])])
            return f

        if self._lutconv.size < 3:
            self.logger.fatal('wrong dimensioning of LUT')

        if(rng is None or rng.size != 2):
            rng = array([self._lutconv[0, 0, 0, 0] - 100, self._lutconv[0, self._n_fact - 1, 0, 0] + 1000]) # min and max wv in atmfile for h=0
        # for the standard MS: range = [ 400-100, 4000+1000 ] = [300, 5000]
        # IF n_elements(range) NE 2 then $
        # range=[400,3000]

        # transform the LUT to the final appearance
        lutcor = zeros([self._n_bands, self._n_fact, self._hlevels], float32)
        lutcor[:,:,:] = self._lutconv[1:self._n_bands+1, 0:self._n_fact,:, 0] - self._lutconv[1:self._n_bands+1, 0:self._n_fact,:, 1]
        measurement_cube = zeros([self._anz_mea, self._n_fact, self._hlevels], float32)
        for i in arange(0, (self._anz_mea)):
            index = where32(ravel(self._measure_ch[i] == self._wvlarr[:, 0]))
            measurement_cube[i,:,:] = lutcor[index,:,:]

        #; transform reference channels:
        reference_cube = zeros([self._anz_ref, self._n_fact, self._hlevels], float32)
        for i in arange(0, (self._anz_ref)):
            index = where32(ravel(self._reference_ch[i] == self._wvlarr[:, 0]))
            reference_cube[i,:,:] = lutcor[index,:,:]

        n_pix_temp = self._n_pixels
        n_lin_temp = self._n_lines
        self._n_pixels = self._hlevels
        self._n_lines = self._n_fact

        lutrat = self.calcapda(reference_cube, measurement_cube)
        self._n_pixels = n_pix_temp
        self._n_lines = n_lin_temp

        #; setting index for first altitude level
        #; rangeind = long([0, 6, 12, 18]) & n_range = 4 ; old version, restricted to nwv=4
        n_range = self._n_fact # = nwv = number of water vapor grid points in database (RR)
        rangeind = arange(n_range, dtype=int16) * self._hlevels # [0, 6, 12, 18] for nwv=4, hlevels=6, i.e. 0(500)2500 m
        # [0, 8, 16, 24, 32] for nwv=5, hlevels=8, i.e. 0(500)3500 m
        x = zeros([n_range], float32)
        y = x

        if (two is not None):
            b = array([1.0, 0.01])
            # routine froexp:
            def func(x, a0, b0):
                return froexp(x, a0, b0)
        else:
            b = array([1, 0.01, 0.5])
            # routine triexp:
            def func(x, a0, b0, c0):
                return triexp(x, a0, b0, c0)

        outpars = zeros([self._hlevels, b.size], float32)

        for hh in arange(0, (self._hlevels)):
            x = self._lutconv[0,:, hh, 0]
            y = lutrat[:, hh]
            #; reducing weight of out of range points
            #; settihg the weights
            w = ones([n_range], float32)
            offrange = where32(ravel(bitwise_or(x < rng[0], x > rng[1])))
            if offrange.size > 0:
                w[offrange] = w[offrange] / 5.

            # attention!!! curve fit needs 64 bit floats, it fails with float 32 !!!
            popt, pcov = curve_fit(func, x.astype(float), y.astype(float), p0=b, sigma=w)
            outpars[hh,:] = popt
            rangeind = rangeind + 1

        return outpars

    #-----------------------------------------------------------------------
    def atmprecor(self, w_pos, h_pos, group=None, iter=None):

    # PURPOSE:
    # Program to calculate atmospheric precorrection
    # INPUTS:
    # w_pos: relative position in water wapor amount for
    # each pixel with respect to LUT
    # h_pos: relative position in altitude for each pixel
    # with respect to LUT
    # reference_cube: reference channels
    # measurement_cube: measurement channels
    # reference_ch: channel numbers of reference bands (Common apdavar)
    # measure_ch: channel numbers of measurement bands " "
    # optional keywords: group: widget group
    # iter: number of iterations
    #
    # This Procedure subtracts the pathradiance from the raw-image
    # under consideration of water vapor amount (w_pos) and height (h_pos)

    # calculate position of the pathradiance in the array

        from scipy.ndimage import map_coordinates
        for i in arange(0, (self._anz_ref)):
            interarr = zeros([self._n_fact, self._hlevels], float32)
            channel_position = self._reference_ch[i]
            #; search channel in LUT
            interarr[:,:] = self._lutconv[channel_position, 0:self._n_fact,:, 1] # get Lp for all n_alt, 0:nwv-1
            #; extract 2D pathrad at channel position and subtract from
            coordinates = array([w_pos, h_pos])
            ipr1 = map_coordinates(interarr, coordinates, order=1, mode='nearest')
            self._reference_cube[i,:,:] = self._reference_cube[i,:,:] - ipr1

            #; preserve original state during iteration
            if (iter is not None):
                coordinates = array([self._w_pos_it, h_pos])
                ipr2 = map_coordinates(interarr, coordinates, order=1, mode='nearest')
                self._reference_cube[i,:,:] = self._reference_cube[i,:,:] + ipr2

        for i in arange(0, (self._anz_mea)):
            interarr = zeros([self._n_fact, self._hlevels], float32)
            channel_position = self._measure_ch[i]
            interarr[:,:] = self._lutconv[channel_position, 0:self._n_fact,:, 1] # Lp(n_alt,0:n_wv) all 6 alti, all 4 wv
            coordinates = array([w_pos, h_pos])
            ipm1 = map_coordinates(interarr, coordinates, order=1, mode='nearest')
            self._measurement_cube[i,:,:] = self._measurement_cube[i,:,:] - ipm1
            if (iter is not None):
                coordinates = array([self._w_pos_it, h_pos])
                ipm2 = map_coordinates(interarr, coordinates, order=1, mode='nearest')
                self._measurement_cube[i,:,:] = self._measurement_cube[i,:,:] + ipm2

        self._w_pos_it = w_pos.copy()

        return

    #-----------------------------------------------------------------------
    def apda_iter(self, n_iter, h_pos, dem, group=False, two=False):

    # APDA calculation and iteration
    # INPUTS:
    # n_iter: number of iterations required
    # h_pos: relative position in height levels of LUT in each pixel
    # and Common apdavar
    #
    # OUTPUTS:
    # water vapor contents
    #
    # calculate transformation coefficients
    # -------------------------------------------------

        coef = self.apda_qualut(group, two)
        w_pos = ones([self._n_lines, self._n_pixels], float32)
        hposr = ravel(h_pos)
        wposr = ravel(w_pos)

        # Calculate Correction and APDA and one iteration
        # -------------------------------------------------------------
        self.atmprecor(w_pos, h_pos)
        ratim = ravel(self.calcapda(self._reference_cube, self._measurement_cube))

        #; convert the ratio to water vapor
        apdaarr = zeros([self._n_lines * self._n_pixels], float32)
        ind = where32(ravel(bitwise_and(ratim > 0.0, ratim <= 1.0)))
        outliers = where32(ravel(bitwise_or(ratim <= 0.0, ratim > 1.0)))
        #; evaluation for 1 pixel might be required in spectra module
        if (ind.size < 2):
            self.logger.info('problematic tile detected, reset scaled water vapor map to W=1000')
            return array(apdaarr+1000, dtype=int32) # apdaarr = wv scaled with 1000, stored as 16 bit integer

        if (two is not None):
            coef_0 = linear_interpolation(coef[:, 0], hposr)
            coef_1 = linear_interpolation(coef[:, 1], hposr)

            apdaarr[ind] = float32(((log(ratim[ind]) - log(coef_0[ind])) / (-coef_1[ind])) ** 2) #; froexp
        else:
            coef_0 = linear_interpolation(coef[:, 0], hposr)
            coef_1 = linear_interpolation(coef[:, 1], hposr)
            coef_2 = linear_interpolation(coef[:, 2], hposr)

            apdaarr[ind] = float32(((log(ratim[ind]) - log(coef_0[ind])) / (-coef_1[ind])) ** (1 / coef_2[ind])) #; triexp

        # goto second iteration
        # -----------------------------------------------------------------

        for n_it in arange(2, (n_iter)+(1)):

        # calculating range and update coefficients
        # -----------------------------------------
            maxwv = max(apdaarr) ; minwv = min(apdaarr)
            coef1 = self.apda_qualut(group, two, array([minwv, maxwv]))
            if coef1.size > 0:
                coef = coef1

            #; calculate w_pos (water vapor position index for interpolation,
            #; updated with each iteration)
            #; --------------------------------------------------------------
            #; more than one altitude level hlevels=n_alt+2
            for j in arange(0, (self._hlevels- 1)): # hlevels=n_alt+2
                indh = where32(ravel(bitwise_and(dem >= self._heights[j], dem < self._heights[j + 1])))
                if indh.size > 0:
                    wvrange = self._lutconv[0,:, j, 0] # index 0: wv, altit j, all wv values
                    # j=0: e.g. wv_range=[400, 1000, 2000, 2900]
                    for i in arange(0, (self._n_fact- 1)): # n_fact = 4, 5, or 6 = n_wv levels
                        index = where32(ravel(bitwise_and(apdaarr[indh] >= wvrange[i], apdaarr[indh] < wvrange[i + 1])))
                        if index.size > 0:
                            wposr[indh[index]] = i + (apdaarr[indh[index]] - wvrange[i]) / (wvrange[i + 1] - wvrange[i])
                    index = where32(ravel(apdaarr[indh] < wvrange[0]))
                    if index.size > 0:
                        wposr[indh[index]] = 0
                    #; extrapolate?
                    index = where32(ravel(apdaarr[indh] >= wvrange[self._n_fact - 1]))
                    if index.size > 0:
                        wposr[indh[index]] = self._n_fact - 1 + (apdaarr[indh[index]] - wvrange[self._n_fact - 1]) / (wvrange[self._n_fact - 1] - wvrange[self._n_fact - 2])

            siz = w_pos.shape
            nf = min([10, min(siz)/2]) # nf must be smaller than min(n_rows,n_cols), correction for NumPy 2.1.3
            w_pos = reshape(wposr, array([siz[0], siz[1]]))

            if (self._n_lines > 100):
                w_pos = smooth(w_pos, nf, edge_truncate=True) # average of wposr

            self.atmprecor(w_pos, h_pos, iter=True) # update (L-Lp) for ref and mea channels
            ratim = ravel(self.calcapda(self._reference_cube, self._measurement_cube)) # APDA ratio of channels
            ind = where32(ravel(bitwise_and(ratim > 0, ratim <= 1)))

            #; transform to water vapor contents
            if ind.size > 0:
                if (two is not None):
                    coef_0 = linear_interpolation(coef[:, 0], hposr)
                    coef_1 = linear_interpolation(coef[:, 1], hposr)
                    apdaarr[ind] = float32(((log(ratim[ind]) - log(coef_0[ind])) / (-coef_1[ind])) ** 2) #; froexp
                else:
                    coef_0 = linear_interpolation(coef[:, 0], hposr)
                    coef_1 = linear_interpolation(coef[:, 1], hposr)
                    coef_2 = linear_interpolation(coef[:, 2], hposr)
                    apdaarr[ind] = float32(((log(ratim[ind]) - log(coef_0[ind])) / (-coef_1[ind])) ** (1 / coef_2[ind])) #; triexp

        if(outliers.size > 0):
            apdaarr[outliers] = 0

        return array(apdaarr, dtype=int32) # apdaarr = wv scaled with 1000, stored as 16 bit integer

    #-----------------------------------------------------------------------
    def atap_lut(self, rt_all, nbnds, wavel):

    # PURPOSE:
    # generate a lookup-table to be used with the iterative APDA
    # with ATCOR atm files
    #
    # INPUTS:
    # rt_all = fltarr(4,n_alt,n_wv,nbnds)
    # 4 fcts = {Lp, Edift, E0t, spha}
    # n_alt= 6, hlevels=n_alt+2
    # n_wv = water vapor grid points (4 - 6)
    # nbnds: number of spectral bands
    # u500m: water vapor columns for n_wv values and n_alt+2 height levels
    # (wv=400, 1000, 2000, 2900, 4000) and heights 0(500)3500 m
    # sz solar zenith angle (degrees)
    # albedo: estimated average albedo of the image
    #
    # OUTPUTS:
    # hlevels: indicates the height levels to be used, starting
    # from the bottom level (default: 8)
    # a binary dataset containing the Look-Up-Table for iterative
    # APDA calculations of the dimension:
    # [2: albedo=0.3 (or specified albedo) and albedo=0
    # n_alt: number of height levels (=6) in database (height range 0-2500m)
    # n_wv: 4 to 6 default water vapor concentrations at each height level
    # depending on the number of '*wv*.atm'files
    # band_index+1: all channels + one layer containing the column water vapor ]
    #
    #
    # MODIFICATION HISTORY:
    # January, 1996: apdalut created by Daniel Schlaepfer RSL/LANL
    # Nov, 2007: adapted to ATCOR, RR
    # Feb. 2008 enhanced to account for off-nadir view angles in the .atm database (R. Richter)
    # Aug. 2010 use rt_all array calculated previously instead of .atm database (R. Richter)

    #; initialize
        albedo = 0.3

        siz = array(rt_all.shape) # lp_fit=fltarr(ncf,n_alti,n_bands)
        n_wv = siz[1] # water vapor grid points (4 - 6)
        n_alt = siz[2] # (6 altitudes in database 0, 0.5, 1, 1.5, 2, 2.5 km)

        self._heights = arange(n_alt + 2) * 500 # 0(500)3500 m
        self._hlevels = self._heights.size

        #; create LUTs from the atm file
        #; assign output LUT
        outlut = zeros([nbnds + 1, n_wv, self._hlevels, 2], float32) #; 2 alb values, hlevels=n_alt+2, wv,nbnds+1
        # hlevels = 8, i.e., 0(500)3500m
        # (with extrapolation)

        outlut[1:,:, 0:(n_alt - 1)+1, 1] = rt_all[:,:,:, 0] # Lp for all bands
        outlut[1:,:, 0:(n_alt - 1)+1, 0] = rt_all[:,:,:, 0] + albedo / pi * (rt_all[:,:,:, 2] *
                 cos(self._solze /self._radeg) + rt_all[:,:,:, 1]) / (1. - (albedo - 0.15) * rt_all[:,:,:, 3])

        # extrapolate to 3000 and 3500 m
        # a(2500m) + a(2500m) - a(2000m) = 2*a(2500m) - a(2000m)
        outlut[1:,:, n_alt, 1] = 2 * outlut[1:,:, n_alt - 1, 1] - outlut[1:,:, n_alt - 2, 1]
        outlut[1:,:, n_alt, 0] = 2 * outlut[1:,:, n_alt - 1, 0] - outlut[1:,:, n_alt - 2, 0]
        # extrapolate to 3500 m
        outlut[1:,:, n_alt + 1, 1] = 2 * outlut[1:,:, n_alt, 1] - outlut[1:,:, n_alt - 1, 1]
        outlut[1:,:, n_alt + 1, 0] = 2 * outlut[1:,:, n_alt, 0] - outlut[1:,:, n_alt - 1, 0]


        # write the trace gas contents to first array in wavelength dimension
        # -------------------------------------------------------------------
        # outlut corresponds to uu1_altit in load_commons of ATCOR, but only in 500 m steps not 100 m steps
        # these values are stored in u500m = fltarr(n_wv,n_alt+2) (0-3500m)

        outlut[0,:,:, 0] = self._u500m
        outlut[0,:,:, 1] = rebin(self._heights, n_alt + 2, n_wv)
        return outlut

    #-----------------------------------------------------------------------
    def wv_retrieval_apda2_constvis(self, rt_all, subset):
    # INPUTS:
    # measure_ch: the numbers of the measurement bands, given as
    # vector, starting with number 1 as first band!
    # reference_ch: the numbers of the reference bands, given as vector
    # vector, starting with number 1 as first band!
    # rt_all = fltarr(4,n_alt,n_wv,n_bands)
    # 4 fcts = {Lp, Edift, E0t, spha}
    # u500m: water vapor columns for n_wv values and n_alt+2 height levels
    # (wv=400, 1000, 2000, 2900, 4000) and heights 0(500)3500 m
    # sz: solar zenith angle (degrees)
    #
    # KEYWORD PARAMETERS:
    # demfile: filename of the DEM or altitude [m] of surface
    # cal_fact: calibration factor applied to the data for
    # distribution
    # two: frouin's two parameter approximation is taken
    # instead of three of them
    # subset: [first_col, n_cols, first_row, n_rows] subset file coordinates (starting from 1)
    # liback list of geocoded background pixels
    # icirrus = 0, 1 (no, yes)
    # t945 two-way transmittance sun-cirrus-sensor, 945 nm band of Sentinel-2,
    # t945=1 for non-Sentinel-2
    # rhoc apparent cirrus reflectance image (1.38 micron)
    # gamma two-way transmittance sun-cirrus-sensor (1.38 um), from scatterplot red/1.38 um bands
    # esol solar irradiance for each channel
    #
    # OUTPUTS:
    # columnar water vapor content as signed 16 bit integer, scaled with 1000,
    # e.g. value 250 corresponds to content 0.250 cm
    # structure of APDA routines:
    #
    # wv_retrieval_apda2_constvis
    # !- atap_lut (read .atm file)
    # !- apda_iter (iteration on wv)
    # !- atmprecor (Lpath)
    # !- apda_qualut (exp. interpolation of LUT)
    # !- calcapda (apda ratio)

        #; search for 'real'measurement and reference bands:
        # Initialisation
        # number of performed iterations (usually 2)
        n_iter = 2
        dc = self._c0
        cal_fact = self._c1
        wvl = self._wvlsen
        t945 = self._trwv945
        rhoc = self._rho_cir_app
        gamma = self._gamma_cir
        keep_wpos = False
        sz = self._solze

        li = where32(ravel(wvl < 2.8))  # n_bands = only reflective bands
        nbands = li.size
        wavel = wvl[li]

        #; create LUT for this scene, all relevant bands in
        #; reflective range.
        #; hlevels = 6 (height levels in LUT database)
        self._lutconv = self.atap_lut(rt_all, nbands, wavel)
        self._n_fact = self._lutconv[0,:, 0, 0].size # n_wv number of wv levels (4 - 6)
        self._n_bands = self._lutconv[:, 0, 0, 0].size - 1
        self._heights = self._lutconv[0, 0,:, 1] # 0(500)3500 meter

        #----------------------------------------------------------------------
        # read in the wavelengths stored in a segment containing one column of the
        # characteristics
        #

        self._wvlarr = zeros([self._n_bands, 2], float32)
        self._wvlarr[:, 0] = arange(self._n_bands) + 1
        self._wvlarr[:, 1] = wavel

        # check of allowed values of reference_ch and measure_ch is partially done in ATCOR
        n_mea = self._measure_c.size
        #; take only values of measure_c between first and last wavelength (for
        #; detector transition range...
        index = where32(ravel(bitwise_and(wavel[self._measure_c - 1] >= \
                wavel[self._measure_c[0] - 1], wavel[self._measure_c - 1] <= \
                wavel[self._measure_c[n_mea - 1] - 1])))
        if index[0] == -1:
            self.logger.fatal('invalid measurement band numbers')

        self._measure_ch = self._measure_c[index]
        self._anz_mea = self._measure_ch.size
        self._reference_ch = self._reference_c
        self._anz_ref = self._reference_ch.size
        self._n_pixels = self.config.ncols
        self._n_lines = self.config.nrows

        self._datyp = 0

        DEM_OK = False
        # read the DEM in [m] and its relative position in the heightarray
        # -----------------------------------------------------------------
        if(self.tables.hasBand(Band.DIGITAL_ELEVATION_MAP) == False): #self.tables.DEM
            DEM_OK = False
        else:
            n_rows, n_cols, count = self.tables.getBandSize(self.tables.DEM, resampled=True) #self.tables.DEM
            if n_rows != self.config.nrows or n_cols != self.config.ncols:
                self.logger.fatal('incompatibable DEM dimensions')
                DEM_OK = False
            else:
                DEM_OK = True

        if(DEM_OK == False):
            surfaceheight = self._heights[0]
            nrows = self.config.nrows
            ncols = self.config.ncols
            dem = zeros([nrows, ncols], float32) + surfaceheight
        else:
            #dt = self.tables.getDataType(self.tables.DEM)
            dem = clip(self.tables.getBand(Band.DIGITAL_ELEVATION_MAP), 0, 3500) #self.tables.DEM

        # take DEM subset if required, adapt n_pixels, n_lines
        if (subset.size == 4):
            # self._n_lines = (subset[1] - subset[0] + 1)
            # self._n_pixels = (subset[3] - subset[2] + 1)
            # dem = dem[subset[0] - 1:(subset[1] - 1)+1, subset[2] - 1:(subset[3] - 1)+1]
            self._n_lines = int(subset[1] - subset[0] + 1)
            self._n_pixels = int(subset[3] - subset[2] + 1)
            dem = dem[int(subset[0] - 1): int((subset[1] - 1) + 1), int(subset[2] - 1): int((subset[3] - 1) + 1)]

        # read the image data, include subset option
        # measurement bands
        self._measurement_cube = zeros([self._anz_ref, self._n_lines, self._n_pixels], dtype=float32)

        # account for cirrus
        if (self.cirrus_correction):
            rhoc1 = rhoc
            # rhoc1 is needed in case of a subset: the original rhoc is needed in the calling routine
            if (subset.size == 4):
                # rhoc1 = rhoc1[subset[0] - 1:(subset[1] - 1)+1, subset[2] - 1:(subset[3] - 1)+1]
                rhoc1 = rhoc1[int(subset[0] - 1): int((subset[1] - 1) + 1), int(subset[2] - 1): int((subset[3] - 1) + 1)]

        for i in arange(0, (self._anz_mea)):
            dn = self.tables.getBand(self._measure_ch[i]-1) #band is now read from 0-index
            if (subset.size == 4):
                # dn = dn[subset[0] - 1:(subset[1] - 1)+1, subset[2] - 1:(subset[3] - 1)+1]
                dn = dn[int(subset[0] - 1): int((subset[1] - 1) + 1), int(subset[2] - 1): int((subset[3] - 1) + 1)]
            if (self.cirrus_correction):
                rho_app = dn
                rho_app = rho_app / t945 - rhoc1 / gamma
                # calculate cirrus subtracted DN
                dn = (rho_app * self._es[self._measure_ch[i] - 1] * cos(sz * self._dtor) / \
                (pi * self.config.d2) - dc[self._measure_ch[i] - 1]) / cal_fact[self._measure_ch[i] - 1]
            else:
                dn = (dn * self._es[self._measure_ch[i] - 1] * cos(sz * self._dtor) / \
                (pi * self.config.d2) - dc[self._measure_ch[i] - 1]) / cal_fact[self._measure_ch[i] - 1]

            self._measurement_cube[i,:,:] = dc[self._measure_ch[i] - 1] + dn * cal_fact[self._measure_ch[i] - 1]

        #; reference bands
        self._reference_cube = zeros([self._anz_ref, self._n_lines, self._n_pixels], float32)
        for i in arange(0, (self._anz_ref)):
            dn = self.tables.getBand(self._reference_ch[i]-1) #band is now read from 0-index
            if (subset.size == 4):
                # dn = dn[subset[0] - 1:(subset[1] - 1)+1, subset[2] - 1:(subset[3] - 1)+1]
                dn = dn[int(subset[0] - 1): int((subset[1] - 1) + 1), int(subset[2] - 1): int((subset[3] - 1) + 1)]
            if (self.cirrus_correction):
                rho_app =  dn
                rho_app = rho_app - rhoc1 / gamma

                # calculate cirrus subtracted DN
                dn = (rho_app * self._es[self._reference_ch[i] - 1] * cos(sz * self._dtor) / \
                (pi * self.config.d2) - dc[self._reference_ch[i] - 1]) / cal_fact[self._reference_ch[i] - 1]
            else:
                dn = (dn * self._es[self._reference_ch[i] - 1] * cos(sz * self._dtor) / \
                (pi * self.config.d2) - dc[self._reference_ch[i] - 1]) / cal_fact[self._reference_ch[i] - 1]

            self._reference_cube[i,:,:] = dc[self._reference_ch[i] - 1] + dn * cal_fact[self._reference_ch[i] - 1]
        dn = 0

        #; prepare for iteration - doubles the data amount (!)
        #;raw_ref = reference_cube
        #;raw_mea = measurement_cube

        # calculate the relative height-position of each pixel
        # within the LUT (no meters,
        # ranges from 0 to n_height-levels
        # ---------------------------------------------------------------
        h_pos1d = zeros((self._n_lines * self._n_pixels), float32)
        dem1d = ravel(dem)
        for i in arange(0, (self._hlevels - 1)):
            index = where32(ravel(bitwise_and(dem >= self._heights[i], dem < self._heights[i + 1])))
            if(index.size > 0):
                h_pos1d[index] = float32(i) + (dem1d[index] - self._heights[i]) / (self._heights[i + 1] - self._heights[i])

        h_pos = reshape(h_pos1d, array([self._n_lines, self._n_pixels]))
        #; do iterative APDA calculation
        #; -----------------------------
        apdaarr = self.apda_iter(n_iter, h_pos, dem, group=None, two=None)

        # do not modify very low wv values (high mountains)
        # very high wv values might indicate problems (e.g. strange TOA reflectance/radiance spectra)
        band = self.tables.getBand(Band.BLUE) #self.tables.B02
        # liback = where32(ravel(band[subset[0] - 1:(subset[1] - 1)+1, subset[2] - 1:(subset[3] - 1)+1] <= 0))  # list of background pixels
        liback = where32(ravel(band[int(subset[0] - 1): int((subset[1] - 1) + 1), int(subset[2] - 1): int((subset[3] - 1) + 1)] <= 0)) # list of background pixels
        apdaarr[liback] = 0

        if(keep_wpos == False):
            self._w_pos_it = 0

        return apdaarr

    #@profile
    def masking_a3(self):
    # --------------------------------------------------------
    # The masking routine first calculates mlist_haze (list of haze pixels)
    # The list of cloud pixels (mlist_cloud) is taken from the (VEGA) preclassification.
    # The list of DDV reference pixels is calculated in mask_veget_swir .
    #
    # The haze masking and correction consists of 9 steps.
    # Steps 1 - 7 are performed in "masking"
    # Steps 8 - 9 are performed in "haze_pixel"
    # --------------------------------------------------------
    # 1. mask clear scene with "inverted" tass.cap" (reduced image)
    # 2. calculate "clear line" (reduced image)
    # 3. calculate HOT image (reduced image)
    # 4. haze mask
    # Options
    # (a) hot > mean-0.5 *sigma: standard large haze mask (ihot_mask=2)
    # (b) hot > mean: compact smaller haze mask (ihot_mask=1)
    # 5. expand haze mask generously (reduced image)
    # 6. resize haze mask to full image size
    # remove water and cloud pixels from haze mask

    #
    # Function masking_a3_skip1() replaces the initial ATCOR GOTO SKIP1 statement

        # background pixels from geocoding have bandm1=0
        # list of background pixels calculated in dtm_flat or dtm_array
        self._np_water = 0
        i_hcw = 0
        a = self.getClassificationMap()
        if(a.size > 0):
            self._mlist_haze = array(0)
            self._np_haze = 0
            self._mlist_shad = where32(ravel([a == 2]))
            self._np_shadow = self._mlist_shad.size
            self._mlist_clshad = where32(ravel([a == 3]))
            self._np_clshad = self._mlist_clshad.size
            self._mlist_clear = where32(ravel([(a == 4) | (a == 5)]))
            self._np_clear = self._mlist_clear.size
            self._mlist_water = where32(ravel([a == 6]))
            self._np_water = self._mlist_water.size
            # comment_JL on mlist_cloud:  a==10 cirrus has been added as cirrus could disturb adjacency correction.  a==7 is kept for now in the cloud list.
            self._mlist_cloud = where32(ravel([(a == 7) | (a == 8) | (a == 9) | (a == 10)]))
            self._np_cloud = self._mlist_cloud.size
            self._mlist_cloudw = array(0)
            self._np_cloudw = 0
            mlist_snow = where32(ravel([a == 11]))
            self._np_snow = mlist_snow.size

            if (self.cirrus_correction):
                self.calc_rho_cir_app()

        self.masking_a3_skip1()
        return

    def masking_a3_skip1(self):
        if (self._n_bands == 4):
            return # 4 S2 bands with 10m resolution

        self._np_water = self._mlist_water.size

        if 'LANDSAT' not in self.config.spacecraftName:
            if self.cirrus_correction and self.config.iwaterwv == 1:
                # Switch off cirrus detection and removal if W(average) < 0.6 cm.
                # Check W for a small subset (500x500 pixels) in the center of the image.
                # This also reduces the influence of smile, i.e., smile effect is neglected here!
                self.wv_regions_940_1130_a3() # sets reference_c, measure_c
                self.apda1_lut_constvis_a3()

                rt_all = zeros([self._n_refl, self._nuu1, self._n_alt, 4], float32) # 4 fcts={Lp, edift, e0t, spha}
                for j in arange(0, (self._n_alt)):
                    rt_all[:,:, j, 0] = self._lp_all[:, j * 25,:] # index j*25 uses the 500m grid instead of the 20m grid
                    rt_all[:,:, j, 1] = self._edift_all[:, j * 25,:]
                    rt_all[:,:, j, 2] = self._e0t_all[:, j * 25,:]
                    rt_all[:,:, j, 3] = self._spha_all[:, j * 25,:]

                self._u500m = transpose(self._uu1_altit[arange(self._n_alt + 2) * 5, 0:(self._nuu1 - 1)+1]) # wv 0(500)3500 m
                col1 = self._first_col + (self.config.ncols - self._first_col) / 4
                col2 = (minimum((col1 + 500), self.config.ncols))
                lin1 = self._first_row + (self.config.nrows - self._first_row) / 4
                lin2 = (minimum((lin1 + 500), self.config.nrows))
                subs = array([lin1, lin2, col1, col2])
                wvs = self.wv_retrieval_apda2_constvis(rt_all, subs)
                # UMW: bug in original ATCOR? average should exclude background, fixed.
                self._wv_av = int16(wvs[wvs>0].mean())
                #if self._wv_av < self._wv_thr_cirrus:
                #    self._icirrus = 0 # (wv below threshold)

        # ref. pixels are masked with 1.6 or 2.2 micron band (dark vegetation)
        # -------------------------------------------------------------------
        # get non-background pixels:
        list_non_background = where32(ravel(self.tables.getBand(Band.SCENE_CLASSIFICATION) > self.config.saturatedDefective))
        self.mask_veget_swir_a3(list_non_background.size) # the non-background pixels are required
        # as input to calculate percentage of ref. pixels
        # mlistref = list of ref. pixels is now available

        # Remove cloud shadow pixels from reference pixel map.
        # Works only if cloud shadow map is not modified interactively.
        # Sequence: masking_cloud_shadow_init, etc, then masking, then ref_pixel,
        # then "interactive thresholding".

        if (self._npref > 0 and self._np_clshad > 0):
            x1 = zeros([self.config.nrows * self.config.ncols], dtype=uint8) # array is initialized with zero
            x1[self._mlistref] = 1
            x1[self._mlist_clshad] = 0 # remove cloud shadow pixels from ref.pixel map
            x2 = zeros([self.config.nrows * self.config.ncols], dtype=float32) # to update reflref_vect
            x2[self._mlistref] = self._reflref_vect
            self._mlistref = where32(ravel(x1 == 1))  # update mlistref
            self._npref = self._mlistref.size
            self._reflref_vect = x2[self._mlistref] # update reflref_vect
        return

    def interp1_a3(self, refl0, npr_sec):
    # Input:
    # refl0 = reflref_vect = vector of DDV reference pixel reflectances, length npr_sec
    # reflvis = fltarr(npr_sec,nvis) matrix of reflectance vectors for nvis visibilities
    # (Common block)
    # npr_sec = number of DDV pixels
    #
    # Output:
    # vis.index vector for the DDV pixels
    #
    # calculate vis.index by interpolating the optical thickness values
    # obtained for the reference pixels
    # -------------------------------------------------------------

        if self._mlistref.shape != (): # check the case when interp1_a3() is called by DBV algorithm and _mlistref has been reset to 0
            ml = self._mlistref.copy()
        else:
            ml = self._mlistref_dbv.copy() # DBV algorithm, ml should be a 1D vector between 5 and 10 elements

        ivpix = zeros([npr_sec], uint8) + 255 # 255 marks pixels that were not treated
        # these 'strange'pixel are finally assigned the mean vis.index

        reflv0 = self._reflvis[0,:] # vis=5 km --> visindex=nvisx-1
        li1 = where32(ravel(reflv0 > refl0))
        if (li1.size > 0):
            ivpix[li1] = uint8(self._nvisx - 1)
            # fix for SIIMPC-998, UMW: ml is always 0:
            # ml[li1] = -1 # remove processed pixels from list

        self._depthinc = self._depthx[1] - self._depthx[0]
        dr = reverse(self._depth)

        reflv0 = self._reflvis[self._nvis - 1,:] # rho for vis = 120 km
        li1 = where32(ravel(bitwise_and(ml > 0, reflv0 < refl0)))
        if (li1.size > 0):
            ivpix[li1] = 0 # default visindex (corresponds to visib=190 km)

            # refl0 = reflref_vect is vector of reflectances of DDV reference pixels

            # now calculate if pixels occur in the visibility range 120-190 km, i.e., visext[0:3]
            # (compare vis_reg definition in load_commons)
            # gradient in AOT from 80 to 120 km = gradient from 120 to 190 km
            # refl(190km) = refl(120km) + (refl(120km) - refl(80km))
            # 120km = visarr[nvis-1], 80km = visarr[nvis-2]
            reflv1 = reflv0 + (refl0 - self._reflvis[self._nvis - 2,:])
            dr190 = self._depthx[0] ; dr120 = self._depth[0] # AOT for visib=190km, and 120 km
            vis1 = 120.0
            vis2 = 240.0
            #ind1 = asarray(where32(ravel(self._vis_reg[:,1] <= vis2))).min()  # not used
            #ind2 = asarray(where32(ravel(self._vis_reg[:,1] <= vis1))).min()  # not used

            list1 = where32(ravel(self._visext > 122.0))# not used  # (small margin to 120 km) list1=[0, 1, 2, 3] currently
            for i in arange(0, (list1[list1.size - 1])+(1)):  # (i.e. i=0, 3 for current visext)
                li1 = where32(ravel(bitwise_and(bitwise_and(ml > 0, refl0 >= reflv0), refl0 <= reflv1)))
                if (li1.size > 0):
                    vv = ((dr120 + (dr190 - dr120) * (refl0[li1] - reflv0[li1]) / (reflv1[li1] - reflv0[li1])) - self._depthx[0]) / self._depthinc + 0.5
                    li2 = where32(ravel(bitwise_and(vv > 0, vv < 3)))
                    if (li2.size > 0):
                        ivpix[li1[li2]] = uint8(vv[li2])
                        ml[li1[li2]] = -1 # remove processed pixels from list

        # remaining visibilities
        # -----------------------
        for j in arange(1, (self._nvis)):

            reflv0 = self._reflvis[j - 1,:]
            reflv1 = self._reflvis[j,:]
            vis1 = self._visarr[j - 1]
            vis2 = self._visarr[j]
            ind1 = asarray(where32(ravel(self._vis_reg[:, 1] <= vis2))).min()
            ind2 = asarray(where32(ravel(self._vis_reg[:, 1] <= vis1))).min()

            for i in arange(ind1, (ind2)+(1)):
                li1 = where32(ravel(bitwise_and(bitwise_and(ml > 0, refl0 >= reflv0), refl0 <= reflv1)))
                if (li1.size > 0):
                    vv = ((dr[j - 1] + (dr[j] - dr[j - 1]) * (refl0[li1] - reflv0[li1]) / (reflv1[li1] - reflv0[li1])) - self._depthx[0]) / self._depthinc + 0.5
                    li2 = where32(ravel(bitwise_and(vv > 0, vv < self._nvisx - 0.5)))  # check valid range
                    if (li2.size > 0):
                        ivpix[li1[li2]] = uint8(vv[li2])
                        ml[li1[li2]] = -1 # remove processed pixels from list

        li1 = where32(ravel(ivpix == 255))
        if (li1.size > 0):
            li2 = where32(ravel(ivpix < 255))  # replace 'strange'pixels with mean vis.index
            if (li2.size > 0):
                ivpix[li1] = uint8(ivpix[li2].mean() + 0.5)
            else:
                ivpix[li1] = indexvis(abs(self._visibility), self._nvisx, self._vis_reg)
            # the "if (cnt2 gt 0)" is necessary if no pixels are assigned to a vis.index
            # i.e. ELSE part, all pixels have no vis.index assignment: all pixels have ivpix=255b
            # In this case the input (original) visibility is assigned to those pixels

        return ivpix

    #----------------------------------------------------------------------------
    def scale_path_radiance_wfov(self):

    # Second part of the aerosol DDV algorithm if a blue band exists
    # Surface reflectance rho(blu) = ratio_blu_red*rho(red) (default=0.5)
    # If no blue band exists, but a green band:
    # rho(green) = 1.3*rho(red)
    # Assumption yields reasonable results, tested with SPOT4/5 scenes in Europe + Africa
    #
    # Purpose:
    # Scale path radiance for bands in the blue-to-red region if it deviates more than 3%
    # from the path radiance of the selected standard (LibRadtran) aerosol.
    # The path radiance in the blue band is calculated by subtracting the reflected DDV radiance
    # from the total at-sensor radiance. The surface reflectance rho(blue) is obtained from the
    # spectral correlation rho(blue) = fact * rho(red) (default fact=0.5).
    # Then the scene-derived ratio of Lpath(blue)/Lpath(red) (mean for all dark ref.pixels) is
    # compared to the corresponding ratio for the selected model aerosol.
    # Note:
    # To speed up execution and avoid a cell loop the evaluation is performed for the near-nadir
    # view angles, i.e., VZA <= 10 degrees. If the corresponding number of ref.pixels is less than
    # a quarter of the original ref.pixels, then do not modify the path radiances and return.
    #
    # Input: npref, mlistref, meanvi, reflref_vect (for red band, %)
    # Output: re-scaled path radiance lph in blue-to-red bands, if blue_band > 0
    # " green-to-red bands, if blue_band = 0
    #

    # Attention: path radiance lph is modified for bands 1 to red_band
    # make sure routine altitv_atm (which calculates lph from database) is not called later !
    # ( call of altitv_atm in variable_atm only if no blue band band exists)
        self._iscale_path = -1
        # reflref_vect is reflectance of ref.pixels in red band (percent unit)
        # extract reflref_vect for ref.pixels in the nadir region (VZA < 10 degrees)
        mask = zeros([self.config.nrows*self.config.ncols], float32)
        mask[self._mlistref] = self._reflref_vect # ref.pixels for complete scene
        #mask[arange(mlistref.shape[0])[:,None], reflref_vect]

        # define subset with VZA < 10 deg
        mask = reshape(mask, [self.config.nrows, self.config.ncols])
        mlistref = self._mlistref
        npRef1 = float32(mlistref.size)
        # fix for SIIMPC-892 UMW: making stop criteria independent on tile size, new configuration parameters for AC:
        # AC_Spr_Refl_Percentage default = 0.25
        # AC_Spr_Refl_Promille default = 0.3
        minRefPercentage = self.config.AC_Spr_Refl_Percentage
        minRefPromille = self.config.AC_Spr_Refl_Promille
        npMask = float32(mask.size)
        npRef1Promille = npRef1  * 1000.0 / npMask
        # fix for SIIMPC-890 UMW: mlistref.size was used instead of self._npref
        if (npRef1 / float32(self._npref) < minRefPercentage) | (npRef1Promille < minRefPromille):
            return
        # end fix for SIIMPC-892 UMW
        # set nadir geometry (view and sun angles)
        x = set_nadir_geometry(self.config.solze_arr, self.config.solaz_arr, self.config.vza_arr, self.config.vaa_arr, self._itilt)
        self._thv = x[0]
        self._phiv = x[1]
        self._solze = x[2]
        self._solaz = x[3]

        # get path radiance for interpolated extended vis. lph[iav_ele_ref,nvisx,n_bands_all)
        # iav_ele_ref is index of average elevation of ref. pixels (from ref_pixel3)
        keff = self._iav_ele_ref
        self.altit3v_atm() # calls read3v_atm

        reflref_vect1 = ravel(mask)[mlistref] # ref.pixels for subset with VZA < 10 deg
        self._refl_red_av = reflref_vect1.sum() / npRef1

        # fix according to RR., applied for V.2.4.0:
        # convert into 0-1 range, add offset 0.005
        # avoids very small rho(blue) in bands 1 & 2
        # refl_blu_av  = ratio_blu_red * refl_red_av * 0.01 + 0.005:
        self._refl_blu_av = self._ratio_blu_red * self._refl_red_av * 0.01 + 0.005  # modified according to Bringfrieds remark
        band_blu = self.tables.getBand(Band.BLUE) # blue Band #self.tables.B02
        band_red = self.tables.getBand(Band.RED) # red Band # self.tables.B04
        band_coastal_aerosol = self.tables.getBand(Band.COASTAL_AEROSOL) # self.tables.B01
        if (self.cirrus_correction):
            # red band
            band_red = self.rmCirrus(Band.RED, radiance=True) #(band_red, self.tables.B04)
            # blue band
            band_blu = self.rmCirrus(Band.BLUE, radiance=True) #(band_blu, self.tables.B02)
            # b00 band
            band_coastal_aerosol = self.rmCirrus(Band.COASTAL_AEROSOL, radiance=True) #(band_b00, self.tables.B01)
        else:
            band_red = self.tables.getBand(Band.RED, radiance=True) #self.refl2rad(band_red, self.tables.B04)
            band_blu = self.tables.getBand(Band.BLUE, radiance=True) #self.refl2rad(band_blu, self.tables.B02)
            band_coastal_aerosol = self.tables.getBand(Band.COASTAL_AEROSOL, radiance=True) #self.refl2rad(band_b00, self.tables.B01)

        # average DN of ref. pixels in the blue & red band
        # fix for SIIMPC-890, UMW: variables need not to be global
        dn_red_ref_av = float32(mean(ravel(band_red)[mlistref]))
        dn_blu_ref_av = float32(mean(ravel(band_blu)[mlistref]))
        dn_coastal_aerosol_ref_av = float32(mean(ravel(band_coastal_aerosol)[mlistref]))

        # Exclude only medium thickness haze, not thin haze,
        # because thin haze often occupies the whole scene !!
        clear_scene = ones([self.config.nrows*self.config.ncols], uint8) # matrix to indicate clear areas

        # fix for SIIMPC-552, UMW:
        th_bright = dn_red_ref_av * 5.0
        list_br = where32(ravel(band_red > th_bright))
        if(list_br.size > 1):
            clear_scene[list_br] = 0
        # end fix SIIMPC-552
        
        if (self._np_hazew > 1):
            clear_scene[self._mlist_hazew] = 0
        if (self._np_cloud > 1):
            clear_scene[self._mlist_cloud] = 0
        list1 = where32(ravel(band_blu == 0))
        if (list1.size > 0):
            clear_scene[list1] = 0

        clear_scene = reshape(clear_scene, [self.config.nrows, self.config.ncols])
        list_clear_scene = where32(ravel(clear_scene == 1))  # clear scene pixels

        if (list_clear_scene.size > 0):
            # fix for SIIMPC-890, UMW: variables need not to be global
            dn_blu_scene_av = float32(mean(ravel(band_blu)[list_clear_scene]))
            dn_red_scene_av = float32(mean(ravel(band_red)[list_clear_scene]))
            dn_coastal_aerosol_scene_av = float32(mean(ravel(band_coastal_aerosol)[list_clear_scene]))
        else:
            return

        # calculate scale factor for total path radiance (Rayleigh+aerosol, red, blue bands)
        visin_mean = self._meanvi # current meanvi(DDV) of ref_pixel_vi_map

        iv1 = asarray(visin_mean).astype(int16)
        ib = Band.BLUE.value -1 #self._blue_band
        ir = Band.RED.value -1 #self._red_band
        index_coastal_aerosol = Band.COASTAL_AEROSOL.value -1 # ib00 = 0

        # The adjacency effect is considered with weighing 0.5*qh instead of qh
        # The factor 0.5 is used here because at least half the neighborhood of ref.pixels is
        # also ref.pixels, and dn_av(whole scene) might not be representative of the neighborhood
        # of DDV patches, i.e. dn_av is probably too high.
        # fix for SIIMPC-890, UMW: variables need not to be global
        dn_blu = dn_blu_ref_av
        lmeas_blu = self.config.d2 * (dn_blu + 0.5 * self._qh[ib, iv1, keff] * (dn_blu - dn_blu_scene_av))

        # subtract reflected radiance to obtain total path radiance
        lp_blu_scene = lmeas_blu - self._refl_blu_av * (self._e0th[ib, iv1, keff] * cos(radians(self._solze)) + self._edifth[ib, iv1, keff]) / pi
        self._lp_blu = lp_blu_scene

        if (self._lp_blu < 0): # negative path radiance, retrieval results not valid for blue band
            self._iscale_path = 1 # negative path radiance, retrieval results not valid for blue band
            return # is reported in .log file

        # fix for SIIMPC-552, UMW - making option configurable:
        if (self.config.scaling_disabler == True): #Modification JL 03/06/2016
            self.config.timestamp('L2A_AtmCorr: the rescaling of path radiance in blue band has been disabled by configuration')
            return

        self._iscale_path = 0
        # fix for SIIMPC-890, UMW: variables need not to be global
        dn_blu = dn_coastal_aerosol_ref_av
        wvl_blu = self.config.wvlsen[index_coastal_aerosol]
        lmeas_coastal_aerosol = self.config.d2 * (dn_blu + 0.5 * self._qh[index_coastal_aerosol, iv1, keff] * (dn_blu - dn_coastal_aerosol_scene_av))

        refl_blu_av = uint16(self._refl_blu_av * self.config.dnScale)
        refl_blu_av = interpol(array([0.6, 1.0] * refl_blu_av), array([0.4, self._wvlsen[index_coastal_aerosol]]), wvl_blu)
        refl_blu_av /= float32(self.config.dnScale)
        lp_coastal_aerosol = lmeas_coastal_aerosol - refl_blu_av[0] * (self._e0th[index_coastal_aerosol, iv1, keff] * cos(radians(self._solze)) + self._edifth[ib00, iv1, keff]) / pi

        dn_red = dn_red_ref_av
        lmeas_red = self.config.d2 * (dn_red + 0.5 * self._qh[ir, iv1, keff] * (dn_red - dn_red_scene_av))

        # subtract reflected radiance to obtain total path radiance
        self._lp_red_scene = lmeas_red - self._refl_red_av * 0.01 * (self._e0th[ir, iv1, keff] * cos(radians(self._solze)) + self._edifth[ir, iv1, keff]) / pi

        # only relevant for automatic aerosol detection:
        if self.aerosolDetection == 'STARTED':
            self._dratio_aeros = (lp_blu_scene / self._lp_red_scene) / (self._lph[ib, iv1, keff] / self._lph[ir, iv1, keff])
            return

        # calculate total path radiance in red region
        # (for mean vis. index)
        self._lp_red = self._lph[ir, iv1, keff]
        sc_blu = self._lp_blu / self._lph[ib, iv1, keff]
        sc_red = 1.0

        # fix for SIIMPC-552, UMW - making options configurable:
        # limit path radiance blue scaling to +/-10% scaling. Modification JL 03/06/2016
        if (self.config.scaling_limiter == True):
            self.logger.info('Computed path radiance for blue band rescaling factor: ' + str(sc_blu))
            min_sc_blu = self.config.min_sc_blu # default: 0.9
            max_sc_blu = self.config.max_sc_blu # default 1.1
            if (sc_blu < min_sc_blu) | (sc_blu > max_sc_blu):
                sc_blu = clip(sc_blu, min_sc_blu, max_sc_blu)
                self.config.timestamp('The rescaling factor is clipped as it exceeds +/- 10 % variation')

        self.config.sc_lp_blu = sc_blu # save for *log file
        self.logger.info('path radiance for blue band rescaled from 1.0 to: ' + str(sc_blu))

        # exponential interpolation/extraploation (blue to red)
        # (1) take log
        # (2) linear interpolation for log
        # (3) take exponential

        a1_sc = log( [ sc_blu, sc_red ] )
        a1_sc_intpl = interpol(a1_sc, [self._wvlsen[ib], self._wvlsen[ir]], self._wvlsen[0:ir])
        a1_sc_new  = exp(a1_sc_intpl)

        for k in arange(0, ir):
            self._lph[k,:,:] = self._lph[k,:,:] * a1_sc_new[k]

        a2_sc = array([lp_coastal_aerosol, self._lp_blu]) / array([self._lph[index_coastal_aerosol, iv1, keff], self._lph[ib, iv1, keff]])
        wv_grid = array([self._wvlsen[index_coastal_aerosol], self._wvlsen[ib]])

        for k in arange(0, ib):
            self._lph[k,:,:] = self._lph[k,:,:] * interpol(a2_sc, wv_grid, self._wvlsen[k])

        return

    #----------------------------------------------------------------------------
    def ref_pixel_wfov(self, iflag_haze):
    # Purpose: calculate the visibility index (AOT) for the dark reference pixels
    # on the per-cell basis.
    # mlistref = list of reference pixels from masking routine (whole scene)
    # The average elevation of the ref.pixels per cell is taken into account.
    # This is a first step, the visindex might later be modified (check_negative_refl)
    # if it leads to negative reflectance pixels in the red /nir bands.
    # The visindex map is calculated for the red band (not blue, as blue band
    # is critical wrt atmosphere and calibration). The blue band is later included
    # to scale the path radiance in the blu-to-red region (scale_path_radiance)
    # Steps:
    # 1. The loop over cells calculates the visindex (AOT) for each cell
    # 2. The mean visindex is calculated (whole scene)
    #
    # Input: everything from Common atcor3.inc
    # iflag_haze = 0 visindex calculated without de-hazing red band
    # = 1 visindex calculated with de-hazed red band
    # Output: ivisrpix = vector of vis.index corresponding to mlistref
    #
    # Structure:
    # ! read_channel_bsq
    # ! read_hcw_file_get_haze
    # ! read_hcw_file_get_shadow
    # ! read_hcw_file_get_cirrus
    # ! altit3v_atm
    # ! interp1_a3

        if(self._npref == 0):
            return

        nrows = self.config.nrows
        ncols = self.config.ncols
        try: # agular values are arrays:
            x = arange(nrows, dtype=float32) / (nrows-1) * self.config.solze_arr.shape[0]
            y = arange(ncols, dtype=float32) / (ncols-1) * self.config.solze_arr.shape[1]
            szi = rectBivariateSpline(x, y, self.config.solze_arr)
            x = arange(nrows, dtype=float32) / (nrows-1) * self.config.solaz_arr.shape[0]
            y = arange(ncols, dtype=float32) / (ncols-1) * self.config.solaz_arr.shape[1]
            sai = rectBivariateSpline(x, y, self.config.solaz_arr)
            del x
            del y
        except: # angular value is a scalar:
            szi = ones([nrows, ncols], dtype=float32) * self.config.solze_arr.mean()
            sai = ones([nrows, ncols], dtype=float32) * self.config.solaz_arr.mean()

        vzi = ones([nrows, ncols], dtype=float32) * self.config.vza_arr
        vai = ones([nrows, ncols], dtype=float32) * self.config.vaa_arr

        # cell loop for RT terms needs lph, e0th(n_alti,nvisx,n_bands) etc.
        # calculated in altit3v_atm
        # if (self._itarget == 1):
        #     nk = self._red_band # get RED band (vegetation)
        # elif (self._itarget == 2):
        #     nk = self._nir_band-1 # get NIR band (water)
        # band = self.tables.getBand(nk)
        #
        # if (self.cirrus_correction): # subtract cirrus contribution
        #     band = self.rmCirrus(band, nk)
        # band = self.refl2rad(band, nk)

        if (self._itarget == 1):
            nk = Band.RED  # get RED band (vegetation)
        elif (self._itarget == 2):
            nk = Band.NEAR_INFRARED  # get NIR band (water) # S2 does not have NEAR_INFRARED. however given that self._itarget = 1 never goes here

        if self.cirrus_correction:  # subtract cirrus contribution
            band_red_radiance = self.rmCirrus(nk, radiance=True)
        else:
            band_red_radiance = self.tables.getBand(nk, radiance=True)

        sc1 = 1.0 / self._sc_bet # scale factor sc_bet for cbeta

        # calculate average DN in red band for adjacency correction
        sc = self.tables.getBand(Band.SCENE_CLASSIFICATION) #Band.SCENE_CLASSIFICATION
        clear_scene = zeros_like(sc)
        clear_scene[(sc == self.config.water) | (sc == self.config.bareSoils) | (sc == self.config.vegetation)] = 1

        # add the cirrus pixels if no cirrus correction is selected:
        # disabled as cirrus pixels are not reliable enough. from sen2cor 2.10.2
        # if not self.cirrus_correction:
        #     clear_scene[sc == self.config.thinCirrus] = 1

        mask_ref = zeros([self.config.nrows*self.config.ncols], uint8)
        mask_ref[self._mlistref] = 1

        mask_rho = zeros([self.config.nrows*self.config.ncols], float32)
        mask_rho[self._mlistref] = self._reflref_vect
        self._reflref_vect_save = self._reflref_vect.copy() # vector of reflectance values of dark reference pixels
        self._mlistref_save = self._mlistref.copy()

        keffi = self._iav_ele
        self._iav_ele_ref = self._iav_ele

        bandvis = zeros([self.config.nrows, self.config.ncols], uint8)

        # read DEM height file
        if (self._iter_terrain > 0):
            if(self.tables.hasBand(Band.DIGITAL_ELEVATION_MAP)): #self.tables.DEM
                dtm_h = clip(self.tables.getBand(Band.DIGITAL_ELEVATION_MAP), 0, 3500) #self.tables.DEM
            else:
                self.logger.fatal('no DEM found')
                return False

            # needed in scale_path_radiance, the 20 in the denominator converts 20 m height grid into index
            #self._iav_ele_ref = asarray(sum(maximum((ravel(dtm_h)[self._mlistref]), 0)) / (20. * self._npref)).astype(int16)
            self._iav_ele_ref = uint16(dtm_h[dtm_h > 0].mean() / 20. + .5)

        if 'LANDSAT' in self.config.spacecraftName:
            index = self.tables.reindex(nk) -1  #Band.RED
        else:
            index = self.tables.reindex(nk)
        for jx in arange(0, (self._nx_cell)):
            for jy in arange(0, (self._ny_cell)):
                subset = array([self._ycell[jy, 0], self._ycell[jy, 1], self._xcell[jx, 0], self._xcell[jx, 1]])
                # fix for SIIMPC-897-1, UMW: take only non background cells into account:
                nonback_cell = sc[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                li_nonback_cell = where32(ravel(nonback_cell > 0))
                # fix for SIIMPC-897-1, end
                mask_ref = reshape(mask_ref, (self.config.nrows, self.config.ncols))
                self._mlistref = where32(ravel(mask_ref[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1] > 0))
                self._npref = self._mlistref.size
                if (self._npref > 0):
                    #self._reflref_vect = mask_rho[self._mlistref] # there seems to me that there is an error here: mask_ref[subset] and mask_rho don't have the same size (JL)
                    mask_rho = reshape(mask_rho, (self.config.nrows, self.config.ncols)) # Corrected JL20171013
                    self._reflref_vect = ravel(mask_rho[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1])[self._mlistref] # Corrected JL20171013
                    band = reshape(band_red_radiance, (self.config.nrows, self.config.ncols))
                    band1 = band[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                    clear_scene = reshape(clear_scene, (self.config.nrows, self.config.ncols))
                    clear_sub = clear_scene[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                    li_clear = where32(ravel(clear_sub == 1))
                    if (self._iter_terrain > 0):
                        ecl = self._eclass[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                        eclm = ravel(ecl)[self._mlistref]
                    bandvis1 = zeros([subset[1]-subset[0]+1, subset[3]-subset[2]+1]) + 255
                    dn = ravel(band1)[self._mlistref]
                    # fix for SIIMPC-897-2, UMW: dn_av must always be a scalar value:
                    # otherwise adjacency effect is too small, according to RR.
                    if (li_clear.size > 1):
                        mean_radiance = ravel(band1)[li_clear].mean()
                    else:
                        mean_radiance = ravel(band1)[li_nonback_cell].mean()
                    # fix for SIIMPC-897-2 end

                    vsky1 = self._vsky[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                    cbetac1 = self._cbeta[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                    self._vskyc = ravel(vsky1)[self._mlistref] * 0.01
                    self._cbetac = ravel(cbetac1)[self._mlistref] * sc1

                    if (self._iter_terrain == 0):
                        k = self._iav_ele # quasi-flat terrain, scalar index
                    else:
                        k = eclm # vector of height indices for ref.pixels

                    # update geometry for current cell
                    self._solze = float32(mean(szi[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]))
                    self._solaz = float32(mean(sai[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]))
                    self._thv = float32(mean(vzi[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]))
                    self._phiv = float32(mean(vai[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]))

                    self.altit3v_atm() # calculate RT fcts, calls read3v_atm_hyper_a3
                    # yields lpx(n_alti,nvis,n_bands)

                    # reflectance of reference pixels for nvis visibilities
                    self._reflvis = zeros([self._nvis, self._npref], float32)

                    for j in arange(0, (self._nvis)):

                    # reflvis is input to interp1_a3
                    # Adjacency effect is accounted for with 0.5*qx and global dn_av
                    # The factor 0.5 at qx is used here because at least half the neighborhood of ref.pixels is
                    # also ref.pixels, and dn_av(current cell) might not be representative of the neighborhood
                    # of DDV patches, i.e. is probably too high.

                        self._reflvis[j,:] = 100. * pi * (self.config.d2 * \
                        (dn + 0.5 * self._qx[index, j, k] * (dn - mean_radiance)) - self._lpx[index, j, k]) / \
                        (self._e0tx[index, j, k] * self._cbetac + self._ediftx[index, j, k] * (self._tsunx[index, j, k] * \
                        self._cbetac / cos(radians(self._solze)) + (1.0 - self._tsunx[index, j, k]) * \
                        self._vskyc) + (self._e0tx[index, j, k] * cos(radians(self._solze)) + self._ediftx[index, j, k]) * \
                        self._reflter[index] * (1.0 - self._vskyc))

                    # ivpix is list of visibility index of reference pixels
                    ivpix = self.interp1_a3(self._reflref_vect, self._npref)
                    # place ivpix into bandvis1
                    ravel(bandvis1)[self._mlistref] = ivpix
                    # place bandvis1 into bandvis
                    bandvis[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1] = bandvis1

        # restore reflref_vect
        self._reflref_vect = (self._reflref_vect_save)

        # restore mlistref, npref
        self._mlistref = self._mlistref_save
        self._npref = self._mlistref.size

        self._ivisrpix = ravel(bandvis)[self._mlistref]
        bandvis = zeros([self.config.nrows, self.config.ncols], uint8) # vis.index map
        ravel(bandvis)[self._mlistref] = self._ivisrpix
        self.tables.setBand(Band.VISIBILITY, bandvis) #self.tables.VIS

        self._meanvis = float32(mean(self._ivisrpix))
        self._meanvi = uint8(0.5 + self._meanvis) # mean vis. index
        self._visibility = self._visext[self._meanvi] # mean visibility in km

        # restrict visibility values to <= 120 km (needed for wv routine)
        if (self._visibility > self._visarr[self._nvis - 1]):
            self._visibility = self._visarr[self._nvis - 1]
        self._visibility_ddv = self._visibility
        self._meanvi_ddv = indexvis(self._visibility_ddv, self._nvisx, self._vis_reg)
        return

    #============================================================================
    # ref_pixel_vi_map_wfov
    # Purpose:
    # Calculation of visibility index map based on reference pixels (using the red band).
    # In case of haze removal the de-hazed red band is used.
    # Is similar to ref_pixel_wfov but raises the visindex map if required by check_negative_refl routine.
    # - Calls scale_path_radiance to update the path radiance in the blue-to-red region.
    # - Smoothes the visindex map and performs spatial triangular interpolation if set.
    # - Stores visindex map = bandvis, and writes it to file.
    #
    # Input: everything from Common atcor3.inc
    # Output: bandvis and visindex file filin + '_visindex.bsq'
    #
    # Structure
    # ! ref_pixel_wfov
    # ! altit3v_atm
    # ! scale_path_radiance_wfov
    # ! write_visindex_file_a3
    # ! write_ddv_map
    #============================================================================
    def ref_pixel_vi_map_wfov(self):

        # nk = self._red_band
        self._meanvi_check_neg = self._meanvi # from DBV and / or check_negative_refl (red & NIR bands)

        # meanvi_check_neg is compliant with DBV (red band) and water (NIR)
        # (calculated in ref_pixel_dbv or check_negative_refl)
        # update ivisrpix if meanvi_check_neg < meanvi_ddv, i.e., visib_check_neg > visib_ddv

        # s1 = stdev (ivisrpix,meanvis)    ; from ref_pixel_wfov
        # meanvi_ddv = byte (0.5+meanvis)  ; average visibility index
        # visib_ddv  = visext[meanvi_ddv]
        #    ; meanvi=meanvi_dbv is compliant with DBV (red band) and water (NIR)
        #    ; (calculated in ref_pixel_dbv or check_negative_refl)
        #    ; update ivisrpix if meanvi < meanvi_ddv, i.e., visib > visib_ddv
        # IF (meanvi LT meanvi_ddv) THEN BEGIN
        #     ; meanvi  is result from DBV and water pixels
        #     ; dbv or water requires higher visib, i.e., lower vis.index
        #     ; shift pixels with vi > meanvi to vi(new) = vi(old) + (meanvi-meanvi_ddv)
        #     li = where(ivisrpix GT fix(meanvi), cnt)
        #     IF (cnt GT 0) THEN ivisrpix[li] = byte ( (ivisrpix[li] + (fix(meanvi) - meanvi_ddv)) > 0b)
        #     li = 0
        #     meanvi = byte(0.5+total(ivisrpix) / n_elements(ivisrpix))  ; update meanvi
        #     visib = visext[meanvi]  ; corresponding visib from DBV and water
        # ENDIF ELSE BEGIN
        #     meanvi = meanvi_ddv
        #     visib = visext[meanvi]
        # ENDELSE

        if (self._meanvi_check_neg < self._meanvi_ddv):
            self._meanvi = self._meanvi_check_neg
            # meanvi_check_neg is result from DBV and water pixels (check_negative_refl)
            # dbv or water requires higher visib, i.e., lower vis.index
            # shift pixels with vi > meanvi to vi(new) = vi(old) + (meanvi-meanvi_ddv)
            li = where32(ravel(self._ivisrpix > self._meanvi))
            if (li.size > 0):
                self._ivisrpix[li] = uint8(maximum(int16(self._ivisrpix[li]) + self._meanvi - self._meanvi_ddv + 0.5, 0))
            self._meanvi = uint8(0.5 + self._ivisrpix.mean()) # update meanvi
            self._visibility = self._visext[self._meanvi] # corresponding visib from DBV and water

            if self.logger.level == logging.DEBUG:
                print('visibility index reduced to %f according to check in: ref_pixel_vi_map_wfov()' % self._visibility)
        else:
            self._meanvi = self._meanvi_ddv
            self._visibility = self._visext[self._meanvi]

        # restrict visibility values to <= 120 km
        # required for water vapor algorithm
        if (self._visibility > self._visarr[self._nvis - 1]):
            self._visibility = self._visarr[self._nvis - 1]

        # for ratio_blu_red < 0 no re-scaling of path radiance in the blue-red region
        # if (self._ratio_blu_red > 0.0 and (Band.BLUE.value > 0 or Band.GREEN.value > 0)): #self._green_band
        if (self._ratio_blu_red > 0.0 and ((Band.BLUE.value -1) > 0 or (Band.GREEN.value -1) > 0)):  # self._green_band
            self.scale_path_radiance_wfov() # with current meanvi

        # reflref_vect can be freed now, because "scale_path_radiance" which needs reflref_vect is
        # no more called during "run" and image processing
        self._reflref_vect = 0

        # Calculation of spatial visibility index map (total optical thickness)
        # replace the 255 tag in bandvis by meanvi and smooth vis.index array
        bandvis = zeros([self.config.nrows*self.config.ncols], uint8) + 255
        bandvis[self._mlistref] = self._ivisrpix

        list_gap = where32(ravel(bandvis == 255))
        cnt_gap = list_gap.size
        if (cnt_gap > 0):
            bandvis[list_gap] = self._meanvi

        # smooth with filter size nf pixels (3000 m was default, now configurable via AC_Ddv_Smooting_Window)
        np = array(asarray([self.config.ncols / 2 - 2, self.config.nrows / 2 - 2])).min()
        nf = (maximum(asarray(self.config.AC_Ddv_Smooting_Window * 1000.0 / self.config.pixelsize).astype(int16), 3)) # low pass filter size (nf*nf)
        if (nf > np):
            nf = np
        if (nf % 2 == 0):
            nf = nf + 1

        bandvis = bandvis.reshape(self.config.nrows, self.config.ncols)
        # JL20200309: set bandvis[no_data] to meanvi value before bandvis smoothing, then revert bandvis[no_data] to 0
        scl = self.getClassificationMap()
        bandvis[scl <= self.config.saturatedDefective] = self._meanvi
        # bandvis values range from 0-182=nvisx-1
        #bandvis = uint8(0.5 + smooth(asarray(bandvis).astype(float32), nf, edge_truncate=True))
        ### suggestion JL20171012:
        bandvis = uint8(0.5 + gaussian_filter(bandvis.astype(float32), nf))
        bandvis[scl <= self.config.saturatedDefective] = 0
        # write visindex map to file (not for tiff mode)
        self.write_visindex_file_a3(bandvis)
        return

    def write_ddv_map(self):
        # Input:
        # mlistref = list of DDV reference pixels (coded green)
        #
        # Output:
        # Map of DDV[4], dark features[2] and water pixels [6]
        CM = self.getClassificationMap()
        # set all other classes to background pixels
        self._background = self.config.bareSoils #5
        CM[CM == self.config.saturatedDefective] = self._background
        CM[CM == self.config.cloudShadows] = self._background
        CM[CM == self.config.vegetation]  = self._background
        CM[CM == self.config.bareSoils]  = self._background
        CM[CM == self.config.lowProbaClouds] = self._background
        CM[CM == self.config.medProbaClouds] = self._background
        CM[CM == self.config.highProbaClouds] = self._background
        CM[CM == self.config.thinCirrus] = self._background
        CM[CM == self.config.snowIce] = self._background
        # keep dark features and water
        #reset new classification DDV to self._vegetation
        ravel(CM)[self._mlistref] = self.config.vegetation
        self.tables.setBand(Band.DARK_DENSE_VEGETATION, CM) #self.tables.DDV
        return True

    #----------------------------------------------------------------------------
    def ref_pixel_dbv_wfov(self):
    # Analogous to "ref_pixel" for dense dark vegetation
    # but dense bright vegetation pixels are searched, because
    # scene does not contain enough ddv pixels.
    # The DBV pixels are determined with
    #
    # li_dbv = where( rho_app_nir gt 0.35 and ndvi gt 0.66)
    #
    # Then the 10 DBV pixel(s) with the smallest values of
    # rho_app_red are used to determine the visibility assuming
    # rho_red(DBV) = 0.02. This leads to a lower bound of the calculated
    # visibility vis_dbv. If, in reality rho_red > 0.02 (e.g., 0.03)
    # then the true visibility would be higher, still the spectral
    # shape of vegetation will be preserved and negative rho_red(veg)
    # will be avoided.
    # Additionally, rho_vegetation(Red) is tested (rho > 0.005)
    # using NDVI_app > 0.45 and rho_nir_app > 0.15
    # Additionally, rho_water_nir is tested, and if it is
    # negative for too many pixels ( > 1% of scene) then vis_dbv
    # is raised iteratively.
    # The search for DBV pixels is restricted to the close-to-nadir part of the scene with VZA < 10 deg
    # to speed up the calculation.
    #
    # Input:
    # npref_dbv = 0: initialization for dbv
    # Output:
    # (A)
    # npref_dbv = 10: this module found DBV pixels and updates visibility
    # visib_check_neg = visib: updated visibility
    # meanvi_check_neg= meanvi: updated meanvi
    # (B)
    # npref_dbv =-1: no DBV pixels found
    # visib_check_neg = visib: updated visibility
    # meanvi_check_neg= meanvi: updated meanvi
    #
    # Structure:
    # ! read_channel_bsq
    # ! altit3v_atm
    # ! interp1_a3
    # ! indexvis
    # ! check_negative_refl_wfov

        self._npref_dbv = -1

        # check is performed for the close-to-nadir part of the scene with VZA < 10 deg
        # n_cols is simply scaled assuming pixel 1 corresponds to VZA[0,0] = UL
        # and pixel n_cols corresponds to VZA[0,1] = UR
        # This is usually also sufficient to guarantee a small SZA difference.

        # set nadir geometry (view and sun angles)
        x = set_nadir_geometry(self.config.solze_arr, self.config.solaz_arr, self.config.vza_arr, self.config.vaa_arr, self._itilt)
        self._thv = x[0]
        self._phiv = x[1]
        self._solze = x[2]
        self._solaz = x[3]

        # get RED band (vegetation)
        # index = self.tables.B04
        # B04 = self.tables.getBand(index)
        # if (self.cirrus_correction): # subtract cirrus contribution
        #     B04 = self.rmCirrus(B04, index)

    # radiance is required for reflvis computation:
        if self.cirrus_correction:  # subtract cirrus contribution
            band_red_radiance = self.rmCirrus(Band.RED, radiance=True)
        else:
            band_red_radiance = self.tables.getBand(Band.RED, radiance=True)

    # calculate average DN in red band for adjacency correction
        # since ref. pixels are surrounded by ref. pixels and other pixels the assumption is
        # made for the adjacency correction that 50% are other pixels, i.e. weighting is 0.5*qh
        clear_scene = ones([self.config.nrows, self.config.ncols], uint8) # matrix to indicate clear areas
        if (self._np_hazew > 1):
            ravel(clear_scene)[self._mlist_hazew] = 0
        if (self._np_cloud > 1):
            ravel(clear_scene)[self._mlist_cloud] = 0
        if (self._np_clshad > 1):
            ravel(clear_scene)[self._mlist_shad] = 0
        list1 = where32(ravel(band_red_radiance == 0))  # background pixels
        if (list1.size > 0):
            ravel(clear_scene)[list1] = 0

        # fix for SIIMPC-897-1, UMW: take only non background cells into account:
        nonback_cell = self.tables.getBand(Band.SCENE_CLASSIFICATION) #self.tables.SCL
        li_nonback_cell = where32(ravel(nonback_cell > self.config.saturatedDefective))
        # fix for SIIMPC-897-1, end
        list_clear_scene = where32(ravel(clear_scene == 1))  # clear scene pixels
        if (list_clear_scene.size > 0):
            # fix for SIIMPC-897-2, UMW: dn_av must always be a scalar value:
            mean_radiance = ravel(band_red_radiance)[list_clear_scene].mean() # self.refl2rad(ravel(band_red_radiance)[list_clear_scene].mean(), index) # fix JL20171016: radiance is required for reflvis computation
        else:
            mean_radiance = ravel(band_red_radiance)[li_nonback_cell].mean() #self.refl2rad(ravel(B04)[li_nonback_cell].mean(), index) # fix JL20171016: radiance is required for reflvis computation
            # fix for SIIMPC-897-2, end

        # now the DBV (dense bright vegetation) specific part begins
        # get NIR band
        # Todo: check for sentinel, there Vegetation 4 is used!
        if self.cirrus_correction:  # subtract cirrus contribution
            band_red_reflectance = self.rmCirrus(Band.RED)
            if 'LANDSAT' in self.config.spacecraftName:
                band_nir_reflectance = self.rmCirrus(Band.NEAR_INFRARED)
            else:
                band_nir_reflectance = self.rmCirrus(Band.VEGETATION_4)
        else:
            band_red_reflectance = self.tables.getBand(Band.RED)
            if 'LANDSAT' in self.config.spacecraftName:
                band_nir_reflectance = self.tables.getBand(Band.NEAR_INFRARED)
            else:
                band_nir_reflectance = self.tables.getBand(Band.VEGETATION_4)
        # B8A = self.tables.getBand(self.tables.B8A)
        # index = self.tables.B8A - 1
        # if (self.cirrus_correction): # subtract cirrus contribution
        #     B8A = self.rmCirrus(B8A, index)

        # apparent reflectance
        ndvi = (band_nir_reflectance - band_red_reflectance) / maximum((band_nir_reflectance + band_red_reflectance), 0.01)

        sc1 = 1.0 / asarray(self._sc_bet).astype(float32)
        cbeta1 = self._cbeta

        li_dbv = where32(ravel([(band_nir_reflectance > self.config.AC_Dbv_Nir_Refl_Th) &
                              (ndvi > self.config.AC_Dbv_Ndvi_Th) & (band_red_reflectance > 0.01) &
                              (abs(arccos(cbeta1 * sc1) * self._radeg - self._solze) < 15.0)]))

        # fix for SIIMPC-917-1: DBV algorithm was not called as already described for SIIMPC-880, self._npref_dbv was always empty:
        self._npref_dbv = li_dbv.size

        if (self._npref_dbv < 5): # minimum is 5 DBV pixels
            self._npref_dbv = -1 # indicates failure of DBV search, npref_dbv=0 is default
            if (self._npref == 0):
                # keep current visibility, update meanvi
                # To be replaced: SIIMPC-1223, reported by VD, TPZF for 2.6.3:
                index = interpol(arange(self._nvisx-1, -1, -1, dtype=float32), sorted(self._visext), self._visibility)
                self._meanvi = (minimum(uint8(maximum(index, 0)), uint8(self._nvisx - 1))) # (must be byte data!)

        vsky1 = self._vsky
        self._eclass1 = self._eclass

        # fix for SIIMPC-917-2: various fixes in DBV iteration:
        if (self._npref_dbv >= 5):
            # get the 10 pixels with lowest reflectance in the red band
            rh_red = ravel(band_red_reflectance)[li_dbv]
            li_sort = argsort(rh_red)
            rh_red = rh_red[li_sort]
            self._npref_dbv = (minimum(self._npref_dbv, 10))
            self._mlistref_dbv = li_dbv[li_sort[0:self._npref_dbv]]# list of 10 dbv pixels in image coordinates
            li10_dbv = self._mlistref_dbv
            self._iav_el = int16(sum(ravel(self._eclass1)[self._mlistref_dbv]) / self._npref_dbv + 0.5)
            self._vskyc = ravel(vsky1)[self._mlistref_dbv] * 0.01
            self._cbetac = ravel(cbeta1)[self._mlistref_dbv] * sc1
            eclm = ravel(self._eclass1)[self._mlistref_dbv]

            self.altit3v_atm() # calculate RT fcts, calls read3v_atm_hyper_a3

            # reflectance of 10 reference pixels for nvis visibilities
            self._reflvis = zeros([self._nvis, self._npref_dbv], float32)

            self._reflref_vect = zeros([self._npref_dbv], float32) + self.config.AC_Red_Ref_Refl_Th * 100.0
            # DBV surface reflectance 2% in the red band
            # nk = self.tables.B04
            # B04 = self.tables.getBand(nk)
            # dn = self.refl2rad(ravel(B04)[li10_dbv], nk) # fix JL20171016: radiance is required for reflvis computation
            band_red_radiance = ravel(self.tables.getBand(Band.RED, radiance=True))[li10_dbv]
            if 'LANDSAT' in self.config.spacecraftName:
                index = self.tables.reindex(Band.RED) -1
            else:
                index = self.tables.reindex(Band.RED)
            # visibility loop
            #----------------
            # Adjacency effect is accounted for with 0.5*qx and global dn_av
            # The factor 0.5 at qx is used here because at least half the neighborhood of ref.pixels is
            # also ref.pixels, and dn_av(whole scene) might not be representative of the neighborhood
            # of DDV patches, i.e. is probably too high.
            for j in arange(0, (self._nvis)):
            # reflvis is input to interp1_a3
                self._reflvis[j,:] = 100. * pi * (self.config.d2 * \
                    (band_red_radiance + 0.5 * self._qx[index, j, eclm] * (band_red_radiance - mean_radiance)) - self._lpx[index, j, eclm]) / \
                    (self._e0tx[index, j, eclm] * self._cbetac + self._ediftx[index, j, eclm] * (self._tsunx[index, j, eclm] * \
                     self._cbetac / cos(radians(self._solze)) + (1.0 - self._tsunx[index, j, eclm]) * self._vskyc) + \
                    (self._e0tx[index, j, eclm] * cos(radians(self._solze)) + self._ediftx[index, j, eclm]) * \
                     self._reflter[index] * (1.0 - self._vskyc))

            # calculate vis.index for current sector
            # based on reflectance reflref_vect of ref.pixels
            iv_dbv = self.interp1_a3(self._reflref_vect, self._npref_dbv)

            self._meanvi_dbv = uint8(sum(iv_dbv) / self._npref_dbv) # average of the vi for the 10 DBV pixels, rounded as byte
            self._visib_dbv = (minimum(self._visext[self._meanvi_dbv], self._visarr[self._nvis - 1])) # water vapor requires visib <=120 km
            self._meanvi_dbv = uint8(indexvis(self._visib_dbv, self._nvisx, self._vis_reg))

            # update visib with visib_dbv if visib_dbv < 80 to prevent artificially high visib (only 10 pixels are used!)
            if (self._visib_dbv > self._visibility and self._visib_dbv <= 80.0):
                self._visibility = self._visib_dbv
                self._meanvi = self._meanvi_dbv

        # end fix for SIIMPC-917
        self.check_negative_refl_wfov(Band.RED)
        if 'LANDSAT' in self.config.spacecraftName:
            self.check_negative_refl_wfov(Band.NEAR_INFRARED)
        else:
            self.check_negative_refl_wfov(Band.VEGETATION_4)
        return

    #----------------------------------------------------------------------------
    def check_negative_refl_wfov(self, band_identifier):

    # Purpose: check whether visib from ref_pixel leads to negative reflectance for
    # other pixels (red band and nir band)
    # If so then the visib is increased iteratively up to visib=80 km.
    # No check is performed if visib >= 80.
    # If the visibility is too low (rho < thresh) it is increased iteratively
    # until percentage of pixels with reflectance < thresh is < 1% of scene pixels.
    # thresh = + 0.010 (red band, intended for vegetation, but also reasonable for water)
    # thresh = - 0.005 (NIR band, water)
    # In the NIR band all land pixels in shadow regions will have an apparently
    # small, but still positive rerflectance. Only for water and shadow over water
    # negative reflectances are likely to occur if the visibility is too low.
    #
    # Input: band_identifier (= red_band or nir_band)
    # n_rows, n_cols: image size (Common)
    # visib: (Common)
    # c0, c1: cal. coefficients (Common)
    # lp, e0t, edift: (Common)
    # dastr: Common
    # solze: (Common)
    # config.pixelsize: (Common)
    # cntback: number of background pixels (Common)
    #
    # Output: visib, meanvi, if percent_neg > 1% (meanvi = mean visibility index)
    #
        if (self._visibility >= 120.0):
            return

        # check is performed for the close-to-nadir part of the scene with VZA < 10 deg
        # n_cols is simply scaled assuming pixel 1 corresponds to VZA[0,0] = UL
        # and pixel n_cols corresponds to VZA[1,0] = UR
        # This is usually also sufficient to guarantee a small SZA difference.

        # set nadir geometry (view and sun angles)
        x = set_nadir_geometry(self.config.solze_arr, self.config.solaz_arr, self.config.vza_arr, self.config.vaa_arr, self._itilt)
        self._thv = x[0] ; self._phiv = x[1] ; self._solze = x[2] ; self._solaz = x[3]

        # current visibility from ref_pixel:
        visib = self._visibility
        gamma = self._gamma_cir

        if band_identifier == Band.RED:
            thresh = self.config._AC_Cut_Off_Aot_Iter_Vegetation
        else:  # Band.NEAR_INFRARED
            thresh = self.config._AC_Cut_Off_Aot_Iter_Water

        eclass_rd = self._eclass
        cbeta_rd = self._cbeta
        vsky_rd = self._vsky

        cnt_water_rd = 0
        if (
            band_identifier == Band.RED
            and self._np_water > 1
            and self._np_water / (asarray(self.config.ncols).astype(
                    float32) * self.config.nrows - self._cntback) > 0.03
        ):
            # exclude water if more than 3% of scene, this is treated for the NIR band
            mask1 = zeros([self.config.nrows, self.config.ncols], uint8)
            mask2 = zeros([self.config.nrows, self.config.ncols], uint8)
            if self._np_water > 1:
                ravel(mask1)[self._mlist_water] = 1
                ravel(mask2)[self._mlistref] = 1
            mlist_water_rd = where32(ravel(mask1 == 1))
            cnt_water_rd = mlist_water_rd.size

        n_cols_rd = self.config.ncols
        n_rows_rd = self.config.nrows
        nf = (maximum((minimum((minimum(asarray(1000. / self.config.pixelsize).astype(int16), 500)), 51)), 3))
        nf = (maximum(array(asarray([nf, n_cols_rd / 2, n_rows_rd / 2])).min(), 3))
        if (nf % 2 == 0):
            nf = nf - 1
        if (self._nadj <= 1):
            nf = 1

        if self.cirrus_correction:
            band_selected = self.rmCirrus(band_identifier, radiance=True)
        else:
            band_selected = self.tables.getBand(band_identifier, radiance=True)

        liback_rd = where32(ravel(band_selected <= 0))
        mask = ones([n_rows_rd, n_cols_rd], uint8)
        if (liback_rd.size > 0):
            ravel(mask)[liback_rd] = 0

        # protoCAMS_v1 : do negative check only on vegetation (scl==4) and not vegetated (scl==5) classes
        scl = self.tables.getBand(Band.SCENE_CLASSIFICATION) #self.tables.SCL
        li_nonback_rd = where32(ravel(bitwise_or(scl == 4, scl == 5)))
        """
        if (cnt_water_rd == 0):
            li_nonback_rd = where32(ravel(mask > 0))
        else:
            li_nonback_rd = where32(ravel32(bitwise_and(mask > 0, mask1 == 0))) # exclude water for red band
        """
        sc1 = 1.0/self._sc_bet

        # set start values:
        percent_neg = 10.0
        iter = 0

        # check negative reflectance
        if 'LANDSAT' in self.config.spacecraftName:
            index = self.tables.reindex(band_identifier) -1
        else:
            index = self.tables.reindex(band_identifier)
        while (percent_neg > 1.0) & (iter <= 12) & (visib < 120) & (li_nonback_rd.size > 0):
            iter = iter + 1
            self._visibility = visib
            self.altit3_atm() # update lp, e0t, edift, etc
            fluxpix = self._e0th[index, eclass_rd] * cbeta_rd * sc1 + self._edifth[index, eclass_rd] * (
                self._tsunh[index, eclass_rd] * cbeta_rd * sc1 / cos(radians(self._solze))
                + (1.0 - self._tsunh[index, eclass_rd]) * vsky_rd * 0.01
            )  # (binary matrix b=1)
            fluxter = maximum(
                (
                    (self._e0th[index, eclass_rd] * cos(radians(self._solze)) + self._edifth[
                        index, eclass_rd])
                    * (100 - vsky_rd)
                    * 0.01
                    * self._reflter[index]
                ),
                (self._e0th[index, eclass_rd] * cos(radians(self._solze)) * 0.01),
            )
            fluxm = pi * (self.config.d2 * band_selected - self._lph[index, eclass_rd])

            rho = ravel(fluxm / (fluxpix + fluxter)).astype(float32)

            # JL20200309: set no_data to mean value before rho average computation
            scl = self.getClassificationMap()
            rho_matrix = rho.reshape(band_selected.shape[0], band_selected.shape[1])
            rho_matrix[scl <= self.config.saturatedDefective] = \
                rho_matrix[scl > self.config.saturatedDefective].mean()
            rho = ravel(rho_matrix)
            # JL: smooth replaced by gaussian more adapted to large kernels
            rho_av = ravel(gaussian_filter(minimum(rho.reshape(band_selected.shape[0], band_selected.shape[1]), 0.40), nf)) # (exclude clouds with threshold 0.40)
            rho = rho + self._qh[index, self._iav_ele] * (rho - rho_av)
            li_neg = where32(ravel(rho[li_nonback_rd] < thresh))

            percent_neg = 100.0 * li_neg.size / float(n_cols_rd * n_rows_rd - liback_rd.size)
            if self.logger.level == logging.DEBUG:
                print('iteration, visibility, perc neg. pixels: ', iter, visib, percent_neg)
            if percent_neg > 1.0:
                if   visib  < 23.0: visib = 23.0
                elif visib  < 26.0: visib += 3.0
                elif visib  < 30.0: visib += 4.0
                elif visib  < 40.0: visib += 5.0
                elif visib < 100.0: visib += 10.0
                else: visib = 120.0

                visib = (minimum(visib, self._visarr[self._nvis - 1]))
                if self.logger.level == logging.DEBUG:
                    print('visbility increased to: ', visib)
                
            if visib == 120:
                break

        # if visib=29, 39, etc then round to visib=30, 40
        if int((visib + 0.5) % 10 == 9):
            visib = 10.0 * int(visib / 10 + 0.5)

        self._visib_check_neg = visib
        self._meanvi = uint8(indexvis(visib, self._nvisx, self._vis_reg))
        self._meanvi_check_neg = self._meanvi

        # if green band exists: iterate VIS until rho(veg, green, DDV) > 1.15*rho(veg, red, DDV)
        if (self._mlistref.size > 1) & (band_identifier == Band.RED) & (visib < 120.0): # red band
            self._visibility = visib
            # calculate DDV ref.pixels in close-to-nadir region
            mask = zeros([self.config.nrows, self.config.ncols], uint8)
            ravel(mask)[self._mlistref] = 1
            mlistref = self._mlistref
            if (mlistref.size < 4):
                return
            
            # red band
            # --------
            band_red_radiance = self.tables.getBand(Band.RED, radiance=True)
            if self.cirrus_correction:
                rho_cir_app = self.tables.getBand(Band.CIRRUS, radiance=True)
                rho_cir_app = median_filter_2d(rho_cir_app, 3) * self.config.dnScale
                ravel(band_selected)[mlistref] -= ravel(rho_cir_app)[mlistref] / gamma

            # fix for SIIMPC-890, UMW: variables need not to be global
            # evaluate rho_red(DDV) with last visib
            r_eclass_rd = ravel(eclass_rd)[mlistref]
            r_cbeta_rd = ravel(cbeta_rd)[mlistref]
            r_vsky_rd = ravel(vsky_rd)[mlistref]

            if 'LANDSAT' in self.config.spacecraftName:
                index = self.tables.reindex(Band.RED) -1
            else:
                index = self.tables.reindex(Band.RED)

            fluxpix = self._e0th[index, r_eclass_rd] * r_cbeta_rd * sc1 + self._edifth[index, r_eclass_rd] * (
                self._tsunh[index, r_eclass_rd] * r_cbeta_rd * sc1 / cos(radians(self._solze))
                + (1.0 - self._tsunh[index, r_eclass_rd]) * r_vsky_rd * 0.01
            )  # (binary matrix b=1)

            fluxter = maximum(
                (
                    (
                    self._e0th[index, r_eclass_rd] * cos(radians(self._solze)) + self._edifth[index, r_eclass_rd])
                    * (100 - r_vsky_rd)
                    * 0.01
                    * self._reflter[index]
                ),
                (self._e0th[index, r_eclass_rd] * cos(radians(self._solze)) * 0.01),
            )

            fluxm = pi * (self.config.d2 * ravel(band_red_radiance)[mlistref] - self._lph[index, r_eclass_rd])
            rho_red = fluxm / (fluxpix + fluxter) # (neglect adj. correction)
            fluxpix = 0 ; fluxter = 0 ; fluxm = 0

            # green Band
            # ----------
            band_green_radiance = self.tables.getBand(Band.GREEN, radiance=True)
            if self.cirrus_correction:
                ravel(band_green_radiance)[mlistref] -= ravel(rho_cir_app)[mlistref] / gamma
            if 'LANDSAT' in self.config.spacecraftName:
                index = self.tables.reindex(Band.GREEN) -1
            else:
                index = self.tables.reindex(Band.GREEN)
            # evaluate rho_gre(DDV) with last visib
            fluxpix = self._e0th[index, r_eclass_rd] * r_cbeta_rd * sc1 + self._edifth[index, r_eclass_rd] * (
                self._tsunh[index, r_eclass_rd] * r_cbeta_rd * sc1 / cos(radians(self._solze))
                + (1.0 - self._tsunh[index, r_eclass_rd]) * r_vsky_rd * 0.01
            )  # (binary matrix b=1)

            fluxter = maximum(
                (
                    (
                    self._e0th[index, r_eclass_rd] * cos(radians(self._solze)) + self._edifth[index, r_eclass_rd])
                    * (100 - r_vsky_rd)
                    * 0.01
                    * self._reflter[index]
                ),
                (self._e0th[index, r_eclass_rd] * cos(radians(self._solze)) * 0.01),
            )

            fluxm = pi * (self.config.d2 * ravel(band_green_radiance)[mlistref] - self._lph[index, r_eclass_rd])
            rho_green = fluxm / (fluxpix + fluxter) # (neglect adj. correction)

            thr_gre_red = 1.08 # was initially 1.15
            if (rho_green.sum() >= rho_red.sum() * thr_gre_red):
                return # OK, no visib iteration necessary

            if visib   < 26.0:
                visib += 3.0
            elif visib < 30.0:
                visib += 4.0
            elif visib < 40.0:
                visib += 5.0
            elif visib < 40.0:
                visib += 5.0
            elif visib < 100.0:
                visib += 10.0
            else: visib = 120.0

            iter = 0
            # index_red = self.tables.B04
            # index_green = self.tables.B03
            rhom_red = 0.02
            rhom_green = 0.01

            while (iter < 6) & (visib < 120) & (rhom_green < thr_gre_red * rhom_red):
                iter = iter + 1
                self._visibility = visib
                self.altit3_atm() # update lp, e0t, edift, etc

                # red band (index_red, dn_red)
                if 'LANDSAT' in self.config.spacecraftName:
                    index = self.tables.reindex(Band.RED) -1
                else:
                    index = self.tables.reindex(Band.RED)
                fluxpix = self._e0th[index, r_eclass_rd] * r_cbeta_rd * sc1 + self._edifth[index, r_eclass_rd] * (
                    self._tsunh[index, r_eclass_rd] * r_cbeta_rd * sc1 / cos(radians(self._solze))
                    + (1.0 - self._tsunh[index, r_eclass_rd]) * r_vsky_rd * 0.01
                )  # (binary matrix b=1)

                fluxter = maximum(
                    (
                        (self._e0th[index, r_eclass_rd] * cos(radians(self._solze)) + self._edifth[
                            index, r_eclass_rd])
                        * (100 - r_vsky_rd)
                        * 0.01
                        * self._reflter[index]
                    ),
                    (self._e0th[index, r_eclass_rd] * cos(radians(self._solze)) * 0.01),
                )
                # fix for SIIMPC-890, UMW: dn_red need not to be global
                fluxm = pi * (self.config.d2 * ravel(band_red_radiance)[mlistref] - self._lph[index, r_eclass_rd])
                rho_red = fluxm / (fluxpix + fluxter) # (neglect adj. correction)
                fluxpix = 0 ; fluxter = 0 ; fluxm = 0

                # green band (Band.GREEN, radiance_green)
                if 'LANDSAT' in self.config.spacecraftName:
                    index = self.tables.reindex(Band.GREEN) -1
                else:
                    index = self.tables.reindex(Band.GREEN)
                fluxpix = self._e0th[index, r_eclass_rd] * r_cbeta_rd * sc1 + self._edifth[index, r_eclass_rd] * (
                    self._tsunh[index, r_eclass_rd] * r_cbeta_rd * sc1 / cos(radians(self._solze))
                    + (1.0 - self._tsunh[index, r_eclass_rd]) * r_vsky_rd * 0.01
                )  # (binary matrix b=1)

                fluxter = maximum(
                    (
                        (self._e0th[index, r_eclass_rd] * cos(radians(self._solze)) + self._edifth[
                            index, r_eclass_rd])
                        * (100 - r_vsky_rd)
                        * 0.01
                        * self._reflter[index]
                    ),
                    (self._e0th[index, r_eclass_rd] * cos(radians(self._solze)) * 0.01),
                )

                fluxm = pi * (self.config.d2 * ravel(band_green_radiance)[mlistref] - self._lph[index, r_eclass_rd])
                rho_green = fluxm / (fluxpix + fluxter) # (neglect adj. correction)
                fluxpix = 0 ; fluxter = 0 ; fluxm = 0

                rhom_red = rho_red.mean()
                rhom_green = rho_green.mean()
                if self.logger.level == logging.DEBUG:
                    print('iteration, visibility: ', iter, visib)

                if visib < 26.0:
                    visib += 3.0
                elif visib < 30.0:
                    visib += 4.0
                elif visib < 40.0:
                    visib += 5.0
                elif visib < 40.0:
                    visib += 5.0
                elif visib < 100.0:
                    visib += 10.0
                else: visib = 120.0

                if self.logger.level == logging.DEBUG:
                    print('visbility increased to: ', visib)
                if visib == 120:
                    break

        # if visib=29, 39, etc then round to visib=30, 40
        if int((visib + 0.5) % 10 == 9):
            visib = 10.0 * int(visib / 10 + 0.5)
            if self.logger.level == logging.DEBUG:
                print('visbility rounded to: ', visib)

        self._visib_check_neg = visib
        self._meanvi = uint8(indexvis(visib, self._nvisx, self._vis_reg))
        self._meanvi_check_neg = self._meanvi
        self._visibility = visib
        return

    def dtm_flat(self):
    # Flat terrain, provide same arrays as for rugged terrain
    # to treat both cases in the same code (although there is a time penalty)
    #
    # Input: (from Common blocks)
    # altit = average ground elevation of scene [km]
    # first_band, n_bands, n_bands: first, last band, number of bands
    # n_rows, n_cols: number of image columns and rows
    # config.solze_arr: SZA (deg) at scene corners, fltarr(2,2)
    #
    # Output: (Common blocks in atcor3.inc)
    # n_alti1 = number of altitude (elevation) classes with 20 m grid up to 2500 m = 126
    # n_alti = 151,176 (for max height=3000m, 3500m, respectively)
    # iav_ele = elevation index for average elevation region
    # cntback: number of background pixels
    # cnt_nonb: " non- " "
    # liback: list of background pixels
    # li_nonback list of non-background pixels
    # dtm_h: height, elevation in [m], integer*2: 2*n_rows*n_cols
    # vsky: sky view factor byte: n_rows*n_cols
    # cbeta: cos of incidence angle * 255 byte: n_rows*n_cols
    # eclass: elevation class for each pixel byte: n_rows*n_cols
    # ele_class = zeros(n_alti), -1: no pixels in this class

        av_ele = self._altit * 1000 # elevation height in meter
        elmin = (maximum((minimum(asarray(av_ele + 0.5).astype(int16), 3500)), 0))
        self._elmax = elmin
        #dtm_h = zeros((self.config.nrows, self.config.ncols),int) + elmin  # commented because not used in dtm_flat() function (JL)

        self._n_alti = 126 # 126 grid points, 0(100)2500 m (range of LUTs)
        self._n_alti1 = 126
        if (self._elmax > 2500 and self._elmax <= 3000):
            self._n_alti = 151 # extrapolation
        if (self._elmax > 3000):
            self._n_alti = 176 # "

        imin_alti = asarray(0.5 + elmin / 20.).astype(int16)

        self._altitude_grid_km = arange(self._n_alti, dtype=float32) * 0.02 # 0.02 km grid

        self._ele_class = zeros(self._n_alti) - 1 # default: negative index means no pixels in this elevation class
        self._ele_class[imin_alti] = 1 # all pixels in this class

        self._eclass = zeros((self.config.nrows, self.config.ncols), uint8) + imin_alti  # casted to uint8 like when terrain is not flat
        self._iav_ele = imin_alti # index for arrays ele_class, eclass
        self._iav_ele_ref = self._iav_ele

        # search for geocoded background pixels with DN=0
        map = self.getNodataMap()
        self._liback = where32(ravel([map == 0]))
        self._cntback = self._liback.size
        self._li_nonback = where32(ravel([map > 0]))
        self._cnt_nonb = self._li_nonback.size

        # cbeta = cos(local illumination)
        # a bilinear interpolation of config.solze_arr is employed
        nrows = self.config.nrows
        ncols = self.config.ncols

        if 'LANDSAT' in self.config.spacecraftName:
            sza_arr2 = zeros([nrows, ncols], dtype=float32) + self.config.solze
        else:
            try:
                x = arange(nrows, dtype=float32) / (nrows - 1) * self.config.solze_arr.shape[0]
                y = arange(ncols, dtype=float32) / (ncols - 1) * self.config.solze_arr.shape[1]
                # casted to float32 because rectBivariateSpline returns float64 (JL)
                sza_arr2 = rectBivariateSpline(x, y, self.config.solze_arr).astype(float32)
                del x
                del y
            except:
                sza_arr2 = ones([nrows, ncols], dtype=float32) * self.config.solze_arr.mean()
        # try:
        #     x = arange(nrows, dtype=float32) / (nrows - 1) * self.config.solze_arr.shape[0]
        #     y = arange(ncols, dtype=float32) / (ncols - 1) * self.config.solze_arr.shape[1]
        #     # casted to float32 because rectBivariateSpline returns float64 (JL)
        #     sza_arr2 = rectBivariateSpline(x, y, self.config.solze_arr).astype(float32)
        #     del x
        #     del y
        # except:
        #     sza_arr2 = ones([nrows, ncols], dtype=float32) * self.config.solze_arr.mean()
        self._cbeta = (self._sc_bet * cos((sza_arr2) * self._dtor) + 0.5).astype(uint8)
        del sza_arr2
        self._vsky = zeros((self.config.nrows, self.config.ncols), uint8) + 100 # sky view factor 0-100, 100=hemisphere
        self._ibrdf = 0 # reset to 0 for flat terrain
        self._iter_terrain = 0 # used to simplify rho_retrieval_step2
        # (part with terrain influence is skipped)
        return

    #@profile
    def dtm_array(self):
    # update coordinates of selected sub-image (rows, columns)
    # determine elevation class for each non-background pixel
    # there are up to 176 elevation classes:
    # the number of required classes is determined from the max. elevation
    #
    # class 0: elevation < 10 m center: 0 m
    # class 1: elevation 10 - 30 m " 20 m
    #
    # class 125 elevation 2490 - 2510 m 2500 m
    #
    # class 150 elevation 2490 - 3010 m 3000 m
    # class 175 elevation > 3490 m 3500 m


    # input data:
    # -----------
    # dtm_h: height, elevation in [m], integer*2: 2*n_rows*n_cols
    # dtm_s: slope[degree], byte, integer or float, if float: 4*n_rows*n_cols
    # dtm_a: aspect[degree], integer or float, if float: 4*n_rows*n_cols
    # eclass: elevation class for each pixel, byte: n_rows*n_cols
    # vsky: sky view factor (optional) byte: n_rows*n_cols
    # toposhad: topographic shadow, cast shadow (optional) byte: n_rows*n_cols
    # input data are processed in such a sequence that the max. memory requirement
    # depends on 3 float arrays: dtm_s, dtm_a, cbeta: 3*4*n_rows*n_cols=12*n_rows*n_cols

    # thv=sensor view angle=0, phiv=view azimuth angle=0 (only for ibrdf > 2)

    # output data:
    # -----------
    # ierror = 1: error with DEM file(s)
    # ierror = 2: stripes in cbeta=illumination and cancel (only interactive mode)
    # ierror = 3: less than 1% of image has slope angles > 6 deg and max height difference < 500m
    # in this case a flat DEM with average height is used to avoid possible slope artifacts
    #
    # n_alti = number of altitude classes
    # = 126 if max. elevation is el < 2500 m
    # = 151 if max. elevation is 2500 < el <= 3000 m
    # = 176 if max. elevation is el <= 3500 m

    # n_alti1 = number of altitude (elevation) classes with 20 m grid up to 2500 km = 126

    # altit = average terrain altitude [km]
    # iav_ele = elevation index for average elevation region

    # cntback ; number of background pixels
    # cnt_nonb: " non- " "
    # liback: list of background pixels
    # li_nonback list of non-background pixels
    # dtm_h: height, elevation in [m], integer*2: 2*n_rows*n_cols (or float: 4*n_rows*n_cols)
    # vsky: sky view factor byte: n_rows*n_cols
    # cbeta: cos of incidence angle * 255 byte: n_rows*n_cols
    # eclass: elevation class for each pixel byte: n_rows*n_cols
    # ---------------------
    # total = 5 *n_rows*n_cols
    #
    # cbetr: cos of reflected angle * 255 (ibrdf>2) byte: n_rows*n_cols
    # (obsolete, removed)
    #
    # array cbeta = illumination image is written to file
    # scaling of cbeta is 100 for the illumination file (easy to interpret)
    # scaling of cbeta is sc_bet=255 internally to have improved accuracy
    #
    # If "xxx" is the input image file name (without extension) then the name
    # of the illumination file is "xxx_ilu.bsq" . It is written to the directory
    # of the output image file.
    # An existing float ilu file with internal raw geometry (IGM) is used
    # and it will not be calculated again (and no byte ilu file will be written).

        av_ele = self.config.altit * 1000 # elevation height in meter
        elmin = (maximum((minimum(asarray(av_ele + 0.5).astype(int16), 3500)), 0))
        ierror = 0

        # 1. read DEM slope
        # -----------------
        if(self.tables.hasBand(Band.SLOPE)): #self.tables.SLP
            #dt = self.tables.getDataType(self.tables.SLP)

            dtm = self.tables.getBand(Band.SLOPE)
            mask = self.getNodataMap()
            dtm[mask == 0] = 0
            # convert slope from "degree" into "radian" and cast it to float32:
            if self.config.resolution == 10:
                dtm_s = radians(dtm).astype(float32)
            else:    
                dtm_s = radians(dtm).astype(float32)
            self.tables.setBand(Band.SLOPE, dtm)
            del dtm

        else:
            self.logger.fatal('no slope found for DEM')
            return False

        # 2. read DEM aspect (degree with respect to north, 90=east)
        # ----------------------------------------------------------
        #if(self.tables.hasBand(self.tables.ASP)):
        #    #dt = self.tables.getDataType(self.tables.ASP)
        #    dtm_a = clip(self.tables.getBand(self.tables.ASP),0,4000)
        #    dtm_a[mask == 0] = 0
        #    self.tables.setBand(self.tables.ASP,dtm_a)

        #else:
        #    self.logger.fatal('no aspect found for DEM')
        #    return False

        # search for geocoded background pixels with DN=0
        self._liback = where32(ravel([mask == 0]))
        self._cntback = self._liback.size
        self._li_nonback = where32(ravel([mask > 0]))
        self._cnt_nonb = self._li_nonback.size

        '''
        # cbeta = angle between the solar ray and the surface normal for each pixel
        # if cbeta < 0 the direct solar beam does not reach the pixel (shadow region)
        # in this case cbeta is reset to 0 to suppress the direct solar contribution
        # cbeta is also set to 0 if shadow is cast from surrounding topography,
        # The unit of solze, solaz and dtm_a is degree
        # make arrays for 2-D interpolation
        nrows = self.config.nrows
        ncols = self.config.ncols
        cbeta1 = zeros([nrows,ncols],dtype=float32)
        x = arange(nrows, dtype=float32) / (nrows-1) * self.config.solze_arr.shape[0]
        y = arange(ncols, dtype=float32) / (ncols-1) * self.config.solze_arr.shape[1]
        sz = rectBivariateSpline(x,y,self.config.solze_arr)
        x = arange(nrows, dtype=float32) / (nrows-1) * self.config.solaz_arr.shape[0]
        y = arange(ncols, dtype=float32) / (ncols-1) * self.config.solaz_arr.shape[1]
        az = rectBivariateSpline(x,y,self.config.solaz_arr)

        for i in range(self.config.nrows):
            rad_sz_i = radians(sz[i,:])
            dtm_s_i = dtm_s[i,:]
            rad_az_dtm_a_i = radians(az[i,:] - dtm_a[i,:])
            cbeta1[i,:] = cos(rad_sz_i) * cos(dtm_s_i) + sin(rad_sz_i) * sin(dtm_s_i) * cos(rad_az_dtm_a_i)
        '''

        # attention, this is implemented new due to adaptation of GDAL hillshadow utility
        # and replaces the algorithm above
        if(self.tables.hasBand(Band.SHADOW_MAP)): #self.tables.SDW
            #dt = self.tables.getDataType(self.tables.SDW)
            toposhad = self.tables.getBand(Band.SHADOW_MAP)
            toposhad[mask == 0] = 0
            self.tables.setBand(Band.SHADOW_MAP, toposhad)
            self._cbeta = toposhad.astype(float32) / 255.0
            del toposhad

        self._cbeta = clip(self._cbeta, 0, self._cbeta.max())
        if(len(self._cbeta[mask > 0]) > 0):
            sig0 = self._cbeta[mask > 0].std() * 100
        else:
            sig0 = 1.0

        # get elevation data (height unit: meter, decimeter, or centimeter)
        # decimeter or centimeter are converted into meter
        # DEM_cell_size = pixel_size has unit meter !
        # ----------------------------------------------------------
        if(self.tables.hasBand(Band.DIGITAL_ELEVATION_MAP)): #self.tables.DEM
            #dt = self.tables.getDataType(self.tables.DEM)
            dtm_h = self.tables.getBand(Band.DIGITAL_ELEVATION_MAP) #dtm_h = self.tables.getBand(self.tables.DEM)
            dtm_h[mask == 0] = 0
            self.tables.setBand(Band.DIGITAL_ELEVATION_MAP, dtm_h)
        else:
            self.logger.fatal('no DEM found')
            return False

        self._elmax = int(dtm_h.max())  # get max & min elevation for specified scene
        li = where32(ravel(self._cbeta < 0.0))
        if (li.size > 1):
            shad = zeros([self.config.nrows, self.config.ncols], uint8) + 1
            shad[li] = 0
            shad = median_filter_2d(shad, 3)  # remove isolated shadow pixels
            self._cbeta = self._cbeta * shad

        sig1 = std(uint8(100 * ravel(self._cbeta)[self._li_nonback] + 0.5))
        if (sig1 > 1.0e-4):
            if (sig0 / sig1 >= 1.25 or sig0 / sig1 <= 0.80):
                self.logger.warning('potential problem with illumination channel')
                self.logger.warning('reason: DEM slope might have a lot of steps')
                self.logger.warning('if true: smooth DEM files (ele, slope, aspect')
                ierror = 2

        # cbeta is multiplied with sc_bet=255 and reduced to a byte array to save memory.
        # Therefore, in subsequent equations the term "c1=1./sc_bet" is used.
        self._cbeta *= self._sc_bet
        cbeta1 = clip(self._cbeta, 0, 255)
        self._cbeta = (cbeta1 + 0.5).astype(uint8)
        del cbeta1

        # end of ATCOR Goto SKIP1
        # entry point for"raw" geometry ilu.bsq (originally 4 bytes/pixel, but here already byte-scaled)

        # default sky view factor is derived from local slope angle
        # vsky = cos(slope/2)*cos(slope/2) = 0.5*(1 + cos(slope))
        # vsky is also multiplied with 100 and converted into a byte array
        # to save memory (dtm_s is already in "radian" unit here)
        # ------------------------------------------------------------------
        self._vsky = (0.5 * (1.0 + cos(dtm_s)) * 100.0 + 0.5).astype(uint8)
        # Check how many pixels have slope > 6 degrees
        # If percentage less than 1% and height difference < 300 m then switch to flat terrain
        # fix for SIIMPC-550.2, SIIMPC-802, UMW:
        try:
            hdiff = ptp(dtm_h[dtm_h>0])  # ignore background pixels -> This may also ignore elevation < 0 like dead sea (JL)
            # hdiff = ptp(dtm_h[dtm_h != 0])  # proposed modification (JL)
        except:
            hdiff = 0
        if (hdiff < self.config.AC_Dem_P2p_Val):
            cntx = dtm_s[dtm_s > self.config.AC_Slope_Th].size # count of all pixels where slope is above threshold
            if (cntx / self._cnt_nonb) < self.config.AC_Topo_Corr_Th:
                self._altit = 0.001 * dtm_h[dtm_h>0].mean()
                ierror = 3
                return ierror
        # end fix for SIIMPC-550.2

        del dtm_s
        # 4. read sky view factor from file if available, overwrite default.
        # sky view factor file is already scaled in the 0-100 range
        # ---------------------------------------------------------------
        # if(tables.hasBand(tables.SKY)):
        # vsky = tables.getChannel(tables.SKY)
        # vsky = vsky[first_row - 1:(n_rows - 1)+1,first_col - 1:(n_cols - 1)+1]

        # elevations < 0 (e.g. Dead Sea area) are reset to 0
        # array dtm_h must be >= 0
        dtm_h = asarray(maximum((dtm_h + 0.5), 0)).astype(int16)  # [m] unit

        if (self._cntback == 0 or self._li_nonback.size < 3):
            av_ele = dtm_h.mean()
        else:
            if(self._cnt_nonb != 0):
                av_ele = (ravel(dtm_h)[self._li_nonback]).mean()

        self._elmax = asarray(self._elmax).astype(int16)
        elmin = (minimum(asarray(elmin).astype(int16), 2500))

        # interpolated elevation increment is 20 m
        self._n_alti1 = 126
        self._n_alti = 126
        if (self._elmax > 2500 and self._elmax <= 3000):
            self._n_alti = 151
        if (self._elmax > 3000):
            self._n_alti = 176
        if (self._elmax > 3500):
            self._elmax = 3500

        imin_alti = (maximum(asarray(0.5 + elmin / 20.).astype(int16), 0))
        self._altitude_grid_km = arange(self._n_alti, dtype=float32) * 0.02
        delta_h = 20  # (m)

        self._ele_class = zeros(self._n_alti) - 1 # default: negative index means no pixels in this elevation class
        self._eclass = zeros([self.config.nrows, self.config.ncols], uint8)
        self._iav_ele = 0

        mask = ones([self.config.nrows, self.config.ncols], uint8)
        if (self._cntback > 0):
            ravel(mask)[self._liback] = 0

        for j in arange(imin_alti, (self._n_alti)):
            h1 = (j - 1) * delta_h + delta_h / 2
            h2 = h1 + delta_h
            ll = where32(ravel(bitwise_and(bitwise_and(dtm_h >= h1, dtm_h < h2), mask == 1))) # do not include background pixels!
            if (ll.size > 0):
                ravel(self._eclass)[ll] = uint8(j)
                self._ele_class[j] = 1
                # ele_class is a binary array (yes/no)
                # ele_class[j] =-1: no pixel in class j
                # = 1: there are pixels in elevation class j

        limax = where32(ravel(bitwise_and(dtm_h >= h2, mask == 1)))
        if (limax.size > 0):
            ravel(self._eclass)[limax] = self._n_alti - 1
            ravel(self._ele_class)[self._n_alti - 1] = 1

        # average elevation index iav_ele, weighted with frequency of pixels
        # in each altitude region
        # av_ele = average terrain elevation [m]
        # altit = average terrain altitude [km]
        self._altit = av_ele * 0.001
        self._iav_ele = (minimum(asarray(av_ele / delta_h).astype(int16), (self._n_alti - 1)))
        self._iav_ele_ref = self._iav_ele

        self._litopo_shad = where32(ravel(self._cbeta <= 1))  # topo shadow for ilu > 89.42 deg
        return ierror


    def write_output_file_a3(self, count, band):
    # for the current channel "band"
    # Input:
    # nk: band number (starting from 1)
    # band: current channel
    # wvlsen: sensor wavelength vector (Common block)
    # fcref: scale factor reflectance ( " " )
    # cntback: number of background pixels (" " )
    # liback: list of background pixels (" " )
    # Output:
    # "band" is written to file, then memory for band is freed
    # no other variables are modified

    # The lower bound for reflectance is set to 0.25% (veg. in the blue region)
    # 0.50% (veget. in the green region), and 0.25% (veg. in red region),
    # The default lower bound is 0% . The default is not always used because of
    # small possible overcorrections due to haze removal, too low visibility, or
    # small calibration errors.

        if(count < 0 or count > 12):
            self.logger.fatal('wrong band index: %d', count)


        # if 'LANDSAT' in self.config.spacecraftName:
        #     rho_low = 1.0 / self._dnScale  # 0.01% low reflectance with _dnScale = 10000.0
        #     band = clip(band, rho_low, band.max())

        if self.config.productVersion <= float32(14.6) :
            # not necessary for higher Versions with offset:
            rho_low = 1.0 / self._dnScale  # 0.01% low reflectance with _dnScale = 10000.0
            band = clip(band, rho_low, band.max())

        mask = self.getNodataMap()
        band[mask == 0] = 0
        band_index = count # self._band_index[count]

  #       if 'LANDSAT' in self.config.spacecraftName:
  #           band_index = count  # self._band_index[count]
  #           list_band=[Band.COASTAL_AEROSOL, Band.BLUE, Band.GREEN, Band.RED, \
  # Band.VEGETATION_1, Band.VEGETATION_2, Band.VEGETATION_3, Band.NEAR_INFRARED, Band.VEGETATION_4,
  # Band.WATER_VAPOUR_INPUT, Band.CIRRUS, Band.SHORT_WAVE_INFRARED_1, Band.SHORT_WAVE_INFRARED_2]
  #           self.tables.setTmpBand(list_band[band_index], (band).astype(uint16))

        if self.config.productVersion <= float32(14.6):
            self.tables.setTmpBand(band_index, (band * self._dnScale  + 0.5).astype(uint16))
        else:
            # note that offset values have a negative sign: SCOR-49
#            band[band!=0]=(band[band!=0]* self._dnScale + self.config.radio_add_offset_list[band_index] * -1 + 0.5)
            band_tmp = ndarray.copy(band)  # create a temporary copy of band array distinct from band
            band_tmp[band_tmp != 0] = (band_tmp[band_tmp != 0] * self._dnScale + self.config.radio_add_offset_list[band_index] * -1 + 0.5)

            # to replace all potential negative values with 1, 0 is kept for the background values:
#            band[band < 0] = 1
            band_tmp[band_tmp < 0] = 1
            band_tmp = clip(band_tmp, 0, 32767)  # SIIMPC 1803 65535
            # self.tables.setTmpBand(band_index, band.astype(uint16))
            self.tables.setTmpBand(band_index, band_tmp.astype(uint16))
            # self.tables.setTmpBand(band_index, (band * self._dnScale + self.config.radio_add_offset_list[band_index]*-1 + 0.5).astype(uint16))

        return

    def gap_processing2_a3(self, igap, iflag_interp, ibns, nk, band, chan2win):
    # All parameters are input parameters
    # igap is also output parameter, igap is modified upon entering the second time.
    #
    # Actions when first entering this routine (start of gap region):
    # 1a) the "left" gap channel array "band" is saved as "refl_w1" (in Common block).
    # 1b) the band index ibns is saved as ibns_save, nkl as nkl_save (in Common)
    # 1c) refl_w1 is written to file as the current reflectance channel
    #
    # Actions when entering this routine the second time (end of gap region):
    # 2a) The "right" gap channel "band" is available.
    # 2b) It is assigned to "refl_w2", and written to file at the appropriate positon ibns.
    # 2c) ibns = ibns_save is restored
    # 2d) The gap bands are interpolated using refl_w1 and refl_w2
    # Different weighting is applied for vegetation and non-vegetation during interpolation


    # Upon first entering:

        if (igap == 0 and iflag_interp == 0):
            self._refl_w1 = band
            nkl = nk # save left channel number
            self._nkl_save = nkl

        # Upon second entering
        if (iflag_interp == 1):
            refl_w2 = band # save rho(right window)
            nkr = nk # save right channel number
            nkl = self._nkl_save

            # nkr = 1600 nm band
            ib1 = nkr - 1 # Sentinel-2 case, interpolation for cirrus band
            ib2 = nkl # 940 nm band
            w1 = (self._wvlsen[nkr] - self._wvlsen[ib1]) / (self._wvlsen[nkr] - self._wvlsen[ib2])
            w2 = (self._wvlsen[ib1] - self._wvlsen[ib2]) / (self._wvlsen[nkr] - self._wvlsen[ib2])
            band = w1 * self._refl_w1 + w2 * refl_w2 #

            # write interpolated band (1.38 um cirrus) into file (B10)
            # -------------------------------------------------
            self.write_output_file_a3(ib1, band)

            # write 1.6 um band into file (B11)
            # ----------------------------
            self.write_output_file_a3(nkr, refl_w2)
            nvl = refl_w2[refl_w2 < 0].size
            # print('negative pixels from gap function ->',nvl)
            self.config.L2A_BOA_NEGATIVE_VALUES_LIST[nkr] = nvl
            igap = 0
            iflag_interp = 0

        return igap, iflag_interp


    def find_vegetation_pixels_vari_a3(self, b, ga):

    # IDL2PY_TBD @atcor3.inc

    # Calculate list of vegetation pixels for brdf correction
    # in order to perform a separate BRDF correction for vegetation.
    # This requires the atm/topo corrected surface reflectance in the Red/NIR
    # to obtain the vegetation index and thus the vegetation pixel positions.
    #
    # Same routine as "find_vegetation_pixels..." but for variable visibility
    # The atmospheric correction functions (e0tx, etc.) are evaluated for the mean vis. index

    # Input parameters:
    # b = binary shadow matrix b(n_rows,n_cols)
    # ga = ga(n_rows,n_cols) without vegetation consideration, range 0-1, float
    # (and quantities from Common blocks)
    #
    # Output parameters
    # list_vege: list of vegetation pixels

        sc1 = 1.0 / self._sc_bet

        self._visib_save = self._visibility.copy()
        self._visibility = self._visext[self._meanvi]
        self.altit3_atm()
        self._visibility = self._visib_save

        # RED band
        # --------
        nk = self._red_band
        band = self.tables.getBand(nk)
        band = self.refl2rad(band, nk)
        vsky = self._vsky
        fluxpix = (self._e0th[nk, self._eclass] * self._cbeta * sc1 + self._edifth[nk, self._eclass] * (b * self._tsunh[nk, self._eclass] * self._cbeta * sc1 / cos(radians(self._solze)) + (1.0 - b * self._tsunh[nk, self._eclass]) * vsky * 0.01))
        fluxter = (maximum(((self._e0th[nk, self._eclass] * cos(radians(self._solze)) + self._edifth[nk, self._eclass]) * (100 - vsky) * 0.01 * self._reflter[nk]), (self._e0th[nk, self._eclass] * cos(radians(self._solze)) * 0.01)))
        fluxm = pi * (self.config.d2 * band - self._lph[nk, self._eclass])
        rho_red = (maximum((ga * fluxm / (fluxpix + fluxter)), 0.01))

        # NIR band
        # --------
        nk = self._nir_band
        band = self.tables.getBand(nk)
        band = self.refl2rad(band, nk)
        fluxpix = (self._e0th[nk, self._eclass] * self._cbeta * sc1 + self._edifth[nk, self._eclass] * (self._tsunh[nk, self._eclass] * self._cbeta * sc1 / cos(radians(self._solze)) + (1.0 - b * self._tsunh[nk, self._eclass]) * vsky * 0.01))
        fluxter = (maximum(((self._e0th[nk, self._eclass] * cos(radians(self._solze)) + self._edifth[nk, self._eclass]) * (100 - vsky) * 0.01 * self._reflter[nk]), (self._e0th[nk, self._eclass] * cos(radians(self._solze)) * 0.01)))
        fluxm = pi * (self.config.d2 * band - self._lph[nk, self._eclass])
        rho_nir = (maximum((ga * fluxm / (fluxpix + fluxter)), 0.01))

        fluxpix = 0
        fluxter = 0
        fluxm = 0 # free memory

        # list of vegetation pixels: rho_nir > 0.1 and rho_nir/rho_red > 3
        # -----------------------------------------------------------------
        list_vege = where32(ravel(bitwise_and(rho_nir > 0.1, rho_nir / rho_red > self.config.AC_Vegetation_Index_Th)))
        cnt_vege = list_vege.size
        rho_nir = 0
        rho_red = 0

        if (cnt_vege > 0):
            # mask = zeros([self.config.ncols * self.config.nrows], uint8)
            mask = zeros((self.config.ncols , self.config.nrows), uint8)
            # mask[list_vege] = 255
            ravel(mask)[list_vege] = 255

            # remove isolated vegetation pixels
            mask = median_filter_2d((mask), 5)
            list_vege = where32(ravel(mask == 255))

            mask = 0 # free memory

        return list_vege

    #@profile
    def prepare_rho_retrieval(self):

    # Purpose: do all preparations independent of cell loops
    # This includes haze/water and sunglint/water correction,
    # the de-haze/de-glinted scene is written to filetmp which is later
    # deleted (end of rho_retrieval_step1)
    #
    # Output:
    # nw = number of bands in strong wv regions (940, 1130, 1400, 1800 nm)
    # ju = water vapor interval (0 - 6)
    # cnt_nonback = number of non-background pixels
    # chan2win = zeros(4) for the 940/1130, and 1400/19000 nm regions
    # 1: left wv window channel, 2: left and right window channel available
    # li_stat = list of non-background pixels used for statistics (with a pixel skip factor)
    # icloud = 0 (cloud percentage < 1%) otherwise icloud=1
    # li_clear = list of clear scene pixels (no cloud, for terrain and adj. correction)
    # b = binary shadow matrix (1=sunlit, 0=shadow)
    # cnt_vege = number of vegetation pixels for BRDF correction
    # ga = matrix for BRDF correction
    # ga_nir = matrix for BRDF correction (vegetation)
    # nch_wvp = zeros(n_bands) (0= no wv band, 1=wv band)
    #
    # Structure
    # ! read_hcw_file_get_cirrus
    # ! find_vegetation_pixels_vari_a3

        self._nth_out = self._n_bands - self._first_band + 1
        self._band_saturated = zeros([self._n_bands], float32) # percent of saturated pixels (percent)
        # update lp_all, e0t_all, lp_fit, q_fit etc.
        # the update is performed for the scene center geometry. It is not critical as it only
        # serves to determine vegetation and non-vegetation pixels which are treated
        # differently in the spectral reflectance interpolation in the 1400, 1800 nm regions
        self._solze = float32(mean(self.config.solze_arr))
        self._solaz = float32(mean(self.config.solaz_arr))
        self._thv = float32(mean(self.config.vza_arr))
        self._phiv = float32(mean(self.config.vaa_arr))
        self.apda1_lut_constvis_a3() # update lp_all, e0t_all, lp_fit, q_fit etc.
        # list of non-background pixels
        if (self._cntback > 0):
            mask = zeros([self.config.nrows, self.config.ncols], uint8)
            ravel(mask)[self._liback] = 255
            self._li_nonback = where32(ravel(mask == 0))
            cnt_nonback = self._li_nonback.size
        else:
            cnt_nonback = self.config.ncols * self.config.nrows
            #cnt_nonback = asarray(self.config.ncols).astype(int16) * self.config.nrows
            self._li_nonback = arange(cnt_nonback, dtype=int32)

        if (self._ibrdf > 0 and self._ibrdf_ini > 0 and asarray(self._np_snow).astype(float32) / cnt_nonback > 0.5):
            self._ibrdf = 0 # a note is given in the .log file that brdf correction was switched off
        # ibrdf_ini < 0 forces BRDF correction

        # binary shadow matrix
        b = ones([self.config.nrows, self.config.ncols], uint8) # b(i,j) = 1 (unit matrix)
        ll = where32(ravel(self._cbeta <= 0))  # determine shadow pixels
        if (ll.size > 0):
            ravel(b)[ll] = 0 # b(i,j) = 0 for shadow pixels

        # if number of cloud pixels is > 1% of image then determine non-cloud=clear pixels
        # and replace the reflectance of cloud pixels by average reflectance of clear pixels,
        # but only for the adjacency correction: clouds are high in the atmosphere and do not
        # contribute to the adjacency effect.
        icloud = 0
        np_cirrus = 0
        li_clear = array(0)

        if (self.cirrus_correction):
            level = 1 # thick cirrus pixels
            mlist_cirrus = self.read_hcw_file_get_cirrus(level)
            np_cirrus = mlist_cirrus.size

        if ((np_cirrus + self._np_cloud) / float32(cnt_nonback) > 0.01):
            icloud = 1
            x1 = ones([self.config.nrows*self.config.ncols], uint8)
            if (self._np_cloud > 0):
                x1[self._mlist_cloud] = 0
            if (np_cirrus > 0):
                x1[mlist_cirrus] = 0
            self._mlist_cloud = where32(ravel(x1 == 0))  # contains "normal" cloud and thick cirrus cloud !
            self._np_cloud = self._mlist_cloud.size
            if (self._cntback > 0):
                ravel(x1)[self._liback] = 0
            li_clear = append(self._mlist_clear, self._mlist_water) #from sen2cor 2.10.2
            # li_clear = where32(ravel(x1 == 1)) from sen2cor 2.10.2

        self._iflag_interp725 = 0
        self._iflag_interp825 = 0

        nch_wvp = array([0])
        self._gap = zeros([4, 2], uint8) # start/stop channels for 940, 1130, 1440, 1900 regions
        nw = 0
        ju = 0

        if (self.config.iwaterwv > 0 and self.config.iwaterwv <= 3):

            # use average wv to update lp, e0t etc
            # this also provides the q, spha values for adj. correction based on scene-averaged wv
            # In addition, the scene-average visib is used for the wv channels (as calculated in ref_pixel)
            if (self._wv_av <= self._uu1_altit[self._iav_ele // 5, 1]):
                j1 = 0
            else:
                if (self._wv_av <= self._uu1_altit[self._iav_ele // 5, 3]):
                    j1 = 2
                else:
                    j1 = 4 # (last else required for nuu1=4,5)
            if (self._nuu1 > 5):
                if (self._wv_av > self._uu1_altit[self._iav_ele // 5, 3] and self._wv_av <= self._uu1_altit[self._iav_ele // 5, 4]):
                    j1 = 4
                if (self._wv_av > self._uu1_altit[self._iav_ele // 5, 4]):
                    j1 = 6
            if (self._nuu1 == 4 and j1 > 2):
                j1 = 2 # (old version has only 4 wv grid points)
            ju = j1 # (used in polx3)

            nw = asarray(sum(self._iabs_region[0:4])).astype(int16) # number of bands in strong water vapor regions
            nch_wvp = zeros(self._n_bands_all)
            if (nw >= 1): # (i.e. in 940, 1130, 1400-1500, 1800-1980 nm)
                self._gap = zeros([4, 2], int8)-1 # start/stop channels for 1440, 1900 regions
                # channels in gap regions will be interpolated
                # bands in strong wv regions 1440 and 1900 nm exist

                if (self.config.intpol940_1130 > 0):
                    j = 0
                    li = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= self.config.wl940a[0], \
                                                 self._wvlsen[self._first_band:self._n_bands] <= self.config.wl940a[1])))
                    if (li.size > 1):
                        nch_wvp = li # count bands from 0
                        self._gap[j, 0] = li[0]
                        self._gap[j, 1] = li[li.size - 1]


                    j = 1
                    li = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= self.config.wl1130a[0], \
                                                 self._wvlsen[self._first_band:self._n_bands] <= self.config.wl1130a[1])))
                    if (li.size > 1):
                        if (nch_wvp.size == 0):
                            nch_wvp = li
                        else:
                            nch_wvp = array([nch_wvp, array([li])])
                        self._gap[j, 0] = li[0]
                        self._gap[j, 1] = li[li.size - 1] # ( end of config.intpol940_1130 section)

                if (self.config.intpol1400 > 0):
                    for j in arange(2, 4):
                        li = where32(ravel(self._iabs_chan[j,:] > 0))
                        if (li.size > 1):
                            iac = self._iabs_chan[j, li]
                            # concatenate nch_wvp channels
                            if (nch_wvp.size == 0):
                                nch_wvp = self._iabs_chan[j, li]
                            else:
                                nch_wvp = array([nch_wvp, iac])
                            self._gap[j, 0] = self._iabs_chan[j, li[0]]
                            self._gap[j, 1] = self._iabs_chan[j, li[li.size - 1]] # ( end of config.intpol1400 section)
                    # (of nw)


        self._cnt_brdf = 0 # number of pixels involved in brdf correction
        sc1 = 1.0 / self._sc_bet
        ga = 0
        ga_nir = 0 # default, no brdf
        cnt_vege = 0

        if (self._ibrdf > 0):

            bthr = cos(self._beta_thr * self._dtor)
            # cbeta ge 0 was checked in routine "dtm_array"
            if (self._ibrdf == 1 or self._ibrdf == 11 or self._ibrdf == 12):
                ga = (maximum((self._cbeta * sc1 / bthr), self.config.thr_g))
            if (self._ibrdf == 2 or self._ibrdf == 21 or self._ibrdf == 22):
                ga = (maximum(sqrt(maximum((self._cbeta * sc1 / bthr), 0)), self.config.thr_g))

            if (self._ibrdf > 10):
                list_vege = self.find_vegetation_pixels_vari_a3(b, ga)
                cnt_vege = list_vege.size

            ga_nir = 0
            if (cnt_vege > 0):
                ga_nir = ga
                # insert ga values for vegetation (lambda < 750 nm)
                # ga = (cbeta/cos(sz)) ^ 0.75
                ravel(ga)[list_vege] = (maximum((maximum((ravel(self._cbeta)[list_vege] * sc1 / bthr), 0)) ** 0.75, self.config.thr_g))

                # insert ga values for vegetation for lambda >= 750 nm
                # (a)=(1) ga = (cbeta/cos(beta_thr)) ^ 0.333 or
                # (b)=(2) ga = (cbeta/cos(beta_thr))
                if (self._ibrdf % 10 == 1):
                    ravel(ga_nir)[list_vege] = (maximum((maximum((ravel(self._cbeta)[list_vege] * sc1 / bthr), 0)) ** 0.3333, self.config.thr_g))
                if (self._ibrdf % 10 == 2):
                    ravel(ga_nir)[list_vege] = (maximum((maximum((ravel(self._cbeta)[list_vege] * sc1 / bthr), 0)), self.config.thr_g))

            # background pixels are assigned ga=1.0
            if (self._cntback > 0):
                ravel(ga)[self._liback] = 1.0

            ll1 = where32(ravel(ga > 1.0))  # pixels not involved in BRDF corr.
            if (ll1.size > 0):
                ravel(ga)[ll1] = 1.0 # " " "
            ll1 = 0
            ga = uint8(ga * self._sc_bet + 0.5) # make byte array to save memory

            if (cnt_vege > 0):
                ll1 = where32(ravel(ga_nir > 1.0))
                if (ll1.size > 0):
                    ravel(ga_nir)[ll1] = 1.0
                ga_nir = uint8(ga_nir * self._sc_bet + 0.5)
                ll1 = 0

            self._dn_band4 = 0

            # cnt_brdf is printed in log file; ga < sc1: BRDF correction takes place
            ll1 = where32(ravel(ga < self._sc_bet))
            self._cnt_brdf = ll1.size
            ll1 = 0

        # 1: indicates left wv window channel, 2: left and right window channel available
        chan2win = ones(4) # for the 940/1130, and 1400/19000 nm regions
        if (self._ch940w1 + self._ch940w2 == 0):
            chan2win[0] = 0 # only SWIR wv channels
        if (self._ch1130w1 + self._ch1130w2 == 0):
            chan2win[1] = 0
        if (self._ch940w1 > 0 and self._ch940w2 > 0):
            chan2win[0] = 2
        if (self._ch1130w1 > 0 and self._ch1130w2 > 0):
            chan2win[1] = 2
        if (self._iabs_region[2] > 0):
            # 1400 nm
            if (self._n_bands > self._iabs_chan[2, self._iabs_region[2] - 1]):
                chan2win[2] = 2
        if (self._iabs_region[3] > 0):
            # 1900 nm
            if (self._n_bands > self._iabs_chan[3, self._iabs_region[3] - 1]):
                chan2win[3] = 2

        li = where32(ravel(bitwise_and(self._wvlsen >= 1.25, self._wvlsen <= 1.5)))
        if (li.size > 0 and self._n_bands_all < self._n_bands_min_interp):
            # Sentinel-2 case: a cirrus band but only 12 channels
            if (li.size > 0):
                li1 = where32(ravel(self._wvlsen < 1.25))
                li2 = where32(ravel(self._wvlsen > 1.50))
                if (li1.size > 0 and li2.size > 0):
                    chan2win = zeros(4) + 2 # two window bands per absorption region
                    nch_wvp = li # band(s) in (cirrus) absorption region
                    self.config.intpol1400 = 1
                    self._gap[2, 0] = li1[li1.size - 1] # 1400 nm region, band number starts with 0
                    self._gap[2, 1] = li2[0]

        # li_stat = list of non-background pixels used for statistics
        skipf = 10
        if (cnt_nonback < 1.0e6):
            skipf = 1
        if (cnt_nonback > 50.0e6):
            skipf = 50
        n1 = (maximum((cnt_nonback / skipf), 1))
        li_stat = self._li_nonback[arange(n1, dtype=int16) * skipf]
        #li_stat = li_stat.astype(uint32)

        # free memory of self._li_nonback only if no background pixels are present
        # otherwise self._li_nonback is required for rho_retrieval step 2 (adjacency)
        if (self._cntback == 0):
            self._li_nonback = 0


        # cbeta < 26 means beta = acos(26./255)*!RADEG = 84.1 degree
        # cbeta is reset to 26 if smaller than 26 to avoid overcorrection during
        # standard topographic correction. cos(84 deg) = 0.1028, the inverse is a
        # factor of 9.73 (for the assumed Lambertian reflector). A very large factor
        # is difficult to compensate during the empirical BRDF correction.
        self._cbeta = (maximum(self._cbeta, 26))

        return(nw, ju, cnt_nonback, chan2win, li_stat, icloud, li_clear, b, cnt_vege, ga, ga_nir, nch_wvp)

    #@profile
    def rho_retrieval_step1(self, nw, ju, li_stat, b, cnt_vege, ga, ga_nir):

    # Purpose: first step of surface reflectance retrieval (without terrain iteration, without adj. correction)
    #
    # Input:
    # nw = number of bands in strong wv regions (940, 1130, 1400, 1800 nm)
    # ju = water vapor interval (0 - 6)
    # li_stat = list of non-vegetation pixels used for statistics (with a pixel skip factor)
    # cnt_vege = number of vegetation pixels for BRDF correction
    # b = binary shadow matrix (1=sunlit, 0=shadow)
    # ga = matrix for BRDF correction
    # ga_nir = matrix for BRDF correction (vegetation)
    # list_clear_water = list of clear water pixels (for haze removal over water)
    #
    # Output:
    # filou_tmp: file containing surface reflectance cube of step1
    #
    # Structure:
    # ! at_log
    # ! altit3v_atm_a3
    # ! read3v_atm_hyper_a3

    # filetmp: file for haze/water and sunglint/water corrected data.
    # Contrary to haze over land, ALL bands are sun-glint/water corrected
    # although for haze/water a correction of the visisble bands is sufficient.
    # But ihaze=2, 3 includes haze/water and sunglint/water.

        iswitch_nir = 0

        if 'LANDSAT' in self.config.spacecraftName:
            wv = self.tables.getBand(Band.WATER_VAPOUR) #/ 1000
        else:
            wv = array(0)

        if(self.tables.hasBand(Band.VISIBILITY) == False): #self.tables.VIS
            self.logger.fatal('no visibility channel present, giving up')

        bandvis = self.tables.getBand(Band.VISIBILITY)
        minvis = bandvis.min()
        maxvis = bandvis.max()
        sc1 = 1.0/self._sc_bet
        del bandvis
        if self.config.resolution == 60:
            factor = 1
        elif self.config.resolution == 30:
            factor = 2
        elif self.config.resolution == 20:
            factor = 3
        elif self.config.resolution == 10:
            factor = 6
        nrows = int(self.config.nrows / factor)
        ncols = int(self.config.ncols / factor)
        xcell = (self._xcell / factor).astype(int)
        ycell = (self._ycell / factor).astype(int)

        try:  # agular values are arrays:
            x = arange(nrows, dtype=float32) / (nrows - 1) * self.config.solze_arr.shape[0]
            y = arange(ncols, dtype=float32) / (ncols - 1) * self.config.solze_arr.shape[1]
            szi = rectBivariateSpline(x, y, self.config.solze_arr)
            x = arange(nrows, dtype=float32) / (nrows - 1) * self.config.solaz_arr.shape[0]
            y = arange(ncols, dtype=float32) / (ncols - 1) * self.config.solaz_arr.shape[1]
            sai = rectBivariateSpline(x, y, self.config.solaz_arr)
            del x
            del y
        except:  # angular value is a scalar:
            szi = ones([nrows, ncols], dtype=float32) * self.config.solze_arr.mean()
            sai = ones([nrows, ncols], dtype=float32) * self.config.solaz_arr.mean()

        vzi = ones([nrows, ncols], dtype=float32) * self.config.vza_arr
        vai = ones([nrows, ncols], dtype=float32) * self.config.vaa_arr

        # cell loop for RT terms needs lph, e0th(n_alti,nvisx,n_bands) etc.
        # calculated in altit3v_atm
        self._lph_cell = zeros([self._ny_cell, self._nx_cell, self._n_refl, self._nvisx, self._n_alti], float32)
        self._e0th_cell = zeros([self._ny_cell, self._nx_cell, self._n_refl, self._nvisx, self._n_alti], float32)
        self._edifth_cell = zeros([self._ny_cell, self._nx_cell, self._n_refl, self._nvisx, self._n_alti], float32)
        self._tsunh_cell = zeros([self._ny_cell, self._nx_cell, self._n_refl, self._nvisx, self._n_alti], float32)
        self._sphah_cell = zeros([self._ny_cell, self._nx_cell, self._n_refl, self._nvisx, self._n_alti], float32)
        self._qh_cell = zeros([self._ny_cell, self._nx_cell, self._n_refl, self._nvisx, self._n_alti], float32)

        for jx in arange(0, (self._nx_cell)):
            for jy in arange(0, (self._ny_cell)):
                # update geometry for current cell
                sub_r = array([ycell[jy, 0], ycell[jy, 1], xcell[jx, 0], xcell[jx, 1]])
                self._solze = float32(mean(szi[sub_r[0]:sub_r[1]+1, sub_r[2]:sub_r[3]+1]))
                self._solaz = float32(mean(sai[sub_r[0]:sub_r[1]+1, sub_r[2]:sub_r[3]+1]))
                self._thv = float32(mean(vzi[sub_r[0]:sub_r[1]+1, sub_r[2]:sub_r[3]+1]))
                self._phiv = float32(mean(vai[sub_r[0]:sub_r[1]+1, sub_r[2]:sub_r[3]+1]))

                self.altit3v_atm() # calculate RT fcts, calls read3v_atm_hyper_a3
                # yields lph(n_alti,nvisx,n_bands)
                self._lph_cell[jy, jx,:,:,:] = self._lph # fltarr(n_alti,nvisx,n_bands)
                self._e0th_cell[jy, jx,:,:,:] = self._e0th
                self._edifth_cell[jy, jx,:,:,:] = self._edifth
                self._tsunh_cell[jy, jx,:,:,:] = self._tsunh
                self._sphah_cell[jy, jx,:,:,:] = self._sphah
                self._qh_cell[jy, jx,:,:,:] = self._qh

        del vai
        del vzi
        del sai
        del szi
        del xcell
        del ycell

        # for ratio_blu_red < 0 no re-scaling of path radiance in the blue-red region
        # otherwise, re-scaling is only performed if sc_lp_blu differs more than 0.03 from 1.0
#        if (self._ratio_blu_red > 0.0 and Band.BLUE.value > 0 and (self.config.sc_lp_blu < (1 - self.config.AC_Aerosol_Type_Ratio_Th) or self.config.sc_lp_blu > (1 + self.config.AC_Aerosol_Type_Ratio_Th))):
        if (self._ratio_blu_red > 0.0 and (Band.BLUE.value -1) > 0 and (self.config.sc_lp_blu < (1 - self.config.AC_Aerosol_Type_Ratio_Th) or self.config.sc_lp_blu > (1 + self.config.AC_Aerosol_Type_Ratio_Th))):

            # update lph according to routine scale_path_radiance_wfov
            # because lph was overwritten by calling altit3v_atm in the previous cell loop
            if self.logger.level == logging.DEBUG: 
                self.logger.info('update path radiance according to routine scale_path_radiance_wfov.')
            wvl1 = self._wvlsen[0:(Band.RED.value)-1]

            # fix for SIIMPC-952: wrong index for 10m, UMW:
            if self.config.resolution == 10:
                sc = interpol(array([self.config.sc_lp_blu, 1.0]), array([self._wvlsen[Band.BLUE.value-2], self._wvlsen[Band.RED.value-2]]), wvl1)
            else:
                sc = interpol(array([self.config.sc_lp_blu, 1.0]), array([self._wvlsen[Band.BLUE.value -1], self._wvlsen[Band.RED.value -1]]), wvl1)
            # end fix SIIMPC-952
            for k in arange(0, (Band.RED.value -1)):
                self._lph_cell[:,:, k,:,:] = self._lph_cell[:,:, k,:,:] * sc[k]
            del sc
            del wvl1

        if (self.config.iwaterwv > 0 and self.config.iwaterwv <= 3):
            # read water vapor map (scaled with 1000)
            if(self.tables.hasBand(Band.WATER_VAPOUR) == False): #self.tables.WVP
                self.logger.fatal('no water vapor channel present, giving up')

            wv = self.tables.getBand(Band.WATER_VAPOUR) #self.tables.WVP this is for Sentinel2
            if (self.config.smooth_wvmap > 0):
                # in case of smooth the wv map in the output file is smoothed
                nff = int(minimum(max([(self.config.smooth_wvmap / self.config.pixelsize), 1]), 51)+0.5)
                if (nff > min([self.config.ncols / 2, self.config.nrows / 2])):
                    nff = min([self.config.ncols / 2, self.config.nrows / 2])
                if (nff % 2 == 0):
                    nff = nff + 1
                if (nff > 1):
                    mask = self.getNodataMap()
                    wv[mask == 0] = 0
                    #wv = smooth(wv, nff, edge_truncate=True)
                    wv = median_filter(wv, (3, 3))  # modif JL20160216
                    self.tables.setBand(Band.WATER_VAPOUR, wv) #self.tables.WVP
                del nff

        # loop for reflective bands, write results to file filou_tmp
        # this file is read in rho_retrieval_step1 to perform the terrain and adj. correction
        # and possibly band interpolation
        trwv = 1.0
        self._iflag_interp725 = 0
        self._iflag_interp825 = 0
        fluxter = 0.0

    # http://jira.acri-cwa.fr/browse/SIIMPC-1019: disable DEM for high and mean cloud probability and invalid pixel:
    # self._cbeta contains the terrain infomation. self._vsky contains the slope. This is filtered with the information
    # on clouds and invalid pixel as taken from the classification mask.
    # For these values the same calculation is taken as in routine: dtm_flat. self._vsky (slope) is set to max (default: 100).
    # the terrain correction cannot be performed earlier, as the information goes also in the AOT calculation where it should be kept.
    # for http://jira.acri-cwa.fr/browse/SIIMPC-557: the same algorithm is applied, but for all pixels.

        nrows = self.config.nrows
        ncols = self.config.ncols
        try:
            x = arange(nrows, dtype=float32) / (nrows - 1) * self.config.solze_arr.shape[0]
            y = arange(ncols, dtype=float32) / (ncols - 1) * self.config.solze_arr.shape[1]
            # casted to float32 because rectBivariateSpline returns float64 (JL)
            sza_arr2 = rectBivariateSpline(x, y, self.config.solze_arr).astype(float32)
            del x
            del y
        except:
            sza_arr2 = ones([nrows, ncols], dtype=float32) * self.config.solze_arr.mean()

        if self.config.dem_terrain_correction == True:
            # SIIMPC-1019 - disable DEM only for clouds and invalid pixels:
            CM = self.tables.getBand(Band.SCENE_CLASSIFICATION) #self.tables.SCL
            filt = (CM == self.config.highProbaClouds) \
                   | (CM == self.config.medProbaClouds) \
                   | (CM == self.config.thinCirrus) \
                   | (CM == self.config.saturatedDefective) \
                   | (CM == self.config.noData)
            del CM

            filt_large_elements = zeros(filt.shape).astype('bool')
            label_filt, nb_labels = ndimage.label(filt)
            del filt
    
            resolution = self.config.resolution
            blob_size = (2000//resolution)**2 # 2km by 2km clouds define the thershold for large elements of the binary filtering
            hist_label_filt, hist_label_filt_edges = histogram(label_filt, bins=arange(label_filt.max()+2)-0.5)
            list_of_large_elements = where(hist_label_filt[1:] > blob_size)[0] + 1  # Discard background = 0 (first index of histogram)
            label_filt_ravel = ravel(label_filt)	    
            ravel(filt_large_elements)[in1d(label_filt_ravel, list_of_large_elements)] = True    
            del label_filt_ravel, label_filt

            # Modification in case of input dem resolution is 30 m (COPDEM30): Corresponds to a dem native pixel size of 0.00027778 degrees
            try:
                if self.tables._input_dem_resolution == 0.00027778: #self.tables._input_dem_resolution
                    if resolution == 10:
                        cop_dem_30_filter_size = 9
                    elif resolution == 20:
                        cop_dem_30_filter_size = 5
                    else:
                        cop_dem_30_filter_size = 3

                    if self._cntback > 0:  # Fix for bright swath border effect (Collection-1)
                        ravel(self._cbeta)[self._liback] = ravel(self._cbeta)[self._li_nonback].mean()

                    self._cbeta = smooth(self._cbeta, cop_dem_30_filter_size, edge_truncate=True).astype(uint8)
            except:
                pass

            self._cbeta[filt_large_elements] = (self._sc_bet * cos((sza_arr2[filt_large_elements]) * self._dtor) + 0.5).astype(uint8)
            self._vsky[filt_large_elements] = self._vsky.max() # flatten slope for clouds and invalid pixels.
            del filt_large_elements 

        else: # SIIMPC-557 - disable DEM for all pixels:
            self._cbeta = (self._sc_bet * cos((sza_arr2) * self._dtor) + 0.5).astype(uint8)
            self._vsky[:] = self._vsky.max() # flatten slope for clouds and invalid pixels.
            self._iter_terrain = 0

        del sza_arr2	       

        # fixed together with SIIMPC-1019: VIS band must not be called inside loop.
        bandvis = self.tables.getBand(Band.VISIBILITY) #self.tables.VIS

        # for ibnd in arange(self._first_band, self._n_bands):
        #     band_index = self._band_index[ibnd]
        #     iwv_dependence = 0 # default: no wv correction
        #     if ((self._n_bands > 4) and
        #         (self.config.iwaterwv > 0) and
        #         (self._band_index_wvdepend[ibnd] == 1) and
        #         (ibnd != 7)):
        #         iwv_dependence = 1 # wv correction

        band_index = -1

        for idx, band_identifier in enumerate \
                    ([Band.COASTAL_AEROSOL, Band.BLUE, Band.GREEN, Band.RED, \
                      Band.VEGETATION_1, Band.VEGETATION_2, Band.VEGETATION_3, Band.NEAR_INFRARED, Band.VEGETATION_4,
                      Band.WATER_VAPOUR_INPUT, Band.CIRRUS, Band.SHORT_WAVE_INFRARED_1, Band.SHORT_WAVE_INFRARED_2]):
            if not self.tables.hasBand(band_identifier):
                continue
            if 'LANDSAT' in self.config.spacecraftName and band_identifier == Band.CIRRUS:
                continue
            band_index += 1
            iwv_dependence = self.getWvDependencyIndex(band_index)
            # print('idx: ', idx, ' - band_identifier: ', band_identifier, ' - band_index: ', band_index,' - wv dependency:', iwv_dependence) #, ' - _wvlsen[band_index]',self._wvlsen[band_index])

            # check for saturated pixels
            # criterion is DN > rel_saturation*(digital encoding range)
            thresh_sat = 0.0 # no threshold for saturation
            # FCP 20211020 Commented for Landsat and Sentinel2
            # dataType = self.tables.getDataType(band_identifier) #band_index
            # if dataType == uint8:
            #     thresh_sat = ones(self._rel_saturation, uint8) * 255
            # # fix for SIIMPC-1038, UMW: avoid floats as index, depecated for numpy > 1.11
            # elif dataType == int16:
            #     thresh_sat = ones(self._rel_saturation, int16) * 32767
            # elif dataType == uint16:
            #     thresh_sat = ones(self._rel_saturation, uint16) * 65535
            # else:
            #     self.logger.fatal('no match found for data type of bands')

            band = self.tables.getBand(band_identifier) #band_index
            if self.config.resolution == 10:
                band = band.astype(float16)

            if (thresh_sat > 0):
                li = where32(ravel(band)[li_stat] >= thresh_sat)
                if(li.size > 0):
                    percent = 100.0 * li.size / li_stat.size
                    if (percent >= 1.0):
                        self._band_saturated[band_index] = percent #ibnd
            # ToDo check for saturated pixels ...
            if (self.cirrus_correction):
                # subtract cirrus contribution on the DN level
                gamma = interpol(self._arr_gamma, self._wvl_gamma, self._wvlsen[band_index])
                if (self._wvlsen[band_index] >= 1.50):
                    gamma = gamma * 2 # reduction factor 2

                # wv band of Sentinel-2
                if(abs(self._wvlsen[band_index]-0.945) < 0.010):
                    trwv = self._trwv945
                else:
                    trwv = 1.0

                # ToDo: check this!!!
                #band = self.tables.getBand(band_identifier, radiance=True) #changed with lines below
                if 'LANDSAT' in self.config.spacecraftName:
                    band = self.tables.getBand(band_identifier, radiance=True)
                else:
                    # Two steps (old fashion) to get radiance computed with solze specific to cell being processed (not solze at scene center)
                    band = self.tables.getBand(band_identifier) #JL 05/01/2022
                    if self.config.resolution == 10:
                        band = band.astype(float16)
                    band = self.refl2rad(band, band_identifier)
                band_cirr = self.tables.getBand(Band.CIRRUS, radiance=True)
                band_cirr = median_filter_2d(band_cirr, 3) * self.config.dnScale

                if self.config.resolution == 10:
                    band = band.astype(np.float32) / trwv - band_cirr
                    band = band.astype(np.float16)
                else:
                    band = band / trwv - band_cirr
                # rho_app = band

                # if self.config.resolution == 10:
                #     rho_app = rho_app.astype(float32) / trwv - self._rho_cir_app
                #     rho_app = rho_app.astype(float16)
                # else:
                #     rho_app = rho_app / trwv - self._rho_cir_app
                #
                # band = self.refl2rad(rho_app, ibnd)
                # del rho_app

            else:
                if 'LANDSAT' in self.config.spacecraftName:
                    band = self.tables.getBand(band_identifier, radiance=True)
                else:
                    # Two steps (old fashion) to get radiance computed with solze specific to cell being processed (not solze at scene center)
                    band = self.tables.getBand(band_identifier) #JL 05/01/2022
                    if self.config.resolution == 10:
                        band = band.astype(float16)
                    band = self.refl2rad(band, band_identifier)

                #band = self.tables.getBand(band_identifier, radiance=True) #band = self.refl2rad(band, ibnd)  # band in float16 for 10 m and in float32 for 20 m or 60 m


            # band is input DN of whole scene (cirrus, haze subtracted)

            rho_scene = zeros([self.config.nrows, self.config.ncols], float32)
            # cell loop
            # -----------
            for jx in arange(0, (self._nx_cell)):
                for jy in arange(0, (self._ny_cell)):
                # update RT functions
                    self._lph = self._lph_cell[jy, jx,:,:,:] # accounts for variable vis (2nd index)
                    self._e0th = self._e0th_cell[jy, jx,:,:,:]
                    self._edifth = self._edifth_cell[jy, jx,:,:,:]
                    self._tsunh = self._tsunh_cell[jy, jx,:,:,:]
                    #; sphah = sphah_cell[*,*,*,jx,jy] ; only needed in rho_retrieval_step2
                    #; qh = qh_cell[*,*,*,jx,jy]
                    if (self.config.iwaterwv > 0):
                        self._lp_fit = self._lp_fit_cell[jy, jx,:,:,:] # uses scene-average visibility
                        self._e0t_fit = self._e0t_fit_cell[jy, jx,:,:,:] # (only for atm. absorption bands,
                        self._edift_fit = self._edift_fit_cell[jy, jx,:,:,:] # therefore: small influence as lambda > 900 nm)
                        self._tsun_fit = self._tsun_fit_cell[jy, jx,:,:,:]

                    # take cell subset
                    subset = array([self._ycell[jy, 0], self._ycell[jy, 1], self._xcell[jx, 0], self._xcell[jx, 1]])
                    band_sub = band[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1].astype(float32)  # DN of input scene
                    li_scene = where32(ravel(band_sub > 0))
                    if(li_scene.size > 0): # GOTO SKIPC in ATCOR, line 8649
                        fluxpix = zeros([subset[1]-subset[0]+1, subset[3]-subset[2]+1], float32)
                        if (self._iter_terrain > 0):
                            fluxter = zeros([subset[1]-subset[0]+1, subset[3]-subset[2]+1], float32)
                        fluxm = zeros([subset[1]-subset[0]+1, subset[3]-subset[2]+1], float32)
                        self._eclass_sub = self._eclass[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                        b_sub = b[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                        bvis_sub = bandvis[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                        cbeta_sub = self._cbeta[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                        vsky_sub = self._vsky[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                        if (wv.size > 1):
                            wv_sub = wv[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]

                        if (self._ibrdf > 0):
                            ga_sub = ga[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]

                            if (cnt_vege > 0 and iswitch_nir == 0 and self._wvlsen[band_index] >= 0.72):
                                # start is with ga = ga_vis channels
                                # switch to ga_nir if lambda >=0.72 um
                                ga_sub = ga_nir[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                                iswitch_nir = 1

                        if (iwv_dependence == 0):
                            # constant water vapor section (band does not depend on wv)
                            # --------------------------------------------------------

                            if (self._wvlsen[band_index] < 1.36):
                                # --------------------------------------
                                # account for vis.index if wvl < 1.36 um
                                # for longer wvl the average vis.index is used to save time !
                                # --------------------------------------

                                for j in arange(minvis, (maxvis)+(1)):
                                # get pixels with vis.index j
                                    ll = where32(ravel(bvis_sub == j))

                                    if (ll.size > 0):
                                        # anisotropic diffuse flux edifh
                                        # Sentinel-2: important: cbeta_sub includes config.solze_arr, config.solaz_arr (direct flux) (see dtm_array)
                                        # For the diffuse flux the average solze is used (difference to config.solze_arr < 1.5 deg)
                                        # tsunh is for solze (no bilinear interp.with config.solze_arr, elevation dependence more important)
                                        ravel(fluxpix)[ll] = float32(self._e0th[band_index, j, ravel(self._eclass_sub)[ll]] * ravel(cbeta_sub)[ll] * \
                                                             sc1 + self._edifth[band_index, j, ravel(self._eclass_sub)[ll]] * (ravel(b_sub)[ll] * \
                                                             self._tsunh[band_index, j, ravel(self._eclass_sub)[ll]] * ravel(cbeta_sub)[ll] * sc1 / cos(radians(self._solze)) + \
                                                             (1.0 - ravel(b_sub)[ll] * self._tsunh[band_index, j, ravel(self._eclass_sub)[ll]]) * ravel(vsky_sub)[ll] * 0.01))

                                        # reflected neighboring terrain flux > 0.01*direct flux
                                        if (self._iter_terrain > 0):
                                            ravel(fluxter)[ll] = float32((maximum(((self._e0th[band_index, j, ravel(self._eclass_sub)[ll]] * cos(radians(self._solze)) + \
                                                                 self._edifth[band_index, j, ravel(self._eclass_sub)[ll]]) * (100 - ravel(vsky_sub)[ll]) * 0.01 *
                                                                 self._reflter[band_index]), (self._e0th[band_index, j, ravel(self._eclass_sub)[ll]] * cos(radians(self._solze)) * 0.01))))

                                        # measured radiance: L=c0[nk-1]+c1[nk-1]*band
                                        ravel(fluxm)[ll] = float32(pi * (self.config.d2 * \
                                                           ravel(band_sub)[ll] - self._lph[band_index, j, ravel(self._eclass_sub)[ll]]))
                                        # ( cnt loop of pixels found for current visibility) # (end of vis loop)
                            else:
                                # wvl > 1.36 um: use mean vis.index=meanvi now
                                fluxpix = float32(self._e0th[band_index, self._meanvi, self._eclass_sub] * cbeta_sub * sc1 + self._edifth[band_index, self._meanvi, self._eclass_sub] * \
                                                    (b_sub * self._tsunh[band_index, self._meanvi, self._eclass_sub] * cbeta_sub * sc1 / cos(radians(self._solze)) + (1.0 - b_sub * \
                                                    self._tsunh[band_index, self._meanvi, self._eclass_sub]) * vsky_sub * 0.01))

                                # reflected neighboring terrain flux > 0.01*direct flux
                                if (self._iter_terrain > 0):
                                    fluxter = float32((maximum(((self._e0th[band_index, self._meanvi, self._eclass_sub] * cos(radians(self._solze)) + self._edifth[band_index, self._meanvi, self._eclass_sub]) * \
                                              (100 - vsky_sub) * 0.01 * self._reflter[band_index]), (self._e0th[band_index, self._meanvi, self._eclass_sub] * cos(radians(self._solze)) * 0.01))))

                                fluxm = float32(pi * (self.config.d2 * band_sub - self._lph[band_index, self._meanvi, self._eclass_sub]))

                            band1 = (minimum((fluxm / (fluxpix + fluxter)), self._refl_cutoff))

                            if (self._ibrdf > 0):
                                band1 = band1 * ga_sub * sc1
                        else:
                            # variable water vapor section
                            self._e0tw = polx3(ju, wv_sub, self._e0t_fit[band_index, self._eclass_sub,:])
                            self._ediftw = polx3(ju, wv_sub, self._edift_fit[band_index, self._eclass_sub,:])
                            self._tsunw = polx3(ju, wv_sub, self._tsun_fit[band_index, self._eclass_sub,:])

                            fluxpix = (self._e0tw * cbeta_sub * sc1 + self._ediftw * (b_sub * self._tsunw * cbeta_sub * sc1 / \
                                      cos(radians(self._solze)) + (1.0 - b_sub * self._tsunw) * vsky_sub * 0.01))
                            
                            del self._tsunw
 
                            if (self._iter_terrain > 0):
                                fluxter = (maximum(((self._e0tw * cos(radians(self._solze)) + self._ediftw) *
                                          (100 - vsky_sub) * 0.01 * self._reflter[band_index]), (self._e0tw * cos(radians(self._solze)) * 0.01)))

                            del self._e0tw
                            
                            fluxm = pi * (self.config.d2 * band_sub - polx3(ju, wv_sub, self._lp_fit[band_index, self._eclass_sub,:]))

                            band1 = (minimum((fluxm / (fluxpix + fluxter)), self._refl_cutoff))
                            if (self._ibrdf > 0):
                                band1 = band1 * ga_sub * sc1 # ( of iwv section)
                            del self._ediftw

                        rho_scene[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1] = band1
                # end y
            # end x
            # Version 02.10.02:
            # Set a high cloud coverage flag to disable full rho_retrieval_step 2 or only adjacency correction if image is too cloudy
            if (self._np_cloud / float(self._np_cloud + self._np_clear + 1)) > 0.95:  # +1 to avoid division by zero
                self._high_cloud_coverage = True  # disable only adjacency correction, keep terrain correction
                #self.config.rho_retrieval_step2 = False  # uncomment to disable adjacency correction and terrain correction -> save processing time

            # write results to file filou_tmp, unit assvouv (float reflectance)

            if 'LANDSAT' in self.config.spacecraftName: #force landsat
                # if (not self.config.rho_retrieval_step2): #force landsat
                self.tables.setTmpBand(band_identifier, rho_scene) #force landsat
                nvl_landsat = 0
                nvl_landsat = rho_scene[rho_scene < 0].size
                self.config.L2A_BOA_NEGATIVE_VALUES_LIST[band_index] = nvl_landsat
            else:     #force landsat
                if(self._iter_terrain == 0 & self._nadj == 1):
                    nvl = rho_scene[rho_scene < 0].size
                    self.config.L2A_BOA_NEGATIVE_VALUES_LIST[band_index] = nvl
                    self.write_output_file_a3(band_index, rho_scene)
                elif(not self.config.rho_retrieval_step2):
                    nvl = rho_scene[rho_scene < 0].size
                    self.config.L2A_BOA_NEGATIVE_VALUES_LIST[band_index] = nvl
                    self.write_output_file_a3(band_index, rho_scene)
                else:
                    self.tables.setTmpBand(band_identifier, rho_scene)
        # fixed together with SIIMPC-1019:
        try:
            del bandvis
            if self.config.resolution == 10:
                # VIS band can now be removed for 10 m processing:
                self.tables.removeBandRes(band.VISIBILITY) #self.tables.VIS
        except:
            pass
        return

    #@profile
    def rho_retrieval_step2(self, nw, ju, cnt_nonback, chan2win, icloud, li_clear, b, cnt_vege, ga, ga_nir, nch_wvp):

    # Purpose: second step of surface reflectance retrieval (with terrain iteration, with adj. correction)
    #
    # Input:
    # nw = number of bands in strong wv regions (940, 1130, 1400, 1800 nm)
    # ju = water vapor interval (0 - 6)
    # cnt_nonback = number of non-background pixels
    # chan2win = zeros(4) for the 940/1130, and 1400/19000 nm regions
    # 1: left wv window channel, 2: left and right window channel available
    # icloud = 0 (cloud percentage < 1%) otherwise icloud=1
    # li_clear = list of clear scene pixels (no cloud, for terrain and adj. correction)
    # b = binary shadow matrix (1: sunlit, 0: shadow)
    # cnt_vege = number of vegetation pixels for BRDF correction
    # ga = matrix for BRDF correction
    # ga_nir = matrix for BRDF correction (vegetation)
    # nch_wvp = zeros(n_bands) (0= no wv band, 1=wv band)
    #
    # Output: final surface reflectance cube
    #
    # Structure:
    # ! at_log
    # ! gap_processing2_a3
    # ! write_output_file_a3

        # Convert li_clear list (int32) to li_clear_mask (byte) to save RAM
        albedo_reference = 0.15
        li_clear_mask = zeros([self.config.nrows, self.config.ncols], bool)
        ravel(li_clear_mask)[li_clear] = True
        del li_clear

        if self._iter_terrain == 0 and self._nadj == 1:
            return # flat terrain and no adjacency correction
        # (no de-shadowing is performed in this case)

        if (self._iter_terrain > 0):
            tt = 'surface reflectance retrieval, step 2 (terrain, adjacency effect)'
        else:
            tt = 'surface reflectance retrieval, step 2 (adjacency effect)'  # flat terrain
        self.logger.info(tt)

        iswitch_nir = 0
        sc1 = 1.0 / self._sc_bet

        cnt_clear = 0
        if(li_clear_mask.size != 0):
            cnt_clear = li_clear_mask.sum()

        # No cell loop for adjacency correction: take qh for scene center geometry..
        # Terrain view factor is independent of sun/observer geometry and e0th, edifth for
        # fluxter are evaluated for scene center (small influence compared to approximations
        # involved in simplified terrain model)

        self._lph = self._lph_cell[self._ny_cell // 2, self._nx_cell // 2,:,:,:]
        self._e0th = self._e0th_cell[self._ny_cell // 2, self._nx_cell // 2,:,:,:]
        self._edifth = self._edifth_cell[self._ny_cell // 2, self._nx_cell // 2,:,:,:]
        self._sphah = self._sphah_cell[self._ny_cell // 2, self._nx_cell // 2,:,:,:]
        self._qh = self._qh_cell[self._ny_cell // 2, self._nx_cell // 2,:,:,:]

        del self._e0th_cell
        del self._edifth_cell

        del self._ele_class
        del self._e0t
        del self._e0t_all
        del self._e0th1
        del self._e0th2
        del self._e0tx
        del self._ediftx

        wvl_nir = 0.850 # (includes sensors without and with a NIR band)
        if 'LANDSAT' in self.config.spacecraftName:
            # wvl_nir = self._wvlsen[self._nir_band-1] before merging
            wvl_nir = self._wvlsen[self.tables.reindex(Band.NEAR_INFRARED)-1] #Band.VEGETATION_4
        else:
            if self.config.resolution == 10:
                wvl_nir = self._wvlsen[self.tables.reindex(Band.NEAR_INFRARED)]
            else:
                wvl_nir = self._wvlsen[self.tables.reindex(Band.VEGETATION_4)]  # consistent with Sen2Cor 2.10.2

        nf_rad = (maximum(asarray((self.config.AC_Rng_Nbhd_Terrain_Corr * 1000.0) / self.config.pixelsize).astype(int16), 3)) # min = 3 pixels
        if (nf_rad > array(asarray([self.config.ncols / 2 - 2, self.config.nrows / 2 - 2])).min()):
            nf_rad = array(asarray([self.config.ncols / 2 - 2, self.config.nrows / 2 - 2])).min()
        nf_diameter = 2 * nf_rad + 1  # usually 7 pixels for wide FOV sensors

        # For adjacency range 1.0 km and config.pixelsize= 300m we get nadj=1000./ 300= 3 pixels
        # " 1.0 km " config.pixelsize=1200m " nadj=1000./1200= 1 pixels
        # For nadj=1 no adjacency correction is performed, but a terrain view factor correction!
        igap = 0
        iflag_interp = 0
        self._iflag_interp725 = 0
        self._iflag_interp825 = 0

        if 'LANDSAT' in self.config.spacecraftName:
            wv = self.tables.getBand(Band.WATER_VAPOUR) #/ 1000 # read water vapor map (scaled with 1000)
        else:
            wv = array(0)

        if (self.config.iwaterwv > 0) & (self.config.iwaterwv <= 3):
            if (self.tables.hasBand(Band.WATER_VAPOUR) == False): #self.tables.WVP
                self.logger.fatal('no water vapor channel present, giving up')
            else:
                wv = self.tables.getBand(Band.WATER_VAPOUR) #self.tables.WVP #in this case reading for Sentinel 2
        if (self._cntback > 0):
            scl = self.tables.getBand(Band.SCENE_CLASSIFICATION) #self.tables.SCL
        # band loop
        band_index = -1
        for idx, band_identifier in enumerate \
            ([Band.COASTAL_AEROSOL, Band.BLUE, Band.GREEN, Band.RED, \
              Band.VEGETATION_1, Band.VEGETATION_2, Band.VEGETATION_3, Band.NEAR_INFRARED, Band.VEGETATION_4,
              Band.WATER_VAPOUR_INPUT, Band.CIRRUS, Band.SHORT_WAVE_INFRARED_1, Band.SHORT_WAVE_INFRARED_2]):
            if not self.tables.hasBand(band_identifier):
                continue
            if 'LANDSAT' in self.config.spacecraftName and band_identifier == Band.CIRRUS:
                continue

            band_index += 1
            # print(idx, band_identifier, band_index)
            iwv_dependence = self.getWvDependencyIndex(band_index)

            # Band interpolation is performed in rho_retrieval_step2, but skipping of bands
            # is also done here
            # skip bands in 1400/1900 nm region (will be interpolated)
            # (for config.intpol940_1130 > 0: 940/1130 nm region will also be interpolated)
            if nw >= 1 or self.config.intpol1400 > 0:
                # search for first band in absorption region
                if (igap == 0 and ((band_index >= self._gap[0, 0] and band_index <= self._gap[0, 1]) or (band_index >= self._gap[1, 0] and band_index <= self._gap[1, 1]) or (band_index >= self._gap[2, 0] and band_index <= self._gap[2, 1]) or (band_index >= self._gap[3, 0] and band_index <= self._gap[3, 1]))):
                    if self.config.intpol940_1130 > 0:
                        igap = 1
                        # reset igap=0 if no 940 left window band, only 940 absorption bands (Hyspex-SWIR)
                        if chan2win[0] == 0 and band_index >= self._gap[0, 0] and band_index <= self._gap[0, 1]:
                            igap = 0
                        if band_index >= self._gap[1, 0] and band_index <= self._gap[1, 1]:
                            igap = 2
                    if self.config.intpol1400 > 0:
                        if band_index >= self._gap[2, 0] and band_index <= self._gap[2, 1]:
                            igap = 3
                        if band_index >= self._gap[3, 0] and band_index <= self._gap[3, 1]:
                            igap = 4

                if igap > 0:
                    if chan2win[igap - 1] > 1:  # left and right window channels exist
                        li = where32(ravel(nch_wvp == band_index))
                        cnt = li.size
                    else:
                        # only left window channel exists (e.g., CASI 400-970 nm)
                        # if the last water vapor channel is processed set iflag_interp=1
                        if band_index == nch_wvp[nch_wvp.size - 1]:
                            cnt = 0
                        else:
                            cnt = 1

                    if (cnt == 0):
                        iflag_interp = 1 # flag to perform interpolation
                    # first band in next window region found, do band interpolation now

            if igap > 0 and iflag_interp == 0:
                # skip normal calculation for gap bands
                # emulate the GOTO SKIP1 statement of ATCOR:
                continue

            if self._ibrdf > 0 and cnt_vege > 0 and iswitch_nir == 0 and self._wvlsen[band_index] >= 0.72:
                # start is with ga = ga_vis channels
                # switch to ga_nir if lambda >=0.72 um
                ga = ga_nir
                iswitch_nir = 1

            # adjacency correction is only performed for wvl < wvl_adj and excluding the
            # 1.35-1.45 micron region
            wvlc = self._wvlsen[band_index]

            if (wvlc > self._wvl_adj or (wvlc >= 1.35 and wvlc <= 1.45) or (wvlc >= 1.80 and wvlc <= 1.95) or wvlc > 2.46):
                iadj = 0
            else:
                iadj = 1

            if self.config.resolution == 10:
                if iadj == 1:
                    if self._iter_terrain > 0:
                        #vsky = self._vsky
                        # test radiative transfer functions in float16 only if iwv_dependence == 0 (JL)
                        e0th = self._e0th.astype(float16)
                        edifth = self._edifth.astype(float16)
                        tsunh = self._tsunh.astype(float16)
                        lph = self._lph.astype(float16)

                        if iwv_dependence == 0:
                            fluxpix = (e0th[band_index, self._meanvi, self._eclass] * self._cbeta * sc1 + edifth[band_index, self._meanvi, self._eclass] *
                                      (b * tsunh[band_index, self._meanvi, self._eclass] * self._cbeta * sc1 / cos(radians(self._solze)) +
                                      (1.0 - b * tsunh[band_index, self._meanvi, self._eclass]) * self._vsky * 0.01)).astype(float16)
                            band= self.tables.getBand(band_identifier) # band = self.tables.getBand(band_index)
                            # JL20200309: set background to mean value before fluxm computation (Halo issue Jira-1569)
                            # JL20210630: updated for L1C quality masks handling (missing data)
                            if (self._cntback > 0):
                                ravel(band)[self._liback] = ravel(band)[self._li_nonback].mean()
                            # band = self.tables.getBand(band_index, radiance=True).astype(float16) # band = self.refl2rad(band, ibnd).astype(float16)
                            band = self.refl2rad(band, band_identifier) #to mimick the old version
                            fluxm = (pi * (self.config.d2 * band - lph[band_index, self._meanvi, self._eclass])).astype(float16)
                        else:
                            fluxpix = (polx3(ju, wv, self._e0t_fit[band_index, self._eclass,:]) * self._cbeta * sc1 +
                                       polx3(ju, wv, self._edift_fit[band_index, self._eclass,:]) * (b * polx3(ju, wv, self._tsun_fit[band_index, self._eclass,:]) *
                                       self._cbeta * sc1 / cos(radians(self._solze)) + (1.0 - b * polx3(ju, wv, self._tsun_fit[band_index, self._eclass,:])) * self._vsky * 0.01)).astype(float16)
                            band = self.tables.getBand(band_identifier)
                            # JL20200309: set background to mean value before fluxm computation (Halo issue Jira-1569)
                            # JL20210630: updated for L1C quality masks handling (missing data)
                            if (self._cntback > 0):
                                ravel(band)[self._liback] = ravel(band)[self._li_nonback].mean()
                            # band = self.tables.getBand(band_index, radiance=True).astype(float16) #self.refl2rad(band, ibnd).astype(float16)
                            band = self.refl2rad(band, band_identifier)  # to mimick the old version
                            fluxm = (pi * (self.config.d2 * band - polx3(ju, wv, self._lp_fit[band_index, self._eclass,:]))).astype(float16)

                        del band
                        # replace background pixels with band average value
                        # geocoded background with band1=0 is included in "total"
                        # (does not contribute to the sum, but is excluded in cnt_nonback ! )
                        # open file containing the step1 surface reflectance cube
                        band1 = self.tables.getTmpBand(band_identifier)
                        # note that band1 corresponds to rho_step1 in previous versions <=2.6.2
                        if (self._cntback > 0):
                            ravel(band1)[self._liback] = ravel(band1)[self._li_nonback].mean()

                        # consider cloud if more than 1% of image pixels
                        if (icloud == 0 or cnt_clear == 0):
                            # include edge=border pixels for smooth

                            refav = smooth16(band1, nf_diameter, edge_truncate=True)

                        else:
                            # cloud reflectance is replaced by average of
                            # clear pixel reflectance for terrain corr
                            #rho_step1 = self.tables.getTmpBand(band_index,float32).astype(float16) #rho_step1 looks already open (JL)

                            ravel(band1)[self._mlist_cloud] = band1[li_clear_mask].mean()
                            refav = smooth16(band1, nf_diameter, edge_truncate=True)

                        del band1

                        # multiple terrain reflection as geometric series E(terrain) = Eg*rho_av*Vt/(1-rho_av*Vt_av)
                        # where Vt is the terrain view factor = 1 - skyview, Vt_av = smooth(Vt, nf_diameter)
                        # (P. Sirguey, RSE, Vol. 113, pp.160-181, 2009)

                        x = (uint8(100) - self._vsky) * float16(0.01) * refav  # V(terrain) * rho(av)
                        xa = (uint8(100) - smooth16(self._vsky, nf_diameter, edge_truncate=True)) * float16(0.01) * refav  # V(terrain,av)* rho(av)

                        del refav

                        fluxter = ((maximum(((e0th[band_index, self._meanvi, self._eclass] *
                                   cos(radians(self._solze)) + edifth[band_index, self._meanvi, self._eclass]) *
                                   x / (1.0 - xa)), (e0th[band_index, self._meanvi, self._eclass] * cos(radians(self._solze)) * 0.01)))).astype(float16)

                        del x
                        del xa

                        band1 = (minimum((fluxm / (fluxpix + fluxter)), self._refl_cutoff)).astype(float16)

                        del fluxm
                        del fluxpix
                        del fluxter

                        if self._ibrdf > 0:
                            band1 = band1 * ga * sc1  # is required for step1 and step2 !

                        #rho_step1 = band1  # replace with topographic iteration

                    else:
                        # open file containing the step1 surface reflectance cube
                        band1 = self.tables.getTmpBand(band_identifier).astype(float16)
                        # JL20220128: set background to mean value before refav computation (Halo issue Jira-1569)
                        if (self._cntback > 0):
                            ravel(band1)[self._liback] = ravel(band1.astype(float32))[self._li_nonback].mean()
                    # (end of local terrain neighborhood)
                    # low pass filtering of updated reflectance array band1
                    # using adjacency filter size nad=2*nadj corresponding to 2*adj_km
                    if (icloud == 0 or cnt_clear == 0):
                        # include edge=border pixels for smooth
                        refav = smooth16(band1.astype(float32), 2 * self._nadj + 1, edge_truncate=True) # rho_step1 is updated in iter_terrain section
                        #rho_step1 = rho_step1.astype(float16)
                    else:
                        # cloud reflectance is replaced by average of
                        # clear pixel reflectance for adj. corr.
                        band1_tmp = copy(band1) # band1 is copied to avoid modification of original band1 during refav computation
                        ravel(band1_tmp)[self._mlist_cloud] = band1_tmp[li_clear_mask].astype(float32).mean()
                        refav = smooth16(band1_tmp.astype(float32), 2 * self._nadj + 1, edge_truncate=True) # rho_step1 is updated in iter_terrain section
                        del band1_tmp

                    refav = (maximum(refav, 0.0))

                    # SIIMPC-998 UMW: section rewritten.
                    # adjacency correction
                    # test qh and sphah in float16 (JL)
                    qh16 = self._qh.astype(float16)
                    qh = qh16[band_index, self._meanvi, self._eclass]
                    sphah = self._sphah.astype(float16)

                    # Perform adjacency correction only if not in case of high_cloud_coverage Version 2.10.2
                    if not self._high_cloud_coverage:
                        band1 += qh * (band1 - refav)

                    if (self._cntback > 0):
                        mask = zeros([self.config.nrows, self.config.ncols], uint8) + 255
                        mask[scl <= self.config.saturatedDefective] = 0

                        mask = smooth16(mask, 2 * self._nadj + 1, edge_truncate=True)  # border region has 0 < mask1 < 255
                        maskF = (mask > 0) & (mask < 251)  # mask filter must be limited to 250 due to used filter
                        del mask

                        b_r = band1[maskF] - refav[maskF]
                        qh = qh[maskF]
                        band1[maskF] += 0.1 * qh * b_r
                        del qh
                        del b_r
                        del maskF

                    # SIIMPC-998 UMW end
                    # spherical albedo correction
                    band1 *= (1.0 - (refav - albedo_reference) * sphah[band_index, self._meanvi, self._eclass])

                    del refav

                else:
                    # (end of terrain /adjacency correction based on iadj=1)
                    # spherical albedo correction must still take place

                    # open file containing the step1 surface reflectance cube
                    band1 = self.tables.getTmpBand(band_identifier).astype(float16)
                    band1 *= (1.0 - (band1 - albedo_reference) * sphah[band_index, self._meanvi, self._eclass])

                # interpolation for bands in strong water vapor regions
                if (nw >= 1 and (self.config.intpol940_1130 > 0 or self.config.intpol1400 > 0)):
                    iflag_interp_save = iflag_interp
                    igap, iflag_interp = self.gap_processing2_a3(igap, iflag_interp, band_index, band_index, band1, chan2win)
                    if iflag_interp_save == 1:
                        continue

                thr_neg = 0
                if self._neg_channels < 30 and self._wvlsen[band_index] >= 0.450 and self._wvlsen[band_index] <= wvl_nir:
                    if self._cntback > 0:
                        ravel(band1)[self._liback] = 255 # (background is reset to 0 later)
                    # allow slightly neg. values (-0.5%) if lamb > 0.7 (water) or lamb < 0.5 um.
                    # for lamb < 0.5 um: errors in the aerosol type (Lpath) can cause slightly
                    # negative rho values in tree/building shadow regions for dark pixels.
                    if self._wvlsen[band_index] < 0.500 or self._wvlsen[band_index] > 0.700:
                        thr_neg = -0.005

                    li_neg = where32(ravel(band1 < thr_neg))
                    if 100.0 * li_neg.size / cnt_nonback > 1.0:
                        # more than 1% of scene pixels has neg. reflectance
                        self._neg_pixels_percent[self._neg_channels] = 100.0 * li_neg.size / cnt_nonback
                        self._neg_pixels_channels[self._neg_channels] = band_index
                        self._neg_channels += 1
                    if self._cntback > 0:
                        ravel(band1)[self._liback] = 0  # ; (works for 0b, 0, 0.0)

            # end of 10 m processing optimized for limited RAM usage

            else:  # for 60m and 20m resolutions: less restriction on RAM and priority to processing time
                if iadj == 1:
                    if self._iter_terrain > 0:
                        #vsky = self._vsky
                        # test radiative transfer functions in float16 only if iwv_dependence == 0 (JL)
                        e0th = self._e0th
                        edifth = self._edifth
                        tsunh = self._tsunh
                        lph = self._lph
                        # band = self.tables.getBand(band_index)
                        band = self.tables.getBand(band_identifier)
                        # JL20200309: set background to mean value before fluxm computation (Halo issue Jira-1569)
                        # JL20210630: updated for L1C quality masks handling (missing data)
                        if (self._cntback > 0):
                            ravel(band)[self._liback] = ravel(band)[self._li_nonback].mean()
                        # band = self.tables.getBand(band_identifier, radiance=True) #old version
                        if 'LANDSAT' in self.config.spacecraftName:
                            band = self.tables.getBand(band_identifier, radiance=True)
                        else:
                            # Two steps (old fashion) to get radiance computed with solze specific to cell being processed (not solze at scene center)
                            band = self.tables.getBand(band_identifier)  # JL 05/01/2022
                            band = self.refl2rad(band, band_identifier)

                        if iwv_dependence == 0:
                            fluxpix = (e0th[band_index, self._meanvi, self._eclass] * self._cbeta * sc1 + edifth[band_index, self._meanvi, self._eclass] *
                                      (b * tsunh[band_index, self._meanvi, self._eclass] * self._cbeta * sc1 / cos(radians(self._solze)) +
                                      (1.0 - b * tsunh[band_index, self._meanvi, self._eclass]) * self._vsky * 0.01))
                            fluxm = float32(pi) * (self.config.d2 * band - lph[band_index, self._meanvi, self._eclass])
                        else:
                            fluxpix = (polx3(ju, wv, self._e0t_fit[band_index, self._eclass,:]) * self._cbeta * sc1 +
                                       polx3(ju, wv, self._edift_fit[band_index, self._eclass,:]) * (b * polx3(ju, wv, self._tsun_fit[band_index, self._eclass,:]) *
                                       self._cbeta * sc1 / cos(radians(self._solze)) + (1.0 - b * polx3(ju, wv, self._tsun_fit[band_index, self._eclass,:])) * self._vsky * 0.01)).astype(float16)
                            fluxm = float32(pi) * (self.config.d2 * band - polx3(ju, wv, self._lp_fit[band_index, self._eclass,:]))

                        del band

                        # replace background pixels with band average value
                        # geocoded background with band1=0 is included in "total"
                        # (does not contribute to the sum, but is excluded in cnt_nonback ! )
                        # open file containing the step1 surface reflectance cube
                        band1 = self.tables.getTmpBand(band_identifier)
                        # note that band1 corresponds to rho_step1 in previous versions <=2.6.2

                        if (self._cntback > 0):
                            ravel(band1)[self._liback] = ravel(band1)[self._li_nonback].mean()

                        # consider cloud if more than 1% of image pixels
                        if (icloud == 0 or cnt_clear == 0):
                            # include edge=border pixels for smooth

                            refav = smooth(band1, nf_diameter, edge_truncate=True)

                        else:
                            # cloud reflectance is replaced by average of
                            # clear pixel reflectance for terrain corr
                            #rho_step1 = self.tables.getTmpBand(band_index,float32).astype(float16) #rho_step1 looks already open (JL)

                            ravel(band1)[self._mlist_cloud] = band1[li_clear_mask].mean()
                            refav = smooth(band1, nf_diameter, edge_truncate=True)

                        del band1

                        # multiple terrain reflection as geometric series E(terrain) = Eg*rho_av*Vt/(1-rho_av*Vt_av)
                        # where Vt is the terrain view factor = 1 - skyview, Vt_av = smooth(Vt, nf_diameter)
                        # (P. Sirguey, RSE, Vol. 113, pp.160-181, 2009)

                        x = (uint8(100) - self._vsky) * float32(0.01) * refav  # V(terrain) * rho(av)
                        xa = (uint8(100) - smooth(self._vsky, nf_diameter, edge_truncate=True)) * float32(0.01) * refav  # V(terrain,av)* rho(av)

                        del refav

                        fluxter = maximum(((e0th[band_index, self._meanvi, self._eclass] *
                                   cos(radians(self._solze)) + edifth[band_index, self._meanvi, self._eclass]) *
                                   x / (1.0 - xa)), (e0th[band_index, self._meanvi, self._eclass] * cos(radians(self._solze)) * 0.01))

                        del x
                        del xa

                        band1 = minimum((fluxm / (fluxpix + fluxter)), self._refl_cutoff)

                        del fluxm
                        del fluxpix
                        del fluxter

                        if self._ibrdf > 0:
                            band1 = band1 * ga * sc1  # is required for step1 and step2 !

                        #rho_step1 = band1  # replace with topographic iteration

                    else:
                        # open file containing the step1 surface reflectance cube
                        band1 = self.tables.getTmpBand(band_identifier)
                        # JL20200309: set background to mean value before refav computation (Halo issue Jira-1569)
                        # JL20210630: updated for L1C quality masks handling (missing data)
                        if (self._cntback > 0):
                            ravel(band1)[self._liback] = ravel(band1)[self._li_nonback].mean()

                    # (end of local terrain neighborhood)
                    # low pass filtering of updated reflectance array band1
                    # using adjacency filter size nad=2*nadj corresponding to 2*adj_km
                    if (icloud == 0 or cnt_clear == 0):
                        # include edge=border pixels for smooth
                        refav = smooth(band1, 2 * self._nadj + 1, edge_truncate=True) # rho_step1 is updated in iter_terrain section
                        #rho_step1 = rho_step1.astype(float16)
                    else:
                        # cloud reflectance is replaced by average of
                        # clear pixel reflectance for adj. corr.
                        band1_tmp = copy(band1) # band1 is copied to avoid modification of original band1 during refav computation
                        ravel(band1_tmp)[self._mlist_cloud] = band1_tmp[li_clear_mask].mean()
                        refav = smooth(band1_tmp, 2 * self._nadj + 1, edge_truncate=True) # rho_step1 is updated in iter_terrain section
                        del band1_tmp

                    refav = maximum(refav, 0.0)

                    # SIIMPC-998 UMW: section rewritten.
                    # adjacency correction
                    # test qh and sphah in float16 (JL)
                    qh32 = self._qh.astype(float32)
                    qh = qh32[band_index, self._meanvi, self._eclass]
                    del qh32
                    sphah = self._sphah.astype(float32)

                    # Perform adjacency correction only if not in case of high_cloud_coverage Version 2.10.2
                    if not self._high_cloud_coverage:
                        band1 += qh * (band1 - refav)

                    if (self._cntback > 0):
                        mask = zeros([self.config.nrows, self.config.ncols], uint8) + 255
                        mask[scl <= self.config.saturatedDefective] = 0

                        mask = smooth16(mask, 2 * self._nadj + 1, edge_truncate=True)  # border region has 0 < mask1 < 255
                        maskF = (mask > 0) & (mask < 251)  # mask filter must be limited to 250 due to used filter
                        del mask

                        b_r = band1[maskF] - refav[maskF]
                        qh = qh[maskF]
                        band1[maskF] += 0.1 * qh * b_r
                        del qh
                        del b_r
                        del maskF

                    # SIIMPC-998 UMW end
                    # spherical albedo correction
                    band1 *= (1.0 - (refav - albedo_reference) * sphah[band_index, self._meanvi, self._eclass])

                    del refav

                else:
                    # (end of terrain /adjacency correction based on iadj=1)
                    # spherical albedo correction must still take place

                    # open file containing the step1 surface reflectance cube
                    band1 = self.tables.getTmpBand(band_identifier)
                    band1 *= (1.0 - (band1 - albedo_reference) * sphah[band_index, self._meanvi, self._eclass])

                # interpolation for bands in strong water vapor regions
                if (nw >= 1 and (self.config.intpol940_1130 > 0 or self.config.intpol1400 > 0)):
                    iflag_interp_save = iflag_interp
                    igap, iflag_interp = self.gap_processing2_a3(igap, iflag_interp, band_index, band_index, band1, chan2win)
                    if iflag_interp_save == 1:
                        continue

                thr_neg = 0
                if self._neg_channels < 30 and self._wvlsen[band_index] >= 0.450 and self._wvlsen[band_index] <= wvl_nir:
                    if self._cntback > 0:
                        ravel(band1)[self._liback] = 255 # (background is reset to 0 later)
                    # allow slightly neg. values (-0.5%) if lamb > 0.7 (water) or lamb < 0.5 um.
                    # for lamb < 0.5 um: errors in the aerosol type (Lpath) can cause slightly
                    # negative rho values in tree/building shadow regions for dark pixels.
                    if self._wvlsen[band_index] < 0.500 or self._wvlsen[band_index] > 0.700:
                        thr_neg = -0.005

                    li_neg = where32(ravel(band1 < thr_neg))
                    if 100.0 * li_neg.size / cnt_nonback > 1.0:
                        # more than 1% of scene pixels has neg. reflectance
                        self._neg_pixels_percent[self._neg_channels] = 100.0 * li_neg.size / cnt_nonback
                        self._neg_pixels_channels[self._neg_channels] = band_index
                        self._neg_channels += 1
                    if self._cntback > 0:
                        ravel(band1)[self._liback] = 0  # ; (works for 0b, 0, 0.0)

            if self._cntback > 0:
                ravel(band1)[self._liback] = 0  # ; (works for 0b, 0, 0.0)
            # end of 20 m and 60 m processing optimised for processing time

            # write data into file
            if 'LANDSAT' in self.config.spacecraftName: #force landsat
                    self.tables.setTmpBand(band_identifier, band1) #force landsat
                    nvl_landsat= 0
                    nvl_landsat = band1[band1 < 0].size
                    self.config.L2A_BOA_NEGATIVE_VALUES_LIST[band_index] = nvl_landsat
            else:     #Sentinel - Hyper
                    nvl = band1[band1 < 0].size
                    # print(band_identifier, band_index, 'negative pixel from end-rho-step-2', nvl)
                    self.config.L2A_BOA_NEGATIVE_VALUES_LIST[band_index] = nvl

                    self.write_output_file_a3(band_index, band1)
        try:
            del wv
            del scl
        except:
            pass
        return


    def wv_regions_940_1130_a3(self): # determine channels in wv regions
    # assign channels from .ini file or from wv selection panel
    #
    # iabs_region = zeros(5): number of absorption (measurement) channels per region
    # iabs_region[0]: 940 nm region
    # [1]: 1130 nm region
    # [2]: 1400 nm region
    # [3]: 1800 nm region
    # [4]: 10 micron region
    #
    # iabs_chan = zeros(n,5): list of absorption channels for the 5 regions
    # n = max number of absorption channels (all regions)
    # the 1400, 1800 nm bands are assigned for interpolation
    #
    # iwin1_region = zeros(5): number of window channels for the 5 window regions
    # (1) 850 - 890 nm
    # (2) 1000 - 1095 nm
    # (3) 1200 - 1250 nm
    # (4) 11.00 - 11.50 um
    # (5) 11.90 - 12.30 um
    # For TSR multiple channels are used per window region
    # otherwise only one channel is used
    # TSR is currently not implemented, but the program structure
    # supports TSR if iwin1_region is properly defined.
    #
    # iwin1_chan = zeros(n,5): channel numbers for each window region (starting from 1)
    # iwin1_chan[*,0] = ch940w1, ch940w1 window channels 850 - 890 nm
    # iwin1_chan[*,1] = ch940w2, ch1130w1 1000 - 1095 nm
    # iwin1_chan[*,2] = ch1130w2, ch1130w2 1180 - 1250 nm
    #
        n = 1000
        # fix for SIIMPC-1038, UMW: avoid floats as index, depecated for numpy > 1.11
        self._iabs_region = zeros(5, dtype=uint16)
        self._iabs_chan = zeros([5, n], dtype=uint16)
        self._iwin1_region = zeros(5, dtype=uint16)
        self._iwin1_chan = zeros([5, n], dtype=uint16)

        i1 = self._first_band
        i2 = self._n_bands

        # config.iwaterwv = 1: only 940 nm bands
        # = 2: only 1130 nm bands
        # = 3: both regions used during wv retrieval

        if (self.config.iwaterwv >= 1 and self.config.iwaterwv <= 3):

            # 940 nm
            if (self._ch940w1 == 0):
                # wv bands are defined for 1130 nm, but specify bands for 940nm interpolation
                self._ch940w2 = 0

                li = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= 0.850, self._wvlsen[self._first_band:self._n_bands] < 0.899)))
                cnt1 = li.size
                if (cnt1 > 0):
                    self._ch940w1 = li[cnt1 - 1] + 1

                li = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= 0.995, self._wvlsen[self._first_band:self._n_bands] < 1.080)))
                cnt1 = li.size
                if (cnt1 > 0):
                    self._ch940w2 = li[0] + 1
            else:
                if (self._ch940w1 > 0 and self._ch940a1 >= 1 and self._ch940a2 >= 1):
                    #li = where32(ravel(bitwise_and(wvlsen[i1:(i2)+1] >= wvlsen[ch940a1 - 1], wvlsen[i1:(i2)+1] <= wvlsen[ch940a2 - 1])))
                    li = where32(ravel((self._wvlsen[i1:(i2)+1] >= self._wvlsen[self._ch940a1 - 1]) & (self._wvlsen[i1:(i2)+1] <= self._wvlsen[self._ch940a2 - 1])))

                    cnt = li.size
                    if (cnt > 0):
                        self._iabs_chan[0, 0:(cnt - 1)+1] = li + 1 # list of bands, add +1 to start with band 1

                    self._iabs_region[0] = cnt # number of 940 nm absorption bands
                    # but the 940 nm window channels (from ch940w1+1) are used for interpolation if set

                    li = where32(ravel(bitwise_and(self._wvlsen[i1:(i2)+1] >= 0.850, self._wvlsen[i1:(i2)+1] <= 0.899)))
                    cnt = li.size
                    if (cnt == 0):
                        li = where32(ravel(bitwise_and(self._wvlsen[i1:(i2)+1] >= 0.820, self._wvlsen[self._first_band:self._n_bands] < 0.899)))
                        cnt = li.size
                    cnt2 = max([cnt, 2])
                    self._iwin1_region[0] = cnt2 # cnt2 channels in 850 - 899 nm
                    self._iwin1_chan[0, 0:(cnt - 1)+1] = li + 1
                    self._iwin1_chan[0, 0] = self._ch940w1
                    self._iwin1_chan[0, cnt2 - 1] = self._ch940w1 # make sure the right-most channel is ch940w1

                    # find and remove double elements (spectrometer overlap)
                    ch1 = list(set(self._iwin1_chan[0, 0:(cnt2 - 1)+1]))
                    self._iwin1_chan[0, 0:(cnt2 - 1)+1] = 0 # reset to 0
                    self._iwin1_chan[0, 0:(size(ch1) - 1)+1] = ch1 # update channel list
                    self._iwin1_region[0] = size(ch1) # update number of valid window channels,
                    # left shoulder of 940 nm
                    if (self._ch940w2 > 0):
                        li = where32(ravel(bitwise_and(self._wvlsen[i1:(i2)+1] >= self._w995, self._wvlsen[i1:(i2)+1] <= 1.095)))
                        cnt = li.size
                        cnt2 = max([cnt, 2])
                        self._iwin1_region[1] = cnt2 # cnt2 channels in 995 - 1095 nm
                        self._iwin1_chan[1, 0:(cnt - 1)+1] = li + 1
                        self._iwin1_chan[1, 0] = self._ch940w2 # make sure the left-most channel is ch940w2
                        self._iwin1_chan[1, cnt2 - 1] = self._ch1130w1 # " " right-most " " ch1130w1

                        # find and remove double elements (spectrometer overlap)
                        ch1 = list(set(self._iwin1_chan[1, 0:(cnt2 - 1)+1]))
                        self._iwin1_chan[1, 0:(cnt2 - 1)+1] = 0 # reset to 0
                        self._iwin1_chan[1, 0:(size(ch1) - 1)+1] = ch1 # update channel list
                        self._iwin1_region[1] = size(ch1) # update number of valid channels in 940 nm region
                        # right shoulder of 940 nm

            # 1130 nm
            if (self._ch1130w1 == 0):
                # wv bands are defined for 940 nm, but specify bands for 1130nm interpolation
                self._ch1130w2 = 0

                li = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= self._w995, self._wvlsen[self._first_band:self._n_bands] < 1.095)))
                cnt1 = li.size
                if (cnt1 > 0):
                    self._ch1130w1 = li[cnt1 - 1] + 1

                li = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= 1.180, self._wvlsen[self._first_band:self._n_bands] < 1.255)))
                cnt1 = li.size
                if (cnt1 > 0):
                    self._ch1130w2 = li[0] + 1
            else:

                if (self._ch1130w1 > 0 and self._ch1130a1 >= 1 and self._ch1130a2 >= 1):
                    li = where32(ravel(bitwise_and(self._wvlsen[i1:(i2)+1] >= self._wvlsen[self._ch1130a1 - 1], self._wvlsen[i1:(i2)+1] <= self._wvlsen[self._ch1130a2 - 1])))
                    cnt = li.size
                    if (cnt > 0):
                        self._iabs_chan[1, 0:(cnt - 1)+1] = li + 1 # list of bands, add +1 to start with band 1
                    self._iabs_region[1] = cnt # number of 1130 nm absorption bands
                    # but the 1130 nm window channels (from ch1130w1+1) are used for interpolation if set
                    if (self.config.iwaterwv == 1):
                        cnt = 0 # 1130 nm absorption bands are not used during wv retrieval: iabs_region[1] = 0

                    li = where32(ravel(bitwise_and(self._wvlsen[i1:(i2)+1] >= self._w995, self._wvlsen[i1:(i2)+1] <= 1.095)))
                    cnt = li.size
                    cnt2 = max(array([cnt, 2]))
                    self._iwin1_region[1] = cnt2 # cnt2 channels in 995 - 1095 nm
                    self._iwin1_chan[1, 0:(cnt - 1)+1] = li + 1
                    self._iwin1_chan[1, 0] = (maximum(self._ch940w2, 1)) # make sure the left-most channel is ch940w2
                    # the > 1 is required because of compress iwin1_chan later
                    self._iwin1_chan[1, cnt2 - 1] = self._ch1130w1 # make sure the right-most ch is ch1130w1
                    if (self._ch1130w2 > 0):
                        li = where32(ravel(bitwise_and(self._wvlsen[i1:(i2)+1] >= 1.180, self._wvlsen[i1:(i2)+1] <= 1.250)))
                        cnt = li.size
                        cnt2 = max(array([cnt, 2]))
                        self._iwin1_region[2] = cnt2 # cnt2 channels in 1200 - 1250 nm
                        self._iwin1_chan[2, 0:(cnt - 1)+1] = li + 1
                        self._iwin1_chan[2, 0] = self._ch1130w2 # make sure the left-most channel is ch1130w2
                        self._iwin1_chan[2, cnt2 - 1] = self._ch1130w2

        # config.iwaterwv = 4 (thermal region, to be implemented later)
        # assign the 1400, 1900 nm channels for interpolation
        abs_reg1 = zeros([2, 2], float32)
        abs_reg1[0,:] = array([self.config.wl1400a[0], self.config.wl1400a[1]]) # 1400 nm absorption region
        abs_reg1[1,:] = array([self.config.wl1900a[0], self.config.wl1900a[1]]) # 1900 nm absorption region
        for j in arange(0, 2):
            li = where32(ravel(bitwise_and(self._wvlsen[self._first_band:self._n_bands] >= abs_reg1[j, 0], self._wvlsen[self._first_band:self._n_bands] <= abs_reg1[j, 1])))
            cnt = li.size
            if (cnt > 0):
                self._iabs_region[2 + j] = cnt # 940, 1130 are stored in iabs_region[0:1]
                # 1400, 1800 nm region: iabs_region[2:3]
                self._iabs_chan[2 + j, 0:(cnt - 1)+1] = li + 1 # start channel number at 1

        # compress iabs_chan = zeros(n,5) to required n
        nmax = 1
        for j in arange(0, 5):
            li = where32(ravel(self._iabs_chan[j,:] > 0))
            cnt = li.size
            if (cnt > nmax):
                nmax = cnt

        ixx = zeros((5, nmax), int)

        for j in arange(0, 5):
            li = where32(ravel(self._iabs_chan[j,:] > 0))
            cnt = li.size
            if (cnt > 0):
                ixx[j, cnt-1] = self._iabs_chan[j, 0:(cnt - 1)+1]
        self._iabs_chan = ixx

        nmax = 1
        # compress iwin1_chan = zeros(n,5)
        for j in arange(0, 5):
            li = where32(ravel(self._iwin1_chan[j,:] > 0))
            cnt = li.size
            if (cnt > nmax):
                nmax = cnt

        ixx = zeros((5, nmax), int)
        for j in arange(0, 5):
            li = where32(ravel(self._iwin1_chan[j,:] > 0))
            cnt = li.size
            if (cnt > 0):
                ixx[j, 0:cnt] = self._iwin1_chan[j, 0:cnt]
        self._iwin1_chan = ixx

        # Attention:
        # ch940w1, ch940w2 must not be reset to 0 if config.iwaterwv=2 (1130 nm wv retrieval) because they
        # indicate the interpolation region (if 940/1130 nm interpolation is set)
        # similarly, ch1130w1, ch1130w2 must not be reset to 0 if config.iwaterwv=1 !

        # update the defaults from "load_common" routine
        # (ch940w1, ch940w2) indicates wavelength interval for interpolation of bands in 940 nm absorption region
        # ch940w1, ch940w2, and ch1130w1, ch1130w2 are read from .ini file and may have been updated
        # in routine "select_wv_options"
        if (self._ch940w1 > 0):
            wl1 = min([self._wvlsen[self._ch940w1], 0.880])
            # Allow a larger margin in the 940nm region for the hull correction (if 940nm interpolation is set).
            # The hull correction is very small (< 1% for rho) in the 880-900 nm region, but still improves
            # the visual appearance and makes hyperspectral spectra smoother.
            # Start with channel ch940w1+1 channel (zero-index ch940w1) or a 0.880 nm channel (whichever has the
            # smaller wavelength). wl1 enters config.wl940a which is used in "spectrum".
            # Defaults of config.wl940a, config.wl1130a were already set in "load_commons"
            if (self.config.wl940a[0] > wl1):
                wl1 = self.config.wl940a[0]
            self.config.wl940a[0] = wl1
            if (self._ch940w2 <= self._ch940w1):
                self._ch940w2 = 0

            if (self._ch940w2 > 0):
                wl2 = self._wvlsen[self._ch940w2 - 1] - 0.001
                if (self.config.wl940a[1] < wl2):
                    wl2 = self.config.wl940a[1]
                if (wl2 > self._w995):
                    wl2 = self._w995
                self.config.wl940a[1] = wl2

        if (self._ch1130w1 > 0):
            wl1 = self._wvlsen[self._ch1130w1 - 1] + 0.001
            if (wl1 < 1.080):
                wl1 = 1.080
            if (self.config.wl1130a[0] > wl1):
                wl1 = self.config.wl1130a[0]
            self.config.wl1130a[0] = wl1
            if (self._ch1130w2 <= self._ch1130w1):
                self._ch1130w2 = 0

            if (self._ch1130w2 > 0):
                wl2 = self._wvlsen[self._ch1130w2 - 1] - 0.001
                if (self.config.wl1130a[1] < wl2):
                    wl2 = self.config.wl1130a[1]
                if (wl2 > 1.195):
                    wl2 = 1.195
                self.config.wl1130a[1] = wl2

        wv_reg = zeros([6, 2], float32)
        wv_reg[0, 0:2] = array([0.700, 0.740]) # wv absorption regions
        wv_reg[1, 0:2] = array([0.800, 0.840]) # for a channel in this region band_index_wvdepend is set to 1
        wv_reg[2, 0:2] = array([0.890, self._w995])
        wv_reg[3, 0:2] = array([1.075, 1.240])
        wv_reg[4, 0:2] = array([1.290, 2.110])
        wv_reg[5, 0:2] = array([2.260, 2.600])
        siz = len(wv_reg)
        if (self._band_index_wvdepend[0] <= 0):
            self._band_index_wvdepend = zeros(self._n_bands_all)
        for j in arange(0, siz-1):
            li = where32(ravel(bitwise_and(self._wvlsen + self._fwhm > wv_reg[j, 0], self._wvlsen - self._fwhm < wv_reg[j, 1])))
            cnt = li.size
            if (cnt > 0):
                self._band_index_wvdepend[li] = 1 # flag to perform wv-dependent atm. correction

        if (self.config.iwaterwv == 1 or self.config.iwaterwv == 3):
            # IF 940 and 1130 regions are selected, restrict APDA-LIRR to 940 region
            # which usually yields better results.
            # ch940 = [ wl1, wl2, a1, a2, wr1, wr2 ] w=window, l=left, a=absorption, r=right
            if (self.config.ch940[0] == self.config.ch940[1]):
                self._reference_c = array([self.config.ch940[1]])
            else:
                self._reference_c = zeros((self.config.ch940[1] - self.config.ch940[0] + 1), dtype=int)
                for j in arange(0, (self._reference_c.size)):
                    self._reference_c[j] = self.config.ch940[0] + j
            if (self.config.ch940[4] == self.config.ch940[5]):
                self._reference_c2 = array([self.config.ch940[5]])  # array bands later
            else:
                self._reference_c2 = zeros((self.config.ch940[5] - self.config.ch940[4] + 1), dtype=int)
                for j in arange(0, (self._reference_c2.size)):
                    self._reference_c2[j] = self.config.ch940[4] + j
            if (self.config.ch940[4] > 0):
                self._reference_c = array([self._reference_c, self._reference_c2])  # channels [ wl1:wl2, wr1:wr2]
            self._reference_c = self._reference_c[where32(ravel(self._reference_c > 0))]  # only channels > 0

            self._measure_c = zeros((self._ch940a2 - self._ch940a2 + 1), dtype=int)  # measurement = absorption channels
            n_mea = self._measure_c.size
            for j in arange(0, (n_mea)):
                self._measure_c[j] = self._ch940a1 + j
            index = where32(ravel(bitwise_and(self._wvlsen[self._measure_c - 1] >= self._wvlsen[self._measure_c[0] - 1],
                                              self._wvlsen[self._measure_c - 1] <= self._wvlsen[
                                                  self._measure_c[n_mea - 1] - 1])))
            self._measure_ch = self._measure_c[index]
        else:
            # ch1130 = [ wl1, wl2, a1, a2, wr1, wr2 ] w=window, l=left, a=absorption, r=right
            if (self._ch1130[0] == self._ch1130[1]):
                self._reference_c = array([self._ch1130[1]])
            else:
                self._reference_c = zeros((self._ch1130[1] - self._ch1130[0] + 1), dtype=int)
                for j in arange(0, (self._reference_c.size)):
                    self._reference_c[j] = self._ch1130[0] + j
            if (self._ch1130[4] == self._ch1130[5]):
                self._reference_c2 = array([self._ch1130[5]])  # array bands later
            else:
                self._reference_c2 = zeros((self._ch1130[5] - self._ch1130[4] + 1), dtype=int)
                for j in arange(0, (self._reference_c2.size)):
                    self._reference_c2[j] = self._ch1130[4] + j
            if (self._ch1130[4] > 0):
                self._reference_c = array([self._reference_c, self._reference_c2])  # channels [ wl1:wl2, wr1:wr2]
            self._reference_c = self._reference_c[where32(ravel(self._reference_c > 0))]  # only channels > 0

            self._measure_c = zeros((self._ch1130a2 - self._ch1130a1 + 1), dtype=int)
            n_mea = self._measure_c.size
            for j in arange(0, (n_mea)):
                self._measure_c[j] = self._ch1130a1 + j
            index = where32(ravel(bitwise_and(self._wvlsen[self._measure_c - 1] >= self._wvlsen[self._measure_c[0] - 1],
                                              self._wvlsen[self._measure_c - 1] <= self._wvlsen[
                                                  self._measure_c[n_mea - 1] - 1])))
            self._measure_ch = self._measure_c[index]

        return


    def apda1_lut_constvis_a3(self, ishort=None):

    # calculate LUT, the radiance ratio L(C) / L(D), for each absorption channel
    # L(C-D, wv) is calculated for u=0 (point C) to u=4500 (points D)
    # The ratio {L(C)-Lp} / {L(D)-Lp} is nearly independent of rho
    # apda_index = 1000*L(C) / L(D)

    # adapt uu1 to current ground altitude "altit" by interpolation
    # ATCOR standard grid: 0(0.1)3.5 km and 4.0(0.5)8.5 km

        if (ishort is None):
            ishort = 0

        if self.config.midLatitude == 'SUMMER':
            # get lp, e0t, etc for the water vapor columns wv04, wv10, wv20, wv29, wv40, wv50
            wvcols = array(['wv04', 'wv10', 'wv20', 'wv29', 'wv40', 'wv50'])
            uu1 = array([400, 1000, 2000, 2900, 4000, 5000])
        elif self.config.midLatitude == 'WINTER':
            # get lp, e0t, etc for the water vapor columns wv04, wv10, wv20, wv29, wv40, wv50
            wvcols = array(['wv02', 'wv04', 'wv08', 'wv11'])
            uu1 = array([200, 400, 800, 1100])

        self._nuu1 = uu1.size

        # for ATCOR3 the following arrays will be 3-dimensional to include the elevation class
        self._lp_all = zeros([self._n_bands_all, self._n_alti, self._nuu1], float32)
        self._e0t_all = zeros([self._n_bands_all, self._n_alti, self._nuu1], float32)
        self._edift_all = zeros([self._n_bands_all, self._n_alti, self._nuu1], float32)
        self._q_all = zeros([self._n_bands_all, self._n_alti, self._nuu1], float32)
        self._tdir_all = zeros([self._n_bands_all, self._n_alti, self._nuu1], float32)
        self._tdif_all = zeros([self._n_bands_all, self._n_alti, self._nuu1], float32)
        self._spha_all = zeros([self._n_bands_all, self._n_alti, self._nuu1], float32)

        for j in arange(0, (self._nuu1)):
            wv00 = self.config.atmDataFn.rfind('_wv')
            wv01 = self.config.atmDataFn[wv00+1:wv00+5]
            self._atmDataFn = self.config.atmDataFn.replace(wv01, wvcols[j])
            self.logger.info('generating grid for altitude with aerosol file name: %s' % os.path.basename(self._atmDataFn))
            self.altit3_atm()

            self._lp_all[:,:, j] = self._lph
            self._e0t_all[:,:, j] = self._e0th
            self._edift_all[:,:, j] = self._edifth
            self._q_all[:,:, j] = self._qh
            self._tdir_all[:,:, j] = self._tdirh
            self._tdif_all[:,:, j] = self._tdifh
            self._spha_all[:,:, j] = self._sphah

        self._tsun_all = float32(exp(log(maximum(self._tdir_all, 0.01)) / cos(radians(self._solze))))

        self._atmDataFn = self.config.atmDataFn # restore original atmfile
        self.altit3_atm() # RT terms for original atmfile

        if (ishort == 1):
            return

        ncf = 2 * 2 # 2 coefficients for exponential fit, 2 fit regions (old version)
        if (self._nuu1 == 5):
            ncf = 2 * 3 # 2 coefficients for exponential fit, 3 fit regions
        if (self._nuu1 == 6):
            ncf = 2 * 4 # 2 coefficients for exponential fit, 4 fit regions

        # wv grid points j1 index for lp_fit etc
        # region 0: uu1_interp[0:2] ( 400, 1000, 2000) 0:1 (two coefficients per region)
        # region 1: uu1_interp[1:3] (1000, 2000, 2900) 2:3
        # region 2: uu1_interp[3:4] (2000, 2900, 4000) 4:5
        # region 3: uu1_interp[4:5] (2900, 4000, 5000) 6:7

        nb = min(array([self._n_bands, self._n_refl]))
        self._lp_fit = zeros([nb, self._n_alti, ncf], float32)
        self._e0t_fit = zeros([nb, self._n_alti, ncf], float32)
        self._edift_fit = zeros([nb, self._n_alti, ncf], float32)
        self._tsun_fit = zeros([nb, self._n_alti, ncf], float32)
        self._q_fit = zeros([nb, self._n_alti, ncf], float32)
        self._spha_fit = zeros([nb, self._n_alti, ncf], float32)

        self._uu1_altit_save = self._uu1_altit.copy() # save for interpolation with 20 m grid
        if (self._n_alti > 100):
            x = arange(self._n_alti, dtype=float32)/5
            y = arange(self._nuu1, dtype = float32)
            z = self._uu1_altit.copy()

            self._uu1_altit = rectBivariateSpline(x, y, z) # 20 m grid
            #uu1_altit = interpol2d(uu1_altit, arange(nuu1), arange(n_alti, dtype=float32) / 5) # 20 m grid

        for ir in arange(0, (ncf // 2)): # (ncf/2 = 2, 3 or 4)
            for j in arange(0, (nb)):
            # calculate exponential fit coefficients a,b of: y = exp(a + b*sqrt(x) )
                for k in arange(0, (self._n_alti)):
                # get wv for current elevation and the nuu1 water vapor levels
                # convert altitude_grid 0(0.1)4.5 km into array index
                # an NaN usually happens in the 1.37-1.44 or 1.80-1.94 micron region
                # the NaN during regression cannot be prevented, but is replaced in "check_if_finite"

                    self._uu1_interp3 = self._uu1_altit[k, ir:(ir + 2)+1] # 3 wv grid points

                    with errstate(over='ignore'):
                        self._lp_fit[j, k, ir * 2:(ir * 2 + 1)+1] = fit_coeff(self._uu1_interp3, self._lp_all[j, k, ir:(ir + 2)+1]).astype(float32)
                        self._e0t_fit[j, k, ir * 2:(ir * 2 + 1)+1] = fit_coeff(self._uu1_interp3, self._e0t_all[j, k, ir:(ir + 2)+1]).astype(float32)
                        self._edift_fit[j, k, ir * 2:(ir * 2 + 1)+1] = fit_coeff(self._uu1_interp3, self._edift_all[j, k, ir:(ir + 2)+1]).astype(float32)
                        self._tsun_fit[j, k, ir * 2:(ir * 2 + 1)+1] = fit_coeff(self._uu1_interp3, self._tsun_all[j, k, ir:(ir + 2)+1]).astype(float32)
                        self._q_fit[j, k, ir * 2:(ir * 2 + 1)+1] = fit_coeff(self._uu1_interp3, self._q_all[j, k, ir:(ir + 2)+1]).astype(float32)
                        self._spha_fit[j, k, ir * 2:(ir * 2 + 1)+1] = fit_coeff(self._uu1_interp3, self._spha_all[j, k, ir:(ir + 2)+1]).astype(float32)

        self._uu1_altit = self._uu1_altit_save # restore original 100m grid

        return


    def prepare_wv_retrieval(self):

    # Purpose: read the required arrays for the whole image
    # dn1w (band nk1, left window shoulder)
    # dn2w (band nk2, right " " )
    # dnabs (band nka, absorption band)
    # dn0w (band nk1-1, only for MERIS type)
    #
    # Input: n_rows, n_cols, liback, cntback, wvlsen etc from atcor3.inc
    #
    # Output: nk1, nk2, nka, dn1w, dn2w, dnabs, dn0w
    # w1, w2: weight factors based on wavelength band distances (for wv_retrieval_apda1_constvis)
    # Calls:
    # ! apda1_lut_constvis_a3
    # ! read_atm_hyper_a3
    # ! intpol_lut5
    # ! read_channel_bsq

    # update geometry for image center

        dn2w = 0
        dn0w = 0
        w1 = 0
        w2 = 0

        self._solze = float32(mean(self.config.solze_arr))
        self._solaz = float32(mean(self.config.solaz_arr))
        self._thv = float32(mean(self.config.vza_arr))
        self._phiv = float32(mean(self.config.vaa_arr))

        self.apda1_lut_constvis_a3(ishort=1) # updates lp_all(nuu1,n_alti,n_bands_all) for current visib

        # Prerequisite: only 1 absorption channel, only 940nm or 1130nm, not both
        nk_wv = zeros(2)
        n_wv_refch = zeros((2, 1), dtype=int)

        if (self._ch940a1 > 0):
            n_wv_refch[0] = self._ch940w1 ; n_wv_refch[1] = self._ch940w2 # for log file
            if (self._ch940w2 <= 0):
                n_wv_refch[1] = self._ch940w1 # repeat channel for log file
            ir = 0
            nk_wv[0] = self._ch940a1 # (for log file)
        if (self._ch1130a1 > 0):
            n_wv_refch[0] = self._ch1130w1 ; n_wv_refch[1] = self._ch1130w2
            if (self._ch1130w2 <= 0):
                n_wv_refch[1] = self._ch1130w1
            ir = 1
            nk_wv[0] = self._ch1130a1
        nk1 = self._iwin1_chan[ir, self._iwin1_region[ir] - 1] # last window channel on left shoulder
        nk2 = self._iwin1_chan[ir + 1, 0] # first " " right "

        nk = self._iabs_chan[ir, 0]
        nka = nk

        if (nk2 > 0):
            w1 = (self._wvlsen[nk2 - 1] - self._wvlsen[nk]) / (self._wvlsen[nk2 - 1] - self._wvlsen[nk1 - 1])
            w2 = (self._wvlsen[nk] - self._wvlsen[nk1 - 1]) / (self._wvlsen[nk2 - 1] - self._wvlsen[nk1 - 1])

        nk1a = 0
        if ((self._wvlsen[nk1 - 1] - self._wvlsen[nk1 - 2]) < 0.025 and ((self._wvlsen[nk] - self._wvlsen[nk1 - 1]) < 0.020)):
            nk1a = nk1 - 1
            # MERIS type: bands at 865nm (nk1a), 885nm (nk1), 900nm (nk)
            dn0w = self.tables.getBand(nk1a-1) #band is now read from 0-index

        dn1w = self.tables.getBand(nk1-1) #band is now read from 0-index
        dnabs = self.tables.getBand(nk-1) #band is now read from 0-index
        if (nk2 > 0):
            dn2w = self.tables.getBand(nk2-1) #band is now read from 0-index
        if (nk1a > 0):
            dn2w = self.tables.getBand(nk1a-1) #band is now read from 0-index

        return nk1, nk2, nka, dn1w, dn2w, dnabs, dn0w, w1, w2

    def wv_retrieval_apda1(self, nk1, nk2, nka, dn1w, dn2w, dnabs, dn0w, w1, w2):
    # Purpose: calculate water vapor map based on cell processing
    #
    # Approximation: scene average visibility is used visib=visext[meanvi]
    #
    # Input: nk1, nk2, nka: band numbers, left shoulder window band, right window, abs. band,
    # corresponding arrays dn1w, dn2w, dnabs (see prepare_wv_retrieval), full image size
    # dn0w corresponds to band nk1-1 (case of MERIS NIR bands)
    # w1, w2: channel weight factors
    #
    # Output: wv(n_rows,n_cols) map
    #
    # Calls:
    # ! apda1_lut_constvis_a3
    # ! altit3_atm
    # ! read_atm_hyper_a3
    # ! intpol_lut5
    # ! wv_retrieval_apda2_constvis

        self._u500m = transpose(self._uu1_altit[arange(self._n_alt + 2) * 5, 0:(self._nuu1 - 1)+1]) # wv 0(500)3500 m

        wv = zeros([self.config.nrows, self.config.ncols], dtype=uint16) + 1000 # default, background will later be set to 0
        # water will be reset to average of land pixels
        # SIIMPC-1017: The Water Vapour retrieval algorithm is designed to work with pixels
        # of bands B09 and B08A directly illuminated by the sun. When the cloud shadow class
        # is present on the scene those pixels should be excluded from the Water Vapour retrieval
        # algorithm like cloudy pixels.
        mask_land = zeros([self.config.nrows, self.config.ncols], dtype=uint16)
        scl = self.tables.getBand(Band.SCENE_CLASSIFICATION)  #self.tables.SCL
        mask_land[(scl == self.config.vegetation) |
                  (scl == self.config.bareSoils) |
                  (scl == self.config.snowIce)] = 1
        list_land = where32(ravel(mask_land == 1))
        np_land = list_land.size

        factor = self.config.resolution / 60.0
        xcell = (self._xcell * factor + 0.5).astype(int)
        ycell = (self._ycell * factor + 0.5).astype(int)

        if (np_land > -1):
            nrows = self.config.nrows
            ncols = self.config.ncols

            try:  # agular values are arrays:
                x = arange(nrows, dtype=float32) / (nrows - 1) * self.config.solze_arr.shape[0]
                y = arange(ncols, dtype=float32) / (ncols - 1) * self.config.solze_arr.shape[1]
                szi = rectBivariateSpline(x, y, self.config.solze_arr)
                x = arange(nrows, dtype=float32) / (nrows - 1) * self.config.solaz_arr.shape[0]
                y = arange(ncols, dtype=float32) / (ncols - 1) * self.config.solaz_arr.shape[1]
                sai = rectBivariateSpline(x, y, self.config.solaz_arr)
                del x
                del y
            except:  # angular value is a scalar:
                szi = ones([nrows, ncols], dtype=float32) * self.config.solze_arr.mean()
                sai = ones([nrows, ncols], dtype=float32) * self.config.solaz_arr.mean()

            vzi = ones([nrows, ncols], dtype=float32) * self.config.vza_arr
            vai = ones([nrows, ncols], dtype=float32) * self.config.vaa_arr

            # loop over cells
            # Functions lph, e0th, edifth, tsunh, sphah, qh (for window channels)
            # and lp_fit, e0t_fit, edift_fit, tsun_fit (abs. bands) are stored per cell for later use.

            self.apda1_lut_constvis_a3() # dummy call to get size of lp_fit

            siz = array(self._lp_fit.shape) # lp_fit=fltarr(ncf,n_alti,n_bands)
            self._lp_fit_cell = zeros((self._ny_cell, self._nx_cell, siz[0], siz[1], siz[2]), float32)
            self._e0t_fit_cell = zeros((self._ny_cell, self._nx_cell, siz[0], siz[1], siz[2]), float32)
            self._edift_fit_cell = zeros([self._ny_cell, self._nx_cell, siz[0], siz[1], siz[2]], float32)
            self._tsun_fit_cell = zeros([self._ny_cell, self._nx_cell, siz[0], siz[1], siz[2]], float32)

            rt_all = zeros([self._n_refl, self._nuu1, self._n_alt, 4], float32) # 4 fcts={Lp, edift, e0t, spha}
            # UMW rollback of SIIMPC-907 due to SIIMPC-1003:
            for jx in arange(0, (self._nx_cell)):
                for jy in arange(0, (self._ny_cell)):
                    subset = array([self._ycell[jy, 0], self._ycell[jy, 1], self._xcell[jx, 0], self._xcell[jx, 1]])
                    sub_r = array([ycell[jy, 0], ycell[jy, 1], xcell[jx, 0], xcell[jx, 1]])
                    # update geometry for current cell
                    self._solze = float32(mean(szi[sub_r[0]:(sub_r[1])+1, sub_r[2]:(sub_r[3])+1]))
                    self._solaz = float32(mean(sai[sub_r[0]:(sub_r[1])+1, sub_r[2]:(sub_r[3])+1]))
                    self._thv = float32(mean(vzi[sub_r[0]:(sub_r[1])+1, sub_r[2]:(sub_r[3])+1]))
                    self._phiv = float32(mean(vai[sub_r[0]:(sub_r[1])+1, sub_r[2]:(sub_r[3])+1]))
                    self.apda1_lut_constvis_a3()
                    # to get lp_all(nuu1,n_alti,n_bands_all), lp_fit(ncf, n_alti, nb)
                    # and lp_abs_fit(ncf,maxel-minel+1,nabsch_all)
                    # calls altit3_atm, read_atm_hyper_a3, intpol_lut5

                    for j in arange(0, (self._n_alt)): # rt_all has the 500m grid
                        rt_all[:,:, j, 0] = self._lp_all[:, j * 25,:] # index j*25 refers to the 20m grid, i.e.
                        rt_all[:,:, j, 1] = self._edift_all[:, j * 25,:] # lp_all[*,0,*] represents h = 0m
                        rt_all[:,:, j, 2] = self._e0t_all[:, j * 25,:] # lp_all[*,25,*] represents h = 500m
                        rt_all[:,:, j, 3] = self._spha_all[:, j * 25,:] # lp_all[*,50,*] represents h = 1000m etc

                    # store fcts for reflectance retrieval
                    self._lp_fit_cell[jy, jx,:,:,:] = self._lp_fit
                    self._e0t_fit_cell[jy, jx,:,:,:] = self._e0t_fit
                    self._edift_fit_cell[jy, jx,:,:,:] = self._edift_fit
                    self._tsun_fit_cell[jy, jx,:,:,:] = self._tsun_fit
                    li = where32(ravel(mask_land[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1] == 1))  # check for land pixels
                    # cell is not processed if it contains only background and/or water
                    # (but for iwv_watermask=0 water pixels are also processed)
                    if (li.size > 0):
                        dn1w_sub = dn1w[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                        dnabs_sub = dnabs[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                        if (nk2 > 0):
                            dn2w_sub = dn2w[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]
                        if(dn0w != 0):
                            if (dn0w.size > 1):
                                dn0w_sub = dn0w[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1]

                        wv_cell = self.wv_retrieval_apda2_constvis(rt_all, subset+1)
                        # put wv(cell) into wv(scene)
                        wv[subset[0]:(subset[1])+1, subset[2]:(subset[3])+1] = reshape(wv_cell, [dn1w_sub.shape[0], dn1w_sub.shape[1]])

            try: # fix for SIIMPC-1018 UMW: list land can be empty
                self._wv_av = int(mean(ravel(wv)[list_land])+0.5)
            except:
                self._wv_av = 0

        if (self.config.iwv_watermask > 0):
            # SIIMPC-1017: The Water Vapour retrieval algorithm is designed to work with pixels
            # of bands B09 and B08A directly illuminated by the sun. When the cloud shadow class
            # is present on the scene those pixels should be excluded from the Water Vapour retrieval
            # algorithm like cloudy pixels.
            wv[(scl != self.config.vegetation) &
               (scl != self.config.bareSoils) &
               (scl != self.config.snowIce)] = self._wv_av
            wv[(scl <= self.config.saturatedDefective)] = 0

        lix = where32(ravel(wv < 10))
        if (lix.size > 0):
            ravel(wv)[lix] = self._wv_av # avoid numerical difficulties

        # write wv map into file:
        self.tables.setBand(Band.WATER_VAPOUR, wv) #self.tables.WVP

        # restore solze, solaz, thv, phiv for scene center:
        self._solze = float32(mean(self.config.solze_arr))
        self._solaz = float32(mean(self.config.solaz_arr))
        self._thv = float32(mean(self.config.vza_arr))
        self._phiv = float32(mean(self.config.vaa_arr))

        return


    def read_hcw_file_get_cirrus(self, level):
    # Input:
    # level = 1 ; read list of thin cirrus (not used currently)
    # level = 2 ; read list of medium and thick cirrus (for AOT retrieval)
    # level = 3 ; read list of thick cirrus (for cloud definition during reflectance retrieval)
    # level = 4 ; read list of "thin + medium + thick" cirrus (not used currently)
    #
    # Output:
    # mlist_cirrus: list of cirrus pixels corresponding to level
    #
    # Purpose:
    # Read the "_hcw.bsq" or "_out_hcw.bsq" file, get pixels with medium and high cirrus thickness.
    # level=2: Thin cirrus is not included as it might include the whole image, leaving no dark
    # reference areas for the AOT retrieval.
    #
    # see routine "write_out_hcw_file" for the definition of the labels.

        a = self.tables.getBand(Band.SCENE_CLASSIFICATION) #self.getClassificationMap()
        if(a.size < 1): return False
        if (level == 1): mlist_cirrus = where32(ravel(a == 10))
        elif(level == 2): mlist_cirrus = where32(ravel(bitwise_or(a == 8, a == 9)))
        elif(level == 3): mlist_cirrus = where32(ravel(a == 9))
        elif(level == 4): mlist_cirrus = where32(ravel(bitwise_and(a >= 7, a <= 10)))
        return(mlist_cirrus)


    def read_hcw_file_get_shadow(self):
    # Output:
    # mlist_shadow = list of these pixels
    #
        a = self.getClassificationMap()
        if(a.size < 1): return False
        mlist_shadow = where32(ravel(bitwise_or(a == 2, a == 3)))
        return mlist_shadow


    def checkDem(self):
    # check DEM files
    # If the DEM elevation file name is missing then a flat terrain is assumed
    # with the average elevation as specified in the .inn file.
    # Then slope and aspect will be set to 0 arrays. This case is treated
    # later (routine "dtm_flat")

        if(self.tables.hasBand(Band.DIGITAL_ELEVATION_MAP) == False): #self.tables.DEM
            self.logger.info('DEM is not present, flat surface assumed')
        else:
            if(self.tables.hasBand(Band.SHADOW_MAP) == False): #self.tables.SDW
                self.logger.fatal('shadow map is not present')

            if(self.tables.hasBand(Band.SLOPE) == False): #self.tables.SLP
                self.logger.fatal('slope map is not present')

            #if(self.tables.hasBand(self.tables.ASP) == False):
            #    self.logger.fatal('aspect map is not present')
        return

    def getClassificationMap(self):
        mask_tmp = self.tables.getBand(Band.SCENE_CLASSIFICATION) #self.tables.SCL
        mask = ndarray.copy(mask_tmp)
        return mask
        
    def getNodataMap(self):
        mask_tmp = self.tables.getBand(Band.SCENE_CLASSIFICATION) #self.tables.SCL
        mask = ndarray.copy(mask_tmp)
        mask[mask <= self.config.saturatedDefective] = 0
        mask[mask > self.config.saturatedDefective] = 1
        return mask

    def refl2rad(self, band, identifier): #JL 05/01/2022
        index = self.tables.reindex(identifier)
        radiance = (band * self._es[index] * cos(radians(self._solze)) / pi)  # removed integer division of // pi
        return radiance

    def rmCirrus(self, identifier, radiance=False):
        rho_cir_app = self.tables.getBand(Band.CIRRUS, radiance)
        rho_cir_app = median_filter_2d(rho_cir_app, 3) * self.config.dnScale
        index = self.tables.reindex(identifier)
        gamma = interpol(self._arr_gamma, self._wvl_gamma, self._wvlsen[index])
        band = self.tables.getBand(identifier, radiance)
        return band - rho_cir_app / gamma


    def automaticAerosolDetection(self):
        # fix for SIIMPC-672.4, UMW:
        self.aerosolDetection = 'STARTED'

        self.config.aerosolType = 'RURAL'
        if not self.calcDratioAerosol():
            return False

        self.config.aerosolType = 'MARITIME'
        if not self.calcDratioAerosol():
            return False

        self.config.aerosolType = self._aerosolTypeBest
        self.config.timestamp('L2A_AtmCorr: best fit is: %s/%s' % (self.config.midLatitude, self._aerosolTypeBest))
        self.config.createAtmDataFilename()
        self.config.timestamp('L2A_AtmCorr: with aerosol double ratio: %f' % self._dratio_aeros_best)
        self.aerosolDetection = 'STOPPED'
        return


    def calcDratioAerosol(self):
        self.config.createAtmDataFilename()
        self.config.timestamp('L2A_AtmCorr: testing mid latitude: %s, aerosol type: %s' % (
        self.config.midLatitude, self.config.aerosolType))
        self.process()
        if self._dratio_aeros == None:
            self.config.timestamp(
                'L2A_AtmCorr: not enough DDV pixel present, aerosol double ratio could not be determined')
            self.config.timestamp(
                'L2A_AtmCorr: standard model is: %s %s' % (self.config.midLatitude, self.config.aerosolType))
            self.aerosolDetection = 'STOPPED'
            return False

        self.logger.info('aerosol double ratio: %f' % self._dratio_aeros)
        dratio_aeros_abs_1 = abs(self._dratio_aeros - 1.0)
        dratio_aeros_abs_best_1 = abs(self._dratio_aeros_best - 1.0)

        if dratio_aeros_abs_1 < dratio_aeros_abs_best_1:
            self._dratio_aeros_best = self._dratio_aeros
            self.config.createAtmDataFilename()
            self._atmDataFn = self.config.atmDataFn
            self._aerosolTypeBest = self.config.aerosolType

        return True

    def getWvDependencyIndex(self, band_index): #from Atmcorr_merge
        # TBD: check water vapour dependency:
        if self.config.resolution == 10: # Sen2Cor == 10 m
            wvdx = [0, 0, 0, 0]
        elif self.config.resolution == 30: # Landsat or Hyper_MS
            if self.config.Hyper == False: # Landsat
                if self.exluding_wv_dependency > 0:
                    wvdx = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  # [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
                else:
                    wvdx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
            else: # Hyper_MS
                if (self.config.iwaterwv > 0) & (self.config.iwaterwv <= 3):
                    wvdx = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]  # Sen2Cor > 10 m
                else:
                    wvdx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Sen2Cor > 10 m

        else:
            if (self.config.iwaterwv > 0) & (self.config.iwaterwv <= 3):
                wvdx = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1] # Sen2Cor > 10 m
            else:
                wvdx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Sen2Cor > 10 m
        return wvdx[band_index]
