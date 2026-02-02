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

from numpy import *
from scipy import interpolate as sp

try:
    from osgeo import gdal, osr
    from osgeo.gdalconst import *
    gdal.TermProgress = gdal.TermProgress_nocb
except ImportError:
    import gdal, osr
    from gdalconst import *

import sys, os

class Error(EnvironmentError):
    pass

def statistics(arr, comment=''):
    if len(arr) == 0:
        return False
    s = 'object:' + str(comment) + '\n'
    s += '--------------------' + '\n'
    s += 'shape: ' + str(shape(arr)) + '\n'
    s += 'sum  : ' + str(arr.sum()) + '\n'
    s += 'mean : ' + str(arr.mean()) + '\n'
    s += 'std  : ' + str(arr.std()) + '\n'
    s += 'min  : ' + str(arr.min()) + '\n'
    s += 'max  : ' + str(arr.max()) + '\n'
    s += '-------------------' + '\n'
    return s


def showImage(arr):
    from PIL import Image
    if (arr.ndim) != 2:
        sys.stderr.write('Must be a two dimensional array.\n')
        return False
    arrmin = arr.min()
    arrmax = arr.max()
    # arrmin = arr.mean() - 3*arr.std()
    # arrmax = arr.mean() + 3*arr.std()
    arrlen = arrmax - arrmin
    arr = clip(arr, arrmin, arrmax)
    scale = 255.0
    scaledArr = (arr - arrmin).astype(float32) / float32(arrlen) * scale
    arr = (scaledArr + 0.5).astype(uint8)
    img = Image.fromarray(arr)
    img.show()
    return True

def showSceneClassification(scene_classification):
    mpl.use('tkagg')
    from matplotlib.colors import ListedColormap
    sc = scene_classification.astype(uint8)
    sc[sc > 11] = 12
    colors = ['black', 'red', 'dimgray', 'saddlebrown', 'lime', 'yellow', 'blue', 'gray', 'darkgray', 'lightgray',
              'lightskyblue', 'violet', 'black']
    cmap = ListedColormap(colors[sc.min():sc.max() + 1])
    plt.imshow(sc, cmap=cmap)
    plt.show(block=False)
    return True
def showImageEqualized(arr):
    mpl.use('tkagg')
    from PIL import Image, ImageOps
    if (arr.ndim) != 2:
        sys.stderr.write('Must be a two dimensional array.\n')
        return False
    arrmin = arr.min()
    arrmax = arr.max()
    # arrmin = arr.mean() - 3*arr.std()
    # arrmax = arr.mean() + 3*arr.std()
    arrlen = arrmax - arrmin
    arr = clip(arr, arrmin, arrmax)
    scale = 255.0
    scaledArr = (arr - arrmin).astype(float32) / float32(arrlen) * scale
    arr = (scaledArr + 0.5).astype(uint8)
    img = Image.fromarray(arr)
    img = ImageOps.equalize(img)
    plt.imshow(img, cmap='gray')
    plt.show(block=False)
    return True

def rectBivariateSpline(xIn, yIn, zIn):
    x = arange(zIn.shape[0], dtype=float32)
    y = arange(zIn.shape[1], dtype=float32)

    f = sp.RectBivariateSpline(x, y, zIn)
    del x
    del y
    return f(xIn, yIn)


def chmod_recursive(targetdir, mode):
    # fix for SIIMPC-598.2 UMW:
    filelist = os.listdir(targetdir)
    errors = []
    os.chmod(targetdir, mode)
    for fnshort in filelist:
        filename = os.path.join(targetdir, fnshort)
        try:
            if os.path.islink(filename):
                pass
            elif os.path.isdir(filename):
                os.chmod(filename, mode)
                chmod_recursive(filename, mode)
            else:
                os.chmod(filename, mode)
        except Error as err:
            errors.extend(err.args[0])
        except EnvironmentError as why:
            errors.append((filename, str(why)))
    if errors:
        raise Error(errors)
    # end fix for SIIMPC-598.2
    return


def getResolutionIndex(resolution):
    res = resolution
    if res == 10:
        return 0
    elif res == 20:
        return 1
    elif res == 60:
        return 2
    else:
        return False
