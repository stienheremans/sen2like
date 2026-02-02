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

'''
Created on 22.04.2015
@author: TPZV
usage python setup.py build_ext
'''
import platform
from setuptools import setup
from Cython.Build import cythonize

system = platform.system()
if system == 'Windows':
	name = 'L2A_AtmCorr.pyd'
else:
	name = 'L2A_AtmCorr.so'

setup(
	name = name,
    ext_modules = cythonize('L2A_AtmCorr.py'),
)

if __name__ == '__main__':
    pass
