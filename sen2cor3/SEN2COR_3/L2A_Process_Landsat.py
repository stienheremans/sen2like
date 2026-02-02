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

#from L2A_SceneClass import L2A_SceneClass

from L2A_AtmCorr import L2A_AtmCorr
from L2A_BandIO_Landsat import BandIO
from L2A_Quality_Landsat import L2A_Quality_Landsat
from shutil import copyfile
import sys, os
from datetime import datetime

class L2A_ProcessLandsat(object):
    def __init__(self, config):
        self._config = config
        self._logger = config.logger

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

    def get_config(self):
        return self._config

    def set_config(self, value):
        self._config = value

    def del_config(self):
        del self._config

    def __exit__(self):
        sys.exit(-1)

    config = property(get_config, set_config, del_config, "config's docstring")
    tables = property(get_tables, set_tables, del_tables, "tables's docstring")
    logger = property(get_logger, set_logger, del_logger, "logger's docstring")

    def preprocess(self):
        self.logger.info('pre-processing with resolution %d m', self.config.resolution)
        # this is to check the config for the L2A_AtmCorr in ahead.
        # This has historical reasons due to ATCOR porting.
        # Should be moved to the Config for better design:

        # This was  containing only the ozone retrieval that is now in the BandIO_Landsat


        return True

    def process(self):
        self._localTimestamp = datetime.utcnow()
        astr = 'L2A processing for Landsat'
        self.config.timestamp(astr)
        self.config.timestamp('process_landsat: start of pre processing')
        if self.preprocess() == False:
            self.logger.fatal('Module %s failed' % (self.config.file_prefix))
            return False
        self.tables = BandIO(self.config)
        if self.tables.import_aux_band_list() == False:
            self.logger.fatal('Import of band list failed')
            return False
        if self.config.scClassic:
            from L2A_SceneClass import L2A_SceneClass
        else:
            # Force "sc_classic" mode for Landsat (version 3.1.0)
            from L2A_SceneClass import L2A_SceneClass
            #from L2A_SceneClass_evolution import L2A_SceneClass
        sc = L2A_SceneClass(self.config, self.tables)
        self.logger.info('processing Scene Classification for Landsat')
        if sc.process() == False:
            self.logger.fatal('Module %s failed' % (self.config.L2A_TILE_ID))
            return False
        if self.config.scOnly == False:
            ac = L2A_AtmCorr(self.config, self.tables)
            self.config.timestamp('processing Atmospheric Correction for Landsat')
            if ac.process() == False:
                 self.logger.fatal('Module %s failed' % (self.config.L2A_TILE_ID))
                 return False

        self.config.timestamp('start of post processing')
        if self.postprocess() == False:
            self.logger.fatal('Module %s failed' % (self.config.L2A_TILE_ID))
            return False
        # else:
        #     pr.disable()
        #     s = StringIO.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(.25, 'L2A_')
        #     ps.print_stats()
        #     profile = s.getvalue()
        #     s.close()
        #     with open(os.path.join(self.config.logDir, 'runtime_profile.log'), 'w') as textFile:
        #         textFile.write(profile)
        #         textFile.close()

        return True

    def postprocess(self):
        fnLogBase = os.path.basename(self._config.fnLog)
        try:
            qi = L2A_Quality_Landsat(self.config, self.tables) #producing the L2A_Quality.xml for Landsat
            if not qi.update_qi_landsat():
                self.config.timestamp('Warning: L2A_Quality.xml cannot be updated')
                self._config.logger.error('L2A_Quality.xml cannot be updated')
                pass
                #return False
        except:
            self.config.timestamp('Warning: L2A_Quality.xml cannot be updated')
            self._config.logger.error('L2A_Quality.xml cannot be updated')
            pass
            #return False
        if self.tables.export_bands() == False: #exporting bands
            return False
        if self.tables.export_landsat_metadata() == False: #copying and updating the landsat metadata in the L2 folder
            return False
        if self.tables.export_not_used_bands() == False: #coping the not_used_bands in the L2 folder
            return False
        try:
            tMeasure = datetime.utcnow() - self._localTimestamp
            self.config.writeTimeEstimation(tMeasure)
        except:
            pass
        try:
            fnLogIn = self._config.fnLog
            outpath = os.path.join(self._config.output_dir, self._config.L2A_TILE_ID)
            fnGippIn = self._config.configFn
            fnGippOut = os.path.join(outpath, os.path.basename(fnGippIn))
            copyfile(fnGippIn, fnGippOut)
            fnGippIn = self._config.configSC
            fnGippOut = os.path.join(outpath, os.path.basename(fnGippIn))
            copyfile(fnGippIn, fnGippOut)
            fnGippIn = self._config.configAC
            fnGippOut = os.path.join(outpath, os.path.basename(fnGippIn))
            copyfile(fnGippIn, fnGippOut)
            fnLogOut = os.path.join(outpath, fnLogBase)
            self.logger.info('Processing for Landsat completed')
            f = open(fnLogIn, 'a')
            f.write('</Sen2Cor_Level-2A_Report_File>\n')
            f.flush()
            f.close()
            copyfile(fnLogIn, fnLogOut)
            # from L2A_Logger import L2A_Logger, getLevel
            # self.config.logger = L2A_Logger(self.config.spacecraftName, fnLog=fnLogOut \
            #                          , logLevel=self.config.log_level \
            #                          , operation_mode=self.config.operationMode)
        except:
            self._config.logger.error('cannot copy report file: %s' % fnLogIn)
            return False
        return True
