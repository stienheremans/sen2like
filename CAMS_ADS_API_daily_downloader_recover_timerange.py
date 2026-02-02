#!/usr/bin/env /opt/anaconda3/bin/python
#
# updated 03/12/2021 for daily data recovery (by dates)
# version 1.0 written on 27/05/2021 by Jerome LOUIS for Sen2like auxiliary data retrieval

# (C) Copyright 2021 Telespazio France

import cdsapi
import datetime
import time
import os
from datetime import date, timedelta

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def retrieve_CAMS_data(date, target_file):
    
    ### Initialize CDS data server
    
    server = cdsapi.Client()

    # Analysis

    # Name                                      Short Name

    # Total Aerosol Optical Depth at 550nm	    aod550
    # Total Column Water Vapour                 tcwv
    # Mean Sea Level Pressure                   msl
    # Total Column Ozone                        gtco3


    server.retrieve(
    'cams-global-atmospheric-composition-forecasts',
    {
        'variable': [
            'total_aerosol_optical_depth_550nm', 'total_column_water_vapour', 'mean_sea_level_pressure', 'total_column_ozone',            
        ],
        'date': date,
        'time': [
        '00:00', '06:00', '12:00', '18:00',
        ],
        'leadtime_hour': '0',
        'type': 'analysis',
        'format': 'netcdf',        
    },
    target_file)


def main():
    """
    Main function for CAMS ADS API download
    """

    dir_ADS_TEST = "/data/CAMS/daily/ADS/"
    dir_OPER = "/data/CAMS/daily/"
    
    dir_CAMS_daily = dir_ADS_TEST
    #dir_CAMS_daily = dir_OPER

    startdate = "2025-01-27"
    enddate = "2025-02-03"

    startdatetime = datetime.datetime.strptime(startdate, "%Y-%m-%d")
    enddatetime = datetime.datetime.strptime(enddate, "%Y-%m-%d")

    for single_date in daterange(startdatetime, enddatetime):
        startdate = single_date.strftime("%Y-%m-%d")
        
        enddate = startdate
        date = startdate + "/" + enddate
        dirdate = startdate.replace("-", "")
        target_dir = dir_CAMS_daily + dirdate + "/"
        target_file = target_dir + "CAMS_archive_aod550_tcwv_msl_gtco3_analysis_0H_6H_12H_18H_{0}.nc".format(startdate)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        print(date)
        print(target_dir)
        print(target_file)

        #let's do it
        retrieve_CAMS_data(date, target_file)

    return


if __name__ == "__main__":
    main()


