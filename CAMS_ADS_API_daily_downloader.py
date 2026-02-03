#!/usr/bin/env python
#
# version 1.0 written on 27/05/2021 by Jerome LOUIS for Sen2like auxiliary data retrieval

# (C) Copyright 2021 Telespazio France

import cdsapi
from argparse import ArgumentParser
import time
import calendar
import os

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
    Main function for CAMS ADS API download - Gecorrigeerde Expert Versie
    """

    # Gebruik jouw specifieke Windows paden
    dir_OPER = "E:/2026_Sen2Like/Python/sen2cor3/SEN2COR_3/aux_data/ECMWF/daily/"
    dir_CAMS_daily = dir_OPER    

    # --- DATUM AANPASSING ---
    # Hier vullen we de specifieke Landsat datum in
    target_date_str = "2023-06-06"
    # Omzetten naar een structuur die de expert-logica begrijpt
    target_time_obj = time.strptime(target_date_str, "%Y-%m-%d")
    
    # Exacte expert-logica voor variabelen
    startdate = time.strftime("%Y-%m-%d", target_time_obj)
    enddate = startdate
    date = startdate + "/" + enddate
    
    # Foldernaam: YYYYMMDD (bijv. 20230606)
    target_dir = dir_CAMS_daily + time.strftime("%Y%m%d", target_time_obj) + "/"
    
    # --- DE CORRECTE EXPERT NAAMGEVING ---
    # Dit is de string waar Sen2Cor specifiek op scant voor Landsat
    target_file = target_dir + "CAMS_archive_aod550_tcwv_msl_gtco3_analysis_0H_6H_12H_18H_{0}.nc".format(startdate)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Requesting date: {date}")
    print(f"Target directory: {target_dir}")
    print(f"Target file: {target_file}")

    # Start de download met de originele argumenten
    retrieve_CAMS_data(date, target_file)

    return

if __name__ == "__main__":
    main()



