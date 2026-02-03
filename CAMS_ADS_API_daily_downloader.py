#!/usr/bin/env python
# version 1.3 - Recursive Batch retrieval for L8 & L9 by Gemini

import cdsapi
import time
import os

def retrieve_CAMS_data(date, target_file):
    ### Initialize CDS data server
    server = cdsapi.Client()

    server.retrieve(
    'cams-global-atmospheric-composition-forecasts',
    {
        'variable': [
            'total_aerosol_optical_depth_550nm', 'total_column_water_vapour', 
            'mean_sea_level_pressure', 'total_column_ozone',            
        ],
        'date': date,
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'leadtime_hour': '0',
        'type': 'analysis',
        'format': 'netcdf',        
    },
    target_file)

def main():
    """
    Main function for CAMS ADS API download - Recursive scan for L8 & L9
    """

    # 1. CONFIGURATIE
    dir_OPER = "C:/Users/stien_heremans/Documents/GitHub/sen2like/sen2cor3/SEN2COR_3/aux_data/ECMWF/daily/"
    # We scannen de hoofdmap waar beide sensoren onder vallen
    landsat_base_folders = [
        "C:/Users/stien_heremans/Documents/GitHub/Data/Landsat8/",
        "C:/Users/stien_heremans/Documents/GitHub/Data/Landsat9/"
    ]
    
    unique_dates = set()

    # 2. RECURSIEVE SCAN
    print("Bezig met scannen van Landsat mappen...")
    for root_folder in landsat_base_folders:
        if not os.path.exists(root_folder):
            print(f"Overslaan: {root_folder} niet gevonden.")
            continue

        for root, dirs, files in os.walk(root_folder):
            for directory in dirs:
                # We zoeken naar folders die beginnen met LC08 of LC09
                if directory.startswith(("LC08_", "LC09_")):
                    parts = directory.split('_')
                    if len(parts) > 3:
                        date_str = parts[3] # Bijv. "20230606"
                        if len(date_str) == 8 and date_str.isdigit():
                            unique_dates.add(date_str)

    if not unique_dates:
        print("Geen Landsat producten gevonden. Controleer de paden.")
        return

    print(f"Gevonden unieke datums: {sorted(list(unique_dates))}")

    # 3. BATCH DOWNLOAD LOOP
    for d_str in sorted(list(unique_dates)):
        target_date_str = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
        target_time_obj = time.strptime(target_date_str, "%Y-%m-%d")
        
        startdate = time.strftime("%Y-%m-%d", target_time_obj)
        date = startdate + "/" + startdate
        target_dir = dir_OPER + d_str + "/"
        target_file = target_dir + "CAMS_archive_aod550_tcwv_msl_gtco3_analysis_0H_6H_12H_18H_{0}.nc".format(startdate)

        if os.path.exists(target_file):
            print(f"[{startdate}] Bestand bestaat al. Overslaan.")
            continue

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        print(f"\n--- Start aanvraag voor {startdate} ---")
        try:
            retrieve_CAMS_data(date, target_file)
            print(f"Succesvol binnengehaald: {target_file}")
        except Exception as e:
            print(f"Fout bij downloaden van {startdate}: {e}")

    print("\nAlle downloads voltooid.")

if __name__ == "__main__":
    main()