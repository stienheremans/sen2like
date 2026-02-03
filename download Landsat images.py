import requests
import json
import os
import re
import time
import tarfile 

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
USERNAME = "sheremans"
APP_TOKEN = "U4_aabAnSeySJjM!WiJuBr7Tu8LmAM!ry7@pTcgcKlBOM_4gKMMNoCjRSNGcmysq" 

BASE_DOWNLOAD_PATH = r"C:\Users\stien_heremans\Documents\GitHub\Data"
AOI_WKT = "POLYGON ((4.683223669876753 50.80588101162928, 4.683223669876753 50.79911870618196, 4.697704580373454 50.79911870618196, 4.697704580373454 50.80588101162928, 4.683223669876753 50.80588101162928))"
# took Meerdaalwoud to test
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
MAX_CLOUD_COVER = 30
DATASET_ALIAS = "landsat_ot_c2_l1"
API_BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def send_request(endpoint, data, apiKey=None):
    url = f"{API_BASE_URL.rstrip('/')}/{endpoint}"
    headers = {'Content-Type': 'application/json'}
    if apiKey:
        headers['X-Auth-Token'] = apiKey
    
    response = requests.post(url, json=data, headers=headers, timeout=60)
    
    try:
        resp_json = response.json()
    except Exception:
        raise Exception(f"Non-JSON Response from {endpoint}: {response.text}")

    if resp_json.get('errorCode'):
        raise Exception(f"API Error ({endpoint}): {resp_json['errorMessage']}")
    
    return resp_json.get('data')

def parse_wkt_to_geojson_coords(wkt):
    matches = re.findall(r"([\d\.]+) ([\d\.]+)", wkt)
    if not matches:
        raise Exception("Could not parse WKT.")
    return [[[float(lon), float(lat)] for lon, lat in matches]]

def download_stream(url, filepath):
    temp_filepath = filepath + ".tmp"
    
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(temp_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)
        
        print(f"      ⬇️  Downloaded: {filepath}")
        return True

    except Exception as e:
        print(f"      ❌ Download Failed: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return False

# --- NIEUWE FUNCTIE OM UIT TE PAKKEN ---
def extract_tar(filepath, extract_to):
    try:
        print(f"      📦 Extracting...")
        # 'r:*' zorgt dat hij automatisch gzip herkent
        with tarfile.open(filepath, 'r:*') as tar:
            tar.extractall(path=extract_to)
        print(f"      ✅ Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"      ❌ Extraction Failed: {e}")
        return False

# ==============================================================================
# 3. MAIN SCRIPT
# ==============================================================================
if __name__ == '__main__':
    try:
        print("--- STARTING LANDSAT DOWNLOAD & EXTRACT ---")

        # 1. Login
        print("1. Authenticating...")
        api_key = send_request("login-token", {'username': USERNAME, 'token': APP_TOKEN})
        print(f"   ✅ Login Success")

        # 2. Search
        print("2. Searching Scenes...")
        aoi_coords = parse_wkt_to_geojson_coords(AOI_WKT)
        
        search_payload = {
            'datasetName': DATASET_ALIAS,
            'maxResults': 20,
            'sceneFilter': {
                'acquisitionFilter': {'start': START_DATE, 'end': END_DATE},
                'cloudCoverFilter': {'max': MAX_CLOUD_COVER, 'min': 0, 'includeUnknown': False},
                'spatialFilter': {'filterType': 'geojson', 'geoJson': {'type': 'Polygon', 'coordinates': aoi_coords}}
            }
        }
        
        results = send_request("scene-search", search_payload, api_key)
        scenes = results.get('results', []) if results else []
        
        if not scenes:
            print("   ⚠️ No scenes found.")
            exit()
            
        print(f"   ✅ Found {len(scenes)} potential scenes.")

        # 3. Filter for L1TP
        valid_scenes = [s for s in scenes if "L1TP" in s['displayId']]
        print(f"   ✅ {len(valid_scenes)} scenes are L1TP.")

        # 4. Download Loop
        for scene in valid_scenes:
            display_id = scene['displayId']
            entity_id = scene['entityId']
            
            match = re.search(r'_(\d{3})(\d{3})_', display_id)
            path_str, row_str = match.groups() if match else ("000", "000")
            mission_folder = "Landsat9" if "LC09" in display_id else "Landsat8"
            
            # --- WIJZIGING: Maak een mapje PER SCENE ---
            # Dit houdt het overzichtelijk bij het uitpakken
            dest_folder = os.path.join(BASE_DOWNLOAD_PATH, mission_folder, path_str, row_str, display_id)
            
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            
            # Het bestand komt in die map te staan
            filename = f"{display_id}.tar.gz"
            filepath = os.path.join(dest_folder, filename)

            # Check of MTL.txt al bestaat (teken dat het al uitgepakt is)
            # Landsat bestanden hebben altijd een _MTL.txt bestand
            mtl_file = os.path.join(dest_folder, f"{display_id}_MTL.txt")
            if os.path.exists(mtl_file):
                print(f"   Skipping {display_id} (Already Extracted)")
                continue

            print(f"   Processing: {display_id}...")

            try:
                # GET DOWNLOAD OPTIONS
                opts = send_request("download-options", {'datasetName': DATASET_ALIAS, 'entityIds': [entity_id]}, api_key)
                
                product_id = None
                for o in opts:
                    if o['available'] and 'Bundle' in o['productName']: 
                        product_id = o['id']
                        break
                
                if not product_id:
                    possible_products = [o for o in opts if o['available'] and o['downloadSystem'] != 'folder']
                    if possible_products:
                        product_id = possible_products[0]['id']
                    else:
                        print("      ❌ No valid product bundle found.")
                        continue

                # REQUEST DOWNLOAD
                req_label = f"S2L_{entity_id}"
                dl_req = send_request("download-request", {
                    'downloads': [{'entityId': entity_id, 'productId': product_id}],
                    'label': req_label
                }, api_key)

                # HANDLE RESPONSE
                download_urls = []
                
                if dl_req.get('availableDownloads'):
                    download_urls.extend([d['url'] for d in dl_req['availableDownloads']])
                
                elif dl_req.get('preparingDownloads') or dl_req.get('newRecords'):
                    print("      ⚠️ File archived. Polling status...")
                    attempts = 0
                    while attempts < 20:
                        time.sleep(30)
                        attempts += 1
                        retrieve = send_request("download-retrieve", {'label': req_label}, api_key)
                        if retrieve.get('available'):
                            target_dl = next((d for d in retrieve['available'] if d['productId'] == product_id), None)
                            if target_dl:
                                download_urls.append(target_dl['url'])
                                break
                        if retrieve.get('queueSize', 0) == 0 and not retrieve.get('available') and attempts > 2:
                             break
                else:
                    print("      ⚠️ No immediate status. Checking existing queue...")
                    retrieve = send_request("download-retrieve", {'label': req_label}, api_key)
                    if retrieve.get('available'):
                         target_dl = next((d for d in retrieve['available'] if d['productId'] == product_id), None)
                         if target_dl:
                             download_urls.append(target_dl['url'])

                # EXECUTE DOWNLOAD & EXTRACT
                if download_urls:
                    for url in download_urls:
                        # 1. Downloaden
                        success = download_stream(url, filepath)
                        
                        # 2. Als download gelukt is -> Uitpakken
                        if success:
                            extract_success = extract_tar(filepath, dest_folder)
                            
                            # 3. (Optioneel) Verwijder de .tar.gz na uitpakken om ruimte te besparen
                            if extract_success:
                                try:
                                    os.remove(filepath)
                                    print("      🗑️  Removed original .tar.gz file")
                                except:
                                    pass
                else:
                    print("      ❌ Could not retrieve download URL.")

            except Exception as e:
                print(f"      ❌ Error processing scene: {e}")
                continue

        # 5. Logout
        send_request("logout", None, api_key)
        print("\n✅ All tasks finished.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")