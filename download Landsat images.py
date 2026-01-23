import requests
import json
import os
import re
import time

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
USERNAME = "sheremans"
APP_TOKEN = "U4_aabAnSeySJjM!WiJuBr7Tu8LmAM!ry7@pTcgcKlBOM_4gKMMNoCjRSNGcmysq" 

BASE_DOWNLOAD_PATH = r"E:\2026_Sen2Like\Data"
AOI_WKT = "POLYGON ((4.683223669876753 50.80588101162928, 4.683223669876753 50.79911870618196, 4.697704580373454 50.79911870618196, 4.697704580373454 50.80588101162928, 4.683223669876753 50.80588101162928))"

START_DATE = "2023-06-01"
END_DATE = "2023-06-10"
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
    
    response = requests.post(url, json=data, headers=headers)
    
    try:
        resp_json = response.json()
    except Exception:
        raise Exception(f"Non-JSON Response from {endpoint}: {response.text}")

    if resp_json.get('errorCode'):
        # Handle "Duplicate Request" gracefully if needed, otherwise raise
        raise Exception(f"API Error ({endpoint}): {resp_json['errorMessage']}")
    
    return resp_json.get('data')

def parse_wkt_to_geojson_coords(wkt):
    matches = re.findall(r"([\d\.]+) ([\d\.]+)", wkt)
    if not matches:
        raise Exception("Could not parse WKT.")
    return [[[float(lon), float(lat)] for lon, lat in matches]]

def download_stream(url, filepath):
    try:
        # Added timeout to prevent hanging
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"      ✅ Saved to: {filepath}")
        return True
    except Exception as e:
        print(f"      ❌ Download Failed: {e}")
        # Clean up partial file
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

# ==============================================================================
# 3. MAIN SCRIPT
# ==============================================================================
if __name__ == '__main__':
    try:
        print("--- STARTING LANDSAT DOWNLOAD ---")

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
            
            # Parse Path/Row
            match = re.search(r'_(\d{3})(\d{3})_', display_id)
            path_str, row_str = match.groups() if match else ("000", "000")
            mission_folder = "Landsat9" if "LC09" in display_id else "Landsat8"
            
            dest_folder = os.path.join(BASE_DOWNLOAD_PATH, mission_folder, path_str, row_str)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            
            filepath = os.path.join(dest_folder, f"{display_id}.tar.gz")
            if os.path.exists(filepath):
                print(f"   Skipping {display_id} (Exists)")
                continue

            print(f"   Processing: {display_id}...")

            try:
                # GET DOWNLOAD OPTIONS
                opts = send_request("download-options", {'datasetName': DATASET_ALIAS, 'entityIds': [entity_id]}, api_key)
                
                # FIX: Specifically look for the 'Bundle' to avoid getting Browse Images
                product_id = None
                for o in opts:
                    # 'available' means it can be ordered. 'productName' check ensures it is the data bundle.
                    if o['available'] and 'Bundle' in o['productName']: 
                        product_id = o['id']
                        break
                
                if not product_id:
                    # Fallback: take the largest file that isn't a folder
                    possible_products = [o for o in opts if o['available'] and o['downloadSystem'] != 'folder']
                    if possible_products:
                        product_id = possible_products[0]['id']
                    else:
                        print("      ❌ No valid product bundle found.")
                        continue

                # REQUEST DOWNLOAD
                req_label = f"S2L_{entity_id}" # Use EntityID in label to track specific files
                dl_req = send_request("download-request", {
                    'downloads': [{'entityId': entity_id, 'productId': product_id}],
                    'label': req_label
                }, api_key)

                # HANDLE RESPONSE
                download_urls = []
                
                # Check for immediate availability
                if dl_req.get('availableDownloads'):
                    download_urls.extend([d['url'] for d in dl_req['availableDownloads']])
                
                # Check if it needs processing
                elif dl_req.get('preparingDownloads') or dl_req.get('newRecords'):
                    print("      ⚠️ File archived. Polling status...")
                    
                    # Polling Loop with Timeout
                    attempts = 0
                    max_attempts = 20 # 10 minutes max
                    while attempts < max_attempts:
                        time.sleep(30)
                        attempts += 1
                        
                        retrieve = send_request("download-retrieve", {'label': req_label}, api_key)
                        
                        if retrieve.get('available'):
                            target_dl = next((d for d in retrieve['available'] if d['productId'] == product_id), None)
                            if target_dl:
                                download_urls.append(target_dl['url'])
                                break
                        
                        # Stop if queue is empty and nothing is available (Server gave up)
                        q_size = retrieve.get('queueSize', 0)
                        if q_size == 0 and not retrieve.get('available'):
                             # Double check: sometimes q_size is 0 but it's just finishing up.
                             if attempts > 2: 
                                 print("      ❌ Server stopped processing.")
                                 break
                else:
                    # Fallthrough case: API returned success but no immediate lists (e.g. duplicate request)
                    # We try to retrieve immediately to see if it was already ready
                    print("      ⚠️ No immediate status. Checking existing queue...")
                    retrieve = send_request("download-retrieve", {'label': req_label}, api_key)
                    if retrieve.get('available'):
                         target_dl = next((d for d in retrieve['available'] if d['productId'] == product_id), None)
                         if target_dl:
                             download_urls.append(target_dl['url'])

                # EXECUTE DOWNLOAD
                if download_urls:
                    for url in download_urls:
                        download_stream(url, filepath)
                else:
                    print("      ❌ Could not retrieve download URL.")

            except Exception as e:
                print(f"      ❌ Error processing scene: {e}")
                # Continue to next scene instead of crashing script
                continue

        # 5. Logout
        send_request("logout", None, api_key)
        print("\n✅ All downloads finished.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")