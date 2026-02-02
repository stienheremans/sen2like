import os
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# 🟢 REPLACE THESE WITH YOUR NEW CLIENT DETAILS FROM CDSE DASHBOARD
CLIENT_ID = "sh-81b25677-5bc5-4146-a2a8-b7f71122fb3b"
CLIENT_SECRET = "JYXSJUnpELj6Llbb8cewSo2rC7f4gbDY"

# Destination Folder (Files will go into E:\Data\Sentinel2\31UFS\...)
BASE_DOWNLOAD_PATH = r"E:\2026_Sen2Like\data"

# Search Parameters
COLLECTION = "SENTINEL-2"
PRODUCT_TYPE = "S2MSI2A" # L2A Data
START_DATE = "2023-06-01"
END_DATE = "2023-06-10"
CLOUD_COVER = 30 # Max 30% clouds

# AOI polygon (Meerdaalwoud)
AOI = "POLYGON ((4.683223669876753 50.80588101162928, 4.683223669876753 50.79911870618196, 4.697704580373454 50.79911870618196, 4.697704580373454 50.80588101162928, 4.683223669876753 50.80588101162928))"


# ==============================================================================
# 2. AUTHENTICATION (OAUTH2 CLIENT CREDENTIALS)
# ==============================================================================
print("1. Authenticating with Client Credentials...")

try:
    # Create the OAuth Session
    client = BackendApplicationClient(client_id=CLIENT_ID)
    oauth = OAuth2Session(client=client)

    # Fetch Token
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        include_client_id=True
    )
    print("✅ Authentication Successful.")
    
except Exception as e:
    print(f"❌ Authentication Failed: {e}")
    print("Tip: Ensure your Client in CDSE Dashboard is set to 'Confidential' and allows 'Service Accounts'.")
    exit()


# ==============================================================================
# 3. SEARCH CATALOGUE
# ==============================================================================
print("2. Searching Catalogue...")

# OData Filter
odata_filter = (
    f"Collection/Name eq '{COLLECTION}' "
    f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{PRODUCT_TYPE}') "
    f"and OData.CSC.Intersects(area=geography'SRID=4326;{AOI}') "
    f"and ContentDate/Start gt {START_DATE}T00:00:00.000Z "
    f"and ContentDate/Start lt {END_DATE}T23:59:59.999Z "
    f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {CLOUD_COVER})"
)

search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
params = {
    "$filter": odata_filter,
    "$top": 20, # Limit download count
    "$orderby": "ContentDate/Start asc"
}

# Use the 'oauth' session directly - it handles the headers automatically!
response = oauth.get(search_url, params=params)
response.raise_for_status()

products = response.json().get('value', [])
print(f"✅ Found {len(products)} products.")

if not products:
    print("No products found. Exiting.")
    exit()


# ==============================================================================
# 4. DOWNLOAD & ORGANIZE
# ==============================================================================
print("3. Starting Downloads...")

for product in products:
    prod_id = product['Id']
    prod_name = product['Name'] # e.g., S2A_MSIL1C_2023..._T31UFS_...
    
    # Extract Tile ID for folder structure (e.g., T31UFS -> 31UFS)
    # Sentinel-2 names usually contain the tile like "_T31UFS_"
    try:
        parts = prod_name.split('_')
        tile_part = [p for p in parts if p.startswith('T') and len(p) == 6][0] # Find T31UFS
        tile_id = tile_part[1:] # Remove 'T' -> 31UFS
    except:
        tile_id = "Unknown_Tile"

    # Create Folder: E:\Data\Sentinel2\31UFS\
    dest_folder = os.path.join(BASE_DOWNLOAD_PATH, "Sentinel2", tile_id)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    zip_path = os.path.join(dest_folder, f"{prod_name}.zip")

    # Skip if exists
    if os.path.exists(zip_path):
        print(f"Skipping {prod_name} (Already exists)")
        continue

    print(f"Downloading: {prod_name} ...")
    
    # -----------------------------------------------------------
    # THE ZIPPER DOWNLOAD
    # -----------------------------------------------------------
    zipper_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value"
    
    try:
        # We use the oauth session to stream the download
        with oauth.get(zipper_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Saved to: {zip_path}")
        
    except Exception as e:
        print(f"❌ Failed to download {prod_name}: {e}")

print("\n✅ All tasks completed.")