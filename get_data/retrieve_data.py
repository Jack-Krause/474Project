import os
import requests

# base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/{endpoint}"
base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/datasets/LCD"
token = os.environ.get("NOAA_JACK")
headers = {"token": token}

def download_dataset(url=base_url, save_dir='data', filename=base_url):
    print("downloading dataset from NOAA")
    response = requests.get(url, headers=headers)
    print(response.text)


