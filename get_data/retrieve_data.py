import os
import requests

# base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/{endpoint}"
base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/datasets/GSOM"
token = os.environ.get("NOAA_JACK")
headers = {"token": token}

def send_request(url=base_url, save_dir='data', filename=base_url):
    print("sending call: " + url)
    response = requests.get(url, headers=headers)
    print(response.text)


