import csv
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


def get_dataset(url=base_url, save_dir='data', filename=base_url):
    print("get dataset: " + url)
    stations = csv_to_str(os.getcwd() + r'\data\arizona_maricopa_stations.csv')
    print("stations:\n", stations)

    params = {
        "startdate": "2017-01-01",
        "enddate": "2021-01-01",
        "stationid": "GHCND:USC00020060"
    }

    response = requests.get(url, headers=headers, params=params)
    print(response.text)


def csv_to_str(path, save=False):
    print("path of csv file: " + path)
    data = ""
    with open(path, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i > 0:
                data += row[0]
                data += ' & '
                break

    return data

