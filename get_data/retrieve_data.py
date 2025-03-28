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
    # station_ids = csv_to_str(os.getcwd() + r'\data\arizona_maricopa_stations.csv')
    station_ids = ["GHCND:USW00003184"]  # phoenix deer valley muni airport
    print(station_ids)

    params = [
        ("startdate", "2017-01-01"),
        ("enddate", "2021-01-01"),
        ("enddate", "2021-01-01")
    ]

    response = requests.get(url, headers=headers, params=params)
    print(response.text)


def csv_to_str(path, save=False):
    print("path of csv file: " + path)
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row[0])

    return data


def download_csv_data(url, save_path):
    response = requests.get(url)
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        f.write(response.content)











