import csv
import os
import requests
import pandas as pd
import json


def parse_csv(path, save_headers=False, chosen_features=None, save=False):
    print("path of csv file: " + path)

    root_dir = os.environ.get("ROOT_DIR")
    parsed_data_dir = os.path.join(root_dir, "parsed_data")
    if save_headers:
        save_headers_json(path, parsed_data_dir)

    with open(os.path.join(parsed_data_dir, "headers.json")) as headers_json:
        headers_obj = json.load(headers_json)
        data = pd.read_csv(path, dtype=headers_obj)
        header_row = data.columns.tolist()
        return data, header_row



def save_headers_json(path, save_path):
    os.makedirs(save_path, exist_ok=True)
    data = pd.read_csv(path, low_memory=False)
    dtypes_dict = data.dtypes.apply(lambda dt: dt.name).to_dict()

    with open(os.path.join(save_path, "headers.json"), "w") as f:
        json.dump(dtypes_dict, f, indent=4)


def download_csv_data(url, save_path):
    response = requests.get(url)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)
