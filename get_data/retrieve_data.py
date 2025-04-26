import csv
import os
import requests
import pandas as pd
import json
import random


def parse_csv(path, save_headers=False, features_arr=None, composite_sum=False):
    print("path of csv file: " + path)

    root_dir = os.environ.get("ROOT_DIR")
    parsed_data_dir = os.path.join(root_dir, "parsed_data")
    header_path = os.path.join(parsed_data_dir, "feature_headers.json")
    
    if os.path.exists(header_path):
        with open(header_path, 'r') as header_file:
            headers_json = json.load(header_file)
            
            data = pd.read_csv(path, dtype=headers_json)
            return data, headers_json
    else:
        data = pd.read_csv(path)
        
    return data, None
        
        
    
    
    
    
    
            
            


def save_headers_json(path, save_path, features_arr=None):
    os.makedirs(save_path, exist_ok=True)
    data = pd.read_csv(path, low_memory=False)
    dtypes_dict = data.dtypes.apply(lambda dt: dt.name).to_dict()

    if features_arr:
        dtypes_dict = {col: dtypes_dict[col] for col in features_arr if col in dtypes_dict}

    with open(os.path.join(save_path, "feature_headers.json"), "w") as f:
        json.dump(dtypes_dict, f, indent=4)


