import get_data.retrieve_data as get_data
import ml_training.process_data as process_data
import os
import numpy as np

root_dir = os.environ.get("ROOT_DIR")
dataloc = os.path.join(root_dir, "data", "Pavement.csv")
parsed_data_dir = os.path.join(root_dir, "parsed_data")

if not os.path.isdir(parsed_data_dir):
    os.mkdir(parsed_data_dir)

if not os.path.isfile(dataloc):
    raise FileNotFoundError(f"file not found: {dataloc}")

data, feature_names = get_data.parse_csv(dataloc, save_headers=True)
data['years_since_repair'] = 2025 - np.maximum(data['CONYR'], data['RESYR'])

# print(feature_names[:20])
# print(data[:20])
print(f"total data: {len(data)}")

target_features = process_data.extract_features(
    data,
    feature_conditions=[
        "CRACK_INDX",
        "FAULT_INDX",
        "IRI_INDX",
        "STRUCT_NEED80"
    ]
)

x_features = process_data.extract_features(
    data,
    feature_conditions=[
        "AADT",
        "TRUCKS",
        "CONYR",
        "RESYR"
        ""
    ]
)

print(f"target features: \n{target_features}\n")
print(f"x features: \n{x_features}")

