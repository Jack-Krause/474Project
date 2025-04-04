import get_data.retrieve_data as get_data
import ml_training.process_data as process_data
import os
import numpy as np
from sklearn import preprocessing

root_dir = os.environ.get("ROOT_DIR")
dataloc = os.path.join(root_dir, "data", "pave.csv")
parsed_data_dir = os.path.join(root_dir, "parsed_data")

if not os.path.isdir(parsed_data_dir):
    os.mkdir(parsed_data_dir)

if not os.path.isfile(dataloc):
    raise FileNotFoundError(f"file not found: {dataloc}")

data, feature_names = get_data.parse_csv(dataloc, save_headers=True)
data['years_since_repair'] = 2025 - np.maximum(data['CONYR'], data['RESYR'])
print("is null rows")
rows_with_nan = data[data.isna().any(axis=1)]
print(rows_with_nan)
# exit(-1)

print(f"total data: {len(data)}")
print(f"total data arr: \n{data}\n")

target_data = process_data.extract_features(
    data,
    feature_conditions=[
        "CRACK_INDX",
        "FAULT_INDX",
        "IRI_INDX",
        "STRUCT_NEED80"
    ]
)

predictor_data = process_data.extract_features(
    data,
    feature_conditions=[
        # "AADT",
        "CONYR",
        "RESYR",
        "years_since_repair"
    ]
)


x_vectors = predictor_data.to_numpy()
y_vectors = target_data.to_numpy()
x_scaled = preprocessing.StandardScaler().fit_transform(x_vectors)


print(f"target features: \n{target_data}\n")
print(f"x features: \n{predictor_data}\n")
print(f"x scaled: \n{x_scaled}\n")

x_train, x_test = process_data.separate_sets(x_scaled)
y_train, y_test = process_data.separate_sets(y_vectors)

linear_regression_model = process_data.train_lr_model(x_train=x_train, y_train=y_train)
acc = process_data.test_lr_model(linear_regression_model, x_test=x_test, y_test=y_test)

print(f"accuracy: {acc}%")
