import os
import numpy as np
import get_data.retrieve_data as get_data
from ml_training import process_data
from sklearn import preprocessing

root_dir = os.environ.get("ROOT_DIR")
dataloc = os.path.join(root_dir, "data", "pave3.csv")
parsed_data_dir = os.path.join(root_dir, "parsed_data")

if not os.path.isdir(parsed_data_dir):
    os.mkdir(parsed_data_dir)

if not os.path.isfile(dataloc):
    raise FileNotFoundError(f"file not found: {dataloc}")

data, features_json = get_data.parse_csv(dataloc, save_headers=True)

data['years_since_repair'] = 2025 - np.maximum(data['CONYR'], data['RESYR'])
data = process_data.remove_empty_cells(data, features_json)

rows_with_nan = data[data.isna().any(axis=1)]
if rows_with_nan.empty:
    print("no empty rows in dataframe")
else:
    print(f"nan rows: \n{rows_with_nan}\n")

print(f"total data: {len(data)}")
print(f"total data arr: \n{data}\n")

# X
predictor_data = process_data.extract_features(
    data,
    feature_conditions=[
        "AADT",
        "CONYR",
        "RESYR",
        "years_since_repair"
    ]
)

# Y
target_data = process_data.extract_features(
    data,
    feature_conditions=[
        "CRACK_INDX",
        "FAULT_INDX",
        "IRI_INDX",
        "STRUCT_NEED80"
    ]
)

x_vectors = predictor_data.to_numpy()
y_vectors = target_data.to_numpy()
x_scaled = preprocessing.StandardScaler().fit_transform(x_vectors)


# process_data.plot_histogram(x_vectors[0])
for header in data:
    print(f"headers are -> {header[:10]}")
    process_data.plot_histogram(
            x_vector=data[header],
            label=header,
            save_path=os.path.join(parsed_data_dir, header + "data")
    )

print(f"target features: \n{target_data}\n")
print(f"x features: \n{predictor_data}\n")
print(f"x scaled: \n{x_scaled}\n")

x_train, x_test = process_data.separate_sets(x_scaled)
y_train, y_test = process_data.separate_sets(y_vectors)

linear_regression_model = process_data.train_lr_model(x_train=x_train, y_train=y_train)
error = process_data.test_lr_model(linear_regression_model, x_test=x_test, y_test=y_test)

print(f"calculate error: {error}")
