import os
import numpy as np
import get_data.retrieve_data as get_data
from ml_training import process_data
from sklearn import preprocessing
import pandas as pd

root_dir = os.environ.get("ROOT_DIR")
dataloc = os.path.join(root_dir, "data", "data_new_b.csv")
parsed_data_dir = os.path.join(root_dir, "parsed_data")
analysis_path = os.path.join(root_dir, "analysis")

if not os.path.isdir(parsed_data_dir):
    os.mkdir(parsed_data_dir)

if not os.path.isfile(dataloc):
    raise FileNotFoundError(f"file not found: {dataloc}")

if not os.path.isdir(analysis_path):
    os.mkdir(analysis_path)


columns_data = [
    "AADT", "CONYR", "RESYR", "CRACK_INDX", "FAULT_INDX", "IRI_INDX", "STRUC80", "TRUCKS"
]

data, features_json = get_data.parse_csv(dataloc,
                                         save_headers=True,
                                         features_arr=columns_data
                                         )

data = process_data.remove_empty_cells(data, dtypes=features_json)
data['years_since_repair'] = 2025 - np.maximum(data['CONYR'], data['RESYR'])
data.to_csv(os.path.join(parsed_data_dir, "current_data.csv"), index=False)

missing_headers = []
for header in data.columns:
    zero_count = 0
    empty_count = 0
    for cell in data[header]:

        if pd.isna(cell) or cell == "" or cell is None:
            empty_count += 1
        if cell == 0 or cell == 0.0:
            zero_count += 1

    missing_headers.append([header, empty_count, zero_count])

missing_headers.sort(key=lambda pt: pt[1] + pt[2], reverse=False)

header_format = "{:>3}  {:<20}  {:>7}  {:>8}  {:>7}"
print(header_format.format("No.", "Column", "Total", "Missing", "Zeros"))
print("-" * 55)

for i, (header, missing, zero) in enumerate(missing_headers, start=1):
    total = missing + zero
    print(header_format.format(i, header, total, missing, zero))


x_1 = [
    "AADT",
    "CONYR",
    "RESYR",
    "years_since_repair"
]

x_2 = [
    "AADT",
    "CONYR",
    "RESYR",
    "years_since_repair",
    "TRUCKS"
]

y_1 = [
    "CRACK_INDX",
    "FAULT_INDX",  # lots missing
    "IRI_INDX",
    "STRUC80"
]


x_feature_sets = [x_1, x_2]
y_feature_sets = [y_1]

x_n, y_n = 0, 0
for x_feature_set in x_feature_sets:
    x_n += 1
    y_n = 0

    for y_feature_set in y_feature_sets:
        y_n += 1
        print(f"\n\nModel: (x:{x_n}, y:{y_n})")
        # X
        predictor_data = process_data.extract_features(
            data,
            feature_conditions=x_feature_set
        )

        # Y
        target_data = process_data.extract_features(
            data,
            feature_conditions=y_feature_set
        )

        x_vectors = predictor_data.to_numpy()
        y_vectors = target_data.to_numpy()
        x_scaled = preprocessing.StandardScaler().fit_transform(x_vectors)

        # for header in data:
        #     print(f"headers are -> {header[:10]}")
        #     process_data.plot_histogram(
        #             x_vector=data[header],
        #             label=header,
        #             save_path=os.path.join(parsed_data_dir, header + "data")
        #     )

        print(f"target features: \n{target_data}\n")
        print(f"x features: \n{predictor_data}\n")
        print(f"x scaled: \n{x_scaled}\n")

        x_train, x_test = process_data.separate_sets(x_scaled)
        y_train, y_test = process_data.separate_sets(y_vectors)

        linear_regression_model = process_data.train_lr_model(x_train=x_train, y_train=y_train)
        error = process_data.test_lr_model(linear_regression_model, x_test=x_test, y_test=y_test)

        print(f"calculate error: {error}")
        print("\n\n")



