import os
import numpy as np
import matplotlib.pyplot as plt
import get_data.retrieve_data as get_data
from analysis import covariance_analysis
from ml_training import process_data
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd

root_dir = os.environ.get("ROOT_DIR")
dataloc = os.path.join(root_dir, "data", "data_new_b.csv")
parsed_data_dir = os.path.join(root_dir, "parsed_data")

if not os.path.isdir(parsed_data_dir):
    os.mkdir(parsed_data_dir)

if not os.path.isfile(dataloc):
    raise FileNotFoundError(f"file not found: {dataloc}")

x_1 = [
    "AADT",
    "CONYR",
    "RESYR",
    "TRUCKS",
    "years_since_repair"
]

y_1 = [
    "PCI_2",
    "RUT_INDX",
    "IRI_INDX",
    "FAULT_INDX",
    "CRACK_INDX",
    "IRI",
    "FRICT",
    "FAULTAV",
    "RUT",
    "CRACK_RATIO",
    "T_INDX",
    "L_INDX",
    "LW_INDX",
    "LLW_INDX",
    "A_INDX"
]

x_feature_sets = [x_1]
y_feature_sets = [y_1]

headers_arr = []
for f_set in x_feature_sets:
    for header in f_set:
        headers_arr.append(header)

for f_set in y_feature_sets:
    for header in f_set:
        headers_arr.append(header)


data, features_json = get_data.parse_csv(dataloc,
                                         features_arr=headers_arr,
                                         save_headers=True,
                                         composite_sum=True
                                         )

# data = process_data.remove_empty_cells(data, dtypes=features_json)
data['years_since_repair'] = 2025 - np.maximum(data['CONYR'], data['RESYR'])
data.to_csv(os.path.join(parsed_data_dir, "current_data.csv"), index=False)
print(data)
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


model_names = ["linear_regression", "supportvectorregression"]
# model_names = ["supportvectorregression"]
x_n, y_n = 0, 0
for model_name in model_names:
    for y_feature_set in y_feature_sets:
        x_n += 1
        y_n = 0

        # Y
        target_data = process_data.extract_features(
            data,
            feature_conditions=y_feature_set
        )

        pca_headers = [
            "A_INDX",
            "IRI_INDX",
            "FAULT_INDX",
            "LLW_INDX",
            "CRACK_INDX",
            "PCI_2",
            "LW_INDX",
            "L_INDX",
            "RUT_INDX"
        ]


        target_pca_subset = target_data[pca_headers]
        target_pca_scaled = preprocessing.StandardScaler().fit_transform(target_pca_subset)

        pca = PCA(n_components=2)
        y_pca = pca.fit_transform(target_pca_scaled)
        print(f"target (y) explained variance: {pca.explained_variance_}")
        print(f"target (y) explained variance ratio: {pca.explained_variance_ratio_}")

        # Plot the 2D projection
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pca[:, 0], y_pca[:, 1], alpha=0.7)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("2D PCA Projection of Target Data")
        plt.grid(True)
        plt.show()


        for x_feature_set in x_feature_sets:
            y_n += 1
            print(f"\n\nModel: (x:{x_n}, y:{y_n})")
            # X
            predictor_data = process_data.extract_features(
                data,
                feature_conditions=x_feature_set
            )


            # check correlation of features
            covariance_analysis.calculate_plot_covariance(predictor_data, title="correlation of predictor")
            covariance_analysis.calculate_plot_covariance(target_data, title="correlation of target")

            x_vectors = predictor_data.to_numpy()
            # y_vectors = target_data.to_numpy()
            y_vectors = y_pca

            # change this for (a) LR or (b) SVR
            x_scaled = x_vectors
            if model_name.lower() == 'linearregression':
                x_scaled = preprocessing.StandardScaler().fit_transform(x_vectors)  # LR

            # for header in data:
            #     print(f"headers are -> {header[:10]}")
            #     process_data.plot_histogram(
            #             x_vector=data[header],
            #             label=header,
            #             save_path=os.path.join(parsed_data_dir, header + "data")
            #     )

            print(f"target features: \n{y_vectors}\n")
            print(f"x features: \n{predictor_data}\n")
            print(f"x scaled: \n{x_scaled}\n")

            x_train, x_test = process_data.separate_sets(x_scaled)
            y_train, y_test = process_data.separate_sets(y_vectors)

            regression_model = process_data.train_lr_model(x_train=x_train, y_train=y_train, model_name=model_name)
            mse, rmse = process_data.test_lr_model(regression_model, x_test=x_test, y_test=y_test)

            process_data.plot_lr_results(regression_model, x_test, y_test, y_feature_set)

            print(f"model: {model_name}, mse: {mse}, rmse: {rmse}")
            print("\n\n")


