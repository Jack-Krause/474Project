import os
import numpy as np
import matplotlib.pyplot as plt
import get_data.retrieve_data as get_data
from analysis import covariance_analysis
from ml_training import process_data
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd

root_dir = os.environ.get("ROOT_DIR")
dataloc = os.path.join(root_dir, "data", "data_new_b.csv")
parsed_data_dir = os.path.join(root_dir, "parsed_data")

if not os.path.isdir(parsed_data_dir):
    os.mkdir(parsed_data_dir)

if not os.path.isfile(dataloc):
    raise FileNotFoundError(f"file not found: {dataloc}")

x_features = [
    "AADT",
    "CONYR",
    "RESYR",
    "TRUCKS",
    "years_since_repair"
]


y_features = ["IRI", "CRACK_INDX", "PCI_2"]

headers_arr = []

for header in y_features:
    headers_arr.append(header)

for header in x_features:
    headers_arr.append(header)

data, features_json = get_data.parse_csv(dataloc,
                                         features_arr=headers_arr,
                                         save_headers=True,
                                         )
data = process_data.remove_empty_cells(data, dtypes=features_json)
data['years_since_repair'] = 2025 - np.maximum(data['CONYR'], data['RESYR'])
data.to_csv(os.path.join(parsed_data_dir, "current_data.csv"), index=False)
# data['composite_target'] = target_data.sum(axis=1)

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

model_name = "mlpregressor"
# Extract full target data from y_1
target_data = process_data.extract_features(data, feature_conditions=y_features)
# Use a subset of target columns for PCA
pca_headers = ["IRI", "CRACK_INDX", "PCI_2"]
target_pca_subset = target_data[pca_headers]
target_pca_scaled = preprocessing.StandardScaler().fit_transform(target_pca_subset)
pca = PCA(n_components=2)
y_pca = pca.fit_transform(target_pca_scaled)

print(f"Target PCA explained variance ratio: {pca.explained_variance_ratio_}")
predictor_data = process_data.extract_features(data, feature_conditions=x_features)
x_vectors = predictor_data.to_numpy()
x_scaled = preprocessing.StandardScaler().fit_transform(x_vectors)
y_vectors = y_pca

# Plot raw 2D PCA projection
plt.figure(figsize=(8, 6))
plt.scatter(y_pca[:, 0], y_pca[:, 1], alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection of Target Data")
plt.grid(True)
plt.show()

print(f"\n\nModel: (x features: model: {model_name})")

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y_vectors, test_size=.20
)

exit(0)
regression_model = process_data.train_lr_model(x_train, y_train, model_name=model_name)
mse, rmse = process_data.test_lr_model(regression_model, x_test, y_test)
process_data.plot_lr_results(regression_model, x_test, y_test, target_names=["PC1", "PC2"])
process_data.plot_residuals(regression_model, x_test, y_test, target_names=["PC1", "PC2"])
process_data.plot_learning_curve(regression_model, x_scaled, y_vectors)

print(f"Model: {model_name}, MSE: {mse}, RMSE: {rmse}\n")
