import os
import numpy as np
import matplotlib.pyplot as plt
from skops.io import dump, load

from mpl_toolkits.mplot3d import Axes3D

import get_data.retrieve_data as get_data
from analysis import covariance_analysis
from ml_training import process_data
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import dotenv


def env_variable(key, f=".env", required=True):
    value = os.environ.get(key)
    if value is not None:
        return value
    
    if os.path.exists(f):
        dotenv.load_dotenv(f)
        value = os.environ.get(key)
        
        if value is not None:
            return value
        
    if required:
        raise RuntimeError(f"ERROR: {key} env variable not found")
    return None
            


# root_dir = os.environ.get("ROOT_DIR")
root_dir = env_variable("ROOT_DIR")

dataloc = os.path.join(root_dir, "gsom_data", "RDU.csv")
parsed_data_dir = os.path.join(root_dir, "parsed_data")
models_save_dir = os.path.join(root_dir, "PersistentModels")

if not os.path.isdir(parsed_data_dir):
    os.mkdir(parsed_data_dir)

if not os.path.isdir(models_save_dir):
    os.mkdir(models_save_dir)

if not os.path.isfile(dataloc):
    raise FileNotFoundError(f"file not found: {dataloc}")

ml_args = process_data.get_model_args()
print(f"Cmd line args:\n{ml_args}")

x_features = [
    "TAVG", # 314 non-null
    "AWND", # 313 non-null
    "DP01", # 318 non-null
    "DP10", # 318 non-null
    "DP1X",  # 318 non-null
    "EMXP"  # 318
]


y_features = ["PRCP"]

# headers_arr = []
headers_arr = [
    "TAVG",
    "AWND",
    "PRCP"
]

for header in y_features:
    headers_arr.append(header)

for header in x_features:
    headers_arr.append(header)
    

data, features_json = get_data.parse_csv(dataloc,
                                         features_arr=headers_arr,
                                         save_headers=True,
                                         )


# data, features_json = get_data.parse_csv(dataloc,
#                                          save_headers=False,
#                                          )


# print(data.info())

data = process_data.remove_empty_cells(data, dtypes=features_json)
data.to_csv(os.path.join(parsed_data_dir, "current_data.csv"), index=False)

# Extract full target data from y_1
target_df = process_data.extract_features(data, feature_conditions=y_features)
predictor_df = process_data.extract_features(data, feature_conditions=x_features)

combined = pd.concat([predictor_df, target_df], axis=1).dropna(how="any")
predictor_df = combined[x_features]
target_df = combined[y_features]

# print(f"target data:\n{target_df}")
# print(f"predictor data:{predictor_df}")

x_vectors = predictor_df.to_numpy()   # -> shape is (n_samples, n_features)
y_vectors = target_df.to_numpy().ravel()   # -> shape is (n_samples, )

x_train, x_test, y_train, y_test = train_test_split(
    x_vectors, y_vectors, test_size=.20, shuffle=True
)

x_scaled = preprocessing.StandardScaler().fit_transform(x_train)

model_name = ml_args.model_name
regression_model = process_data.train_lr_model(x_train, y_train, args=ml_args)

mse, rmse = process_data.test_lr_model(regression_model, x_test, y_test)

# process_data.plot_lr_results(regression_model, x_test, y_test, target_names=["PRCP"])
# process_data.plot_residuals(regression_model, x_test, y_test, target_names=["PRCP"])
# process_data.plot_learning_curve(regression_model, x_scaled, y_vectors)
print("model params", regression_model.get_params())

model_path = os.path.join(models_save_dir, model_name + str(1) + ".skops")
dump(regression_model, model_path)

print(f"Model: {model_name}, MSE: {mse}, RMSE: {rmse}\n")
