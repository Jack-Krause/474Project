import os
import csv
import random
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class SVRWrapper(svm.SVR):
    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            return super().fit(X, y.ravel(), sample_weight=sample_weight)
        return super().fit(X, y.ravel())


def extract_features(data, feature_conditions):
    # return pd.Series(True, index=data.index)
    return data[feature_conditions]


def compute_vector(data, feature_conditions=None):
    if feature_conditions:
        mask = pd.Series(True, index=data.index)

        for feature_cond in feature_conditions:
            mask &= data.eval(feature_cond)

        return mask.astype(int)

    return None


def load_data(data_path, write_file=False, write_path=None):
    features_array = []
    data = []
    if os.path.isfile(data_path):
        with open(data_path, 'r') as f:
            reader = csv.reader(f)

            i = 0
            for row in reader:
                if i == 0:
                    features_array = row
                    print(f"features: {features_array}")
                else:
                    data.append(row)
                    print(f"data row: {row}")

                i += 1

            features_array = np.array(features_array)
            data = np.array(data)

            if write_file and write_path:
                np.save(write_path + "/data.npy", data)
                np.save(write_path + "/feature_names.npy", features_array)

            return features_array, data

    else:
        raise FileNotFoundError("csv file not found")


def remove_empty_cells(data, dtypes):
    for col, dtype in dtypes.items():
        if 'int' in dtype:
            data[col] = data[col].fillna(0).astype('Int64')
        elif 'float' in dtype:
            data[col] = data[col].fillna(0.0)
        elif dtype == 'bool':
            data[col] = data[col].fillna(False).astype('bool')
        elif dtype == 'object':
            data[col] = data[col].replace('', np.nan).fillna('N/A')

    return data


def get_selected_features(data_path, features_arr, write_file=False, write_path=None):
    data = []
    print(f"features chosen: {features_arr}")

    if os.path.isfile(data_path):
        print(f"choosing columns: {data_path}")
        df = pd.read_csv(data_path, usecols=features_arr)
        data = df.to_numpy()

        if write_file and write_path:
            np.save(write_path + "/selected_data.npy", data)

    return data


def separate_sets(data_arr, seed=42):
    random.seed(42)
    random.shuffle(data_arr)

    split_idx = int(len(data_arr) * 0.80)
    training_arr = data_arr[ :split_idx]
    testing_arr = data_arr[split_idx: ]
    print(f"training size: {len(training_arr)}, testing size: {len(testing_arr)}")

    return training_arr, testing_arr


def train_lr_model(x_train, y_train, model_name=None, pca=False):
    model = None

    if model_name is None:
        model = linear_model.LinearRegression()
    elif model_name.lower() == "linearregression":
        model = linear_model.LinearRegression()
    elif model_name.lower() == "supportvectorregression":
        model = MultiOutputRegressor(
            make_pipeline(StandardScaler(), SVRWrapper(kernel='rbf', C=0.5, epsilon=0.1))
        )
        # model = make_pipeline(StandardScaler(), SVRWrapper(kernel='rbf', C=0.5, epsilon=0.1))
    else:
        model = MultiOutputRegressor(linear_model.LinearRegression())
        print("Warning: model name default not found")

    if model is None:
        raise RuntimeError("Error creating model")

    model.fit(x_train, y_train)
    return model


def test_lr_model(model, x_test, y_test):
    y_predictions = model.predict(x_test)
    mse_error = metrics.mean_squared_error(y_test, y_predictions)
    rmse_error = metrics.root_mean_squared_error(y_test, y_predictions)

    return round(mse_error, 3), round(rmse_error, 3)


def aggregate_target(y_matrix):
    pass


def plot_lr_results(model, x_test, y_test, target_names=None):
    """
    Improved plot for actual vs. predicted values.
    It shows:
      - A scatter plot of predictions vs. actual values.
      - An ideal (perfect prediction) line (y = x).
      - A best-fit regression line to summarize the trend.
      - The R² score as a performance metric.
    """
    # Get predictions
    y_pred = model.predict(x_test)

    # Ensure y_test and y_pred are 2D arrays
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_test.shape[1]
    if target_names is None:
        target_names = [f"Target {i + 1}" for i in range(n_targets)]

    for i in range(n_targets):
        plt.figure(figsize=(8, 6))
        # Scatter plot of actual vs. predicted values
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, label="Data points")

        # Determine limits for the plot based on the data range
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())

        # Plot ideal (perfect prediction) line (y = x)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

        # Compute and plot a best-fit regression line
        coeffs = np.polyfit(y_test[:, i], y_pred[:, i], 1)
        best_fit = np.poly1d(coeffs)
        line_x = np.linspace(min_val, max_val, 100)
        plt.plot(line_x, best_fit(line_x), 'g-', lw=2, label="Best Fit")

        # Optionally, compute the R² score for additional clarity
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        plt.text(0.05, 0.95, f"$R^2={r2:.2f}$", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.xlabel("Actual " + target_names[i])
        plt.ylabel("Predicted " + target_names[i])
        plt.title("Actual vs. Predicted " + target_names[i])
        plt.legend()
        plt.show()


def plot_residuals(model, x_test, y_test, target_names=None):
    """
    Plots residuals (Actual - Predicted) vs. Actual values for each target variable.
    """
    y_pred = model.predict(x_test)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    residuals = y_test - y_pred
    n_targets = y_test.shape[1]
    if target_names is None:
        target_names = [f"Target {i + 1}" for i in range(n_targets)]

    for i in range(n_targets):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test[:, i], residuals[:, i], alpha=0.6)
        plt.axhline(0, color='red', ls='--', lw=2)
        plt.xlabel("Actual " + target_names[i])
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title("Residual Plot for " + target_names[i])
        plt.show()

