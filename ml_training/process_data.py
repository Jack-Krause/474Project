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
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import argparse


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
                    # print(f"features: {features_array}")
                else:
                    data.append(row)
                    # print(f"data row: {row}")

                i += 1

            features_array = np.array(features_array)
            data = np.array(data)

            if write_file and write_path:
                np.save(write_path + "/data.npy", data)
                np.save(write_path + "/feature_names.npy", features_array)

            return features_array, data

    else:
        raise FileNotFoundError("csv file not found")


def remove_empty_cells(data: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    data = data.replace("", np.nan)
    
    cols = list(dtypes.keys())
    data = data.dropna(subset=cols, how="any")

    for col, dtype in dtypes.items():
        if 'int' in dtype:
            data[col] = data[col].astype('Int64')
        elif 'float' in dtype:
            data[col] = data[col].astype(float)
        elif dtype == 'bool':
            data[col] = data[col].astype(bool)
        elif dtype == 'object':
            data[col] = data[col].astype(str)

    return data


def get_selected_features(data_path, features_arr, write_file=False, write_path=None):
    data = []
    # print(f"features chosen: {features_arr}")

    if os.path.isfile(data_path):
        # print(f"choosing columns: {data_path}")
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
    # print(f"training size: {len(training_arr)}, testing size: {len(testing_arr)}")

    return training_arr, testing_arr


def train_lr_model(x_train, y_train, model_name=None, pca=False):
    model = None

    if model_name is None:
        model = linear_model.LinearRegression()
    elif model_name.lower() == "linearregression":
        model = linear_model.LinearRegression()
    elif model_name.lower() == "supportvectorregression":
        model = make_pipeline(StandardScaler(), SVRWrapper(kernel='rbf', C=0.5, epsilon=0.1))
    elif model_name.lower() == "mlpregressor":
        # model_MLP = MLPRegressor(max_iter=1000, random_state=42)

        # param_grid = {
        #     'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        #     # 'hidden_layer_sizes': [(25,), (50,), (25, 25)],
        #     'activation': ['relu', 'tanh'],
        #     'alpha': [0.0001, 0.001, 0.01],  # L2 regularization strength
        #     'learning_rate_init': [0.001, 0.01]
        # }
        #
        # model = GridSearchCV(model_MLP, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        # model = MLPRegressor(
        #                      max_iter=2000,
        #                      random_state=42,
        #                      hidden_layer_sizes=(50,50),
        #                      activation='tanh',
        #                      alpha=0.0001,
        #                      learning_rate_init=0.001
        #                      )
        model = MLPRegressor(
            max_iter=2000,
            random_state=42,
            hidden_layer_sizes=(20, 20),
            early_stopping=True,
            activation='relu',
            alpha=0.0001,
            learning_rate_init=0.001
        )
        model.fit(x_train, y_train)

        plt.figure(figsize=(8, 6))
        plt.plot(model.loss_curve_, marker='o')
        plt.title("MLPRegressor Training Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()
        # print(f" NN params:\n{model.get_params(deep=True)}")
        return model

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


def get_model_args():
    parser = argparse.ArgumentParser(description='Args for model configuration:')
    parser.add_argument('-model_name', type=str, default='linearregression', help='regression model name')
    parser.add_argument('-hidden_layer_sizes')
    
    return parser.parse_args()


def plot_lr_results(model, x_test, y_test, target_names=None):
    """
    Plot actual vs. predicted values for each target variable.
    Shows:
      - Scatter plot of predictions vs. actual values.
      - Ideal line (y = x) and best-fit regression line.
      - Displays the R² score.
    """
    y_pred = model.predict(x_test)

    # Ensure 2D arrays for consistency
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_test.shape[1]
    if target_names is None or len(target_names) != n_targets:
        target_names = [f"Target {i + 1}" for i in range(n_targets)]

    for i in range(n_targets):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, label="Data points")

        # Determine limits for the plot
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

        # Best-fit line
        coeffs = np.polyfit(y_test[:, i], y_pred[:, i], 1)
        best_fit = np.poly1d(coeffs)
        line_x = np.linspace(min_val, max_val, 100)
        ax.plot(line_x, best_fit(line_x), 'g-', lw=2, label="Best Fit")

        # Calculate and display the R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        ax.text(0.05, 0.95, f"$R^2={r2:.2f}$", transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_xlabel(f"Actual {target_names[i]}")
        ax.set_ylabel(f"Predicted {target_names[i]}")
        ax.set_title(f"Actual vs. Predicted for {target_names[i]}")
        ax.legend()
        plt.show()


def plot_residuals(model, x_test, y_test, target_names=None):
    """
    Plot residuals (difference between actual and predicted values) for each target variable.
    """
    y_pred = model.predict(x_test)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    residuals = y_test - y_pred
    n_targets = y_test.shape[1]
    if target_names is None or len(target_names) != n_targets:
        target_names = [f"Target {i + 1}" for i in range(n_targets)]

    for i in range(n_targets):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test[:, i], residuals[:, i], alpha=0.6)
        ax.axhline(0, color='red', ls='--', lw=2)
        ax.set_xlabel(f"Actual {target_names[i]}")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        ax.set_title(f"Residual Plot for {target_names[i]}")
        plt.show()


def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Plot the learning curve indicating training and validation MSE.
    """
    train_sizes, train_scores, validation_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='neg_mean_squared_error'
    )
    train_scores_mean = -np.mean(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_scores_mean, 'o-', label="Training Error")
    ax.plot(train_sizes, validation_scores_mean, 'o-', label="Validation Error")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("MSE")
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
    plt.show()









