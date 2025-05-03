from sklearn import covariance
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_plot_covariance(data, title=None):
    """
    High correlations: search for coefs close to 1 or -1 -> high correlation (> 0.7, or < -0.7)
    -> good candidates for reduction with PCA or consolidation into composite score
    """
    # covariance_matrix = data.cov()
    correlation_matrix = data.corr()

    high_corrs = correlation_matrix.stack()
    mask = high_corrs.index.get_level_values(0) != high_corrs.index.get_level_values(1)
    high_corrs = high_corrs[mask]

    high_corrs = high_corrs[abs(high_corrs) > 0.75].reset_index()
    high_corrs.columns = ['Var1', 'Var2', 'Correlation']
    high_corrs = high_corrs.sort_values(by='Correlation', ascending=False)
    print(f"features with high correlation:\n{high_corrs}")
    # high_corrs.to_csv("correlations.csv", index=False)
    return correlation_matrix


def hierarchical_clustering(corr, n_clusters):





