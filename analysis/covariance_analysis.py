from sklearn import covariance
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
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
    correlation_matrix = data.corr().abs()

    high_corrs = correlation_matrix.stack()
    mask = high_corrs.index.get_level_values(0) != high_corrs.index.get_level_values(1)
    high_corrs = high_corrs[mask]

    high_corrs = high_corrs[abs(high_corrs) > 0.75].reset_index()
    high_corrs.columns = ['Var1', 'Var2', 'Correlation']
    high_corrs = high_corrs.sort_values(by='Correlation', ascending=False)

    print(f"features with high correlation:\n{high_corrs}")
    return correlation_matrix


def hierarchical_clustering(corr):
    corr = corr.fillna(0).abs()

    dist_matrix = 1 - corr

    assert dist_matrix.shape[0] == dist_matrix.shape[1], "dist must be square"

    condensed_dist = squareform(dist_matrix.values, checks=False)
    linkage_matrix = linkage(condensed_dist, method='average')
    corr_threshold = 0.3   # correlation >= 0.7

    cluster_labels = fcluster(
            linkage_matrix, 
            t=corr_threshold, 
            criterion='distance'
            )

    

    feature_groups = dict(zip(corr.columns, cluster_labels))
    for cluster_id in sorted(set(feature_groups.values())):
        members = [feature for feature, cid in feature_groups.items() if cid == cluster_id]
        if len(members) > 1:
            print(f"Group {cluster_id}: {members}")
    # print(f"clustering groups:\n{feature_groups}")










