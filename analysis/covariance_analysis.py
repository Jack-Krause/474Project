from sklearn import covariance
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List


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

    cluster_dict = {}
    for feature, label in zip(corr.columns, cluster_labels):
        if label not in cluster_dict:
            cluster_dict[label] = []

        cluster_dict[label].append(feature)

    cluster_list = list(cluster_dict.values())
    cluster_list = [group for group in cluster_list if len(group) > 1]

    print(f"\nFeature Clusters (threshold={corr_threshold}):")
    for i, group in enumerate(cluster_list, start=1):
        print(f"Group {i}: {group}")

    return cluster_list



def cluster_pca(clusters: List[List[str]], data: pd.DataFrame, n_components: int = 1) -> None:
    transformed_data = pd.DataFrame(index=data.index)

    for cluster_num, features in enumerate(clusters, start=1):
        if len(features) == 1:
            transformed_data[f'Cluster_{cluster_num}'] = data[features[0]]

        else:
            cluster_data = data[features].dropna()
            pca = PCA(n_components=n_components)
            pc = pca.fit_transform(cluster_data)
            transformed_data[f'Cluster_{cluster_num}'] = pc[:, 0]

    return transformed_data
    
    
    










