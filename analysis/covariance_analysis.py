from sklearn import covariance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_plot_covariance(data, title=None):
    """
    High correlations: search for coefs close to 1 or -1 -> high correlation (> 0.7, or < -0.7)
    -> good candidates for reduction with PCA or consolidation into composite score
    """
    # covariance_matrix = data.cov()
    correlation_matrix = data.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt_title = "Correlation matrix of features"
    if title:
        plt_title = title
    plt.title(plt_title)
    plt.show()
    print(f"{title}\n{correlation_matrix}\n\n")

    high_corrs = correlation_matrix.stack()
    mask = high_corrs.index.get_level_values(0) != high_corrs.index.get_level_values(1)
    high_corrs = high_corrs[mask]

    high_corrs = high_corrs[abs(high_corrs) > 0.75].reset_index()
    high_corrs.columns = ['Var1', 'Var2', 'Correlation']
    high_corrs = high_corrs.sort_values(by='Correlation', ascending=False)
    print(f"features with high correlation:\n{high_corrs}")






