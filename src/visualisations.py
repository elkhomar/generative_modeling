import numpy as np
import seaborn as sns
import pandas as pd
import torch
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import itertools


def compute_covariance_matrix(tensor):
    mean_tensor = torch.mean(tensor, dim=0)
    tensor_centered = tensor - mean_tensor
    cov_matrix = torch.matmul(tensor_centered.T, tensor_centered)/(tensor_centered.shape[0] - 1)
    return cov_matrix


def plot_cov_matrices(tensor1, tensor2):
    # Compute covariance matrices
    cov_matrix1 = compute_covariance_matrix(tensor1)
    cov_matrix2 = compute_covariance_matrix(tensor2)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.heatmap(cov_matrix1.numpy(), ax=axes[0], annot=True, cmap="YlGnBu")
    axes[0].set_title('Covariance Matrix of Tensor 1')

    sns.heatmap(cov_matrix2.numpy(), ax=axes[1], annot=True, cmap="YlGnBu")
    axes[1].set_title('Covariance Matrix of Tensor 2')

    plt.show()


def compare_pairplots(generated_distribution, original_distribution):
    """Generate pairplots for two tensors."""
    # Create a grid for the pairplot
    df1 = pd.DataFrame(generated_distribution.to("cpu").numpy())
    df2 = pd.DataFrame(original_distribution.to("cpu").numpy())
    num_features = generated_distribution.shape[1]
    fig, axes = plt.subplots(nrows=num_features, ncols=num_features,
                             figsize=(15, 15))

    # Iterate over all pairs of features
    for i, j in itertools.product(range(num_features), repeat=2):
        # Select the columns for the current pair
        cols = df1.columns[i], df1.columns[j]

        if i == j:
            # Diagonal plots (Histograms)
            sns.histplot(df1.iloc[:, i], ax=axes[i, j], kde=False,
                         color='blue', bins=30, stat='density')
            sns.histplot(df2.iloc[:, i], ax=axes[i, j], kde=False, color='red',
                         bins=30, stat='density')
        else:
            # Heatmap for the original_distribution
            sns.kdeplot(x=df2[cols[0]], y=df2[cols[1]], cmap="Reds",
                        shade=True, bw_adjust=0.5, alpha=0.6, ax=axes[i, j])
            # Scatter plot overlay for the generated_distribution
            sns.scatterplot(x=df1[cols[0]], y=df1[cols[1]], alpha=0.7,
                            edgecolor=None, color='blue', ax=axes[i, j])

        # Set labels
        axes[i, j].set_xlabel(cols[0] if i == num_features - 1 else '')
        axes[i, j].set_ylabel(cols[1] if j == 0 else '')
    return fig


def log_pairplots(generated_distribution, original_distribution, epoch, path):
    """Generates and saves the pairplots of two tensors."""
    fig = compare_pairplots(generated_distribution, original_distribution)
    fig.savefig(path + f"pairplot_{epoch}_epoch.png")


# tensor1 = torch.randn(100, 4)
# tensor2 = 4*torch.randn(1000, 4)+1
# compare_pairplots(tensor1, tensor2)
