import numpy as np


def pca_via_svd(data, n_components):
    # Center the data
    mean = data.mean(axis=0)
    centered_data = data - mean

    # PCA via SVD
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)

    components = Vt[:n_components]          # shape (1, 2)

    # Project onto first principal component
    projection = centered_data @ components.T    # (200, 1)

    return projection, components

