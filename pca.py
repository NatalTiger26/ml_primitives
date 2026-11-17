"""Principal Component Analysis (PCA) via Singular Value Decomposition (SVD).

This module provides an exact, from-scratch implementation of PCA using SVD:
- Explicit data centering (zero-mean column centering)
- Singular Value Decomposition via `np.linalg.svd`
- Extraction of principal directions from the right singular vectors (Vt)
- Linear projection of centered observations onto the orthogonal component subspace
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union
import numpy as np


def pca_via_svd(
    data: Union[np.ndarray, Sequence[Sequence[float]]],
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Principal Component Analysis via Singular Value Decomposition.

    Procedure:
        1. Validates input shape and component count.
        2. Centers the data by subtracting the column-wise mean.
        3. Decomposes centered data matrix X via compact SVD:
           X = U * S * V^T
        4. Selects the first `n_components` rows of V^T as principal axes.
        5. Projects centered observations onto the chosen principal components:
           Projection = (X - mean) @ components.T

    Args:
        data: Input data matrix of shape (n_samples, n_features). Must be 2D.
        n_components: Number of principal components to extract. Must satisfy
            1 <= n_components <= n_features.

    Returns:
        A tuple of (projection, components):
            projection: NumPy array of shape (n_samples, n_components) representing
                the coordinates of data points in the reduced-dimensional space.
            components: NumPy array of shape (n_components, n_features) representing
                the orthonormal principal component row vectors.

    Raises:
        ValueError: If `data` is not 2-dimensional, is empty (0 samples), or
            if `n_components` is outside the valid range [1, n_features].

    Examples:
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> proj, comps = pca_via_svd(X, n_components=1)
        >>> proj.shape
        (3, 1)
        >>> comps.shape
        (1, 2)
    """
    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2:
        raise ValueError(
            f"data must be a 2D array of shape (n_samples, n_features), got ndim={arr.ndim}"
        )

    n_samples, n_features = arr.shape
    if n_samples == 0:
        raise ValueError("data cannot be empty (0 samples provided).")

    if n_components < 1 or n_components > n_features:
        raise ValueError(
            f"n_components must be between 1 and {n_features}, got {n_components}"
        )

    if n_components > n_samples:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed sample count ({n_samples})"
        )

    # 1. Center the data (crucial: omitting centering shifts principal axis)
    mean = np.mean(arr, axis=0)
    centered_data = arr - mean

    # 2. Compact SVD on centered data
    _, _, vt = np.linalg.svd(centered_data, full_matrices=False)

    # 3. Top n_components right singular vectors
    components = vt[:n_components]

    # 4. Project centered data onto selected principal components
    projection = centered_data @ components.T

    return projection, components
