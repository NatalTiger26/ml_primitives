"""k-Nearest Neighbors (k-NN) classifier and distance primitives in NumPy.

This module provides from-scratch implementations of fundamental metric and
classification primitives:
- Euclidean distance (1D scalars and broadcasted 2D arrays)
- Cosine similarity with numerical stability guards
- Fully vectorized distance computation from query points to training sets
- Majority-voting k-NN prediction supporting both single queries and batches
- Grid prediction helper for decision boundary visualization
"""

from __future__ import annotations

from typing import Any, Sequence, Union
import numpy as np


def euclidean_distance(
    a: Union[np.ndarray, Sequence[float], float],
    b: Union[np.ndarray, Sequence[float], float],
) -> Union[float, np.ndarray]:
    """Compute the Euclidean (L2) distance between vectors or array points.

    Supports both 1D coordinate vectors (returning a scalar float) and
    2D coordinate matrices (returning pairwise row-wise Euclidean distances).

    Args:
        a: First vector or array of shape (D,) or (N, D).
        b: Second vector or array broadcastable with `a`.

    Returns:
        Euclidean distance as a float for 1D inputs, or an ndarray of shape (N,)
        for 2D inputs.

    Raises:
        ValueError: If array dimensions are incompatible for broadcasting.

    Examples:
        >>> euclidean_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        5.0
        >>> euclidean_distance(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[3.0, 4.0], [4.0, 5.0]]))
        array([5., 5.])
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)

    if arr_a.ndim == 0 and arr_b.ndim == 0:
        return float(abs(arr_a - arr_b))

    dist = np.linalg.norm(arr_a - arr_b, axis=-1)
    if dist.ndim == 0:
        return float(dist)
    return dist


def cosine_similarity(
    a: Union[np.ndarray, Sequence[float]],
    b: Union[np.ndarray, Sequence[float]],
    eps: float = 1e-12,
) -> float:
    """Compute the cosine similarity between two 1D vectors.

    Calculates:
        cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Safely handles near-zero or zero-norm vectors by returning 0.0 without
    throwing division-by-zero or NaN warnings.

    Args:
        a: First vector of shape (D,).
        b: Second vector of shape (D,).
        eps: Small threshold below which norms are considered zero. Defaults to 1e-12.

    Returns:
        Cosine similarity scalar in the range [-1.0, 1.0].

    Raises:
        ValueError: If input vectors have differing lengths or are not 1D.

    Examples:
        >>> cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        0.0
        >>> cosine_similarity(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
        1.0
    """
    arr_a = np.asarray(a, dtype=float).ravel()
    arr_b = np.asarray(b, dtype=float).ravel()

    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"Vector shapes must match for cosine similarity, got {arr_a.shape} vs {arr_b.shape}"
        )

    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))
    denom = norm_a * norm_b

    if denom < eps:
        return 0.0

    sim = float(np.dot(arr_a, arr_b) / denom)
    return float(np.clip(sim, -1.0, 1.0))


def distance_to_all(
    query: Union[np.ndarray, Sequence[float]],
    points: Union[np.ndarray, Sequence[Sequence[float]]],
) -> np.ndarray:
    """Compute Euclidean distances from a single query point to all dataset points.

    Fully vectorized across training points with zero explicit Python loops.

    Args:
        query: 1D query point of shape (D,) or (1, D).
        points: 2D dataset of points of shape (N, D).

    Returns:
        1D array of distances of shape (N,) where entry i is the distance from
        the query point to points[i].

    Raises:
        ValueError: If points is not 2D or feature dimensions do not match.

    Examples:
        >>> pts = np.array([[1.0, 1.0], [4.0, 5.0], [1.0, 4.0]])
        >>> distance_to_all(np.array([1.0, 1.0]), pts)
        array([0., 5., 3.])
    """
    q = np.asarray(query, dtype=float)
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2:
        raise ValueError(f"points must be a 2D array of shape (N, D), got shape {pts.shape}")

    q_flat = q.ravel()
    if q_flat.shape[0] != pts.shape[1]:
        raise ValueError(
            f"Query feature dimension ({q_flat.shape[0]}) does not match points feature dimension ({pts.shape[1]})"
        )

    return np.linalg.norm(pts - q_flat, axis=1)


def knn_predict(
    query: Union[np.ndarray, Sequence[float]],
    X_train: Union[np.ndarray, Sequence[Sequence[float]]],
    y_train: Union[np.ndarray, Sequence[Any], None] = None,
    k: int = 5,
    *,
    Y_train: Union[np.ndarray, Sequence[Any], None] = None,
) -> Any:
    """Predict the class label of a query point using majority voting among k nearest neighbors.

    Accepts training labels as either `y_train` or `Y_train` (both positional as
    third argument and as keyword arguments) to guarantee compatibility across
    different naming conventions.

    Algorithm:
        1. Compute vectorized distances from query to all points in `X_train`.
        2. Identify the indices of the `k` smallest distances via `np.argsort`.
        3. Retrieve neighbor labels.
        4. Determine the majority label using `np.unique` voting.

    Args:
        query: Coordinate vector of shape (D,).
        X_train: Training data matrix of shape (N, D).
        y_train: Class labels array of length N (positional or keyword).
        k: Number of nearest neighbors to vote. Must be an integer >= 1.
        Y_train: Class labels array alias (keyword-compatible with uppercase convention).

    Returns:
        The predicted majority class label (matching the element type of y_train).

    Raises:
        ValueError: If training labels are missing, k < 1, k > N, or feature
            dimensions mismatch.

    Examples:
        >>> X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
        >>> y = np.array([0, 0, 1, 1])
        >>> knn_predict(np.array([0.05, 0.05]), X, y, k=2)
        0
    """
    labels_raw = y_train if y_train is not None else Y_train
    if labels_raw is None:
        raise ValueError("Training labels (y_train or Y_train) must be provided.")

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    X_mat = np.asarray(X_train, dtype=float)
    if X_mat.ndim != 2:
        raise ValueError(f"X_train must be a 2D array, got shape {X_mat.shape}")

    n_samples = X_mat.shape[0]
    if k > n_samples:
        raise ValueError(
            f"k={k} cannot exceed number of training samples ({n_samples})"
        )

    labels = np.asarray(labels_raw)
    if len(labels) != n_samples:
        raise ValueError(
            f"Length of labels ({len(labels)}) does not match X_train rows ({n_samples})"
        )

    distances = distance_to_all(query, X_mat)
    sorted_indices = np.argsort(distances)
    k_nearest_idx = sorted_indices[:k]
    k_nearest_labels = labels[k_nearest_idx]

    vals, counts = np.unique(k_nearest_labels, return_counts=True)
    majority_idx = int(np.argmax(counts))
    majority_label = vals[majority_idx]

    # Return native scalar if numpy scalar
    if isinstance(majority_label, np.generic):
        return majority_label.item()
    return majority_label


def predict_grid(
    grid_points: Union[np.ndarray, Sequence[Sequence[float]]],
    X_train: Union[np.ndarray, Sequence[Sequence[float]]],
    y_train: Union[np.ndarray, Sequence[Any], None] = None,
    k: int = 5,
    *,
    Y_train: Union[np.ndarray, Sequence[Any], None] = None,
) -> np.ndarray:
    """Classify an array of grid coordinates for decision boundary rendering.

    Evaluates k-NN predictions over all grid points. Supports training labels
    passed as either `y_train` or `Y_train`.

    Args:
        grid_points: Grid sample coordinates of shape (M, D).
        X_train: Training feature matrix of shape (N, D).
        y_train: Class labels of length N.
        k: Number of nearest neighbors. Defaults to 5.
        Y_train: Class labels alias for uppercase convention.

    Returns:
        1D numpy array of predicted labels of shape (M,).

    Raises:
        ValueError: If labels are not provided or parameters are invalid.

    Examples:
        >>> X = np.array([[-2.0, -2.0], [2.0, 2.0]])
        >>> y = np.array([0, 1])
        >>> grid = np.array([[-1.9, -1.9], [1.9, 1.9]])
        >>> predict_grid(grid, X, y, k=1)
        array([0, 1])
    """
    labels_raw = y_train if y_train is not None else Y_train
    if labels_raw is None:
        raise ValueError("Training labels (y_train or Y_train) must be provided.")

    pts = np.asarray(grid_points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"grid_points must be a 2D array, got shape {pts.shape}")

    X_mat = np.asarray(X_train, dtype=float)
    labels = np.asarray(labels_raw)

    predictions = [
        knn_predict(point, X_mat, labels, k=k)
        for point in pts
    ]
    return np.asarray(predictions)
