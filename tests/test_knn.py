"""Unit tests for k-NN classifier and distance primitives."""

import numpy as np
import pytest

from knn import (
    cosine_similarity,
    distance_to_all,
    euclidean_distance,
    knn_predict,
    predict_grid,
)


def test_euclidean_distance_1d():
    """Verify Euclidean distance between 1D vectors returns a scalar float."""
    dist = euclidean_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert isinstance(dist, float)
    assert np.isclose(dist, 5.0)


def test_euclidean_distance_2d():
    """Verify Euclidean distance between 2D point arrays returns an ndarray."""
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[3.0, 4.0], [4.0, 5.0]])
    dists = euclidean_distance(a, b)
    assert isinstance(dists, np.ndarray)
    assert np.allclose(dists, np.array([5.0, 5.0]))


def test_cosine_similarity():
    """Verify cosine similarity for parallel, orthogonal, and opposite vectors."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([2.0, 0.0])
    v3 = np.array([0.0, 4.0])
    v4 = np.array([-1.0, 0.0])

    assert np.isclose(cosine_similarity(v1, v2), 1.0)
    assert np.isclose(cosine_similarity(v1, v3), 0.0)
    assert np.isclose(cosine_similarity(v1, v4), -1.0)

    # Numerical safeguard on zero vectors
    zero_vec = np.array([0.0, 0.0])
    assert np.isclose(cosine_similarity(zero_vec, v1), 0.0)


def test_distance_to_all():
    """Verify vectorized distance from one query point to all points in dataset."""
    query = np.array([1.0, 1.0])
    points = np.array([[1.0, 1.0], [4.0, 5.0], [1.0, 4.0]])
    dists = distance_to_all(query, points)
    assert dists.shape == (3,)
    assert np.allclose(dists, np.array([0.0, 5.0, 3.0]))


def test_knn_predict_majority():
    """Verify majority vote selection among nearest neighbors."""
    X_train = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [5.0, 5.0],
        [5.1, 5.0],
        [5.2, 5.0],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # Query close to cluster 0 with k=3
    pred_0 = knn_predict(np.array([0.05, 0.0]), X_train, y_train, k=3)
    assert pred_0 == 0

    # Query close to cluster 1 with k=2
    pred_1 = knn_predict(np.array([5.05, 5.0]), X_train, y_train, k=2)
    assert pred_1 == 1


def test_knn_predict_kwargs_and_validation():
    """Verify parameter flexibility (y_train / Y_train) and validation guards."""
    X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0, 1])

    # Uppercase Y_train keyword
    pred = knn_predict(np.array([0.1, 0.1]), X_train=X_train, Y_train=y_train, k=1)
    assert pred == 0

    # Missing labels
    with pytest.raises(ValueError, match="must be provided"):
        knn_predict(np.array([0.0, 0.0]), X_train=X_train)

    # Invalid k (< 1)
    with pytest.raises(ValueError, match="k must be at least 1"):
        knn_predict(np.array([0.0, 0.0]), X_train, y_train, k=0)

    # k exceeds sample count
    with pytest.raises(ValueError, match="cannot exceed number of training samples"):
        knn_predict(np.array([0.0, 0.0]), X_train, y_train, k=5)


def test_predict_grid():
    """Verify grid prediction helper runs across 2D coordinate matrices."""
    X_train = np.array([[-2.0, -2.0], [2.0, 2.0]])
    y_train = np.array([0, 1])
    grid = np.array([[-1.9, -1.9], [1.9, 1.9]])

    preds = predict_grid(grid, X_train, y_train, k=1)
    assert np.array_equal(preds, np.array([0, 1]))
