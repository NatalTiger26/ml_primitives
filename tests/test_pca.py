"""Unit tests for Principal Component Analysis (PCA) via SVD."""

import numpy as np
import pytest

from pca import pca_via_svd


def test_pca_via_svd_shapes():
    """Verify output shapes of projection and principal components."""
    data = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0],
    ])
    n_components = 2
    proj, comps = pca_via_svd(data, n_components=n_components)

    assert proj.shape == (4, 2)
    assert comps.shape == (2, 3)


def test_pca_centering_invariance():
    """Verify that shifting the dataset mean does not change the principal direction."""
    rng = np.random.default_rng(42)
    base_data = rng.normal(size=(50, 3)) @ np.diag([5.0, 1.0, 0.2])

    _, comps_base = pca_via_svd(base_data, n_components=1)

    # Add large arbitrary offset
    shifted_data = base_data + np.array([100.0, -50.0, 25.0])
    _, comps_shifted = pca_via_svd(shifted_data, n_components=1)

    # Component directions should be collinear (|dot| == 1.0) regardless of sign
    dot_product = abs(float(np.dot(comps_base[0], comps_shifted[0])))
    assert np.isclose(dot_product, 1.0, atol=1e-5)


def test_pca_orthogonality():
    """Verify that extracted principal component vectors are mutually orthonormal."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 4))
    _, comps = pca_via_svd(data, n_components=3)

    # components @ components.T should equal identity matrix
    gram_matrix = comps @ comps.T
    assert np.allclose(gram_matrix, np.eye(3), atol=1e-5)


def test_pca_variance_capture():
    """Verify PCA identifies the dominant linear direction (y = 2x)."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    y = 2.0 * x + 0.1 * rng.normal(size=200)
    data = np.column_stack((x, y))

    proj, comps = pca_via_svd(data, n_components=1)

    # Primary eigenvector direction should align with line y = 2x (slope ~ 2.0)
    comp = comps[0]
    slope = abs(comp[1] / comp[0])
    assert np.isclose(slope, 2.0, atol=0.1)

    # Projection variance should capture > 99% of total variance
    proj_var = float(np.var(proj, axis=0)[0])
    total_var = float(np.sum(np.var(data, axis=0)))
    assert (proj_var / total_var) > 0.99


def test_pca_validation():
    """Verify error checking on inputs and component specifications."""
    # 1D array input
    with pytest.raises(ValueError, match="must be a 2D array"):
        pca_via_svd(np.array([1.0, 2.0, 3.0]), n_components=1)

    # Empty array
    with pytest.raises(ValueError, match="cannot be empty"):
        pca_via_svd(np.empty((0, 2)), n_components=1)

    # n_components outside [1, n_features]
    data_2d = np.ones((5, 2))
    with pytest.raises(ValueError, match="between 1 and 2"):
        pca_via_svd(data_2d, n_components=0)

    with pytest.raises(ValueError, match="between 1 and 2"):
        pca_via_svd(data_2d, n_components=3)
