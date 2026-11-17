"""Unit tests for 1D and 2D gradient descent optimization routines."""

import numpy as np
import pytest

from gradient_descent import (
    f,
    f2,
    grad_f,
    grad_f2,
    gradient_descent_1d,
    gradient_descent_2d,
)


def test_f_and_grad_f():
    """Verify 1D test function value and analytical gradient at key points."""
    assert np.isclose(f(3.0), 0.0)
    assert np.isclose(grad_f(3.0), 0.0)
    assert np.isclose(f(5.0), 4.0)
    assert np.isclose(grad_f(5.0), 4.0)


def test_gradient_descent_1d_convergence():
    """Verify 1D descent converges cleanly to the analytical minimum (x=3)."""
    start = 10.0
    lr = 0.1
    steps = 50

    final_x, history = gradient_descent_1d(start=start, lr=lr, steps=steps)

    assert len(history) == steps + 1
    assert np.isclose(final_x, 3.0, atol=0.001)
    assert f(final_x) < f(start)


def test_gradient_descent_1d_oscillation():
    """Verify that a high learning rate (lr=0.9) induces oscillatory behavior."""
    start = 4.0
    lr = 0.9
    steps = 10

    _, history = gradient_descent_1d(start=start, lr=lr, steps=steps)

    # Count sign flips of (x - 3.0) across successive iterations
    diffs = [x - 3.0 for x in history]
    sign_flips = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
    assert sign_flips >= 8


def test_gradient_descent_1d_custom_grad():
    """Verify gradient_descent_1d supports custom gradient callbacks."""
    # Minimize g(x) = (x - 2)^2 with g'(x) = 2*(x - 2)
    final_x, _ = gradient_descent_1d(
        start=8.0,
        lr=0.1,
        steps=50,
        grad_fn=lambda x: 2.0 * (x - 2.0),
    )
    assert np.isclose(final_x, 2.0, atol=0.001)


def test_gradient_descent_1d_validation():
    """Verify input validation for 1D gradient descent."""
    with pytest.raises(ValueError, match="steps must be non-negative"):
        gradient_descent_1d(start=0.0, lr=0.1, steps=-1)

    with pytest.raises(ValueError, match="lr must be positive"):
        gradient_descent_1d(start=0.0, lr=-0.1, steps=10)


def test_f2_and_grad_f2():
    """Verify 2D test function values and analytical gradients."""
    assert np.isclose(f2(0.0, 0.0), 0.0)
    assert grad_f2(0.0, 0.0) == (0.0, 0.0)

    # At (2, 1): f2 = 2^2 + 5*1^2 = 9; grad = (4, 10)
    assert np.isclose(f2(2.0, 1.0), 9.0)
    assert grad_f2(2.0, 1.0) == (4.0, 10.0)


def test_gradient_descent_2d_convergence():
    """Verify 2D gradient descent trajectory and anisotropic convergence rates."""
    start = [4.0, 4.0]
    lr = 0.05
    steps = 10

    final_pt, path = gradient_descent_2d(start=start, lr=lr, steps=steps)

    assert path.shape == (steps + 1, 2)
    fx, fy = final_pt

    # Analytical geometric decay rates:
    # x_{t+1} = x_t - lr * (2 * x_t) = x_t * (1 - 0.1) = 0.9 * x_t
    # y_{t+1} = y_t - lr * (10 * y_t) = y_t * (1 - 0.5) = 0.5 * y_t
    assert np.isclose(fx, 4.0 * (0.9**steps), atol=0.0001)
    assert np.isclose(fy, 4.0 * (0.5**steps), atol=0.0001)

    # Steeper dimension y converges substantially faster than shallower dimension x
    assert abs(fy) < abs(fx)


def test_gradient_descent_2d_validation():
    """Verify input validation for 2D gradient descent."""
    with pytest.raises(ValueError, match="exactly 2 coordinates"):
        gradient_descent_2d(start=[1.0], lr=0.1, steps=10)

    with pytest.raises(ValueError, match="steps must be non-negative"):
        gradient_descent_2d(start=[1.0, 2.0], lr=0.1, steps=-1)

    with pytest.raises(ValueError, match="lr must be positive"):
        gradient_descent_2d(start=[1.0, 2.0], lr=0.0, steps=10)
