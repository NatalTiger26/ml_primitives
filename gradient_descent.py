"""Gradient descent optimization algorithms and loss functions in 1D and 2D.

This module provides from-scratch implementations of gradient descent routines:
- 1D scalar optimization with trajectory tracking and learning rate comparison
- 2D vector optimization over anisotropic loss surfaces
- Explicit analytical gradients for test quadratic objectives
- Configurable gradient callbacks and comprehensive parameter validation
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np


def f(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Evaluate 1D scalar test objective function f(x) = (x - 3)^2.

    Global minimum is located at x = 3.0 where f(3) = 0.

    Args:
        x: Scalar float or NumPy array of evaluation coordinates.

    Returns:
        Function value(s) evaluated at x.

    Examples:
        >>> f(3.0)
        0.0
        >>> f(5.0)
        4.0
    """
    return (x - 3.0) ** 2


def grad_f(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute analytical derivative of 1D objective f'(x) = 2 * (x - 3).

    Args:
        x: Scalar float or NumPy array of evaluation coordinates.

    Returns:
        Analytical derivative evaluated at x.

    Examples:
        >>> grad_f(3.0)
        0.0
        >>> grad_f(5.0)
        4.0
    """
    return 2.0 * (x - 3.0)


def gradient_descent_1d(
    start: float,
    lr: float,
    steps: int,
    grad_fn: Optional[Callable[[float], float]] = None,
) -> Tuple[float, List[float]]:
    """Perform 1D gradient descent optimization starting from an initial point.

    Repeatedly applies the update rule:
        x_{t+1} = x_t - lr * grad(x_t)

    Args:
        start: Starting point coordinate for the optimization.
        lr: Learning rate (step size multiplier). Must be strictly positive.
        steps: Total number of descent steps to perform. Must be non-negative.
        grad_fn: Optional gradient function taking a float coordinate and returning
            its derivative. If None, defaults to `grad_f`.

    Returns:
        A tuple of (final_x, history):
            final_x: Float representing the coordinate reached after `steps` iterations.
            history: List of float values of length `steps + 1` tracking each visited point.

    Raises:
        ValueError: If `steps` is negative or `lr` is non-positive.

    Examples:
        >>> final_x, history = gradient_descent_1d(start=10.0, lr=0.1, steps=50)
        >>> round(final_x, 3)
        3.0
        >>> len(history)
        51
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if lr <= 0:
        raise ValueError(f"Learning rate lr must be positive, got {lr}")

    gradient = grad_fn if grad_fn is not None else grad_f

    current = float(start)
    history: List[float] = [current]

    for _ in range(steps):
        g = float(gradient(current))
        current -= float(lr) * g
        history.append(current)

    return history[-1], history


def f2(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Evaluate 2D anisotropic quadratic objective f2(x, y) = x^2 + 5 * y^2.

    Global minimum is located at (0.0, 0.0) where f2(0, 0) = 0.
    The curvature along the y-axis is 5x steeper than along the x-axis,
    inducing characteristic curved trajectories for first-order gradient descent.

    Args:
        x: Scalar coordinate or array grid along the x-axis.
        y: Scalar coordinate or array grid along the y-axis.

    Returns:
        Function value(s) evaluated across coordinates.

    Examples:
        >>> f2(0.0, 0.0)
        0.0
        >>> f2(2.0, 1.0)
        9.0
    """
    return x**2 + 5.0 * (y**2)


def grad_f2(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute analytical gradient vector of 2D objective grad f2 = (2x, 10y).

    Args:
        x: Scalar coordinate or array grid along the x-axis.
        y: Scalar coordinate or array grid along the y-axis.

    Returns:
        Tuple of (grad_x, grad_y) representing partial derivatives.

    Examples:
        >>> grad_f2(0.0, 0.0)
        (0.0, 0.0)
        >>> grad_f2(2.0, 1.0)
        (4.0, 10.0)
    """
    return 2.0 * x, 10.0 * y


def gradient_descent_2d(
    start: Union[Sequence[float], np.ndarray],
    lr: float,
    steps: int,
    grad_fn: Optional[Callable[[float, float], Tuple[float, float]]] = None,
) -> Tuple[Tuple[float, float], np.ndarray]:
    """Perform 2D gradient descent optimization on an anisotropic surface.

    Iteratively updates 2D coordinates:
        (x_{t+1}, y_{t+1}) = (x_t, y_t) - lr * grad(x_t, y_t)

    Args:
        start: Initial 2D coordinates [x0, y0]. Must have length 2.
        lr: Learning rate. Must be strictly positive.
        steps: Number of optimization steps. Must be non-negative.
        grad_fn: Optional function returning (df/dx, df/dy). If None, defaults to `grad_f2`.

    Returns:
        A tuple of (final_point, path):
            final_point: Tuple (x, y) of coordinates at the final step.
            path: NumPy array of shape (steps + 1, 2) containing all visited points.

    Raises:
        ValueError: If start does not contain exactly 2 coordinates, steps is
            negative, or lr is non-positive.

    Examples:
        >>> final_pt, path = gradient_descent_2d(start=[4.0, 4.0], lr=0.05, steps=10)
        >>> path.shape
        (11, 2)
    """
    start_seq = list(start)
    if len(start_seq) != 2:
        raise ValueError(
            f"start must contain exactly 2 coordinates, got {len(start_seq)}"
        )
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if lr <= 0:
        raise ValueError(f"Learning rate lr must be positive, got {lr}")

    gradient = grad_fn if grad_fn is not None else grad_f2

    curr_x = float(start_seq[0])
    curr_y = float(start_seq[1])
    path: List[Tuple[float, float]] = [(curr_x, curr_y)]

    for _ in range(steps):
        gx, gy = gradient(curr_x, curr_y)
        curr_x -= float(lr) * float(gx)
        curr_y -= float(lr) * float(gy)
        path.append((curr_x, curr_y))

    final_point: Tuple[float, float] = (path[-1][0], path[-1][1])
    return final_point, np.asarray(path, dtype=float)
