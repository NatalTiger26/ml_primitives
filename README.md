# Lab 2 — ML Primitives in NumPy

From-scratch NumPy implementations of three core machine learning building blocks:
a **k-NN classifier**, **gradient descent** (1D and 2D), and **PCA via SVD** — plus a
notebook that runs each one on synthetic data and visualizes the results.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Part A — k-Nearest Neighbors](#part-a--k-nearest-neighbors)
- [Part B — Gradient Descent](#part-b--gradient-descent)
- [Part C — PCA via SVD](#part-c--pca-via-svd)
- [Design Notes](#design-notes)
- [Running Just the Modules](#running-just-the-modules-no-notebook)

---

## Repository Structure

```
.
├── knn.py                        # k-NN distance functions, classifier, grid predictor
├── gradient_descent.py           # 1D and 2D gradient descent
├── pca.py                        # PCA via SVD
├── ml_primitives_starter.ipynb   # Runs everything above + generates all plots
├── requirements.txt              # Dependencies
├── plots/                        # Saved output figures (optional exports from the notebook)
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook ml_primitives_starter.ipynb
```

Run the notebook top to bottom. It imports every function below, generates its own
synthetic data, and produces all the plots described in this README.

---

## Part A — k-Nearest Neighbors

**File:** `knn.py`

| Function | Description |
|---|---|
| `euclidean_distance(a, b)` | Euclidean distance, broadcast-friendly. |
| `cosine_similarity(a, b)` | Cosine similarity between two vectors. |
| `distance_to_all(query, points)` | Distance from one query point to **every** training point in a single vectorized call — no Python loop over the training set. |
| `knn_predict(query, X_train, Y_train, k)` | Classifies a query point: finds the `k` nearest neighbors via `np.argsort`, then takes the majority label via `np.unique(..., return_counts=True)`. |
| `predict_grid(grid_points, X_train, Y_train, k)` | Runs `knn_predict` over a batch of points (used to render the decision boundary below). |

**What the notebook does:**
1. Generates two Gaussian-distributed classes (`class_a` centered at `(2, 2)`, `class_b` at
   `(-2, -2)`) and combines them into `X_train` / `y_train`.
2. Classifies a sample query point with `k=2`.
3. Builds a `200×200` meshgrid over the data's range, classifies every grid cell with
   `predict_grid`, reshapes the flat predictions back to the grid's shape, and shades the
   result with `contourf` — the classic k-NN decision boundary plot, with the training
   points overlaid on top.

**Vectorization note:** `distance_to_all` computes distances to the whole training set in
one NumPy call (`np.linalg.norm(points - query, axis=1)`), which is what keeps `knn_predict`
fast — no `for point in X_train` loop anywhere in the hot path.

---

## Part B — Gradient Descent

**File:** `gradient_descent.py`

**1D:** minimizes `f(x) = (x - 3)²` (derivative `f'(x) = 2(x - 3)`) via
`gradient_descent_1d(start, lr, steps)`, returning both the final `x` and the full history
of values visited.

**2D:** minimizes `f(x, y) = x² + 5y²` (gradient `(2x, 10y)`) via
`gradient_descent_2d(start, lr, steps)`, returning the final point and the full path as an
array — since the `x` and `y` coefficients differ, the descent path curves rather than
moving in a straight line to the minimum, which is a more realistic picture of gradient
descent outside of toy circular loss surfaces.

**What the notebook does:**
1. **Learning-rate comparison (1D):** runs `gradient_descent_1d` from `start = 4` for five
   learning rates — `0.001, 0.01, 0.03, 0.09, 0.9` — and plots the value of `x` over 100
   steps for each on one chart. The smaller rates converge cleanly; `lr = 0.9` overshoots
   the minimum on every step and visibly **oscillates**, demonstrating what an unstable
   learning rate looks like.
2. **2D path over a contour plot:** starts at `(4, 4)` with `lr = 0.05` for 10 steps, plots
   the loss surface `f2` as a contour map, and overlays the descent path as connected
   markers — you can see it curve toward the origin, moving faster along the shallower `x`
   direction than the steeper `y` direction.

---

## Part C — PCA via SVD

**File:** `pca.py`

`pca_via_svd(data, n_components)`:
1. **Centers the data** — subtracts the column-wise mean. This step is required; skipping
   it tilts the recovered direction away from the true axis of variance.
2. Runs `np.linalg.svd` on the centered data.
3. Takes the top `n_components` rows of `Vt` as the principal component(s).
4. Projects the centered data onto those components and returns `(projection, components)`.

**What the notebook does:**
1. Generates a noisy linear dataset: `y = 2x + noise` (200 points), so there's an obvious
   dominant direction of variance to recover.
2. Calls `pca_via_svd(data, n_components=1)`, then reconstructs the 1D projection back into
   2D coordinates (`projection @ components + mean`) so it can be plotted on the same axes
   as the original data.
3. Plots, together on one chart:
   - the original (uncentered, for visual reference) scattered points,
   - the recovered principal direction as a line running through the data,
   - the 1D projection of each point onto that line.

   The direction line visibly runs along the long axis of the scatter, confirming the SVD
   correctly recovered the dominant direction of variance.

---

## Design Notes

- All three modules are dependency-free aside from NumPy — no scikit-learn, so every
  algorithm (distance computation, k-NN voting, gradient step, SVD-based projection) is
  implemented explicitly rather than called from a library.
- `knn_predict` and `distance_to_all` avoid any loop over `X_train`; the only loop in the
  k-NN code is in `predict_grid`, which iterates over *grid* points (one `knn_predict` call
  per pixel of the decision boundary), not training points.
- Every plot in the notebook includes axis labels, a title, and a legend where more than one
  series is shown.

---

## Running Just the Modules (No Notebook)

Each file is a plain, importable module — you don't need the notebook to use the functions:

```python
import numpy as np
from knn import knn_predict
from gradient_descent import gradient_descent_1d
from pca import pca_via_svd

# k-NN
X_train = np.array([[0, 0], [1, 1], [5, 5]])
y_train = np.array([0, 0, 1])
label = knn_predict(np.array([0.5, 0.5]), X_train, y_train, k=2)

# Gradient descent
final_x, history = gradient_descent_1d(start=10, lr=0.1, steps=50)

# PCA
data = np.random.randn(100, 2) @ np.array([[3, 0], [0, 0.5]])
projection, components = pca_via_svd(data, n_components=1)
```