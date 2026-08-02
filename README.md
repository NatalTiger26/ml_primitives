# Lab 2 — ML Primitives in NumPy

Implementations of k-NN, gradient descent, and PCA (via SVD) built from scratch in NumPy.

## Files

- **`knn.py`** — `euclidean_distance`, `cosine_similarity`, `distance_to_all` (vectorized
  distance from a query point to every training point), `knn_predict` (k-nearest neighbors
  classifier with majority vote), and `predict_grid` (classifies a batch of points, used for
  drawing the decision boundary).
- **`gradient_descent.py`** — `gradient_descent_1d` on `f(x) = (x - 3)^2`, and
  `gradient_descent_2d` on `f(x, y) = x^2 + 5y^2`. Both return the final point and the full
  path taken.
- **`pca.py`** — `pca_via_svd`, which centers a dataset, runs `np.linalg.svd`, and returns the
  projection onto the top `n_components` principal component(s) along with the component
  vectors.
- **`ml_primitives_starter.ipynb`** — Notebook that generates data, calls the functions above,
  and produces all required plots:
  - Part A: scatter plot of the training data and a k-NN decision boundary plot.
  - Part B: a learning-rate comparison for 1D gradient descent (including one rate that
    oscillates) and a 2D gradient descent path plotted over a contour plot.
  - Part C: a PCA plot showing the original data, the recovered principal direction, and the
    1D projection.
- **`requirements.txt`** — Python dependencies for this lab (NumPy, Matplotlib, Jupyter, etc.).
- **`plots/`** — Folder for saving output figures, if exporting them from the notebook.

## How to Run

1. Create/activate an environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open and run the notebook top to bottom:

   ```bash
   jupyter notebook ml_primitives_starter.ipynb
   ```

   This imports the functions from `knn.py`, `gradient_descent.py`, and `pca.py`, generates the
   data, and produces all the plots described above.

   Alternatively, the functions in each `.py` file can be imported and called directly from your
   own script without using the notebook.