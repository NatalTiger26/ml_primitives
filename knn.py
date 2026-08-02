import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def distance_to_all(query, points):
    return np.linalg.norm(points - query, axis=1)



def knn_predict(query, X_train, Y_train, k):
    distances = distance_to_all(query, X_train)
    sorted_indices = np.argsort(distances)
    k_nearest_idx = sorted_indices[:k]
    k_nearest_labels = [Y_train[i] for i in k_nearest_idx]
    val, count = np.unique(k_nearest_labels, return_counts=True)
    return val[np.argmax(count)]


def predict_grid(grid_points, X_train, Y_train, k):
    predictions = []
    for point in grid_points:
        predictions.append(knn_predict(point, X_train, Y_train, k))
    return np.array(predictions)