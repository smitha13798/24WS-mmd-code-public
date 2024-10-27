# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np
from scipy.sparse import csr_matrix

def complete_code(message):
    raise Exception(f"Please complete the code: {message}")
    return None

def center_and_nan_to_zero(matrix, axis=0):
    """ Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix, axis=axis)
    # Subtract the mean from each axis
    matrix_centered = matrix - means
    return np.nan_to_num(matrix_centered)

def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def fast_cosine_sim(utility_matrix, vector, axis=0):
    """ Compute the cosine similarity between the matrix and the vector"""
    # Compute the norms of each column
    norms = np.linalg.norm(utility_matrix, axis=axis)
    um_normalized = utility_matrix / norms[:, np.newaxis]  # Ensure correct shape for broadcasting
    dot = np.dot(um_normalized, vector)  # Calculate dot product
    return dot / np.linalg.norm(vector)  # Scale by the norm of the vector

def users_with_highest_similarity(similarities, neighborhood_size):
    """Select indices of users with highest similarity."""
    # Sort the similarities and return the indices of the top scores
    return np.argsort(similarities)[-neighborhood_size:]

def compute_ratings(best_among_who_rated, similarities, utility_matrix, item_index):
    """Compute the weighted rating for an item."""
    if best_among_who_rated.size == 0:
        return np.nan
    # Compute the weighted sum of ratings
    weights = similarities[best_among_who_rated]
    ratings = utility_matrix[item_index, best_among_who_rated]
    return np.dot(weights, ratings) / weights.sum() if weights.sum() > 0 else np.nan

def rate_all_items(orig_utility_matrix, user_index, neighborhood_size):
    print(f"\n>>> CF computation for UM w/ shape: "
          + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n")

    clean_utility_matrix = center_and_nan_to_zero(orig_utility_matrix)
    user_col = clean_utility_matrix[:, user_index]
    similarities = fast_cosine_sim(clean_utility_matrix, user_col)

    def rate_one_item(item_index):
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            return orig_utility_matrix[item_index, user_index]
        users_who_rated = np.where(np.isnan(orig_utility_matrix[item_index, :]) == False)[0]
        best_among_who_rated = users_with_highest_similarity(similarities[users_who_rated], neighborhood_size)
        best_among_who_rated = users_who_rated[best_among_who_rated]
        best_among_who_rated = best_among_who_rated[np.isnan(similarities[best_among_who_rated]) == False]
        if best_among_who_rated.size > 0:
            rating_of_item = compute_ratings(best_among_who_rated, similarities, orig_utility_matrix, item_index)
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings

def centered_cosine_sim(vector_x, vector_y):
    """Compute centered cosine similarity between two sparse vectors."""
    vector_x = csr_matrix(vector_x)
    vector_y = csr_matrix(vector_y)
    mean_x = vector_x.mean()
    mean_y = vector_y.mean()
    vector_x_centered = vector_x - mean_x
    vector_y_centered = vector_y - mean_y
    dot_product = vector_x_centered.dot(vector_y_centered.T).toarray()[0, 0]
    norm_x = np.linalg.norm(vector_x_centered.toarray())
    norm_y = np.linalg.norm(vector_y_centered.toarray())
    similarity = dot_product / (norm_x * norm_y) if norm_x > 0 and norm_y > 0 else 0
    return similarity

def fast_centered_cosine_sim(sparse_matrix, sparse_vector):
    """Compute centered cosine similarity between a sparse matrix and a sparse vector."""
    sparse_matrix = csr_matrix(sparse_matrix)
    sparse_vector = csr_matrix(sparse_vector)
    mean_matrix = sparse_matrix.mean(axis=1)
    mean_vector = sparse_vector.mean()
    matrix_centered = sparse_matrix - mean_matrix
    vector_centered = sparse_vector - mean_vector
    dot_product = matrix_centered.dot(vector_centered.T).toarray().flatten()
    norms_matrix = np.linalg.norm(matrix_centered.toarray(), axis=1)
    norm_vector = np.linalg.norm(vector_centered.toarray())
    similarities = dot_product / (norms_matrix * norm_vector) if norm_vector > 0 else np.zeros_like(dot_product)
    return similarities
