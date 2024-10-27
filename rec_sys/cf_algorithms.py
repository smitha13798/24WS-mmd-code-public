import numpy as np
from scipy.sparse import csr_matrix

def centered_cosine_sim(vector_x, vector_y):
    """
    Compute centered cosine similarity between two sparse vectors.
    Centered cosine similarity removes the average component from each vector.
    """
    if isinstance(vector_x, csr_matrix) and isinstance(vector_y, csr_matrix):
        vector_x = vector_x.toarray().flatten()
        vector_y = vector_y.toarray().flatten()
    
    mean_x = np.nanmean(vector_x)
    mean_y = np.nanmean(vector_y)
    centered_x = np.nan_to_num(vector_x - mean_x)
    centered_y = np.nan_to_num(vector_y - mean_y)

    numerator = np.dot(centered_x, centered_y)
    denominator = np.linalg.norm(centered_x) * np.linalg.norm(centered_y)
    
    return abs(numerator / denominator) if denominator != 0 else 0

def fast_centered_cosine_sim(matrix, vector):
    """
    Compute centered cosine similarity between a sparse matrix and a sparse vector.
    """
    if isinstance(matrix, csr_matrix):
        matrix = matrix.toarray()
    if isinstance(vector, csr_matrix):
        vector = vector.toarray().flatten()

    matrix_means = np.nanmean(matrix, axis=1).reshape(-1, 1)
    matrix_centered = np.nan_to_num(matrix - matrix_means)

    vector_mean = np.nanmean(vector)
    vector_centered = np.nan_to_num(vector - vector_mean)

    numerator = np.dot(matrix_centered, vector_centered)
    matrix_norms = np.linalg.norm(matrix_centered, axis=1)
    vector_norm = np.linalg.norm(vector_centered)
    
    denominator = matrix_norms * vector_norm
    similarity = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    return similarity
