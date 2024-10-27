import numpy as np
from scipy.sparse import csr_matrix
import numpy.testing as npt
from rec_sys.cf_algorithms import centered_cosine_sim

# Define test vectors
k = 100
vector_x_test1 = np.array([i + 1 for i in range(k)])
vector_y_test1 = np.array([vector_x_test1[k - 1 - i] for i in range(k)])  # Reversed

# Sparse format
vector_x_test1_sparse = csr_matrix(vector_x_test1)
vector_y_test1_sparse = csr_matrix(vector_y_test1)

# Test case 1
similarity_test1 = centered_cosine_sim(vector_x_test1_sparse, vector_y_test1_sparse)
print(f"Test 1 - Similarity between test vectors (expected ~1): {similarity_test1}")
npt.assert_almost_equal(similarity_test1, 1, decimal=1)

# Test case 2 with NaNs
vector_x_test2 = np.array([np.nan if i in [2 + j * 10 for j in range(10)] else i + 1 for i in range(k)])
vector_x_test2_sparse = csr_matrix(vector_x_test2)
similarity_test2 = centered_cosine_sim(vector_x_test2_sparse, vector_y_test1_sparse)
print(f"Test 2 - Similarity with NaNs in vector_x: {similarity_test2}")
