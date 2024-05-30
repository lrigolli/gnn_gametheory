import numpy as np
import torch
from scipy.linalg import circulant


def create_id_matrix(n: int) -> torch.Tensor:
    """
    Create identity matrix
    :param n: num rows/columns
    :return: identity matrix
    """
    A = torch.eye(n, dtype=torch.float)
    return A


def create_const_matrix(n: int) -> torch.Tensor:
    """
    Create square constant matrix
    :param n: num rows/columns
    :return: matrix with constant entries
    """
    A = torch.ones((n, n), dtype=torch.float) * 1/n
    return A


def create_circulant_matrix(n: int) -> torch.Tensor:
    """
    Create circulant matrix with rows summing to one
    :param n: num rows/columns
    :return: circulant matrix
    """
    if n > 1:
        circulant_vector = np.zeros(n)
        circulant_vector[0] = 1
        circulant_vector[1] = 1
        A = circulant(circulant_vector)
        for i in range(A.shape[0]):
            A[i, :] = A[i, :]/sum(A[i, :])
    else:
        A = torch.ones(1)
    return torch.tensor(A, dtype=torch.float)


def create_block_matrix_unit_row_sum(
        num_rows: int,
        num_blocks: int,
        prob_link: float,
        seed: int) -> torch.Tensor:
    """
    Create block matrix having columns summing to one
    :param num_rows: number of rows/columns
    :param num_blocks: number of blocks in matrix
    :param prob_link: probability a link is created between two elements of each block
    :param seed: random seed for replicability
    :return: num_rows x num_rows block matrix having rows summing to one
    """
    # Get approximate number of elements per block
    elements_per_block = num_rows // num_blocks
    # Get blocks starting point
    blocks_start = []
    for i in range(num_blocks):
        blocks_start.append(i * elements_per_block)
    blocks_start_ext = blocks_start + [num_rows]
    # Get blocks length
    blocks_length = []
    for i in range(len(blocks_start_ext) - 1):
        blocks_length.append(blocks_start_ext[i + 1] - blocks_start_ext[i])

    # Create blocks with positive random entries whose rows sum to 1
    np.random.seed(seed)
    blocks = []
    for k in blocks_length:
        A = np.random.random((k, k))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if (np.random.binomial(n=1, p=prob_link) == 1) and (i != j):
                    A[i, j] = 0
                if i == j:  # give more importance to node itself compared to neighbours
                    A[i, j] = 3 * A[i, j]
        # rows must sum to 1
        for l in range(A.shape[0]):
            A[l, :] = A[l, :] / sum(A[l, :])
        blocks.append(A)

    # Aggregate blocks in block matrix
    blocks_list = []
    cols_left = 0
    cols_right = num_rows
    for i in range(num_blocks):
        cols_right -= blocks_length[i]
        b = [np.zeros((blocks_length[i], cols_left)), blocks[i], np.zeros((blocks_length[i], cols_right))]
        blocks_list.append(b)
        cols_left += blocks_length[i]

    M = np.block(blocks_list)
    return torch.tensor(M, dtype=torch.float)


def create_random_doubly_stochastic_matrix(n, seed=3):
    """
    Create symmetric square matrix having columns and rows summing to one
    :param n: number of rows/columns
    :param seed: random seed for replicability
    :return: num_rows x num_rows block matrix having rows and columns summing to one
    """
    # https://github.com/djosix/doubly-stochastic-matrix
    np.random.seed(seed)
    if n == 1:
        return np.ones([1, 1])

    M = np.zeros([n, n])
    M[-1, -1] = np.random.uniform(0, 1)

    # Generate submatrix with constraint
    sM = create_random_doubly_stochastic_matrix(n - 1)
    sM *= (M[-1, -1] + n - 2) / (n - 1)

    M[:-1, :-1] = sM
    M[-1, :-1] = 1 - sM.sum(0)
    M[:-1, -1] = 1 - sM.sum(1)

    # Random permutation  if we don't want M to be symmetric
    # M = M @ np.random.permutation(np.eye(n))

    return M
