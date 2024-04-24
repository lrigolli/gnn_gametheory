import numpy as np
import torch


def create_block_matrix_unit_column_sum(num_nodes: int, num_blocks: int, prob_link: float, seed: int) -> torch.Tensor:
    # Get approximate number of elements per block
    elements_per_block = num_nodes // num_blocks
    # Get blocks starting point
    blocks_start = []
    for i in range(num_blocks):
        blocks_start.append(i * elements_per_block)
    blocks_start_ext = blocks_start + [num_nodes]
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
    cols_right = num_nodes
    for i in range(num_blocks):
        cols_right -= blocks_length[i]
        b = [np.zeros((blocks_length[i], cols_left)), blocks[i], np.zeros((blocks_length[i], cols_right))]
        blocks_list.append(b)
        cols_left += blocks_length[i]

    M = np.block(blocks_list)
    return torch.tensor(M, dtype=torch.float)

