import torch
import numpy as np


def create_random_nodes_features(num_nodes: int, num_feats: int, seed=17) -> torch.Tensor:
    """
    Create random nodes features with constraint of being non negative and summing to one
    :param num_nodes: number of nodes
    :param num_feats: number of features
    :param seed: random seed
    :return: num_nodes x num_feats matrix whose rows sum to one
    """
    np.random.seed(seed)
    # Create nodes and corresponding features
    x_list = []
    for n in range(num_nodes):
        x_unnorm = np.random.uniform(size=num_feats)
        x_norm = x_unnorm/sum(x_unnorm)
        x_list.append(x_norm)
    x_tensor = torch.tensor(x_list, dtype=torch.float)
    return x_tensor


def convert_adjacency_matrix_to_edges_list(A: torch.Tensor) -> list[tuple]:
    """
    Represent adjacency matrix as edges list
    :param A: adjacency matrix
    :return: list of edges
    """
    edges_list = []
    for i in range(A.size()[0]):
        for j in range(A.size()[0]):
            if A[i, j] != 0:
                edges_list.append((i, j))
    return edges_list
