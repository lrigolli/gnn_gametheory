import torch
import numpy as np


def create_nodes_features(num_nodes: int, num_feats: int, seed=17) -> torch.Tensor:
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
    edges_list = []
    for i in range(A.size()[0]):
        for j in range(A.size()[0]):
            if A[i, j] != 0:
                edges_list.append((i, j))
    return edges_list


def remove_row_col_matrix(mat: torch.Tensor, idxs_to_remove: list[int]) -> torch.Tensor:
    mat_reduced = []
    for i in range(mat.size(0)):
        row = []
        for j in range(mat.size(1)):
            if (i not in idxs_to_remove) and (j not in idxs_to_remove):
                row.append(mat[i, j])
        if len(row) > 0:
            mat_reduced.append(row)
    tensor_reduced = torch.tensor(mat_reduced)
    return tensor_reduced


def is_ess(p: torch.Tensor, payoff_mat: torch.Tensor) -> bool:
    F = payoff_mat
    n = len(p)

    # phi_p: R^{n-1} -> R^{n} local linear parametrization of unit symplex around point p in int(symplex)
    # phi(x1,...,x{n-1}) = (x1+p1, ..., x{n-1} + p{n-1}, 1 -x1 - ... - x{n-1} + pn)
    phi = torch.zeros((n, n - 1))
    for i in range(n - 1):
        phi[i, i] = 1
    phi[n - 1, :] = -torch.ones(n - 1)

    # f: R^{n} -> R
    # f(y1,...,yn) = p^T*F*y - y^T*F*y

    # g = f \circ phi
    # phi(0) = p
    # f(p) = 0
    # g(0) = 0

    # if g has a maximum in 0 then p is ESS
    # let's compute gradient and hessian

    # we use following formulas for gradient and hessian of composition of functions
    # 1) grad(x^T*A*x) = (A + A^T)x
    # 2) H(x ^ T * A * x) = A + A^T
    # 3) grad(f \circ phi)(x) = Dphi^T(x) grad(f(phi(x)))
    # 4) H_(f\circ phi)(x) = Dphi^T(x)H_f(phi(x))Dphi(x) + \sum_i=1 ^n \partial{f}{y_j} \cdot H_{phi^i}(x)

    # compute gradient of g in 0
    grad = torch.matmul(phi.T, (torch.matmul(p.T, F) - torch.matmul(F.T, p) - torch.matmul(F, p)).T)

    # compute hessian matrix of g in 0
    H = torch.matmul(torch.matmul(-phi.T, F + F.T), phi)

    ess = False
    print(f"gradient: {grad}")
    print(f"Hessian: {H}")
    if min(grad == 0).item():
        print(f"{p} is a critical point")
        if min(torch.flatten(H > 0)).item():
            # if g has local minimum in p then p is ESS
            print(f"{p} is ESS")
            ess = True
        else:
            print(f"{p} is not ESS")
    return ess
