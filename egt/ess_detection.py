import torch
import numpy as np
from functools import partial
import numdifftools as nd  # https://numdifftools.readthedocs.io/en/stable/topics/finite_difference_derivatives.html
from copy import deepcopy

from utils.fitness import fit_mutant_node
from utils.pydantic_types import EGTGraphType


def ess_check_helper(x: np.array, p: torch.Tensor, graph: EGTGraphType, node_idx: int):
    """
    Define function used to check if a point is ESS for EGT graph (generalization to case of graphs  of function in
    Section 7.2 of "Evolutionary Games and Population Dynamics" by Hofbauer and Sigmund).
    It returns a function for each node index. In order to check if point p is ESS we need to make sure each of the
    returned function takes a minimum in point p.
    :param x: point in neighbourood of 0 \in R^{n_nodes\times (n_strat-1)} (stored as vector)
    :param p: point in R^{n_nodes\times n_strat}, which is candidate ESS (stored as matrix)
    :param graph: EGT graph with defined payoff matrices, adjacency matrix and population
    :param node_idx: node index
    :return:
    """
    x = torch.tensor(x)
    n = graph.num_nodes
    m = graph.num_feats
    # reshape flat tensor to matrix for readability
    x = x.reshape((n, m - 1))

    # make a copy of input grpah
    g_new = deepcopy(graph)

    # Parametrization of neighbourhood of p in single node with index i
    # phi_p^i: R^{m-1} -> R^{m} local linear parametrization of unit symplex around point p in int(symplex)
    # phi_p^i(x) = L(x) + p
    # phi_p^i(x1,...,x{m-1}) =
    #   (x1, ..., x{m-1}, 1 -x1 - ... - x{m-1}) + (p1,...,pm) =
    #   (x1+p1, ..., x{m-1} + p{m-1}, 1 -x1 - ... - x{m-1} + pm)
    L = torch.zeros((m, m - 1))
    for i in range(m - 1):
        L[i, i] = 1
    L[m - 1, :] = -torch.ones(m - 1)

    # Parametrize point x
    x_symp_list = [np.matmul(L, x[i, :]) + p[i, :] for i in range(n)]
    x_symp = torch.stack(x_symp_list).reshape((n, m))  # x_symp is nxm matrix

    # let i be node index
    # f_p^i: R^{n}x{m} -> R
    # f_p^i(x_symp) = Fit_{node_i}(p,x_symp) - Fit_{node_i}(x_symp,x_symp)
    # h_p^i = f_p^i \circ phi_p
    # phi_p(0) = p
    # f_p^i(p) = 0
    # h_p^i(0) = 0

    # (function h for single node graph used in Section 7.2:  p^T*F*y - y^T*F*y)

    # get function
    h = 0
    g_new.update_nodes_features_mutant(p)
    h += fit_mutant_node(node_idx=node_idx, graph=g_new, nodes_feats=x_symp).item()
    g_new.update_nodes_features_mutant(x_symp)
    h -= fit_mutant_node(node_idx=node_idx, graph=g_new, nodes_feats=x_symp).item()
    # necessary condition for p being an ESS is that h has a minimum in 0
    return h


def is_inner_ess(p: torch.Tensor, graph: EGTGraphType, tol_float_digits: int = 5):
    """
    Check if a given point in inner of simplices (no extinction allowed) is ESS
    :param p: candidate ESS point
    :param graph: ...
    :param tol_float_digits: ...
    :return: boolean value saying whether the point is ESS or not
    """
    zero = np.zeros(graph.num_nodes * (graph.num_feats - 1))
    p = p.double()
    grads = []
    hessians = []
    hessians_eigenvals = []
    hessians_restricted = []
    for i in range(graph.num_nodes):
        ess_check_fun = partial(ess_check_helper, p=p, graph=graph, node_idx=i)
        grad = nd.Gradient(fun=ess_check_fun)(zero)
        grad = np.round(grad, tol_float_digits)
        hess = nd.Hessian(f=ess_check_fun)(zero)
        hess = np.round(hess, tol_float_digits)
        hess_restr = hess[i*(graph.num_feats - 1): (i+1)*(graph.num_feats - 1), i*(graph.num_feats - 1): (i+1)*(graph.num_feats - 1)]
        eigenvals = np.round(np.linalg.eig(hess_restr).eigenvalues, tol_float_digits)
        print(f"Gradient of Fit_{i} (p,q) - Fit_{i} (q,q) at point p for node {i}: {grad}")
        print(f"Hessian of Fit_{i} (p,q) - Fit_{i} (q,q) at point p for node {i}: {np.matrix(hess)}")
        print(f"Eigenvalues of Hessian restriction: {eigenvals}")
        grads.append(grad)
        hessians.append(hess)
        hessians_restricted.append(hess_restr)
        hessians_eigenvals.append(eigenvals)

    if np.min(np.abs(grads) < 10**(-tol_float_digits)):
        print(f"{p.numpy()} is a critical point")

        # sufficient condition for p to be ESS (in general \partial f_ij can be zero if i!=j. but as long as f_ii is
        # positive for each i = node index, then we still have an ESS)

        min_eigenval = np.min(hessians_eigenvals)

        if min_eigenval > 0:
            ess = True
            print(f"{p.numpy()} is ESS")
        elif min_eigenval < 0:
            ess = False
            print(f"{p.numpy()} is not inner ESS")
        else:
            if graph.num_nodes == 1:
                ess = False
                print(f"{p.numpy()} is not inner ESS")
            else:
                ess = None
                print(f"{p.numpy()} need further check to determine if ESS")

    else:
        print(f"{p.numpy()} is not a critical point")
        print(f"{p.numpy()} is not inner ESS")
        ess = False

    return ess
