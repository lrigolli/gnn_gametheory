import torch
from copy import deepcopy


def ess_check_helper_pytorch(x: torch.Tensor, p: torch.Tensor, graph: EGTGraphType, node_idx: int):
    """
    Define function used to check if a point is ESS for EGT graph (generalization to case of graphs  of function in
    Theorem 6.4.1 of "Evolutionary Games and Population Dynamics" by Hofbauer and Sigmund).
    It returns a function for each node index. In order to check if point p is ESS we need to make sure each of the
    returned function takes a minimum in point p.
    :param x: point in neighburood of 0 \in R^{n_nodes\times (n_strat-1)} (stored as vector)
    :param p: point in R^{n_nodes\times n_strat}, which is candidate ESS (stored as matrix)
    :param graph: EGT graph with defined payoff matrices, adjacency matrix and population
    :param node_idx: node index
    :return:
    """
    # ATTENTION: torch.autograd.functional.hessian returns always zero hessian...there is something wrong, so we use
    # numdifftools instead
    x = torch.tensor(x)
    n = graph.num_nodes
    m = graph.num_feats
    # reshape flat tensor to matrix for readability
    x = x.reshape((n, m - 1))

    # make a copy of input graph
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
    x_symp_list = [L * x[i, :] + p[i, :].reshape((L * x[i, :]).size()) for i in range(n)]
    x_symp = torch.stack(x_symp_list).reshape((n, m))  # x_symp is nxm matrix

    # let i be node index
    # f_p^i: R^{n}x{m} -> R
    # f_p^i(x_symp) = Fit_{node_i}(p,x_symp) - Fit_{node_i}(x_symp,x_symp)
    # h_p^i = f_p^i \circ phi_p
    # phi_p(0) = p
    # f_p^i(p) = 0
    # h_p^i(0) = 0

    # (function h for single node graph used in Thm 6.4.1:  p^T*F*y - y^T*F*y)

    # get function
    h = torch.tensor(0, dtype=torch.double)
    g_new.update_nodes_features_mutant(p)
    #print(g_new.nodes_feats_mutant)
    #print('ris 1')
    #print(fit_mutant_node(node_idx=node_idx, graph=g_new, nodes_feats=x_symp))
    h += fit_mutant_node(node_idx=node_idx, graph=g_new, nodes_feats=x_symp)
    #print(fit_mutant_node(node_idx=node_idx, graph=g_new, nodes_feats=x_symp))
    g_new.update_nodes_features_mutant(x_symp)
    #print(g_new.nodes_feats_mutant)
    #print('ris 2')
    #print(fit_mutant_node(node_idx=node_idx, graph=g_new, nodes_feats=x_symp))
    h -= fit_mutant_node(node_idx=node_idx, graph=g_new, nodes_feats=x_symp)
    #print(fit_mutant_node(node_idx=node_idx, graph=g_new, nodes_feats=x_symp))
    #pdb.set_trace()
    print(h)

    # necessary condition for p being an ESS is that h has a maximum in 0
    return h


# check if point p_i is ESS for node i
# aggregated payoff matrix is componentwise multiplication of adj matrix and sum of payoffs
# payoff_aggr = adj_mat * (payoff_mats[0] + payoff_mats[1])
# Compute an updated payoff matrix for each node. Each payoff matrix comes from aggregation of other matrices in graph.
#  The aggregated payoff matrix for node k is given by P_agg[k] = \sum_{l=1}^n A[k,l] P[l]
payoff_aggr_mats = []
for k in range(num_nodes):
    payoff_aggr = torch.zeros((g.num_feats,g.num_feats))
    for l in range(len(payoff_mats)):
        payoff_aggr += adj_mat[k, l] * g.payoff_matrices[l]
    payoff_aggr_mats.append(payoff_aggr)

# Check if each of the point in node is ESS. In case we have a global ESS for entire system
# (condition on payoff matrix is replaced by condition on aggregated payoff matrix)
for i in range(num_nodes):
    ess_point = is_ess_single_node(pred[i], payoff_aggr_mats[i])


"""import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-0.01,0.01,101)
y = [ess_check_fun(torch.tensor(el)) for el in x]
plt.plot(x,y)"""


"""def get_reduced_original_strategies_match(self) -> dict:
    dominated_strategies = self.get_dominated_strategies()
    original_strategy_inds = []
    for i in range(self.num_feats):
        if i not in dominated_strategies:
            original_strategy_inds.append(i)
    dict_original_red_match = dict(zip(list(range(self.num_feats - len(dominated_strategies))), original_strategy_inds))
    return dict_original_red_match


def remove_dominated_strategies(self):
    payoff_mats_reduced = []
    dominated_strategies = self.get_dominated_strategies()
    for p in self.payoff_matrices:
        p_red = remove_row_col_matrix(mat=p, idxs_to_remove=dominated_strategies)
        payoff_mats_reduced.append(p_red)
    self.payoff_matrices = payoff_mats_reduced
    self.num_feats = self.payoff_matrices[0].size()[0]
    self.nodes_feats = torch.tensor(
        [[self.nodes_feats[:, i] for i in range(self.nodes_feats.size(1)) if i not in dominated_strategies]],
        dtype=torch.float)


def remove_dominated_strategies_iterative(self):
    d_dom_ori_all = []
    while len(self.get_dominated_strategies()) > 0:
        d_dom_ori = self.get_reduced_original_strategies_match()
        self.remove_dominated_strategies()
        d_dom_ori_all.append(d_dom_ori)
    d_dom_ori_all = d_dom_ori_all[::-1]

    final_dict = {}
    for i in d_dom_ori_all[0].keys():
        k = i
        for j in range(len(d_dom_ori_all)):
            k = d_dom_ori_all[j][k]
        final_dict.update({i: k})

    self.original_reduced_strategies_dict = final_dict"""


def get_domination_payoff_matrix(payoff_mat: torch.Tensor) -> torch.Tensor:
    """
    Starting from payoff matrix, create a new matrix with binary entries a_ij that are:
    - 1 if payoff for strategy i is not smaller than payoff for strategy j for every strategy played by opponent
    - 0 otherwise
    :param payoff_mat: payoff matrix
    :return: matrix encoding information of which strategies are dominated
    """
    n = payoff_mat.size()[0]
    dom_mat = torch.zeros(size=(n, n), dtype=torch.int)
    for i in range(n):
        for j in range(n):
            row_comparison = min(payoff_mat[i, :] >= payoff_mat[j, :])
            if row_comparison:
                dom_mat[i, j] = 1
    return dom_mat


def is_inner_ess_single_node(p: torch.Tensor, payoff_mat: torch.Tensor) -> bool:
    """
    Check if a given point in inner of symplex (no extinction allowed) in given node is ESS
    :param p: candidate ESS point
    :param payoff_mat: payoff matrix for given node
    :return: boolean value saying whether the point is ESS or not
    """
    F = payoff_mat
    n = len(p)

    # phi_p: R^{n-1} -> R^{n} local linear parametrization of unit symplex around point p in int(symplex)
    # phi_p(x) = L(x) + p
    # phi_p(x1,...,x{n-1}) =
    #   (x1, ..., x{n-1}, 1 -x1 - ... - x{n-1}) + (p1,...,pn) =
    #   (x1+p1, ..., x{n-1} + p{n-1}, 1 -x1 - ... - x{n-1} + pn)
    L = torch.zeros((n, n - 1))
    for i in range(n - 1):
        L[i, i] = 1
    L[n - 1, :] = -torch.ones(n - 1)

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
    grad = torch.matmul(L.T, (torch.matmul(p.T, F) - torch.matmul(F.T, p) - torch.matmul(F, p)).T)

    # compute hessian matrix of g in 0
    H = torch.matmul(torch.matmul(-L.T, F + F.T), L)

    ess = False
    print(f"gradient of p^T*F*y - y^T*F*y at point p: {grad.numpy()}")
    print(f"Hessian of p^T*F*y - y^T*F*y at point p: {np.matrix(H)}")
    if min(grad == 0).item():
        print(f"{p.numpy()} is a critical point")
        if min(torch.flatten(H > 0)).item():
            # if g has local minimum in p then p is ESS
            ess = True
    if ess:
        print(f"{p.numpy()} is ESS")
    else:
        print(f"{p.numpy()} is not ESS")
    return ess


"""def ess_check(p: torch.Tensor, graph: EGTGraphType):
    zero = np.zeros(graph.num_nodes * (graph.num_feats - 1))
    p = p.double()
    grads = []
    hessians = []
    approx_digits = 5
    for i in range(graph.num_nodes):
        ess_check_fun = partial(ess_check_helper, p=p, graph=graph, node_idx=i)
        grad = nd.Gradient(fun=ess_check_fun)(zero)
        grad = np.round(grad, approx_digits)
        hess = nd.Hessian(f=ess_check_fun)(zero)
        hess = np.round(hess, approx_digits)
        print(f"Gradient of Fit_{i} (p,q) - Fit_{i} (q,q) at point p for node {i}: {grad}")
        print(f"Hessian of Fit_{i} (p,q) - Fit_{i} (q,q) at point p for node {i}: {np.matrix(hess)}")
        grads.append(grad)
        hessians.append(hess)
    return grads, hessians
"""



def remove_row_col_matrix(mat: torch.Tensor, idxs_to_remove: list[int]) -> torch.Tensor:
    """
    Remove row and columns of given indices from given matrix
    :param mat: input matrix
    :param idxs_to_remove: indices to remove
    :return: matrix with removed rows and columns
    """
    mat_reduced = []
    for i in range(mat.size(0)):
        row = []
        for j in range(mat.size(1)):
            if (i not in idxs_to_remove) and (j not in idxs_to_remove):
                row.append(mat[i, j])
        if len(row) > 0:
            mat_reduced.append(row)
    tensor_reduced = torch.tensor(mat_reduced, dtype=torch.float)
    return tensor_reduced

def mean_squared_diff_fitness_graph_old(nodes_feats: torch.Tensor,
                                        graph: EGTGraphType,
                                        tol_extinction=1e-3) -> torch.Tensor:
    """
    Compute the mean squared difference of fitness for each node for varying strategies. This will be used as loss
    function to detect ESS.
    In formula: \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^m (F_{i,j} - F_{i,k})^2 with n number of nodes, m num strategies
    and F_{i,j} fitness of strategy j in node i
    :param nodes_feats: frequency of strategies across population for each node
    :param graph: graph with adjacency matrix
    :param tol_extinction:
    :return: mean squared difference of fitness for each node for varying strategies
    """
    loss = torch.tensor(0, dtype=torch.float)
    for k in range(graph.num_nodes):
        for i in range(graph.num_feats):
            for j in range(graph.num_feats):
                f_ik = fit_strategy_node(
                    strategy_idx=i, node_idx=k, graph=graph, nodes_feats=nodes_feats)
                f_jk = fit_strategy_node(
                    strategy_idx=j, node_idx=k, graph=graph, nodes_feats=nodes_feats)
                # if strategy is not represented in population, it doesn't contribute to loss
                if (nodes_feats[k, i] > tol_extinction) and (nodes_feats[k, j] > tol_extinction):
                    loss += (f_ik - f_jk)**2
    return loss
