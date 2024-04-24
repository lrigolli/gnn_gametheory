import pandas as pd
import torch

from graph import EGTGraph


def fit_strategy_node(strategy_idx: int,
                      node_idx: int,
                      graph: EGTGraph,
                      payoff_matrices: list[torch.Tensor],
                      nodes_feats: torch.Tensor) -> torch.Tensor:
    # compute fitness of strategy i in node j
    F = nodes_feats  # num_nodes x num_feats
    P = payoff_matrices  # list with length num_nodes of num_feats x num_feats matrices
    A = graph.adjacency_matrix  # num_nodes x num_nodes.
    W = graph.nodes_weights  # num_nodes
    i = strategy_idx
    k = node_idx
    fit = torch.zeros(1)
    for l in range(graph.num_nodes):
        for j in range(graph.num_feats):
            fit += P[l][i, j]*F[l, j]*A[k, l]*W[k]
    return fit


def fit_strategy_graph(strategy_idx: int,
                       graph: EGTGraph,
                       payoff_matrices: list[torch.Tensor],
                       nodes_feats: torch.Tensor
                       ) -> float:
    W = graph.nodes_weights
    fit = 0
    for k in range(graph.num_nodes):
        fit += fit_strategy_node(strategy_idx=strategy_idx,
                                 node_idx=k,
                                 graph=graph,
                                 payoff_matrices=payoff_matrices,
                                 nodes_feats=nodes_feats) * W[k]
    return fit


def loss_ess(nodes_feats: torch.Tensor, graph: EGTGraph, payoff_matrices: list[torch.Tensor]) -> torch.Tensor:
    loss = torch.zeros(1)
    for k in range(graph.num_nodes):
        for i in range(graph.num_feats):
            for j in range(graph.num_feats):
                loss += (fit_strategy_node(
                    strategy_idx=i, node_idx=k, graph=graph, payoff_matrices=payoff_matrices, nodes_feats=nodes_feats)
                         - fit_strategy_node(
                            strategy_idx=j, node_idx=k, graph=graph, payoff_matrices=payoff_matrices, nodes_feats=nodes_feats))**2
    return loss


def get_nodes_strategy_fitness_df(X: torch.Tensor, graph: EGTGraph, payoff_matrices: list[torch.Tensor]) -> pd.DataFrame:
    nodes_start_fit = []
    for i in range(X.size(0)):
        nodes_start_fit_row = []
        for j in range(X.size(1)):
            nodes_start_fit_row.append(
                fit_strategy_node(
                    strategy_idx=j,
                    node_idx=i,
                    graph=graph,
                    payoff_matrices=payoff_matrices,
                    nodes_feats=X).item())
        nodes_start_fit.append(nodes_start_fit_row)
    df_nodes_strat_fit = pd.DataFrame(nodes_start_fit)
    return df_nodes_strat_fit


def loss_tot_fit(nodes_feats: torch.Tensor, graph: EGTGraph, payoff_matrices: list[torch.Tensor]) -> float:
    payoff_sum = 0
    for k in range(graph.num_nodes):
        for i in range(graph.num_feats):
            for j in range(graph.num_feats):
                payoff_sum += fit_strategy_node(strategy_idx=i,
                                                node_idx=k,
                                                graph=graph,
                                                payoff_matrices=payoff_matrices,
                                                nodes_feats=nodes_feats)
    loss = -payoff_sum
    return loss

"""    # a_ij * p_ij * w_i. w_i weight given by 1/degree(node_i) :

    id_mat = torch.eye(num_nodes)
    nodes_degree = sum(adjacency_mat)
    for i in range(num_nodes):
        for j in range(num_nodes):
            coeff = torch.matmul(torch.matmul(adjacency_mat, id_mat[i, :]), id_mat[j, :])
            payoff = torch.matmul(torch.matmul(payoff_mat, output[i, :]), output[j, :])
            payoff_sum += coeff * payoff * (1 / nodes_degree[i])
    neg_payoff = - payoff_sum
    return neg_payoff"""
