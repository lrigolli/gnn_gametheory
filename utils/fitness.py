import torch

from utils.pydantic_types import EGTGraphType


def fit_strategy_node(strategy_idx: int,
                      node_idx: int,
                      graph: EGTGraphType,
                      nodes_feats: torch.Tensor) -> torch.Tensor:
    """
    Get fitness of strategy i in node k, where nodes are part of graph.
    Fitness depends on nodes population, strategy, adjaceny matrix and payoffs across various nodes.
    :param strategy_idx: strategy index
    :param node_idx: node index
    :param graph: graph with adjacency matrix
    :param nodes_feats: frequency of strategies across population for each node
    :return: fitness for given pair (node, strategy)
    """
    # compute fitness of strategy i in node j
    g = graph
    F = nodes_feats  # num_nodes x num_feats
    P = g.payoff_matrices  # list with length num_nodes of num_feats x num_feats matrices
    A = g.adjacency_matrix  # num_nodes x num_nodes.
    i = strategy_idx
    k = node_idx
    fit = torch.tensor(0, dtype=torch.float)
    for l in range(g.num_nodes):
        for j in range(g.num_feats):
            fit += P[l][i, j]*F[l, j]*A[k, l]
    return fit


def fit_mutant_node(node_idx: int,
                    graph: EGTGraphType,
                    nodes_feats: torch.Tensor) -> torch.Tensor:
    """
    Get fitness of mutant (can be any player) in node node_idx, where nodes are part of graph.
    Fitness depends on nodes population, strategy, adjaceny matrix and payoffs across various nodes.
    :param node_idx: node index
    :param graph: graph with adjacency matrix
    :param nodes_feats: frequency of strategies across population for each node
    :return: fitness for given pair (node, strategy)
    """
    fit = torch.tensor(0, dtype=torch.double)
    for j in range(graph.num_feats):
        fit += fit_strategy_node(strategy_idx=j,
                                 node_idx=node_idx,
                                 graph=graph,
                                 nodes_feats=nodes_feats) * graph.nodes_feats_mutant[node_idx, j]
    return fit


def mean_squared_diff_fitness_graph(nodes_feats: torch.Tensor,
                                    graph: EGTGraphType,
                                    tol_extinction=1e-3) -> torch.Tensor:
    """
    Compute the mean squared difference of fitness for each node for varying strategies. This will be used as loss
    function to detect ESS.
    In formula: \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^m (F_{i,j} - F_{i,k})^2 with n number of nodes, m num strategies
    and F_{i,j} fitness of strategy j in node i
    :param nodes_feats: frequency of strategies across population for each node
    :param graph: graph with adjacency matrix
    :param tol_extinction: threshold used to determine extinction of a strategy in population
    :return: mean squared difference of fitness for each node for varying strategies
    """
    loss = torch.tensor(0, dtype=torch.float)
    for k in range(graph.num_nodes):
        f_k = []
        for i in range(graph.num_feats):
            f_k.append(fit_strategy_node(strategy_idx=i, node_idx=k, graph=graph, nodes_feats=nodes_feats))
        for i in range(graph.num_feats):
            for j in range(graph.num_feats):
                # if strategy is not represented in population and has lowest payoff, it doesn't contribute to loss
                # compare with replicator equation x_i' = x_i (f_i-avg_fit)
                # x rest point iff x_i = 0 or f_i=avg_fit)
                if (f_k[j] > min(f_k)) or (nodes_feats[k, i] >= tol_extinction):
                    loss += (f_k[i] - f_k[j]) ** 2
    return loss


# WIP: Functions for cooperative games, not for EGT
def neg_sum_fitness_graph(nodes_feats: torch.Tensor, graph: EGTGraphType) -> float:
    """
    Get total fitness of graph as sum of fitnesses for each node. By adding negative sign we can define a loss useful
    to analyze cooperative games.
    :param nodes_feats: frequency of strategies across population for each node
    :param graph: graph with adjacency matrix

    :return: fitness for given pair (node, strategy)
    """
    payoff_sum = 0
    for k in range(graph.num_nodes):
        for i in range(graph.num_feats):
            for j in range(graph.num_feats):
                payoff_sum += fit_strategy_node(strategy_idx=i,
                                                node_idx=k,
                                                graph=graph,
                                                nodes_feats=nodes_feats)
    loss = -payoff_sum
    return loss


def fit_strategy_graph(strategy_idx: int,
                       graph: EGTGraphType,
                       nodes_feats: torch.Tensor
                       ) -> float:
    """
    Get total fitness of given strategy in graph as average of nodes weighted according to their population
    Fitness depends on nodes population, strategy, adjaceny matrix and payoffs across various nodes.
    :param strategy_idx: strategy index
    :param graph: graph with adjacency matrix
    :param nodes_feats: frequency of strategies across population for each node
    :return: total fitness of given strategy in graph
    """
    W = graph.nodes_weights
    fit = 0
    for k in range(graph.num_nodes):
        fit += fit_strategy_node(strategy_idx=strategy_idx,
                                 node_idx=k,
                                 graph=graph,
                                 nodes_feats=nodes_feats
                                 ) * W[k]
    return fit
