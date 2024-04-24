import networkx as nx
import torch

from utils import create_nodes_features, convert_adjacency_matrix_to_edges_list, remove_row_col_matrix
from payoff_matrices import get_domination_payoff_matrix


class EGTGraph:
    def __init__(self,
                 payoff_matrices: list[torch.Tensor],
                 adjacency_matrix: torch.Tensor = None,
                 nodes_weights: torch.Tensor = None,
                 nodes_features: torch.Tensor = None,
                 seed_nodes_features: int = 17):
        self.num_nodes = len(payoff_matrices)
        self.num_feats = payoff_matrices[0].size()[0]
        self.seed_nodes_features = seed_nodes_features
        self.payoff_matrices = payoff_matrices
        self.original_reduced_strategies_dict = None

        if nodes_weights is None:
            self.nodes_weights = torch.ones(self.num_nodes, dtype=torch.float)
        else:
            self.update_node_weigths(nodes_weights)
        if nodes_features is None:
            self.nodes_feats = create_nodes_features(self.num_nodes, self.num_feats, seed_nodes_features)
        else:
            self.update_nodes_features(nodes_features)
        if adjacency_matrix is None:
            self.adjacency_matrix = torch.eye(self.num_nodes, dtype=torch.float)
        else:
            self.update_edges(adjacency_matrix)
        self.edges_list = convert_adjacency_matrix_to_edges_list(self.adjacency_matrix)
        self.edges_tensor = torch.tensor(self.edges_list).T

    def update_node_weigths(self, nodes_weights: torch.Tensor):
        self.nodes_weights = torch.tensor(nodes_weights)

    def update_nodes_features(self, feats_list: torch.Tensor):
        # list of num_nodes elements, each of them being a list with num_features (num_nodes x num_features matrix)
        self.nodes_feats = torch.tensor(feats_list, dtype=torch.float)

    def update_edges(self, adjacency_matrix: torch.Tensor):
        self.adjacency_matrix = adjacency_matrix
        self.edges_list = convert_adjacency_matrix_to_edges_list(adjacency_matrix)
        self.edges_tensor = torch.tensor(self.edges_list).T

    def update_payoff_matrices(self, payoff_matrices: list[torch.Tensor]):
        self.payoff_matrices = payoff_matrices

    def visualize_graph_structure(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(self.edges_list)
        nx.draw_networkx(G)

    def describe_egtgraph(self):
        print(f"num nodes: {self.num_nodes}")
        print(f"num strategies: {self.num_feats}")
        print(f"num edges: {len(self.edges_list)}")
        print(f"adjacency matrix: {self.adjacency_matrix}")
        print(f"payoff matrices: {self.payoff_matrices}")
        print(f"nodes features: {self.nodes_feats}")
        print(f"nodes weights: {self.nodes_weights}")
        print(f"dominated_strategies: {self.get_dominated_strategies()}")
        print(f"match between non-dominated reduced strategies and original strategies:"
              f" {self.original_reduced_strategies_dict}")

        self.visualize_graph_structure()

    def get_node_neighbourhood_payoff_matrix(self, node_idx: int) -> torch.Tensor:
        # get aggregated payoff matrix. goal is removing dominated strategies from graph
        i = node_idx
        P = self.payoff_matrices[i]
        A = self.adjacency_matrix
        payoff_mat_aggr = torch.zeros(size=(self.num_feats, self.num_feats), dtype=torch.float)
        for j in range(self.num_nodes):
            payoff_mat_aggr += A[i, j] * P
        return payoff_mat_aggr

    def get_dominated_strategies(self) -> list[int]:
        dom_mat_aggr = torch.zeros(size=(self.num_feats, self.num_feats), dtype=torch.float)
        for i in range(self.num_nodes):
            node_neigh_payoff_mat = self.get_node_neighbourhood_payoff_matrix(node_idx=i)
            dom_mat_aggr += get_domination_payoff_matrix(node_neigh_payoff_mat)

        dominated_strategies = []
        for j in range(self.num_feats):
            for k in range(self.num_feats):
                if (dom_mat_aggr[j, k] == self.num_nodes) and (j != k):
                    dominated_strategies.append(k)
        dominated_strategies = list(set(dominated_strategies))
        return dominated_strategies

    def get_reduced_original_strategies_match(self) -> dict:
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
            [[self.nodes_feats[:, i] for i in range(self.nodes_feats.size(1)) if i not in dominated_strategies]])

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

        self.original_reduced_strategies_dict = final_dict

