import torch
import numpy as np
import networkx as nx
from pyvis.network import Network
from copy import deepcopy
from IPython.display import display, HTML

from utils.graph_initialization import create_random_nodes_features, convert_adjacency_matrix_to_edges_list
from utils.fitness import fit_strategy_node


class EGTGraph:
    def __init__(self,
                 payoff_matrices: list[torch.Tensor],
                 adjacency_matrix: torch.Tensor = None,
                 nodes_features: torch.Tensor = None,
                 seed_nodes_features: int = 17):
        self.num_nodes = len(payoff_matrices)
        self.num_feats = payoff_matrices[0].size()[0]
        self.seed_nodes_features = seed_nodes_features
        self.payoff_matrices = payoff_matrices
        self.nodes_feats_mutant = None
        self.nodes_strategy_fit = None

        if nodes_features is None:
            self.nodes_feats = create_random_nodes_features(self.num_nodes, self.num_feats, seed_nodes_features)
        else:
            self.update_nodes_features(nodes_features)
        if adjacency_matrix is None:
            self.adjacency_matrix = torch.eye(self.num_nodes, dtype=torch.float)
        else:
            self.update_edges(adjacency_matrix)
        self.edges_list = convert_adjacency_matrix_to_edges_list(self.adjacency_matrix)
        self.edges_tensor = torch.tensor(self.edges_list, dtype=torch.int64).T
        self.update_nodes_strategy_fit(self.nodes_feats)

    def update_nodes_features(self, feats: torch.Tensor):
        # num_nodes x num_features matrix (2d-tensor)
        self.nodes_feats = feats.clone().detach()

    def update_nodes_features_mutant(self, feats: torch.Tensor):
        # num_nodes x num_features matrix (2d-tensor)
        self.nodes_feats_mutant = feats.clone().detach()

    def update_edges(self, adjacency_matrix: torch.Tensor):
        self.adjacency_matrix = adjacency_matrix
        self.edges_list = convert_adjacency_matrix_to_edges_list(adjacency_matrix)
        self.edges_tensor = torch.tensor(self.edges_list, dtype=torch.int64).T

    def update_nodes_strategy_fit(self, feats: torch.Tensor):
        self.nodes_strategy_fit = self.get_nodes_strategy_fitness(X=feats)

    def update_payoff_matrices(self, payoff_matrices: list[torch.Tensor]):
        self.payoff_matrices = payoff_matrices

    def visualize_graph_structure(self, filename='egt_graph.html'):
        graph_filename = f'../{filename}'
        num_digits_round = 3
        # Create graph
        G = nx.DiGraph()
        for i in range(self.num_nodes):
            label_str = f"Node {i} \n"
            label_str += "Pop: " + str(tuple(np.round(self.nodes_feats[i].numpy(), num_digits_round))) + "\n"
            label_str += "Fit: " + str(tuple(np.round(self.nodes_strategy_fit[i].numpy(), num_digits_round)))
            title_str = f"Payoff matrix \n {np.matrix(self.payoff_matrices[i].numpy())}"
            G.add_node(id, size=20, label=label_str, title=title_str)
        for edge in self.edges_list:
            w = np.round(self.adjacency_matrix[edge[0], edge[1]].item(), num_digits_round)
            G.add_edge(edge[0], edge[1], weight=w, title=w)
        # Plot graph
        nt = Network(notebook=True, directed=True, cdn_resources='in_line')
        nt.from_nx(G)
        nt.save_graph(graph_filename)

        # Read the contents of the HTML file
        with open(graph_filename, 'r') as file:
            html_content = file.read()

        # Display the HTML content
        display(HTML(html_content))

    def describe_egtgraph(self, verbose=False):
        self.visualize_graph_structure()

        if verbose:
            print(f"Num nodes: {self.num_nodes}")
            print(f"Num strategies: {self.num_feats}")
            print(f"Num edges: {len(self.edges_list)}")
            print(f"Adjacency matrix")
            print(np.matrix({self.adjacency_matrix}))
            print("Payoff matrices")
            for i in range(len(self.payoff_matrices)):
                print(f"Node {i}")
                print(np.matrix(self.payoff_matrices[i]))
            print(f"Nodes features: {np.matrix(self.nodes_feats)}")

    def get_nodes_strategy_fitness(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get num_nodes x num_strategies with fitness of each pair (node, strategy)
        :param X:
        :return: num_nodes x num_strategies dataframe with fitness of each pair (node, strategy)
        """
        g = deepcopy(self)
        g.update_nodes_features(X)
        nodes_strat_fit_list = []
        for i in range(X.size(0)):
            nodes_strat_fit_row = []
            for j in range(X.size(1)):
                nodes_strat_fit_row.append(
                    fit_strategy_node(
                        strategy_idx=j,
                        node_idx=i,
                        graph=self,
                        nodes_feats=X).item())
            nodes_strat_fit_list.append(nodes_strat_fit_row)
        nodes_strat_fit = torch.Tensor(nodes_strat_fit_list)
        return nodes_strat_fit
