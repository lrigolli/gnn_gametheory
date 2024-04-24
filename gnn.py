import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from graph import EGTGraph
from fitness import loss_ess


class GCN(torch.nn.Module):
    # Graph NN architecture
    def __init__(self, graph: EGTGraph, payoff_matrices: list[torch.Tensor], hid1_in: int, hid2_in: int):
        super().__init__()
        self.graph = graph
        self.conv1 = GCNConv(graph.num_feats, hid1_in)
        self.conv2 = GCNConv(hid1_in, hid2_in)
        self.conv3 = GCNConv(hid2_in, graph.num_feats)
        self.payoff_matrices = payoff_matrices

    def forward(self, x: Tensor, edge_index: Tensor) -> torch.Tensor:
        # x: Node feature matrix of shape [num_nodes, num_features]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        # todo: make number of layers to be a param
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = F.softmax(x, dim=1)
        return x

    def optimize(self, num_epochs: int, diagnostic: bool =True) -> (list, list):
        preds = []
        losses = []
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            pred = self(self.graph.nodes_feats, self.graph.edges_tensor)
            loss = loss_ess(nodes_feats=pred, graph=self.graph, payoff_matrices=self.payoff_matrices)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds.append(pred)
            losses.append(loss.item())
        if diagnostic:
            fig, ax = plt.subplots(1, 1)
            ax.plot(range(num_epochs), losses)
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')

        return preds[-1]

