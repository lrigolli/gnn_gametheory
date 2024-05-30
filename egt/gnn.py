import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.pydantic_types import EGTGraphType
from utils.fitness import mean_squared_diff_fitness_graph


class GCN(torch.nn.Module):
    # Graph NN architecture
    def __init__(self, graph: EGTGraphType, payoff_matrices: list[torch.Tensor], hid1_in: int = 10, hid2_in: int = 10):
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

    def optimize(self, num_epochs: int, diagnostic: bool = True, stop_loss=1e-08, tol_extinction=1e-03) -> (list, list):
        preds = []
        losses = []
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss = 1

        epoch = 0
        while (epoch < num_epochs) and (loss > stop_loss):
            pred = self(self.graph.nodes_feats, self.graph.edges_tensor)
            # accelerate convergence to extinction
            pred = torch.where(pred < tol_extinction, 0, pred)
            pred = F.normalize(pred, p=1.0)
            loss = mean_squared_diff_fitness_graph(nodes_feats=pred,
                                                   graph=self.graph)
            # TODO: can we set initial starting point?? initialize weights using apply function in nn module...
            #  https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
            # if epoch == 0:
            #    print(loss)
            #    print(self.graph.nodes_feats)
            #    print(pred)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds.append(pred.detach())
            losses.append(loss.item())
            epoch += 1

        if diagnostic:
            fig, ax = plt.subplots(1, 1)
            ax.plot(range(len(losses)), losses)
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')

        print(f"loss: {loss}")
        if loss > stop_loss:
            print("point is not ESS for all nodes, two strategies have different payoffs in at least one node")
        else:
            print(
                "point satisfies necessary condition to be ESS for all nodes. further checks are needed to find it out")
        return preds[-1], losses[-1]
