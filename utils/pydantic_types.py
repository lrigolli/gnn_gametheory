from pydantic import BaseModel, ConfigDict
import torch


class EGTGraphType(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    payoff_matrices: list[torch.Tensor]
    adjacency_matrix: torch.Tensor
    nodes_features: torch.Tensor
    seed_nodes_features: int


class CoopGraphType(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    payoff_matrices: list[torch.Tensor]
    adjacency_matrix: torch.Tensor
    nodes_weights: torch.Tensor
    nodes_features: torch.Tensor
    seed_nodes_features: int
