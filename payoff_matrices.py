import torch


# For single node if v > c all hawks, otherwise v/d hawks in population
def define_hawk_dove_payoff(v: float, c: float) -> torch.Tensor:
    return torch.tensor([[(v-c)/2, v], [0, v/2]], dtype=torch.float)  # H, D


# For single node there are two ESSs. All R or mixed strategy 1/2 H + 1/2 D (?)
# if v > c retailator first dominates dove and then also hawk
# if v < c dove dominates retailator
def define_hawk_dove_retailator_payoff(v: float, c: float) -> torch.Tensor:
    return torch.tensor([[(v-c)/2, v, (v-c)/2], [0, v/2, v/2], [(v-c)/2, v/2, v/2]], dtype=torch.float)  # H, D, R


def define_hawk_dove_bully_payoff(v: float, c: float) -> torch.Tensor:
    return torch.tensor([[(v-c)/2, v, v], [0, v/2, 0], [0, v, v/2]], dtype=torch.float)  # H, D, B
# bully dominates dove
# 1/2H + 1/2 B


# For single node there if eps>0 then ESS is given by mixed strategy 1/3 R + 1/3 S + 1/3 P, but pure strategies don't
# lead to ESS
# If eps<0 then no ESS at all
def define_rock_scissors_paper_payoff(eps: float) -> torch.Tensor:
    return torch.tensor([[-eps, 1, -1], [-1, -eps, 1], [1, -1, -eps]], dtype=torch.float)  # R, S, P


# evolutionary stable set example
def define_evo_stable_set_payoff() -> torch.Tensor:
    return torch.tensor([[0, 2, 0], [2, 0, 0], [1, 1, 0]], dtype=torch.float)  # R, S, P


# circulant matrix payoff




def get_domination_payoff_matrix(payoff_mat: torch.Tensor) -> torch.Tensor:
    n = payoff_mat.size()[0]
    dom_mat = torch.zeros(size=(n, n), dtype=torch.int)
    for i in range(n):
        for j in range(n):
            row_comparison = min(payoff_mat[i, :] >= payoff_mat[j, :])
            if row_comparison:
                dom_mat[i, j] = 1
    return dom_mat
