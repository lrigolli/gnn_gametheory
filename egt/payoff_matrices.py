import torch


# For single node if v > c all hawks, otherwise v/c hawks in population
def define_hawk_dove_payoff(v: float, c: float) -> torch.Tensor:
    return torch.tensor([[(v-c)/2, v], [0, v/2]], dtype=torch.float)  # H, D


# For single node and params v=2, c=4 there is one ESSs (mixed strategy 1/2 H + 1/2 D) and
# one Nash equilibrium (2/3 D + 1/3 R)
# if v > c retaliator first dominates dove and then also hawk
# if v < c dove dominates retaliator
def define_hawk_dove_retaliator_payoff(v: float, c: float) -> torch.Tensor:
    return torch.tensor([[(v-c)/2, v, (v-c)/2], [0, v/2, v/2], [(v-c)/2, v/2, v/2]], dtype=torch.float)  # H, D, R


def define_hawk_dove_bully_payoff(v: float, c: float) -> torch.Tensor:
    return torch.tensor([[(v-c)/2, v, v], [0, v/2, 0], [0, v, v/2]], dtype=torch.float)  # H, D, B
# bully dominates dove
# 1/2H + 1/2 B


def define_hawk_dove_retaliator_bully_payoff(v: float, c: float) -> torch.Tensor:
    return torch.tensor([[(v-c)/2, v, v, (v-c)/2],
                         [0, v/2, 0, v/2],
                         [0, v, v/2, 0],
                         [(v-c)/2, v/2, v, v/2]], dtype=torch.float)  # H, D, B, R
# no ESS. https://www.cs.rug.nl/~michael/teaching/gametheorysheets.pdf


# For single node there if eps>0 then ESS is given by mixed strategy 1/3 R + 1/3 S + 1/3 P, but pure strategies don't
# lead to ESS
# If eps<0 then no ESS at all
def define_rock_scissors_paper_payoff(eps: float) -> torch.Tensor:
    return torch.tensor([[-eps, 1, -1], [-1, -eps, 1], [1, -1, -eps]], dtype=torch.float)  # R, S, P


# evolutionary stable set example
def define_evo_stable_set_payoff() -> torch.Tensor:
    return torch.tensor([[0, 2, 0], [2, 0, 0], [1, 1, 0]], dtype=torch.float)  # R, S, P

# circulant matrix payoff
# cancer game https://arxiv.org/pdf/1803.00607
#A- A+ P C
#A- 1 1+d 1 1-c
#A+ 1-a+d 1-a+d+f 1-a+d 1-c-a+d
#P 1+g 1+d+g 1+g (1+g)(1-c)
#C 1-b+c 1-b+d+e 1-b+e 1-b


# examples https://michaellevet.wordpress.com/2016/05/14/evolutionary-stable-strategies-part-2/
