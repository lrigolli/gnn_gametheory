import torch
import egt.payoff_matrices as pay
import utils.adjacency_matrix_creation as ad_creation
import utils.adjacency_matrices as ad


class EGTGraphParam:
    """A simple class to hold a payoff matrices and adjacency matrix."""

    def __init__(self,
                 payoff_mats: list[torch.Tensor],
                 adj_mat: torch.Tensor,
                 nodes_feats=torch.Tensor or None,
                 expected_ess=None):
        self.payoff_mats = payoff_mats
        self.adj_mat = adj_mat
        self.expected_ess = expected_ess
        self.nodes_feats = nodes_feats


# Single node examples
single_node_hd = EGTGraphParam(
    payoff_mats=[pay.define_hawk_dove_payoff(v=0.7, c=1.0)],
    adj_mat=ad_creation.create_id_matrix(n=1),
    expected_ess="7/10 H is ESS (in general v/c hawks if v<c and all hawks otherwise)",
    nodes_feats=None)

single_node_hdr = EGTGraphParam(
    payoff_mats=[pay.define_hawk_dove_retaliator_payoff(v=2.0, c=4.0)],
    adj_mat=ad_creation.create_id_matrix(n=1),
    expected_ess="1/2 H + 1/2 D is ESS (2/3 D + 1/3 R is Nash equilibrium, but not ESS)",
    nodes_feats=None)

single_node_evo_stable_set = EGTGraphParam(
    payoff_mats=[pay.define_evo_stable_set_payoff()],
    adj_mat=ad_creation.create_id_matrix(n=1),
    expected_ess="exists 1-dim set of Nash equilibria, but no ESS",
    nodes_feats=None)

single_node_rsp = EGTGraphParam(
    payoff_mats=[pay.define_rock_scissors_paper_payoff(eps=0)],
    adj_mat=ad_creation.create_id_matrix(n=1),
    expected_ess="1/3 R + 1/3 S + 1/3 P is Nash equilibrium, but no ESS",
    nodes_feats=None)

single_node_rsp_perturbed = EGTGraphParam(
    payoff_mats=[pay.define_rock_scissors_paper_payoff(eps=0.001)],
    adj_mat=ad_creation.create_id_matrix(n=1),
    expected_ess="1/3 R + 1/3 S + 1/3 P is ESS",
    nodes_feats=None)

single_node_hdb = EGTGraphParam(
    payoff_mats=[pay.define_hawk_dove_bully_payoff(v=2.0, c=4.0)],
    adj_mat=ad_creation.create_id_matrix(n=1),
    expected_ess="1/2 H + 1/2 B is ESS",
    nodes_feats=None)

single_node_hdbr = EGTGraphParam(
    payoff_mats=[pay.define_hawk_dove_retaliator_bully_payoff(v=2.0, c=4.0)],
    adj_mat=ad_creation.create_id_matrix(n=1),
    expected_ess="ESS does not exist, see https://www.cs.rug.nl/~michael/teaching/gametheorysheets.pdf",
    nodes_feats=None)


# Multiple node examples
# hawk dove, two nodes, symmetric doubly stoch matrix. there is ESS if adj matrix is 0.5, otherwise not

two_nodes_hd_const_adj = EGTGraphParam(
    payoff_mats=[pay.define_hawk_dove_payoff(v=0.2, c=1.0), pay.define_hawk_dove_payoff(v=0.6, c=1.0)],
    adj_mat=ad_creation.create_const_matrix(n=2),
    expected_ess="ESS (v1+v2)/(c1+c2) = 0.4 H, since adjacency matrix is constant",
    nodes_feats=None)

two_nodes_hd_nonconst_adj = EGTGraphParam(
    payoff_mats=[pay.define_hawk_dove_payoff(v=0.2, c=1.0), pay.define_hawk_dove_payoff(v=0.6, c=1.0)],
    adj_mat=ad.n2_22,
    expected_ess="no ESS ('most visited point' should be (v1+v2)/(c1+c2) = 0.4 H)",
    nodes_feats=None)

three_nodes_hd_const_adj = EGTGraphParam(
    payoff_mats=[pay.define_hawk_dove_payoff(v=0.2, c=1.0),
                 pay.define_hawk_dove_payoff(v=0.6, c=1.0),
                 pay.define_hawk_dove_payoff(v=1.1, c=1.0)],
    adj_mat=ad_creation.create_const_matrix(n=3),
    expected_ess="(v1 + v2 + v3)/(c1 +c2 + c3) H is ESS, since adjacency matrix is constant",
    nodes_feats=None)

three_nodes_hd_ess = EGTGraphParam(
    payoff_mats=[pay.define_hawk_dove_payoff(v=0.4, c=1.0),
                 pay.define_hawk_dove_payoff(v=0.6, c=1.0),
                 pay.define_hawk_dove_payoff(v=0.8, c=1.0)],
    adj_mat=ad.n3_232,
    expected_ess="(v1 + v2 + v3)/(c1 +c2 + c3) H is ESS",
    nodes_feats=None)
