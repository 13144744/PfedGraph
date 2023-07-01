"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import zeros
from torch.nn import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
import hgcn.manifolds.poincare as poincareball
from hgcn.layers.att_layers import DenseAtt
from hgcn.layers.pool_layers import NodeInformationScore
from sparse_softmax import Sparsemax


def edge_to_adj(edge_index, x):
    row, col = edge_index
    xrow, xcol = x[row], x[col]
    cat = torch.cat([xrow, xcol], dim=1).sum(dim=-1).div(2)
    weights = (torch.cat([x[row], x[col]], dim=1)).sum(dim=-1).div(2)
    adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
    adj[row, col] = weights
    return adj


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.hid_dim] + ([args.emb_dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if torch.cuda.is_available():
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HypAddDistance:
    pass


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        # self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att)
        self.agg = NodeSelect(manifold, c_in, out_features, out_features)
        # self.agg = HypAddDist(manifold, c_in, out_features, out_features)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, edge_index = input
        h = self.linear.forward(x)
        # h = self.agg.forward(h, edge_index)
        # h = self.agg.forward(h, edge_index)
        h = self.agg.forward(h, edge_index)

        h = self.hyp_act.forward(h)
        output = h
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypLinear2(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear2, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.hyperbolic_bias = nn.Parameter(torch.Tensor(out_features))
        self.hyperbolic_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.hyperbolic_weight, gain=math.sqrt(2))
        init.constant_(self.hyperbolic_bias, 1)

    def forward(self, x):
        # weight = self.manifold.logmap0(self.hyperbolic_weight, self.c)
        # weight = self.manifold.proj(weight, self.c)
        drop_weight = F.dropout(self.hyperbolic_weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            # bias = self.manifold.proj_tan0(self.hyperbolic_bias.view(1, -1), self.c)
            # hyp_bias = self.manifold.expmap0(bias, self.c)
            # hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, self.hyperbolic_bias.view(1, -1), c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, edge_index):
        adj = edge_to_adj(edge_index, x)
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            adj_att = self.att(x_tangent, adj)
            support_t = torch.matmul(adj_att, x_tangent)
            del adj_att
        else:
            support_t = torch.spmm(adj, x_tangent)
        del adj
        # output = support_t
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        # xt = self.manifold.logmap0(x, c=self.c_in)
        # xt = x
        xt = self.act(x)
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        xt = self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)
        return xt

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class NodeSelect(Module):
    """
    Hyperbolic pooling layer.
    """

    def __init__(self, manifold, c_in, feat_in, feat_out, spread=1, bias=False):
        super(NodeSelect, self).__init__()
        self.manifold = manifold
        self.c = c_in
        self.bias = True
        self.ratio = 0.75
        self.layer_weight = nn.Linear(2 * feat_out, 1, bias=self.bias)
        self.aggregator = PROPAGATION_OUT()
        self.calc_information_score = NodeInformationScore()

    def forward(self, x, edge_index):
        x_tan = self.manifold.logmap0(x, c=self.c)
        edge_attr = None
        updated_x = x_tan
        sum_Neigh_x = self.aggregator(x_tan, edge_index)
        x_information_score = self.calc_information_score(updated_x, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)
        values, indices = score.topk(int(len(score) * self.ratio), dim=0, largest=True, sorted=True)
        T = torch.min(values)
        hot_prob = torch.where(score > T, torch.tensor(1).cuda(x.device), torch.tensor(0).cuda(x.device))
        SEL_v = hot_prob.view(-1, 1)

        self.L_flag = torch.zeros(1, 1).cuda(x.device) * 0
        self.L_flag = self.L_flag.float().view(-1, 1) + SEL_v

        # sum_Neigh_x = self.aggregator(updated_x, edge_index)
        #
        #  # SELECTION  <==============================================================
        # random_prob = F.relu(self.p_leader(sum_Neigh_x))
        # random_prob = F.softmax(random_prob, dim=-1)
        #
        # self.prob_i = random_prob[:, 1].unsqueeze(1)
        # hot_prob = torch.where(random_prob[:, 1] > 0.48, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        # SEL_v = hot_prob.view(-1, 1)
        # self.L_flag = torch.zeros(1, 1).cuda(x.device) * 0
        # self.L_flag = self.L_flag.float().view(-1, 1) + SEL_v

        #  SUMMATION + CONCAT <========================================================
        sum_SEL_x = self.aggregator(SEL_v * updated_x, edge_index)
        concat_sums = torch.cat([sum_SEL_x, sum_Neigh_x], dim=-1)

        #  WEIGHT             <========================================================
        weight_SEL_v = torch.sigmoid(self.layer_weight(concat_sums))
        A_x = F.relu(self.aggregator(weight_SEL_v * SEL_v * updated_x, edge_index))

        out = updated_x + A_x
        out = self.manifold.proj_tan0(out, c=self.c)
        out = self.manifold.proj(self.manifold.expmap0(out, self.c), self.c)
        return out

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAddDist(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout):
        super(HypAddDist, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.bias = True
        self.aggregator = PROPAGATION_OUT()
        self.layer_weight = nn.Linear(2 * in_features, 1, bias=self.bias)
        self.nodescore = NodeHyperbolicScore(self.manifold, self.c)
        # self.att = SetAttention()
        # self.att = torch.zeros(1, 1)
        # self.reset_parameters()

    def forward(self, x, edge_index):
        edge_attr = None
        sparse_distance = self.nodescore(x, edge_index, edge_attr)
        # score = torch.sum(torch.abs(sparse_distance), dim=1)
        score = torch.sum(sparse_distance, dim=1)
        values, indices = score.topk(int(len(score) * 0.75), dim=0, largest=True, sorted=True)
        T = torch.min(values)
        hot_prob = torch.where(score > T, torch.tensor(1).cuda(x.device), torch.tensor(0).cuda(x.device))
        SEL_v = hot_prob.view(-1, 1)

        self.L_flag = torch.zeros(1, 1).cuda(x.device) * 0
        self.L_flag = self.L_flag.float().view(-1, 1) + SEL_v

        #  SUMMATION + CONCAT <========================================================
        x_tan = self.manifold.logmap0(x, c=self.c)
        sum_SEL_x = self.aggregator(SEL_v * x_tan, edge_index)
        sum_Neigh_x = self.aggregator(x_tan, edge_index)
        concat_sums = torch.cat([sum_SEL_x, sum_Neigh_x], dim=-1)

        #  WEIGHT             <========================================================
        weight_SEL_v = torch.sigmoid(self.layer_weight(concat_sums))
        # att = self.att(weight_SEL_v)
        A_x = F.relu(self.aggregator(weight_SEL_v * SEL_v * x_tan, edge_index))

        out_tan = x_tan + A_x
        # out_tan = self.manifold.mobius_matvec(score, x, self.c)
        # out_tan = self.manifold.mobius_add(out_tan, x, self.c)
        # for i in sparse_distance:
        #     distance.append(F.softmax(sparse_distance[i]))
        # distance = torch.sum(distance, axis=1)
        out = self.manifold.proj_tan0(out_tan, c=self.c)
        out = self.manifold.proj(self.manifold.expmap0(out, self.c), self.c)
        return out

    def extra_repr(self):
        return 'c={}'.format(self.c)

class SetAttention(Module):
    """
    Hyperbolic Attention.
    """

    def __init__(self):
        super(SetAttention, self).__init__()
        self.att = nn.Parameter(torch.zeros(1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.att, gain=math.sqrt(2))

    def forward(self, att):
        self.att = att


class NodeHyperbolicScore(MessagePassing):
    def __init__(self, manifold, c, improved=False, cached=False, **kwargs):
        super(NodeHyperbolicScore, self).__init__(aggr='add', **kwargs)
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None
        self.manifold = manifold
        self.c = c

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)
        # hyperscore = self.manifold.hyp_proximity(expand_deg, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col], self.c)
        return edge_index, expand_deg, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, expand_deg, deg = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            norm = self.manifold.hyp_proximity(expand_deg, deg, self.c)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)
        #     norm1 = self.propagate(edge_index, x=x, norm=expand_deg)
        #     norm2 = self.propagate(edge_index, x=x, norm=deg)
        #     self.cached_result = edge_index, norm1, norm2
        #     #
        # edge_index, norm1, norm2 = self.cached_result
        # norm = self.manifold.hyp_proximity(norm1, norm2, self.c)
        # re = self.propagate(edge_index, x=x, norm=norm)
        # return re

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class PROPAGATION_OUT(MessagePassing):
    def __init__(self):
        super(PROPAGATION_OUT, self).__init__()

    def forward(self, x, edge_index): return self.propagate(edge_index, x=x)

    def message(self, x_j): return x_j

    def update(self, aggr_out): return aggr_out


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, **kwargs):
        super(GCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class HGCNN(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att):
        super(HGCNN, self).__init__()
        self.linear = HyperbolicLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.beta = HyperbolicLinear(manifold, 2 * out_features, 1, c_in, dropout, use_bias)
        # self.layer_weight = nn.Linear(2 * out_features, 1, bias=use_bias)
        self.manifold = manifold
        self.c = c_in
        self.calc_information_score = NodeInformationScore()
        self.ratio = 0.75
        self.aggregator = PROPAGATION_OUT()

    def forward(self, input):
        x, edge_index = input
        updated_x = self.linear.forward(x)
        #  Node Select <===============================================================
        edge_attr = None
        updated_tanx = self.manifold.logmap0(updated_x, c=self.c)
        x_information_score = self.calc_information_score(updated_tanx, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)
        values, indices = score.topk(int(len(score) * self.ratio), dim=0, largest=True, sorted=True)
        T = torch.min(values)
        hot_prob = torch.where(score > T, torch.tensor(1).cuda(x.device), torch.tensor(0).cuda(x.device))
        SEL_v = hot_prob.view(-1, 1)

        self.L_flag = torch.zeros(1, 1).cuda(x.device) * 0
        self.L_flag = self.L_flag.float().view(-1, 1) + SEL_v

        #  SUMMATION + CONCAT <========================================================
        sum_Neigh_x = self.aggregator(updated_tanx, edge_index)
        sum_SEL_x = self.aggregator(SEL_v * updated_tanx, edge_index)
        concat_sums = torch.cat([sum_SEL_x, sum_Neigh_x], dim=-1)
        hyp_sums = self.manifold.proj(self.manifold.expmap0(concat_sums, c=self.c), c=self.c)

        #  WEIGHT             <========================================================
        weight_SEL_v = torch.sigmoid(self.manifold.logmap0(self.beta(hyp_sums), c=self.c))
        # weight_SEL_v = torch.sigmoid(self.layer_weight(concat_sums))
        A_x = F.relu(self.aggregator(weight_SEL_v * SEL_v * updated_tanx, edge_index))
        out = updated_tanx + A_x
        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        return out


class HyperbolicLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HyperbolicLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        if self.use_bias:
            init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            # hyp_bias = self.bias.view(1, -1)
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HyperbolicAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att):
        super(HyperbolicAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        # x_tangent = self.manifold.logmap0(x, c=self.c)
        x_tangent = x
        if self.use_att:
            adj_att = self.att(x_tangent, adj)
            support_t = torch.matmul(adj_att, x_tangent)
            del adj_att
        else:
            support_t = torch.spmm(adj, x_tangent)
        # output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        output = support_t
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HyperbolicAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HyperbolicAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HyperbolicGraphConvolutionII(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att):
        super(HyperbolicGraphConvolutionII, self).__init__()
        self.linear = HypLinearII(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAggII(manifold, c_in, out_features, dropout, use_att)
        # self.agg2 = NodeSelect(manifold, c_in, out_features, out_features)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, edge_index = input
        adj = edge_to_adj(edge_index, x)
        theta = 0.25
        h, h0 = self.linear.forward(x)
        h = self.agg.forward(h, adj, h0, alpha=0.1, theta=theta)
        h = self.hyp_act.forward(h)
        output = h, edge_index
        return output


class HypLinearII(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinearII, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        res2 = F.dropout(res, p=0.1, training=self.training)
        return res, res2


class HypAggII(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att):
        super(HypAggII, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        # init.constant_(self.bias, 0)

    def forward(self, x, adj, h0, alpha, theta):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        hi = torch.spmm(adj, x_tangent)
        support = (1 - alpha) * hi + alpha * h0
        r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HGATv2(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att):
        super(HGATv2, self).__init__()
        self.heads = 1
        self.dropout = dropout
        self.manifold = manifold
        self.c = c_out
        self.lin_l = HypLinear(manifold, in_features, self.heads * out_features, c_in, dropout, use_bias)
        self.lin_r = HypLinear(manifold, in_features, self.heads * out_features, c_in, dropout, use_bias)
        self.hyp_act = HyperbolicAttention(manifold, out_features, c_in, c_out, dropout, use_bias, act)
        # self.linear1 = HypLinear(manifold, out_features * 2, out_features, c_in, dropout, use_bias)
        # self.linear2 = HypLinear(manifold, out_features, out_features, c_in, dropout, use_bias)

        self.out_channels = out_features

    def forward(self, input):
        H, C = self.heads, self.out_channels
        x, edge_index = input
        x_l = self.lin_l(x).view(-1, H, C)
        h = self.lin_r(x)
        x_r = h.view(-1, H, C)
        alpha = self.hyp_act.forward((x_l, x_r))
        return self.manifold.mobius_matvec(h, alpha, self.c)


class HyperbolicAttention(Module):
    """
        Hyperbolic graph convolution layer.
        """

    def __init__(self, manifold, out_features, c_in, c_out, dropout, use_bias, act):
        super(HyperbolicAttention, self).__init__()
        self.manifold = manifold
        self.c = c_in
        self.linear1 = HypLinear(manifold, out_features, out_features, c_in, dropout, use_bias)
        self.linear2 = HypLinear(manifold, out_features, out_features, c_in, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.out_channels = out_features

    def forward(self, input):
        x_l, x_r = input
        x_cat = self.manifold.mobius_add(x_l, x_r, c=self.c)
        x_cat = x_cat.squeeze()
        # x_cat = torch.cat((x_l, x_r), dim=2).squeeze()

        h = self.linear1.forward(x_cat)
        h = self.manifold.proj(self.manifold.expmap0(h, c=self.c), c=self.c)
        h = self.hyp_act.forward(h)
        h = self.linear2.forward(h)
        alpha = F.softmax(h)
        return alpha
