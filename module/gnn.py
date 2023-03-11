#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch


def coord2radial(edges, coord):
    row, col = edges[0], edges[1]
    coord_diff = coord[row] - coord[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

    norm = torch.sqrt(radial) + 1e-8
    coord_diff = coord_diff / norm

    return radial, coord_diff


def vec2product(edges, vec):
    row, col = edges[0], edges[1]
    product = torch.mul(vec[row], vec[col])     # (n_edge, 3)
    product = product.sum(dim=-1).reshape(-1, 1)    # (n_edge, 1)
    return product


### find k satisfies i->j, i->k, j->k
@torch.no_grad()
def find_triangular(edges):
    source, target = edges[0, :], edges[1, :]
    klist = []
    for idx in range(edges.shape[1]):
        k = []
        i, j = source[idx], target[idx]     # i, j: nodes of No.idx edge
        edge_starts_with_i = torch.where(source == i)[0]
        edge_starts_with_j = torch.where(source == j)[0]
        i_ends = target[edge_starts_with_i]
        j_ends = target[edge_starts_with_j]
        i_ends = i_ends.unsqueeze(1).repeat(1, edge_starts_with_j.shape[0])
        j_ends = j_ends.unsqueeze(1).repeat(1, edge_starts_with_i.shape[0]).T
        idx_pairs = torch.nonzero(i_ends == j_ends)
        for pair in idx_pairs:
            k.append([edge_starts_with_i[pair[0]], edge_starts_with_j[pair[1]]])
        klist.append(k)
    return klist

### test
# edges = torch.tensor([[0, 0, 0, 1, 1, 1],
#                       [1, 2, 3, 2, 3, 4]])
# print(find_triangular(edges))


### chemistry && position pairwise energy GCL
class Ch_Pos_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d, act_fn=nn.SiLU(), dropout=0.1, attention=True):
        super(Ch_Pos_GCL, self).__init__()
        self.attention = attention
        input_edge = input_nf * 2 + hidden_nf * 2   # v_i, v_j, h_i, h_j
        edge_coords_nf = 2          # <n_i, n_j>, d_ij

        self.dropout = nn.Dropout(dropout)

        self.ch_edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        self.pos_edge_mlp = nn.Sequential(
            nn.Linear(edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        self.shallow_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, h, coord, edges, nvecs, node_attr, edge_attr=None):
        row, col = edges[0], edges[1]
        radial, coord_diff = coord2radial(edges, coord)
        nprod = vec2product(edges, nvecs)

        if edge_attr is None:
            chem = torch.cat([h[row], h[col], node_attr[row], node_attr[col]], dim=1)
        else:
            chem = torch.cat([h[row], h[col], node_attr[row], node_attr[col], edge_attr], dim=1)
        chem = self.ch_edge_mlp(chem)

        pos = torch.cat([nprod, radial], dim=1)
        pos = self.pos_edge_mlp(pos)

        out = torch.mul(self.shallow_mlp(chem), pos)   # (n_edge, hidden_nf)

        out = self.dropout(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out, chem, pos, coord_diff


### Triangular self-attention gcl
class Tri_Att_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, att_heads, dropout=0.1):
        super(Tri_Att_GCL, self).__init__()
        self.hidden_nf = hidden_nf
        self.att_heads = att_heads

        self.dropout = nn.Dropout(dropout)

        for h in range(self.att_heads):
            self.add_module(f'linear_q{h}', nn.Linear(input_nf, hidden_nf, bias=False))
            self.add_module(f'linear_k{h}', nn.Linear(input_nf, hidden_nf, bias=False))
            self.add_module(f'linear_v{h}', nn.Linear(input_nf, hidden_nf, bias=False))
            self.add_module(f'linear_b{h}', nn.Linear(input_nf, 1, bias=False))
            self.add_module(f'mlp_g{h}', nn.Sequential(
                nn.Linear(input_nf, hidden_nf),
                nn.Sigmoid()
            ))

        self.linear_out = nn.Linear(self.att_heads * hidden_nf, output_nf)


    def forward(self, Z, edges):
        """
        :param Z: (n_edges, input_nf)
        :param edge_index: (2, E)
        :return:
        """
        row, col = edges[0], edges[1]
        device = Z.device
        output = []

        klist = find_triangular(edges)

        for h in range(self.att_heads):
            q = self._modules[f'linear_q{h}'](Z)    # (n_edges, hidden_nf)
            k = self._modules[f'linear_k{h}'](Z)    # (n_edges, hidden_nf)
            v = self._modules[f'linear_v{h}'](Z)    # (n_edges, hidden_nf)
            b = self._modules[f'linear_b{h}'](Z)    # (n_edges, 1)

            g = self._modules[f'mlp_g{h}'](Z)       # (n_edges, hidden_nf)

            tri_att_val = torch.zeros_like(g)

            for idx in range(edges.shape[1]):
                num_k = len(klist[idx])
                if num_k == 0:  # no triangular found for node i and j
                    tri_att_val[idx, :] = 1.
                    continue
                alpha_ijk = []
                for tri_k in klist[idx]:
                    idx_i2k, idx_j2k = tri_k[0], tri_k[1]
                    alpha_ijk.append(1. / np.sqrt(self.hidden_nf) * torch.dot(q[idx], k[idx_i2k]) + b[idx_j2k])
                alpha_ijk = F.softmax(torch.tensor(alpha_ijk, device=device), dim=-1)
                for i, tri_k in enumerate(klist[idx]):
                    idx_i2k = tri_k[0]
                    tri_att_val += alpha_ijk[i] * v[idx_i2k]

            output.append(torch.mul(g, tri_att_val))

        del klist   # release memory

        output = torch.cat(output, dim=-1)
        output = self.linear_out(output)        # (n_edges, output_nf)
        output = self.dropout(output)

        return output


### Pairwise energy && Triangular self-Attention EGNN
class PTA_EGNN(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d,
                 att_heads=4, act_fn=nn.SiLU(), n_layers=4, dropout=0.1):
        super(PTA_EGNN, self).__init__()

        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.ch_pos_gcl = Ch_Pos_GCL(
            input_nf, hidden_nf, hidden_nf, edges_in_d,
            act_fn, dropout
        )

        self.tri_att_gcl = Tri_Att_GCL(
            hidden_nf, hidden_nf, hidden_nf, att_heads,
            dropout
        )

        self.phi_u = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )

        self.phi_x = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )

        self.phi_n = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )

        self.phi_h = nn.Sequential(
            nn.Linear(hidden_nf * 2 + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )

    def forward(self, h, coord, edges, nvecs, edge_attr, node_attr, init_coord, init_nvecs):
        # z, chem, pos: (n_edge, hidden_nf), coord_diff: (n_edge, 3)
        z, chem, pos, coord_diff = self.ch_pos_gcl(h, coord, edges, nvecs, node_attr, edge_attr)
        # m: (n_edge, hidden_nf)
        m = self.tri_att_gcl(z, edges)

        # update coordinates
        ita = .2    # weight
        row, col = edges[0], edges[1]
        x_trans = coord_diff * np.mul(self.phi_u(chem), self.phi_x(pos))
        x_agg = unsorted_segment_mean(x_trans, row, num_segments=coord.shape[0])    # (N, 3)
        coord = ita * init_coord + (1 - ita) * coord + x_agg

        # update normal vectors
        gamma = .2  # weight
        n_trans = nvecs * np.mul(self.phi_u(chem), self.phi_n(pos))
        n_agg = unsorted_segment_mean(n_trans, row, num_segments=coord.shape[0])    # (N, 3)
        nvecs = gamma * init_nvecs + (1 - gamma) * nvecs + n_agg

        # update h
        beta = .2   # weight
        m_agg = unsorted_segment_mean(m, row, num_segments=coord.shape[0])  # (N, 3)
        m_all = torch.cat([h, node_attr, m_agg], dim=-1)
        h = beta * h + (1 - beta) * self.phi_h(m_all)
        return h, coord, nvecs


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr
