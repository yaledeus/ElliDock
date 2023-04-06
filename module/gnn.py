#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_softmax
from typing import List


def coord2radial(edges, coord):
    row, col = edges
    coord_diff = coord[row] - coord[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

    norm = torch.sqrt(radial) + 1e-8
    coord_diff = coord_diff / norm

    return radial, coord_diff


def coord2radialplus(edges, coord, scale: List):
    row, col = edges
    coord_diff = coord[row] - coord[col]    # (n_edges, 3)

    norm = torch.norm(coord_diff, dim=-1, keepdim=True) + 1e-8
    coord_diff = coord_diff / norm
    init_radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)    # (n_edges, 1)

    radial = []
    for s in scale:
        feat_dist = (-0.5 / (2.25 ** s) * init_radial).exp()    # (n_edges, 1)
        radial.append(feat_dist)

    radial = torch.cat(radial, dim=-1)  # (n_edges, *)

    return radial, coord_diff


def vec2product(edges, vec):
    # vec: [N, *, 3]
    row, col = edges
    n_edges = edges.shape[1]
    product = torch.mul(vec[row], vec[col])     # (n_edges, *, 3)
    product = product.sum(dim=-1).reshape(n_edges, -1)  # (n_edges, *)
    return product


def coord2nforce(edges, coord, order: List):
    row, col = edges
    coord_diff = coord[row] - coord[col]    # (n_edges, 3)
    eps = 1e-8
    nvecs = []

    for alpha in order:
        ordered_norm = torch.norm(coord_diff, dim=-1, keepdim=True).pow(-1 - alpha) # (n_edges, 1)
        ordered_diff = torch.mul(ordered_norm, coord_diff)  # (n_edges, 3)

        nforce = unsorted_segment_mean(ordered_diff, row, num_segments=coord.shape[0])  # (N, 3)
        # normalize
        nforce = nforce / (torch.norm(nforce, dim=-1, keepdim=True) + eps)
        nvecs.append(nforce)

    nvecs = torch.stack(nvecs, dim=1)  # (N, *, 3)
    return nvecs


### find k satisfies i->j, i->k, j->k
@torch.no_grad()
def find_triangular(edges):
    source, target = edges
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

# ## test
# edges = torch.tensor([[0, 0, 0, 1, 1, 1],
#                       [1, 2, 3, 2, 3, 4]])
# print(find_triangular_cython(edges.numpy()))


### chemistry && position pairwise energy GCL
class Ch_Pos_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d, act_fn=nn.SiLU(), dropout=0.1, attention=True):
        super(Ch_Pos_GCL, self).__init__()
        self.attention = attention
        input_edge = input_nf * 2 + hidden_nf * 2   # v_i, v_j, h_i, h_j
        edge_coords_nf = 20          # <n_i, n_j> with order = {2,3,4,5,6}, d_ij with scale {1.5^x|x=0,1,...,14}

        self.dropout = nn.Dropout(dropout)

        self.ch_edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        self.pos_edge_mlp = nn.Sequential(
            nn.Linear(edge_coords_nf + edges_in_d, hidden_nf),
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
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

    def forward(self, h, coord, edges, nvecs, node_attr, edge_attr=None):
        row, col = edges
        # radial, coord_diff = coord2radial(edges, coord)
        radial, coord_diff = coord2radialplus(edges, coord, scale=[s for s in range(15)])
        nprod = vec2product(edges, nvecs)   # (n_edges, *)

        if edge_attr is None:
            chem = torch.cat([h[row], h[col], node_attr[row], node_attr[col]], dim=1)
            pos = torch.cat([nprod, radial], dim=1)
        else:
            chem = torch.cat([h[row], h[col], node_attr[row], node_attr[col], edge_attr], dim=1)
            pos = torch.cat([nprod, radial, edge_attr], dim=1)

        chem = self.ch_edge_mlp(chem)
        pos = self.pos_edge_mlp(pos)

        out = torch.mul(self.shallow_mlp(chem), pos)    # (n_edge, hidden_nf)

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
            torch.nn.init.xavier_uniform_(self._modules[f'linear_q{h}'].weight)
            torch.nn.init.xavier_uniform_(self._modules[f'linear_k{h}'].weight)
            torch.nn.init.xavier_uniform_(self._modules[f'linear_v{h}'].weight)
            torch.nn.init.xavier_uniform_(self._modules[f'linear_b{h}'].weight)

        self.linear_out = nn.Linear(self.att_heads * hidden_nf, output_nf)
        torch.nn.init.xavier_uniform_(self.linear_out.weight)


    def forward(self, Z, klist):
        """
        :param Z: (n_edges, input_nf)
        :param klist: (n_edges, 2, MAX_K)
        :return:
        """
        output = []

        MAX_K = klist.shape[2]
        idx_i2k, idx_j2k = klist[:, 0], klist[:, 1]  # (n_edges, MAX_K)
        redundant = idx_i2k == -1  # (n_edges, MAX_K)

        for h in range(self.att_heads):
            q = self._modules[f'linear_q{h}'](Z)    # (n_edges, hidden_nf)
            k = self._modules[f'linear_k{h}'](Z)    # (n_edges, hidden_nf)
            v = self._modules[f'linear_v{h}'](Z)    # (n_edges, hidden_nf)
            b = self._modules[f'linear_b{h}'](Z)    # (n_edges, 1)

            g = self._modules[f'mlp_g{h}'](Z)       # (n_edges, hidden_nf)

            tri_k = k[idx_i2k]  # (n_edges, MAX_K, hidden_nf)
            tri_b = b[idx_j2k]  # (n_edges, MAX_K, 1)
            tri_v = v[idx_i2k]  # (n_edges, MAX_K, hidden_nf)

            # set redundant node k's value to 0
            tri_k[redundant] = 0.
            tri_b[redundant] = 0.

            alpha_ijk = 1. / np.sqrt(self.hidden_nf) * torch.sum(q.unsqueeze(1).repeat(1, MAX_K, 1)
                                                                 * tri_k, dim=-1) \
                        + tri_b.squeeze(2)  # (n_edges, MAX_K)
            alpha_ijk[redundant] = -1e4 # set redundant node k's attention value to 0
            alpha_ijk = F.softmax(alpha_ijk, dim=1) # (n_edges, MAX_K)
            tri_att_val = torch.matmul(alpha_ijk.unsqueeze(1), tri_v).squeeze() # (n_edges, hidden_nf)

            output.append(torch.mul(g, tri_att_val))

            # free memory
            del alpha_ijk
            torch.cuda.empty_cache()


        output = torch.cat(output, dim=-1)
        output = self.linear_out(output)        # (n_edges, output_nf)
        output = self.dropout(output)

        return output


### Transformer gcl
class Transformer_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, att_heads, dropout=0.1):
        super(Transformer_GCL, self).__init__()
        self.hidden_nf = hidden_nf
        self.att_heads = att_heads

        self.dropout = nn.Dropout(dropout)

        for h in range(self.att_heads):
            self.add_module(f'linear_q{h}', nn.Linear(input_nf, hidden_nf, bias=False))
            self.add_module(f'linear_k{h}', nn.Linear(input_nf, hidden_nf, bias=False))
            self.add_module(f'linear_v{h}', nn.Linear(input_nf, hidden_nf, bias=False))
            self.add_module(f'mlp_g{h}', nn.Sequential(
                nn.Linear(input_nf, hidden_nf),
                nn.Sigmoid()
            ))
            torch.nn.init.xavier_uniform_(self._modules[f'linear_q{h}'].weight)
            torch.nn.init.xavier_uniform_(self._modules[f'linear_k{h}'].weight)
            torch.nn.init.xavier_uniform_(self._modules[f'linear_v{h}'].weight)

        self.linear_out = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            nn.GELU(),
            nn.Linear(hidden_nf, output_nf)
        )


    def forward(self, Z, edges):
        """
        :param Z: (n_edges, input_nf)
        :param edges: (2, n_edges)
        :return:
        """
        row, col = edges
        z_update = torch.zeros_like(Z)

        for h in range(self.att_heads):
            q = self._modules[f'linear_q{h}'](Z)    # (n_edges, hidden_nf)
            k = self._modules[f'linear_k{h}'](Z)    # (n_edges, hidden_nf)
            v = self._modules[f'linear_v{h}'](Z)    # (n_edges, hidden_nf)
            g = self._modules[f'mlp_g{h}'](Z)       # (n_edges, hidden_nf)

            Att = 1. / np.sqrt(self.hidden_nf) * (q[:, None, :] @ k[:, :, None]).squeeze()  # (n_edges)
            Att = scatter_softmax(Att, row).unsqueeze(1)     # (n_edges, 1)
            z_update += Att * v * g                 # (n_edges, hidden_nf)

        output = Z + z_update   # residual, (n_edges, output_nf)
        output = output + self.linear_out(output)
        output = self.dropout(output)

        return output


### Pairwise energy && Triangular self-Attention EGNN
class PTA_EGNN(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d,
                 att_heads=4, act_fn=nn.SiLU(), dropout=0.1):
        super(PTA_EGNN, self).__init__()

        self.hidden_nf = hidden_nf

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

        self.phi_h = nn.Sequential(
            nn.Linear(hidden_nf * 2 + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )

    def forward(self, h, coord, edges, nvecs, edge_attr, node_attr, init_coord, klist):

        # z: (n_edge, hidden_nf), coord_diff: (n_edge, 3)
        z, chem, pos, coord_diff = self.ch_pos_gcl(h, coord, edges, nvecs, node_attr, edge_attr)
        # m: (n_edge, hidden_nf)
        m = self.tri_att_gcl(z, klist)

        # update coordinates
        ita = .2    # weight for initial coordinate
        row, col = edges
        x_trans = coord_diff * torch.mul(self.phi_u(chem), self.phi_x(pos))
        x_agg = unsorted_segment_mean(x_trans, row, num_segments=coord.shape[0])    # (N, 3)
        coord = ita * init_coord + (1 - ita) * coord + x_agg

        # update h
        m_agg = unsorted_segment_mean(m, row, num_segments=coord.shape[0])  # (N, hidden_nf)
        m_all = torch.cat([h, node_attr, m_agg], dim=-1)
        # residual
        h = h + self.phi_h(m_all)

        return h, coord


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
