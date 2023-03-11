#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch import nn

import sys
sys.path.append('..')
from data.bio_parse import AA_NAMES_1
from data.geometry import get_backbone_dihedral_angles, pairwise_dihedrals


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def _construct_knn(X, rule_mats, k_neighbors):
    '''
    :param X: [N, n_channel, 3], coordinates
    :param rule_mats: list of [N, N], valid edges after each filtering
    :param k_neighbors: neighbors of each node
    '''
    src_dst = torch.nonzero(sequential_and(*rule_mats))  # [Ef, 2], full possible edges represented in (src, dst)
    BIGINT = 1e10  # assign a large distance to invalid edges
    N = X.shape[0]
    dist = X[src_dst]  # [Ef, 2, n_channel, 3]
    dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
    dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
    dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]
    dist = (torch.ones(N, N, device=dist.device) * BIGINT).index_put_(tuple([k for k in src_dst]), dist)
    # dist_neighbors: [N, topk], dst: [N, topk]
    dist_neighbors, dst = torch.topk(dist, k_neighbors, dim=-1, largest=False)  # [N, topk]
    del dist  # release memory
    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)
    edges = torch.stack([dst, src])  # direction denotes the message passing direction
    return edges  # [2, E]


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings


class AminoAcidEmbedding(nn.Module):
    '''
    [residue embedding, position embedding]
    '''

    def __init__(self, num_res_type, num_res_id, res_embed_size):
        super().__init__()
        self.residue_embedding = nn.Embedding(num_res_type, res_embed_size)
        self.res_pos_embedding = SinusoidalPositionEmbedding(res_embed_size)  # relative positional encoding
        self.res_id_embedding = nn.Embedding(num_res_id, res_embed_size)

    def forward(self, S, RP, ID):
        '''
        :param S: [N], residue types
        :param RP: [N], residue positions
        :param ID: [N], chain identifier, antibody: H/L, antigen: *
        '''
        res_embed = self.residue_embedding(S)       # (N, res_embed_size)
        rp_embed  = self.res_pos_embedding(RP) + self.res_id_embedding(ID)
        return torch.cat([res_embed, rp_embed], dim=-1)  # (N, res_embed_size * 2)


class ComplexGraph(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        self.num_aa_type = len(AA_NAMES_1)
        self.embed_size = embed_size

        self.aa_embedding = AminoAcidEmbedding(self.num_aa_type, 2, embed_size)

    def dihedral_embedding(self, X):
        return get_backbone_dihedral_angles(X)

    def embedding(self, X, S, RP, ID):
        H = self.aa_embedding(S, RP, ID)
        H = torch.cat(H, self.dihedral_embedding(X))
        return H

    @torch.no_grad()
    def construct_edges(self, X, Seg, bid, k_neighbors):
        N = bid.shape[0]
        same_bid = bid.unsqueeze(-1).repeat(1, N)
        same_seg = Seg.unsqueeze(-1).repeat(1, N)
        same_bid = same_bid == same_bid.transpose(0, 1) # (N, N)
        same_seg = same_seg == same_seg.transpose(0, 1)

        edges = _construct_knn(
            X,
            [same_bid, same_seg],
            k_neighbors
        )

        return edges

    def pairwise_dihedral_embedding(self, X, edges):
        return pairwise_dihedrals(X, edges)         # (n_edge, 2)


    def forward(self, X, S, RP, ID, Seg, bid, k_neighbors):
        H = self.embedding(X, S, RP, ID)                            # (N, 2 * embed_size + 3)
        edges = self.construct_edges(X, Seg, bid, k_neighbors)      # (2, n_edge)
        edge_attr = pairwise_dihedrals(X, edges)                    # (n_edge, 2)
        return H, edges, edge_attr
