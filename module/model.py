#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from e3nn import o3
from torch import nn
import torch.nn.functional as F
import torch

import sys
sys.path.append('..')
from data.bio_parse import CA_INDEX
from data.geometry import *
from .embed import ComplexGraph
from .gnn import PTA_EGNN, TPCL
from utils.logger import print_log


class ExpDock(nn.Module):
    def __init__(self, embed_size, hidden_size, k_neighbors=9,
                 att_heads=4, n_layers=4, n_keypoints=10, dropout=0.1,
                 mean=None, std=None) -> None:
        super(ExpDock, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_keypoints = n_keypoints
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)

        node_in_d, edge_in_d = ComplexGraph.feature_dim(embed_size)

        self.graph_constructor = ComplexGraph(embed_size, k_neighbors)

        self.in_conv = TPCL(
            o3.Irreps(f"{node_in_d - 9}x0e + 3x1e"),
            self.sh_irreps,
            o3.Irreps(f"{self.hidden_size}x0e"),
            edge_in_d,
            self.hidden_size
        )

        for i in range(self.n_layers):
            self.add_module(f'gnn_{i}', PTA_EGNN(
                node_in_d, hidden_size, hidden_size, edge_in_d, att_heads=att_heads,
                act_fn=nn.SiLU(), dropout=dropout
                )
            )
            self.add_module(f'inter_act_{i}', nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.hidden_size)
            ))

        for i in range(self.n_keypoints):
            self.register_parameter(f'w1_mat_{i}', nn.Parameter(torch.rand(hidden_size, hidden_size)))
            self.register_parameter(f'w2_mat_{i}', nn.Parameter(torch.rand(hidden_size, hidden_size)))

        self._init()

        self.normalizer = CoordNomralizer(mean=mean, std=std)

    def _init(self):
        for name, param in self.named_parameters():
            # bias terms
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            # weight terms
            else:
                nn.init.xavier_normal_(param)

    def forward(self, X, S, RP, ID, Seg, center, keypoints, bid, k_bid):
        device = X.device
        bs = bid[-1] + 1

        # center to ligand and normalize
        X = self.normalizer.centering(X, center, bid)
        keypoints = self.normalizer.centering(keypoints, center, k_bid)

        # clone original X
        ori_X = X[:, CA_INDEX].clone().detach()
        # clone keypoints for receptor after transformation in R3
        keypoints = keypoints.float().detach()
        trans_keypoints = keypoints.clone()

        # rotate and translate for antibodies, X' = XR + t
        rot, trans = sample_transformation(bid)
        for i in range(bs):
            ab_idx = torch.logical_and(Seg == 0, bid == i)
            X[ab_idx] = X[ab_idx] @ rot[i] + trans[i] * torch.mean(self.normalizer.std)
            trans_keypoints[k_bid == i] = trans_keypoints[k_bid == i] @ rot[i] + trans[i] * torch.mean(self.normalizer.std)

        X = self.normalizer.normalize(X).float()
        ori_X = self.normalizer.normalize(ori_X).float()
        keypoints = self.normalizer.normalize(keypoints).float()
        trans_keypoints = self.normalizer.normalize(trans_keypoints).float()

        node_attr, edges, edge_attr = self.graph_constructor(
            X, S, RP, Seg, bid
        )

        src, dst = edges

        # CA atoms only
        X = X[:, CA_INDEX]      # (N, 3)
        init_X = X.clone()

        coord_diff = init_X[src] - init_X[dst]  # (n_edges, 3)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, coord_diff,
                                         normalize=True, normalization='component')  # (n_edges, 9)

        H = self.in_conv(
            node_attr, edges, edge_attr, edge_sh
        )   # (N, hidden_size)

        for i in range(self.n_layers):
            UH, X = self._modules[f'gnn_{i}'](
                H, X, edges, edge_attr, node_attr, init_X
            )
            for b_ind in range(bs):
                ab_idx = torch.logical_and(Seg == 0, bid == b_ind)
                ag_idx = torch.logical_and(Seg == 1, bid == b_ind)
                H[ab_idx] = UH[ab_idx] + self._modules[f'inter_act_{i}'](
                    UH[ab_idx] * F.softmax((UH[ab_idx] @ UH[ag_idx].T).mean(dim=1),
                                           dim=0).unsqueeze(1)
                )
                H[ag_idx] = UH[ag_idx] + self._modules[f'inter_act_{i}'](
                    UH[ag_idx] * F.softmax((UH[ag_idx] @ UH[ab_idx].T).mean(dim=1),
                                           dim=0).unsqueeze(1)
                )

        # optimal transport loss & keypoint loss & dock loss & rmsd loss &
        # stable loss & match loss
        ot_loss = 0.
        dock_loss = 0.
        rmsd_loss = 0.
        stable_loss = 0.
        match_loss = 0.
        # threshold (loose)
        threshold = 8. / torch.mean(self.normalizer.std)

        for i in range(bs):
            ab_idx = torch.logical_and(Seg == 0, bid == i)
            ag_idx = torch.logical_and(Seg == 1, bid == i)
            H1, H2 = H[ab_idx], H[ag_idx]   # H1: (N, hidden_size) H2: (M, hidden_size)
            # refined_H1 = constrain_refine_hidden_space(H1, H2, ori_X[ab_idx], ori_X[ag_idx])
            # balance_loss += F.kl_div(F.log_softmax(H1, dim=1), F.softmax(refined_H1, dim=1), reduction="batchmean")
            X1, X2 = X[ab_idx], X[ag_idx]   # X1: (N, 3) X2: (M, 3)
            Y1 = [F.softmax(1. / np.sqrt(self.hidden_size) *
                            (H1 @ self._parameters[f'w1_mat_{j}'] @ H2.T)
                             .mean(dim=1), dim=0).unsqueeze(0) @ X1
                  for j in range(self.n_keypoints)]
            Y2 = [F.softmax(1. / np.sqrt(self.hidden_size) *
                            (H2 @ self._parameters[f'w2_mat_{j}'] @ H1.T)
                            .mean(dim=1), dim=0).unsqueeze(0) @ X2
                  for j in range(self.n_keypoints)]
            Y1 = torch.cat(Y1, dim=0)
            Y2 = torch.cat(Y2, dim=0)
            P1, P2 = trans_keypoints[k_bid == i], keypoints[k_bid == i]
            D1 = torch.cdist(Y1, P1)    # (K, S)
            min_indices_1 = torch.argmin(D1, dim=1)     # (K,)
            del D1  # free memory
            ot_loss += F.mse_loss(Y1, P1[min_indices_1])
            D2 = torch.cdist(Y2, P2)    # (K, S)
            min_indices_2 = torch.argmin(D2, dim=1)     # (K,)
            ot_loss += F.mse_loss(Y2, P2[min_indices_2])
            del D2  # free memory
            torch.cuda.empty_cache()
            # compute dock loss
            # _, R1, t1 = quadratic_surface_fit(Y1)
            # _, R2, t2 = quadratic_surface_fit(Y2)
            # R = R1 @ R2.T
            # t = (t1 - t2) @ R2.T
            _, R, t = kabsch_torch(Y1, Y2)
            ct = self.normalizer.mean / torch.mean(self.normalizer.std)
            dock_loss += F.mse_loss(rot[i] @ R, torch.eye(3).to(device))
            dock_loss += F.mse_loss((trans[i] - ct) @ rot[i].T + t + ct, torch.zeros(3).to(device))
            # compute stable loss
            stable_loss += F.softplus(-max_triangle_area(Y1))
            stable_loss += F.softplus(-max_triangle_area(Y2))
            stable_loss += 1. / (torch.mean(self.normalizer.std) ** 2) * \
                           protein_surface_intersection(init_X[ab_idx] * torch.mean(self.normalizer.std),
                                                    Y1 * torch.mean(self.normalizer.std)).clip(min=0.).mean()
            stable_loss += 1. / (torch.mean(self.normalizer.std) ** 2) * \
                           protein_surface_intersection(init_X[ag_idx] * torch.mean(self.normalizer.std),
                                                    Y2 * torch.mean(self.normalizer.std)).clip(min=0.).mean()
            # compute match loss
            D_abag = torch.cdist(ori_X[ab_idx], ori_X[ag_idx])  # (N, M)
            ab_dock_scope, ag_dock_scope = torch.where(D_abag < threshold)
            ab_irr_scope, ag_irr_scope = torch.where(D_abag >= threshold)
            match_loss += F.smooth_l1_loss(
                # (H1[ab_irr_scope][:, None, :] @ H2[ag_irr_scope][:, :, None]).squeeze(),
                # torch.ones_like(ab_irr_scope) * -1
                F.softplus(H1[ab_irr_scope][:, None, :] @ H2[ag_irr_scope][:, :, None]).squeeze(),
                torch.zeros_like(ab_irr_scope)
            )
            match_loss += F.smooth_l1_loss(
                # (H1[ab_dock_scope][:, None, :] @ H2[ag_dock_scope][:, :, None]).squeeze(),
                # torch.ones_like(ab_dock_scope)
                F.softplus(-H1[ab_dock_scope][:, None, :] @ H2[ag_dock_scope][:, :, None]).squeeze(),
                torch.zeros_like(ab_dock_scope)
            )
            # compute rmsd loss
            X1_aligned = init_X[ab_idx] @ R + t   # (N, 3)
            rmsd_loss += F.mse_loss(X1_aligned, ori_X[ab_idx])

        # normalize
        ot_loss /= len(rot)
        dock_loss /= len(rot)
        stable_loss /= len(rot)
        match_loss /= len(rot)
        rmsd_loss /= len(rot)

        bind_loss = ot_loss * dock_loss

        loss = ot_loss + dock_loss + stable_loss + match_loss

        return loss, (ot_loss, dock_loss, bind_loss, stable_loss, match_loss, rmsd_loss)

    def dock(self, X, S, RP, ID, Seg, center, bid, **kwargs):
        device = X.device
        bs = bid[-1] + 1
        # center to antigen
        X = self.normalizer.centering(X, center, bid)
        # normalize X to approximately normal distribution
        X = self.normalizer.normalize(X).float()

        node_attr, edges, edge_attr = self.graph_constructor(
            X, S, RP, Seg, bid
        )

        src, dst = edges

        # CA atoms only
        X = X[:, CA_INDEX]  # (N, 3)
        init_X = X.clone()

        coord_diff = init_X[src] - init_X[dst]  # (n_edges, 3)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, coord_diff,
                                         normalize=True, normalization='component')  # (n_edges, 9)

        H = self.in_conv(
            node_attr, edges, edge_attr, edge_sh
        )  # (N, hidden_size)

        for i in range(self.n_layers):
            UH, X = self._modules[f'gnn_{i}'](
                H, X, edges, edge_attr, node_attr, init_X
            )
            for b_ind in range(bs):
                ab_idx = torch.logical_and(Seg == 0, bid == b_ind)
                ag_idx = torch.logical_and(Seg == 1, bid == b_ind)
                H[ab_idx] = UH[ab_idx] + self._modules[f'inter_act_{i}'](
                    UH[ab_idx] * F.softmax((UH[ab_idx] @ UH[ag_idx].T).mean(dim=1),
                                           dim=0).unsqueeze(1)
                )
                H[ag_idx] = UH[ag_idx] + self._modules[f'inter_act_{i}'](
                    UH[ag_idx] * F.softmax((UH[ag_idx] @ UH[ab_idx].T).mean(dim=1),
                                           dim=0).unsqueeze(1)
                )

        dock_trans_list = []

        for i in range(bs):
            ab_idx = torch.logical_and(Seg == 0, bid == i)
            ag_idx = torch.logical_and(Seg == 1, bid == i)
            H1, H2 = H[ab_idx], H[ag_idx]
            X1, X2 = X[ab_idx], X[ag_idx]
            Y1 = [F.softmax(1. / np.sqrt(self.hidden_size) *
                            (H1 @ self._parameters[f'w1_mat_{j}'] @ H2.T)
                            .mean(dim=1), dim=0).unsqueeze(0) @ X1
                  for j in range(self.n_keypoints)]
            Y2 = [F.softmax(1. / np.sqrt(self.hidden_size) *
                            (H2 @ self._parameters[f'w2_mat_{j}'] @ H1.T)
                            .mean(dim=1), dim=0).unsqueeze(0) @ X2
                  for j in range(self.n_keypoints)]
            Y1 = torch.cat(Y1, dim=0)
            Y2 = torch.cat(Y2, dim=0)
            # _, R1, t1 = quadratic_surface_fit(Y1)
            # _, R2, t2 = quadratic_surface_fit(Y2)
            # R = R1 @ R2.T
            # t = (t1 - t2) @ R2.T
            _, R, t = kabsch_torch(Y1, Y2)
            init_X[ab_idx] = init_X[ab_idx] @ R + t
            dock_trans_list.append(self.normalizer.dock_transformation(center, i, R, t))

        init_X = self.normalizer.unnormalize(init_X)
        init_X = self.normalizer.uncentering(init_X, center, bid)

        return init_X, dock_trans_list


def constrain_refine_hidden_space(H_r, H_l, X_r, X_l):
    """
    :param H_r: receptor hidden space, (N, K)
    :param H_l: ligand hidden space, (M, K)
    :param X_r: groundtruth receptor coordinate, (N, 3)
    :param X_l: groundtruth ligand coordinate, (M, 3)
    :return: refined_H_r, (N, K)
    """
    N = H_r.shape[0]
    device = H_r.device
    receptor_center = torch.mean(X_r, dim=0)    # (3,)
    ligand_center = torch.mean(X_l, dim=0)      # (3,)
    delta_X = ligand_center - X_r               # (N, 3)
    H_l_mean = torch.mean(H_l, dim=0)           # (K,)
    init_prod_H = (H_r @ H_l_mean).unsqueeze(1) # (N, 1)
    A = torch.norm(delta_X, dim=1).pow(-3)[:, None] * delta_X   # (N, 3)
    Force = init_prod_H * A                     # (N, 3)
    # F_U: (N, 3), F_S: (3,), F_Vh: (3, 3)
    F_U, F_S, F_Vh = torch.linalg.svd(Force, full_matrices=False)
    refined_F_S = F_S.clone()
    refined_F_S[-1] = 0. # set the last singular value equals to 0
    refined_F_U = recon_orthogonal_matrix(F_U, torch.ones(N).to(device))
    R2X = X_r - receptor_center     # (N, 3)
    Ra = torch.vstack([
        torch.hstack([torch.zeros(N).to(device), -R2X[:, 2].T, R2X[:, 1].T]),
        torch.hstack([R2X[:, 2].T, torch.zeros(N).to(device), -R2X[:, 0].T]),
        torch.hstack([-R2X[:, 1].T, R2X[:, 0].T, torch.zeros(N).to(device)])
    ])  # (3, 3N)
    T = torch.hstack([
        Ra[:, :N] @ refined_F_U @ torch.diag(refined_F_S),
        Ra[:, N:2*N] @ refined_F_U @ torch.diag(refined_F_S),
        Ra[:, 2*N:] @ refined_F_U @ torch.diag(refined_F_S)
    ])  # (3, 9)
    # T_U: (3, 3), T_S: (3,), T_Vh: (3, 9)
    T_U, T_S, T_Vh = torch.linalg.svd(T, full_matrices=False)
    concat_vh = torch.hstack([F_Vh[:, 0], F_Vh[:, 1], F_Vh[:, 2]])  # (9,)
    # project Vh to zero space of T
    proj_vh = (torch.eye(9).to(device) - T_Vh.T @ T_Vh) @ concat_vh  # (9,)
    refined_F_Vh = torch.vstack([proj_vh[:3], proj_vh[3:6], proj_vh[6:]]).T  # (3, 3)
    # refined_F_Vh = modified_gram_schmidt(refined_F_Vh)  # (3, 3)
    S = refined_F_U @ torch.diag(refined_F_S) @ refined_F_Vh @ torch.linalg.pinv(A) # (N, N)
    refined_prod_H = torch.sum(torch.mul(S @ A, A), dim=1) / torch.norm(A, dim=1).pow(2)    # (N,)
    # Hr_U: (N, K), Hr_S: (K,), Hr_Vh: (K, K) if N >= K
    # Hr_U: (N, N), Hr_S: (N,), Hr_Vh: (N, K) if N < K
    Hr_U, Hr_S, Hr_Vh = torch.linalg.svd(H_r, full_matrices=False)
    refined_Hr_S = Hr_S.clone()
    refined_Hr_S[-1] = 0    # set the last singular value equals to 0
    refined_Hr_Vh = recon_orthogonal_matrix(Hr_Vh.T, H_l_mean).T
    C = Hr_U @ torch.diag(refined_Hr_S) @ refined_Hr_Vh     # (N, K)
    refined_H_r = refined_prod_H.unsqueeze(1) @ torch.linalg.pinv(H_l_mean.unsqueeze(1)) + C
    return refined_H_r


def rotation_loss(pred_R, gt_R):
    cos_sim = (torch.trace(pred_R @ gt_R.T) - 1) / 2.0
    cos_sim = torch.clamp(cos_sim, -1, 1)
    angle_error = torch.acos(cos_sim)
    return angle_error


def sample_transformation(bid):
    device = bid.device
    rot, trans = [], []
    for _ in range(bid[-1] + 1):
        rot.append(rand_rotation_matrix().to(device))
        trans.append(torch.rand(3).to(device))
    return rot, trans
