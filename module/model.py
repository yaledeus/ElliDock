#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

import sys
sys.path.append('..')
from data.bio_parse import CA_INDEX
from data.geometry import CoordNomralizer, rand_rotation_matrix, kabsch_torch, \
    max_triangle_area
from .embed import ComplexGraph
from .gnn import PTA_EGNN, coord2nforce
from .find_triangular import find_triangular_cython


class ExpDock(nn.Module):
    def __init__(self, embed_size, hidden_size, k_neighbors=9,
                 att_heads=4, n_layers=4, n_keypoints=10, dropout=0.1,
                 mean=None, std=None) -> None:
        super(ExpDock, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_keypoints = n_keypoints

        self.normalizer = CoordNomralizer(mean=mean, std=std)

        node_in_d, edge_in_d = ComplexGraph.feature_dim(embed_size)

        self.graph_constructor = ComplexGraph(embed_size, k_neighbors)

        self.linear_in = nn.Linear(node_in_d, hidden_size)

        for i in range(self.n_layers):
            self.add_module(f'gnn_{i}', PTA_EGNN(
                node_in_d, hidden_size, hidden_size, edge_in_d, att_heads=att_heads,
                act_fn=nn.SiLU(), dropout=dropout
                )
            )

        for i in range(self.n_keypoints):
            self.register_parameter(f'w1_mat_{i}', nn.Parameter(torch.rand(hidden_size, hidden_size)))
            self.register_parameter(f'w2_mat_{i}', nn.Parameter(torch.rand(hidden_size, hidden_size)))

    def forward(self, X, S, RP, ID, Seg, center, keypoints, bid, k_bid):
        device = X.device
        # center to antigen
        X = self.normalizer.centering(X, center, bid)
        # normalize X to approximately normal distribution
        X = self.normalizer.normalize(X).float()
        # clone original X, no grad
        ori_X = X[:, CA_INDEX].clone().detach_()
        # note that normalizing shall not change the distance between nodes
        keypoints = self.normalizer.centering(keypoints, center, k_bid)
        keypoints = self.normalizer.normalize(keypoints).float().detach_()
        trans_keypoints = keypoints.clone()

        # rotate and translate for antibodies, X' = XR + t
        rot, trans = sample_transformation(bid)
        for i in range(len(rot)):
            ab_idx = torch.logical_and(Seg == 0, bid == i)
            X[ab_idx] = X[ab_idx] @ rot[i] + trans[i]
            trans_keypoints[k_bid == i] = torch.mm(trans_keypoints[k_bid == i], rot[i]) + trans[i]

        node_attr, edges, edge_attr = self.graph_constructor(
            X, S, RP, ID, Seg, bid
        )

        row, col = edges
        klist = find_triangular_cython(edges.cpu().numpy())
        klist = torch.from_numpy(klist).to(device)

        # CA atoms only
        X = X[:, CA_INDEX]      # (N, 3)
        init_X = X.clone()

        with torch.no_grad():
            nvecs = coord2nforce(edges, init_X, order=[2, 3, 4, 5, 6])

        H = self.linear_in(node_attr)

        # force balance loss
        # fb_loss = 0.
        for i in range(self.n_layers):
            H, X = self._modules[f'gnn_{i}'](
                H, X, edges, nvecs, edge_attr, node_attr, init_X, klist
            )

        # optimal transport loss & keypoint loss & dock loss & rmsd loss &
        # stable loss & match loss
        ot_loss = 0.
        dock_loss = 0.
        rmsd_loss = 0.
        stable_loss = 0.
        match_loss = 0.
        balance_loss = 0.
        # threshold
        threshold = 10. / torch.mean(self.normalizer.std)

        for i in range(len(rot)):
            ab_idx = torch.logical_and(Seg == 0, bid == i)
            ag_idx = torch.logical_and(Seg == 1, bid == i)
            H1, H2 = H[ab_idx], H[ag_idx]   # H1: (N, hidden_size) H2: (M, hidden_size)
            X1, X2 = X[ab_idx], X[ag_idx]   # X1: (N, 3) X2: (M, 3)
            Y1 = torch.zeros(self.n_keypoints, 3).to(device)    # (K, 3)
            Y2 = torch.zeros(self.n_keypoints, 3).to(device)
            for k in range(self.n_keypoints):
                alpha_1k = (1. / np.sqrt(self.hidden_size)) * torch.matmul(
                    (H1 @ self._parameters[f'w1_mat_{k}']),
                    H2.T.mean(dim=-1).unsqueeze(1)
                ).squeeze() # (N,)
                alpha_1k = F.softmax(alpha_1k, dim=0).unsqueeze(1)  # (N, 1)
                Y1[k] = (alpha_1k.T @ X1).squeeze()
                alpha_2k = (1. / np.sqrt(self.hidden_size)) * torch.matmul(
                    (H2 @ self._parameters[f'w2_mat_{k}']),
                    H1.T.mean(dim=-1).unsqueeze(1)
                ).squeeze()  # (M,)
                alpha_2k = F.softmax(alpha_2k, dim=0).unsqueeze(1)  # (M, 1)
                Y2[k] = (alpha_2k.T @ X2).squeeze()
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
            ot_loss /= 2
            # compute dock loss
            _, R, t = kabsch_torch(Y1, Y2)   # minimize RMSD(Y1R + t, Y2)
            dock_loss += F.mse_loss(rot[i] @ R, torch.eye(3).to(device))
            dock_loss += F.mse_loss(trans[i][None, :] @ R, -t[None, :])
            # compute stable loss
            stable_loss += F.softplus(-max_triangle_area(Y1))
            stable_loss += F.softplus(-max_triangle_area(Y2))
            stable_loss /= 2
            # compute match loss
            D_abag = torch.cdist(ori_X[ab_idx], ori_X[ag_idx])  # (N, M)
            ab_dock_scope, ag_dock_scope = torch.where(D_abag < threshold)
            ab_irr_scope, ag_irr_scope = torch.where(D_abag >= threshold)
            match_loss += F.smooth_l1_loss(
                F.softplus(H1[ab_irr_scope][:, None, :] @ H2[ag_irr_scope][:, :, None]).squeeze(),
                torch.zeros_like(ab_irr_scope)
            )
            match_loss += F.smooth_l1_loss(
                (H1[ab_dock_scope][:, None, :] @ H2[ag_dock_scope][:, :, None]).squeeze(),
                torch.ones_like(ab_dock_scope)
            )
            match_loss /= 2
            # compute balance loss
            # force from antigen(ligand) to antibody(receptor)
            force_g2b = ori_X[ag_idx] - ori_X[ab_idx].unsqueeze(1).repeat(1, H2.shape[0], 1)    # (N, M, 3)
            force_g2b = force_g2b * D_abag[:, :, None].pow(-3) * torch.matmul(
                H1.unsqueeze(1).repeat(1, H2.shape[0], 1)[:, :, None, :],
                H2[:, :, None]
            ).squeeze(-1)   # (N, M, 3)
            force_g2b = force_g2b.sum(dim=1)    # (N, 3)
            radial_c2s = ori_X[ab_idx] - ori_X[ab_idx].mean(dim=0)  # (N, 3)
            # angular momentum
            ang_mo = torch.cross(radial_c2s, force_g2b) # (N, 3)
            balance_loss += F.smooth_l1_loss(ang_mo.mean(dim=0), torch.zeros(3).to(device))
            balance_loss += F.smooth_l1_loss(force_g2b.mean(dim=0), torch.zeros(3).to(device))
            balance_loss /= 2
            del D_abag  # free memory
            torch.cuda.empty_cache()
            # compute rmsd loss
            X1_aligned = init_X[ab_idx] @ R + t   # (N, 3)
            rmsd_loss += F.mse_loss(X1_aligned, ori_X[ab_idx])

        # normalize
        ot_loss /= len(rot)
        dock_loss /= len(rot)
        stable_loss /= len(rot)
        match_loss /= len(rot)
        balance_loss /= len(rot)
        rmsd_loss /= len(rot)

        bind_loss = ot_loss * dock_loss

        # print_log(f"fb_loss: {fb_loss}, ot_loss: {ot_loss}, dock_loss: {dock_loss}, "
        #           f"match_loss: {match_loss}, si_loss: {si_loss}", level='INFO')
        # loss = 2 * ot_loss + dock_loss + stable_loss + match_loss
        loss = 2 * bind_loss + stable_loss + match_loss + balance_loss
        return loss, (ot_loss, dock_loss, bind_loss, stable_loss, match_loss, balance_loss, rmsd_loss)

    def dock(self, X, S, RP, ID, Seg, center, keypoints, bid, k_bid):
        device = X.device
        # center to antigen
        X = self.normalizer.centering(X, center, bid)
        # normalize X to approximately normal distribution
        X = self.normalizer.normalize(X).float()

        node_attr, edges, edge_attr = self.graph_constructor(
            X, S, RP, ID, Seg, bid
        )

        klist = find_triangular_cython(edges.cpu().numpy())
        klist = torch.from_numpy(klist).to(device)

        # CA atoms only
        X = X[:, CA_INDEX]  # (N, 3)
        init_X = X.clone()

        with torch.no_grad():
            nvecs = coord2nforce(edges, init_X, order=[2, 3, 4, 5, 6])

        H = self.linear_in(node_attr)

        for i in range(self.n_layers):
            H, X = self._modules[f'gnn_{i}'](
                H, X, edges, nvecs, edge_attr, node_attr, init_X, klist
            )

        for i in range(bid[-1] + 1):
            ab_idx = torch.logical_and(Seg == 0, bid == i)
            ag_idx = torch.logical_and(Seg == 1, bid == i)
            H1, H2 = H[ab_idx], H[ag_idx]
            X1, X2 = X[ab_idx], X[ag_idx]  # X1: (N, 3) X2: (M, 3)
            Y1 = torch.zeros(self.n_keypoints, 3).to(device)  # (K, 3)
            Y2 = torch.zeros(self.n_keypoints, 3).to(device)
            for k in range(self.n_keypoints):
                alpha_1k = (1. / np.sqrt(self.hidden_size)) * torch.mm(
                    torch.mm(H1, self._parameters[f'w1_mat_{k}']),
                    H2.T.mean(dim=-1).unsqueeze(1)
                ).squeeze()  # (N,)
                alpha_1k = F.softmax(alpha_1k, dim=0).unsqueeze(1)  # (N, 1)
                Y1[k] = torch.mm(alpha_1k.T, X1).squeeze()
                alpha_2k = (1. / np.sqrt(self.hidden_size)) * torch.mm(
                    torch.mm(H2, self._parameters[f'w2_mat_{k}']),
                    H1.T.mean(dim=-1).unsqueeze(1)
                ).squeeze()  # (N,)
                alpha_2k = F.softmax(alpha_2k, dim=0).unsqueeze(1)  # (N, 1)
                Y2[k] = torch.mm(alpha_2k.T, X2).squeeze()
            _, R, t = kabsch_torch(Y1, Y2)  # minimize RMSD(Y1R + t, Y2)
            init_X[ab_idx] = init_X[ab_idx] @ R + t

        init_X = self.normalizer.unnormalize(init_X)
        init_X = self.normalizer.uncentering(init_X, center, bid)

        return init_X


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
