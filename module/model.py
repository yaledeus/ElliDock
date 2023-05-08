#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys

import torch

sys.path.append('..')
from data.bio_parse import CA_INDEX
from utils.geometry import *
from .embed import ComplexGraph
from .gnn import PINN, Gated_Equivariant_Block, RBF, coord2radial


class ExpDock(nn.Module):
    def __init__(self, embed_size, hidden_size, k_neighbors=9, n_layers=4,
                 rbf_dim=20, r_cut=1., att_heads=4, dropout=0.1,
                 mean=None, std=None) -> None:
        super(ExpDock, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.rbf_dim = rbf_dim
        self.r_cut = r_cut

        self.quad_const = 1.
        self.scale_factor = 0.015

        node_in_d, edge_in_d = ComplexGraph.feature_dim(embed_size)

        self.graph_constructor = ComplexGraph(embed_size, k_neighbors)

        self.in_conv = nn.Sequential(
            nn.Linear(node_in_d, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        for i in range(self.n_layers):
            self.add_module(f'gnn_{i}', PINN(
                hidden_size, hidden_size, hidden_size, edge_in_d, rbf_dim=self.rbf_dim,
                r_cut=self.r_cut, att_heads=att_heads, act_fn=nn.SiLU(), dropout=dropout
                )
            )
            self.add_module(f'inter_act_{i}', nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.hidden_size)
                )
            )
            self.register_parameter(f'inter_att_{i}', nn.Parameter(torch.rand(hidden_size, hidden_size)))

        self.gated_equiv_block = Gated_Equivariant_Block(
            hidden_size, hidden_size, hidden_size, act_fn=nn.SiLU(), dropout=dropout
        )

        self.re_inv_conv = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 3 + 1)
        )

        self.li_inv_conv = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 3 + 1)
        )

        self.final_att_block = nn.Parameter(torch.rand(hidden_size, hidden_size))

        self.re_equiv_conv = nn.Sequential(
            nn.Linear(self.hidden_size, 3 * 3 + 1, bias=False),
            nn.Dropout(dropout)
        )

        self.li_equiv_conv = nn.Sequential(
            nn.Linear(self.hidden_size, 3 * 3 + 1, bias=False),
            nn.Dropout(dropout)
        )

        self.neg_penalty_loss = NegPenaltyLoss()
        self.out_span_loss = OutSpanLoss()

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

    def forward(self, X, S, RP, Seg, center, keypoints, bid, k_bid, **kwargs):
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
            receptor_idx = torch.logical_and(Seg == 0, bid == i)
            X[receptor_idx] = X[receptor_idx] @ rot[i] + trans[i] * torch.mean(self.normalizer.std)
            trans_keypoints[k_bid == i] = trans_keypoints[k_bid == i] @ rot[i] + trans[i] * torch.mean(
                self.normalizer.std)

        X = self.normalizer.normalize(X).float()
        ori_X = self.normalizer.normalize(ori_X).float()
        keypoints = self.normalizer.normalize(keypoints).float()
        trans_keypoints = self.normalizer.normalize(trans_keypoints).float()

        node_attr, edges, edge_attr = self.graph_constructor(
            X, S, RP, Seg, bid
        )

        # CA atoms only
        X = X[:, CA_INDEX]      # (N, 3)

        node_feat = self.in_conv(node_attr) # (N, hidden_size)
        vec_feat = torch.zeros(X.shape[0], self.hidden_size, 3).to(device) # (N, hidden_size, 3)

        radial, coord_diff = coord2radial(edges, X) # radial: (n_edges, 1), coord_diff: (n_edges, 3)
        norm = torch.sqrt(radial) + 1e-8            # (n_edges, 1)
        rbf = RBF(norm, self.r_cut, self.rbf_dim)   # (n_edges, rbf_dim)

        for i in range(self.n_layers):
            intra_node_feat, vec_feat = self._modules[f'gnn_{i}'](
                node_feat, vec_feat, edges, edge_attr, coord_diff, rbf
            )
            for b_ind in range(bs):
                receptor_idx = torch.logical_and(Seg == 0, bid == b_ind)
                ligand_idx = torch.logical_and(Seg == 1, bid == b_ind)
                node_feat[receptor_idx] = intra_node_feat[receptor_idx] + \
                                          torch.sigmoid((intra_node_feat[receptor_idx] @
                                                        self._parameters[f'inter_att_{i}'] @
                                                        intra_node_feat[ligand_idx].T)
                                                        .mean(dim=1)).unsqueeze(1) * \
                                          self._modules[f'inter_act_{i}'](intra_node_feat[receptor_idx])
                node_feat[ligand_idx] = intra_node_feat[ligand_idx] + \
                                        torch.sigmoid((intra_node_feat[ligand_idx] @
                                                        self._parameters[f'inter_att_{i}'] @
                                                        intra_node_feat[receptor_idx].T)
                                                      .mean(dim=1)).unsqueeze(1) * \
                                        self._modules[f'inter_act_{i}'](intra_node_feat[ligand_idx])

        node_feat, vec_feat = self.gated_equiv_block(node_feat, vec_feat)

        # fit loss && dock loss & rmsd loss & stable loss & match loss
        fit_loss = 0.
        dock_loss = 0.
        rmsd_loss = 0.
        stable_loss = 0.

        threshold = 1. / torch.mean(self.normalizer.std)

        for i in range(bs):
            receptor_idx = torch.logical_and(Seg == 0, bid == i)
            ligand_idx = torch.logical_and(Seg == 1, bid == i)

            re_center = X[receptor_idx].mean(dim=0) # (3,)
            li_center = X[ligand_idx].mean(dim=0)   # (3,)

            inv1 = self.re_inv_conv(
                node_feat[receptor_idx] * torch.sigmoid(
                    (node_feat[receptor_idx] @ self.final_att_block @ node_feat[ligand_idx].T)
                    .mean(dim=1)
                ).unsqueeze(1)
            ).sum(dim=0)    # (3 + 1,)

            inv2 = self.li_inv_conv(
                node_feat[ligand_idx] * torch.sigmoid(
                    (node_feat[ligand_idx] @ self.final_att_block @ node_feat[receptor_idx].T)
                    .mean(dim=1)
                ).unsqueeze(1)
            ).sum(dim=0)    # (3 + 1,)

            eigen = inv1[:3] + inv2[:3]
            eigen = eigen * torch.sgn(eigen)    # (+, +, +)

            # paraboloid constrain
            Lambda = torch.zeros(3).to(device)
            re_prim = torch.zeros(3).to(device)
            li_prim = torch.zeros(3).to(device)
            Lambda[:2] = eigen[:2]
            re_prim[2] = -eigen[2]
            li_prim[2] =  eigen[2]

            # x-y refine
            theta = inv1[3] - inv2[3]
            R_ref = torch.tensor([[ torch.cos(theta), torch.sin(theta)],
                                  [-torch.sin(theta), torch.cos(theta)]], device=device)
            R_ref_3d = torch.eye(3).to(device)
            R_ref_3d[:2, :2] = R_ref

            Y1 = self.re_equiv_conv(
                (vec_feat[receptor_idx] * node_feat[receptor_idx][:, :, None]).transpose(1, 2)
            ).transpose(1, 2).sum(dim=0) / self.scale_factor   # (3*3 + 1, 3)

            Y2 = self.li_equiv_conv(
                (vec_feat[ligand_idx] * node_feat[ligand_idx][:, :, None]).transpose(1, 2)
            ).transpose(1, 2).sum(dim=0) / self.scale_factor   # (3*3 + 1, 3)

            re_P, re_std_trans = Y1[:3] + Y1[3:6] + Y1[6:9], Y1[9]
            li_P, li_std_trans = Y2[:3] + Y2[3:6] + Y2[6:9], Y2[9]

            R1 = modified_gram_schmidt(re_P)  # orthogonal matrix
            R2 = modified_gram_schmidt(li_P)

            re_t_pri = re_std_trans + re_center # prior translation vector
            li_t_pri = li_std_trans + li_center

            # re_A_prime, re_b_prime, re_c_prime = elliptical_paraboloid_std2E3(Lambda, re_prim, R1, re_t_pri)
            # li_A_prime, li_b_prime, li_c_prime = elliptical_paraboloid_std2E3(Lambda, li_prim, R2, li_t_pri)

            t1 = -re_t_pri @ R1 # posterior translation vector
            t2 = -li_t_pri @ R2

            P1 = trans_keypoints[k_bid == i] @ R1 + t1
            P2 = keypoints[k_bid == i] @ R2 + t2
            X1 = X[receptor_idx] @ R1 + t1
            X2 = X[ligand_idx] @ R2 + t2

            key_fit1 = P1**2 @ Lambda + P1 @ re_prim
            key_fit2 = P2**2 @ Lambda + P2 @ li_prim
            # coord_fit1 = X1**2 @ Lambda + X1 @ re_prim
            coord_fit1 = X1**2 @ Lambda + X1 @ li_prim
            coord_fit2 = X2**2 @ Lambda + X2 @ li_prim

            # keypoint fitness
            fit_loss += F.smooth_l1_loss(key_fit1, torch.zeros(P1.shape[0]).to(device))
            fit_loss += F.smooth_l1_loss(key_fit2, torch.zeros(P2.shape[0]).to(device))
            # z-span fitness
            fit_loss += self.out_span_loss(P1[:, 2], low=-threshold, high=threshold)
            fit_loss += self.out_span_loss(P2[:, 2], low=-threshold, high=threshold)
            fit_loss += F.smooth_l1_loss(torch.relu(-coord_fit1), torch.zeros(X1.shape[0]).to(device))
            fit_loss += F.smooth_l1_loss(torch.relu( coord_fit2), torch.zeros(X2.shape[0]).to(device))
            # fit_loss += F.smooth_l1_loss(torch.relu( X1[:, 2]), torch.zeros(X1.shape[0]).to(device))
            # fit_loss += F.smooth_l1_loss(torch.relu(-X2[:, 2]), torch.zeros(X2.shape[0]).to(device))
            # x-y span fitness
            _, R_ref_gt, _ = kabsch_torch(P1[:, :2], P2[:, :2])
            fit_loss += (R_ref - R_ref_gt).pow(2).mean()

            R = R1 @ R_ref_3d @ R2.T
            t = (t1 @ R_ref_3d - t2) @ R2.T

            ct = self.normalizer.mean / torch.mean(self.normalizer.std)
            dock_loss += F.mse_loss(R, rot[i].T)
            dock_loss += F.mse_loss(t, (ct - trans[i]) @ rot[i].T - ct)

            # compute rmsd loss
            X1_aligned = X[receptor_idx] @ R + t   # (N, 3)
            rmsd_loss += F.mse_loss(X1_aligned, ori_X[receptor_idx])

        # normalize
        fit_loss /= bs
        dock_loss /= bs
        stable_loss /= bs
        rmsd_loss /= bs

        loss = 0.5 * fit_loss + dock_loss + rmsd_loss

        return loss, (fit_loss, dock_loss, stable_loss, rmsd_loss)

    def dock(self, X, S, RP, Seg, center, bid, **kwargs):
        device = X.device
        bs = bid[-1] + 1
        # center to antigen
        X = self.normalizer.centering(X, center, bid)
        # normalize X to approximately normal distribution
        X = self.normalizer.normalize(X).float()

        node_attr, edges, edge_attr = self.graph_constructor(
            X, S, RP, Seg, bid
        )

        # CA atoms only
        X = X[:, CA_INDEX]  # (N, 3)

        node_feat = self.in_conv(node_attr)  # (N, hidden_size)
        vec_feat = torch.zeros(X.shape[0], self.hidden_size, 3).to(device)  # (N, hidden_size, 3)

        radial, coord_diff = coord2radial(edges, X)  # radial: (n_edges, 1), coord_diff: (n_edges, 3)
        norm = torch.sqrt(radial) + 1e-8  # (n_edges, 1)
        rbf = RBF(norm, self.r_cut, self.rbf_dim)  # (n_edges, rbf_dim)

        for i in range(self.n_layers):
            intra_node_feat, vec_feat = self._modules[f'gnn_{i}'](
                node_feat, vec_feat, edges, edge_attr, coord_diff, rbf
            )
            for b_ind in range(bs):
                receptor_idx = torch.logical_and(Seg == 0, bid == b_ind)
                ligand_idx = torch.logical_and(Seg == 1, bid == b_ind)
                node_feat[receptor_idx] = intra_node_feat[receptor_idx] + \
                                          torch.sigmoid((intra_node_feat[receptor_idx] @
                                                         self._parameters[f'inter_att_{i}'] @
                                                         intra_node_feat[ligand_idx].T)
                                                        .mean(dim=1)).unsqueeze(1) * \
                                          self._modules[f'inter_act_{i}'](intra_node_feat[receptor_idx])
                node_feat[ligand_idx] = intra_node_feat[ligand_idx] + \
                                        torch.sigmoid((intra_node_feat[ligand_idx] @
                                                       self._parameters[f'inter_att_{i}'] @
                                                       intra_node_feat[receptor_idx].T)
                                                      .mean(dim=1)).unsqueeze(1) * \
                                        self._modules[f'inter_act_{i}'](intra_node_feat[ligand_idx])

        node_feat, vec_feat = self.gated_equiv_block(node_feat, vec_feat)

        dock_trans_list = []

        for i in range(bs):
            receptor_idx = torch.logical_and(Seg == 0, bid == i)
            ligand_idx = torch.logical_and(Seg == 1, bid == i)

            re_center = X[receptor_idx].mean(dim=0) # (3,)
            li_center = X[ligand_idx].mean(dim=0)   # (3,)

            inv1 = self.re_inv_conv(
                node_feat[receptor_idx] * torch.sigmoid(
                    (node_feat[receptor_idx] @ self.final_att_block @ node_feat[ligand_idx].T)
                        .mean(dim=1)
                ).unsqueeze(1)
            ).sum(dim=0)  # (3 + 1,)

            inv2 = self.li_inv_conv(
                node_feat[ligand_idx] * torch.sigmoid(
                    (node_feat[ligand_idx] @ self.final_att_block @ node_feat[receptor_idx].T)
                        .mean(dim=1)
                ).unsqueeze(1)
            ).sum(dim=0)  # (3 + 1,)

            eigen = inv1[:3] + inv2[:3]
            eigen = eigen * torch.sgn(eigen)  # (+, +, +)

            # paraboloid constrain
            Lambda = torch.zeros(3).to(device)
            re_prim = torch.zeros(3).to(device)
            li_prim = torch.zeros(3).to(device)
            Lambda[:2] = eigen[:2]
            re_prim[2] = -eigen[2]
            li_prim[2] = eigen[2]

            # x-y refine
            theta = inv1[3] - inv2[3]
            R_ref = torch.tensor([[ torch.cos(theta), torch.sin(theta)],
                                  [-torch.sin(theta), torch.cos(theta)]], device=device)
            R_ref_3d = torch.eye(3).to(device)
            R_ref_3d[:2, :2] = R_ref

            Y1 = self.re_equiv_conv(
                (vec_feat[receptor_idx] * node_feat[receptor_idx][:, :, None]).transpose(1, 2)
            ).transpose(1, 2).sum(dim=0) / self.scale_factor  # (3*3 + 1, 3)

            Y2 = self.li_equiv_conv(
                (vec_feat[ligand_idx] * node_feat[ligand_idx][:, :, None]).transpose(1, 2)
            ).transpose(1, 2).sum(dim=0) / self.scale_factor  # (3*3 + 1, 3)

            re_P, re_std_trans = Y1[:3] + Y1[3:6] + Y1[6:9], Y1[9]
            li_P, li_std_trans = Y2[:3] + Y2[3:6] + Y2[6:9], Y2[9]

            R1 = modified_gram_schmidt(re_P)  # orthogonal matrix
            R2 = modified_gram_schmidt(li_P)

            re_t_pri = re_std_trans + re_center  # prior translation vector
            li_t_pri = li_std_trans + li_center

            # re_A_prime, re_b_prime, re_c_prime = elliptical_paraboloid_std2E3(Lambda, re_prim, R1, re_t_pri)
            # li_A_prime, li_b_prime, li_c_prime = elliptical_paraboloid_std2E3(Lambda, li_prim, R2, li_t_pri)

            t1 = -re_t_pri @ R1  # posterior translation vector
            t2 = -li_t_pri @ R2

            R = R1 @ R_ref_3d @ R2.T
            t = (t1 @ R_ref_3d - t2) @ R2.T

            X[receptor_idx] = X[receptor_idx] @ R + t
            dock_trans_list.append(self.normalizer.dock_transformation(center, i, R, t))

        X = self.normalizer.unnormalize(X)
        X = self.normalizer.uncentering(X, center, bid)

        return X, dock_trans_list


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


class NegPenaltyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        device = input_tensor.device

        numel = torch.numel(input_tensor)

        if numel == 0:
            return torch.tensor(0., device=device, requires_grad=True)

        negative_mask = input_tensor < 0

        loss = torch.sum(input_tensor[negative_mask].abs())
        loss = loss / numel # mean

        return loss


class OutSpanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, low, high):
        device = input_tensor.device

        lower_loss_mask = input_tensor < low
        upper_loss_mask = input_tensor > high

        numel = lower_loss_mask.sum() + upper_loss_mask.sum()

        if numel == 0:
            return torch.tensor(0., device=device, requires_grad=True)

        lower_loss = torch.sum(torch.pow(input_tensor[lower_loss_mask] - low, 2))
        upper_loss = torch.sum(torch.pow(input_tensor[upper_loss_mask] - high, 2))

        loss = lower_loss + upper_loss
        loss = loss / numel # mean

        return loss


def sample_transformation(bid):
    device = bid.device
    rot, trans = [], []
    for _ in range(bid[-1] + 1):
        rot.append(rand_rotation_matrix().to(device))
        trans.append(torch.rand(3).to(device))
    return rot, trans
