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
                 rbf_dim=20, r_cut=1., dropout=0.1, mean=None, std=None) -> None:
        super(ExpDock, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.rbf_dim = rbf_dim
        self.r_cut = r_cut

        self.quad_const = 1.
        self.scaling_factor = 0.1

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
                r_cut=self.r_cut, act_fn=nn.SiLU(), dropout=dropout
                )
            )
            self.add_module(f'inter_act_{i}', nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.hidden_size)
                )
            )

        self.gated_equiv_block = Gated_Equivariant_Block(
            hidden_size, hidden_size, hidden_size, act_fn=nn.SiLU(), dropout=dropout
        )

        self.final_conv = nn.Sequential(
            nn.Linear(self.hidden_size, 2 * 4 + 1, bias=False),
            nn.Dropout(dropout)
        )

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
            ab_idx = torch.logical_and(Seg == 0, bid == i)
            X[ab_idx] = X[ab_idx] @ rot[i] + trans[i] * torch.mean(self.normalizer.std)
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
                node_feat[receptor_idx] = intra_node_feat[receptor_idx] + self._modules[f'inter_act_{i}'](
                    intra_node_feat[receptor_idx] * F.softmax((intra_node_feat[receptor_idx] @
                                                         intra_node_feat[ligand_idx].T).mean(dim=1), dim=0)
                    .unsqueeze(1)
                )
                node_feat[ligand_idx] = intra_node_feat[ligand_idx] + self._modules[f'inter_act_{i}'](
                    intra_node_feat[ligand_idx] * F.softmax((intra_node_feat[ligand_idx] @
                                                         intra_node_feat[receptor_idx].T).mean(dim=1), dim=0)
                    .unsqueeze(1)
                )

        node_feat, vec_feat = self.gated_equiv_block(node_feat, vec_feat)

        # dock loss & rmsd loss & stable loss & match loss
        dock_loss = 0.
        rmsd_loss = 0.
        stable_loss = 0.

        for i in range(bs):
            receptor_idx = torch.logical_and(Seg == 0, bid == i)
            ligand_idx = torch.logical_and(Seg == 1, bid == i)

            re_center = X[receptor_idx].mean(dim=0) # (3,)
            li_center = X[ligand_idx].mean(dim=0)   # (3,)

            re_pred_vec = self.final_conv(
                (vec_feat[receptor_idx] * node_feat[receptor_idx][:, :, None]).transpose(1, 2)
            ).transpose(1, 2).sum(dim=0)   # (9, 3)

            li_pred_vec = self.final_conv(
                (vec_feat[ligand_idx] * node_feat[ligand_idx][:, :, None]).transpose(1, 2)
            ).transpose(1, 2).sum(dim=0)   # (9, 3)

            # second-order tensor, (3, 3)
            pred_re_A = torch.kron(re_pred_vec[0][:, None], re_pred_vec[1][None, :]) + \
                        torch.kron(re_pred_vec[2][None, :], re_pred_vec[3][:, None]) + \
                        torch.kron(re_pred_vec[4][:, None], re_pred_vec[5][None, :]) + \
                        torch.kron(re_pred_vec[6][None, :], re_pred_vec[7][:, None])
            pred_re_A = pred_re_A / self.scaling_factor**2
            # first-order tensor, (3,)
            pred_re_b = re_pred_vec[8] / self.scaling_factor

            pred_li_A = torch.kron(li_pred_vec[0][:, None], li_pred_vec[1][None, :]) + \
                        torch.kron(li_pred_vec[2][None, :], li_pred_vec[3][:, None]) + \
                        torch.kron(li_pred_vec[4][:, None], li_pred_vec[5][None, :]) + \
                        torch.kron(li_pred_vec[6][None, :], li_pred_vec[7][:, None])
            pred_li_A = pred_li_A / self.scaling_factor**2

            pred_li_b = li_pred_vec[8] / self.scaling_factor

            re_std_params, _, _ = standard_quadratic_transform(pred_re_A, pred_re_b, scale=self.quad_const)
            li_std_params, _, _ = standard_quadratic_transform(pred_li_A, pred_li_b, scale=self.quad_const)

            # compute stable loss
            stable_loss += F.smooth_l1_loss(re_std_params, li_std_params)

            # SO(3) => E(3)
            pred_re_A_e3, pred_re_b_e3 = quadratic_O3_to_E3(pred_re_A, pred_re_b, scale=self.quad_const, t=re_center)
            pred_li_A_e3, pred_li_b_e3 = quadratic_O3_to_E3(pred_li_A, pred_li_b, scale=self.quad_const, t=li_center)

            stable_loss += F.smooth_l1_loss(
                ((trans_keypoints[k_bid == i] @ pred_re_A_e3) * trans_keypoints[k_bid == i])
                .sum(dim=1) + trans_keypoints[k_bid == i] @ pred_re_b_e3 - self.quad_const,
                torch.zeros(keypoints[k_bid == i].shape[0]).to(device)
            )
            stable_loss += F.smooth_l1_loss(
                ((keypoints[k_bid == i] @ pred_li_A_e3) * keypoints[k_bid == i])
                .sum(dim=1) + keypoints[k_bid == i] @ pred_li_b_e3 - self.quad_const,
                torch.zeros(keypoints[k_bid == i].shape[0]).to(device)
            )

            _, R1, t1 = standard_quadratic_transform(pred_re_A_e3, pred_re_b_e3, scale=self.quad_const)
            _, R2, t2 = standard_quadratic_transform(pred_li_A_e3, pred_li_b_e3, scale=self.quad_const)

            R = R1 @ R2.T
            t = (t1 - t2) @ R2.T

            ct = self.normalizer.mean / torch.mean(self.normalizer.std)
            dock_loss += F.mse_loss(rot[i] @ R, torch.eye(3).to(device))
            dock_loss += F.mse_loss((trans[i] - ct) @ rot[i].T + t + ct, torch.zeros(3).to(device))

            # compute rmsd loss
            X1_aligned = X[receptor_idx] @ R + t   # (N, 3)
            rmsd_loss += F.mse_loss(X1_aligned, ori_X[receptor_idx])

        # normalize
        dock_loss /= bs
        stable_loss /= bs
        rmsd_loss /= bs

        loss = 0.2 * dock_loss + stable_loss + 0.2 * rmsd_loss

        return loss, (dock_loss, stable_loss, rmsd_loss)

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
                node_feat[receptor_idx] = intra_node_feat[receptor_idx] + self._modules[f'inter_act_{i}'](
                    intra_node_feat[receptor_idx] * F.softmax((intra_node_feat[receptor_idx] @
                                                               intra_node_feat[ligand_idx].T).mean(dim=1), dim=0)
                    .unsqueeze(1)
                )
                node_feat[ligand_idx] = intra_node_feat[ligand_idx] + self._modules[f'inter_act_{i}'](
                    intra_node_feat[ligand_idx] * F.softmax((intra_node_feat[ligand_idx] @
                                                             intra_node_feat[receptor_idx].T).mean(dim=1), dim=0)
                    .unsqueeze(1)
                )

        node_feat, vec_feat = self.gated_equiv_block(node_feat, vec_feat)

        dock_trans_list = []

        for i in range(bs):
            receptor_idx = torch.logical_and(Seg == 0, bid == i)
            ligand_idx = torch.logical_and(Seg == 1, bid == i)

            re_center = X[receptor_idx].mean(dim=0)  # (3,)
            li_center = X[ligand_idx].mean(dim=0)  # (3,)

            re_pred_vec = self.final_conv(
                (vec_feat[receptor_idx] * node_feat[receptor_idx][:, :, None]).transpose(1, 2)
            ).transpose(1, 2).sum(dim=0)  # (9, 3)

            li_pred_vec = self.final_conv(
                (vec_feat[ligand_idx] * node_feat[ligand_idx][:, :, None]).transpose(1, 2)
            ).transpose(1, 2).sum(dim=0)  # (9, 3)

            # second-order tensor, (3, 3)
            pred_re_A = torch.kron(re_pred_vec[0][:, None], re_pred_vec[1][None, :]) + \
                        torch.kron(re_pred_vec[2][None, :], re_pred_vec[3][:, None]) + \
                        torch.kron(re_pred_vec[4][:, None], re_pred_vec[5][None, :]) + \
                        torch.kron(re_pred_vec[6][None, :], re_pred_vec[7][:, None])
            pred_re_A = pred_re_A / self.scaling_factor ** 2
            # first-order tensor, (3,)
            pred_re_b = re_pred_vec[8] / self.scaling_factor

            pred_li_A = torch.kron(li_pred_vec[0][:, None], li_pred_vec[1][None, :]) + \
                        torch.kron(li_pred_vec[2][None, :], li_pred_vec[3][:, None]) + \
                        torch.kron(li_pred_vec[4][:, None], li_pred_vec[5][None, :]) + \
                        torch.kron(li_pred_vec[6][None, :], li_pred_vec[7][:, None])
            pred_li_A = pred_li_A / self.scaling_factor ** 2

            pred_li_b = li_pred_vec[8] / self.scaling_factor

            # SO(3) => E(3)
            pred_re_A_e3, pred_re_b_e3 = quadratic_O3_to_E3(pred_re_A, pred_re_b, scale=self.quad_const, t=re_center)
            pred_li_A_e3, pred_li_b_e3 = quadratic_O3_to_E3(pred_li_A, pred_li_b, scale=self.quad_const, t=li_center)

            _, R1, t1 = standard_quadratic_transform(pred_re_A_e3, pred_re_b_e3, scale=self.quad_const)
            _, R2, t2 = standard_quadratic_transform(pred_li_A_e3, pred_li_b_e3, scale=self.quad_const)

            R = R1 @ R2.T
            t = (t1 - t2) @ R2.T

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
