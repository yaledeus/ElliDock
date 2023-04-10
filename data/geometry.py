import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .bio_parse import N_INDEX, CA_INDEX, C_INDEX


def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign((torch.cross(v1, v2, dim=-1) * v0).sum(-1))
    dihed = sgn * torch.acos((n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    dihed = torch.nan_to_num(dihed)
    return dihed


def get_backbone_dihedral_angles(coord):
    """
    Args:
        coord: coordinates of backbone atoms, [N, n_channel, 3]
    Returns:
        bb_dihedral: Omega, Phi, and Psi angles in radian, (N, 3).
    """
    coord_N = coord[:, N_INDEX]   # (N, 3)
    coord_CA = coord[:, CA_INDEX]
    coord_C = coord[:, C_INDEX]

    # N-termini don't have omega and phi
    omega = F.pad(
        dihedral_from_four_points(coord_CA[:-1], coord_C[:-1], coord_N[1:], coord_CA[1:]),
        pad=(1, 0), value=0,
    )
    phi = F.pad(
        dihedral_from_four_points(coord_C[:-1], coord_N[1:], coord_CA[1:], coord_C[1:]),
        pad=(1, 0), value=0,
    )

    # C-termini don't have psi
    psi = F.pad(
        dihedral_from_four_points(coord_N[:-1], coord_CA[:-1], coord_C[:-1], coord_N[1:]),
        pad=(0, 1), value=0,
    )

    bb_dihedral = torch.stack([omega, phi, psi], dim=-1)
    return bb_dihedral


def pairwise_dihedrals(coord, edges):
    """
    Args:
        coord:  (N, n_channel, 3).
        edges: (2, n_edges).
    Returns:
        Inter-residue Phi and Psi angles, (n_edges, 2).
    """
    row, col = edges
    coord_N_src = coord[row, N_INDEX]  # (n_edge, 3)
    coord_N_dst = coord[col, N_INDEX]
    coord_CA_src = coord[row, CA_INDEX]
    coord_C_src = coord[row, C_INDEX]
    coord_C_dst = coord[col, C_INDEX]

    ir_phi = dihedral_from_four_points(
        coord_C_dst,
        coord_N_src,
        coord_CA_src,
        coord_C_src
    )
    ir_psi = dihedral_from_four_points(
        coord_N_src,
        coord_CA_src,
        coord_C_src,
        coord_N_dst
    )
    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)
    return ir_dihed


def reletive_position_orientation(coord, edges):
    """
    Args:
        coord:  (N, n_channel, 3).
        edges: (2, n_edges).
    Returns:
        Inter-residue relative position, (n_edges, 3).
    """
    row, col = edges
    coord_N_src = coord[row, N_INDEX]  # (n_edges, 3)
    coord_CA_src = coord[row, CA_INDEX]
    coord_CA_dst = coord[col, CA_INDEX]
    coord_C_src = coord[row, C_INDEX]

    coord_diff = coord_CA_src - coord_CA_dst    # (n_edges, 3)

    eps = 1e-6

    vec_u = coord_N_src - coord_CA_src  # (n_edges, 3)
    vec_u = vec_u / (torch.norm(vec_u, dim=-1, keepdim=True) + eps)
    vec_t = coord_C_src - coord_CA_src  # (n_edges, 3)
    vec_t = vec_t / (torch.norm(vec_t, dim=-1, keepdim=True) + eps)
    vec_n = torch.cross(vec_u, vec_t)   # (n_edges, 3)
    vec_n = vec_n / (torch.norm(vec_n, dim=-1, keepdim=True) + eps)
    vec_v = torch.cross(vec_n, vec_u)   # (n_edges, 3)
    vec_v = vec_v / (torch.norm(vec_v, dim=-1, keepdim=True) + eps)

    local_coord_mat = torch.stack([vec_n, vec_u, vec_v], dim=1)    # (n_edges, 3, 3)

    rp = torch.matmul(local_coord_mat, coord_diff.unsqueeze(-1)).squeeze()  # (n_edges, 3)
    feat_q = torch.matmul(local_coord_mat, vec_n.unsqueeze(-1)).squeeze()   # (n_edges, 3)
    feat_k = torch.matmul(local_coord_mat, vec_u.unsqueeze(-1)).squeeze()  # (n_edges, 3)
    feat_t = torch.matmul(local_coord_mat, vec_v.unsqueeze(-1)).squeeze()  # (n_edges, 3)

    rel_pos_ori = torch.cat([rp, feat_q, feat_k, feat_t], dim=-1)   # (n_edges, 3 * 4)

    return rel_pos_ori


def rand_rotation_matrix():
    a = torch.randn(3, 3)
    q, r = torch.linalg.qr(a)
    d = torch.diag(torch.sign(torch.diag(r)))
    q = torch.mm(q, d)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def kabsch_torch(P: torch.Tensor, Q: torch.Tensor):
    """
    :param P: (N, 3)
    :param Q: (N, 3)
    :return: P @ R + t, R, t such that minimize RMSD(PR + t, B)
    """
    P = P.double()
    Q = Q.double()

    PC = torch.mean(P, dim=0)
    QC = torch.mean(Q, dim=0)

    # centering
    UP = P - PC
    UQ = Q - QC

    # Covariance matrix
    C = UP.T @ UQ
    V, S, W = torch.linalg.svd(C)

    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0

    # avoid inplace modify
    V_R = V.clone()
    if d:
        V_R[:, -1] = -V[:, -1]

    R: torch.Tensor = V_R @ W

    t = QC - PC @ R  # (3,)

    return (UP @ R + QC).float(), R.float(), t.float()

"""
test for kabsch algorithm
# Test Case 1: Simple test case
A = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float)
B = A @ torch.diag(torch.tensor([1, 1, -1]).float()) # Rotate A 180 degrees around Z-axis and flip Z
C, R, t = kabsch_torch(B, A)
assert np.allclose(C, A)

# Test Case 2: Random point cloud with uniform translation and rotation
np.random.seed(0)
A = torch.tensor(np.random.rand(100, 3), dtype=torch.float)
R = rand_rotation_matrix()
t = torch.tensor(np.random.rand(3), dtype=torch.float)
B = A @ R + t
C, R_est, t_est = kabsch_torch(B, A)
assert np.allclose(C, A, atol=1e-3)

# Test Case 3: Identity transformation
A = torch.tensor(np.random.rand(100, 3), dtype=torch.float)
B = A
C, R_est, t_est = kabsch_torch(B, A)
assert np.allclose(C, A, atol=1e-4)
assert np.allclose(R_est, np.eye(3), atol=1e-4)
assert np.allclose(t_est, np.zeros(3), atol=1e-4)
"""

def kabsch_numpy(P: np.ndarray, Q: np.ndarray):
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)

    PC = np.mean(P, axis=0)
    QC = np.mean(Q, axis=0)

    UP = P - PC
    UQ = Q - QC

    C = UP.T @ UQ
    V, S, W = np.linalg.svd(C)

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        V[:, -1] = -V[:, -1]

    R: np.ndarray = V @ W

    t = QC - PC @ R # (3,)

    return (UP @ R + QC).astype(np.float32), R.astype(np.float32), t.astype(np.float32)


def protein_surface_intersection(X, Y, sigma=1.67, gamma=0.67):
    """
    :param X: point cloud  to be referenced, (N, 3)
    :param Y: point cloud to be tested whether it is outside the protein, (M, 3)
    :param sigma, gamma: parameter
    :return:
    """
    return (gamma + sigma * ((-((Y.unsqueeze(1).repeat(1, X.shape[0], 1) - X) ** 2).sum(dim=-1) / sigma)
                             .exp().sum(dim=1)).log())


def max_triangle_area(points):
    """
    :param points: (N, 3)
    :return: maximum triangle area, torch.float
    """
    sorted_indices = torch.argsort(points[:, 0])
    return _max_triangle_area(points[sorted_indices])


# O(nlogn)
def _max_triangle_area(points):
    device = points.device
    # less than 3 points, return 0
    if len(points) <= 2:
        return torch.tensor(0).float().to(device)
    elif len(points) == 3:
        return _triangle_area(points[0], points[1], points[2])

    mid = len(points) // 2
    left_points = points[:mid]
    right_points = points[mid:]
    left_max_area = _max_triangle_area(left_points)
    right_max_area = _max_triangle_area(right_points)

    mid_x = (left_points[-1][0] + right_points[0][0]) / 2
    left_index = mid - 1
    right_index = mid
    mid_max_area = torch.tensor(0).float().to(device)
    while left_index >= 0 and right_index < len(points):
        left_point = points[left_index]
        right_point = points[right_index]
        if left_point[0] < mid_x and right_point[0] >= mid_x:
            area = _triangle_area(left_point, right_point, points[mid])
            if area > mid_max_area:
                mid_max_area = area
            left_index -= 1
        elif left_point[0] >= mid_x:
            left_index -= 1
        else:
            right_index += 1

    if mid_max_area >= left_max_area and mid_max_area >= right_max_area:
        return mid_max_area
    elif left_max_area >= mid_max_area and left_max_area >= right_max_area:
        return left_max_area
    else:
        return right_max_area


def _triangle_area(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = torch.cross(v1, v2)
    area = 0.5 * torch.norm(cross_product)
    return area

"""
### test: max_triangle_area
N = 100
A = torch.randn(N, 3)
print(f"{N} point cloud: {A}")
print(f"maximum triangle area: {max_triangle_area(A)}")

B = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).float()
print(f"maximum area of point cloud B should be 0.5: {max_triangle_area(B)}")
"""


class CoordNomralizer(nn.Module):
    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        self.mean = torch.tensor([-0.49545217, 0.2199743, 0.12866335]) \
                    if mean is None else torch.tensor(mean)
        self.std = torch.tensor([14.85880611, 14.99745863, 17.27655463]) \
                    if std is None else torch.tensor(std)
        self.mean = nn.parameter.Parameter(self.mean, requires_grad=False)
        self.std = nn.parameter.Parameter(self.std, requires_grad=False)

    def centering(self, X, center, bid):
        if X.ndim == 3:   # (N, n_channel, 3)
            X = X - center[bid].unsqueeze(1)
        elif X.ndim == 2: # (N, 3)
            X = X - center[bid]
        else:
            raise ValueError
        return X

    def uncentering(self, X, center, bid):
        X = X + center[bid]
        return X

    def normalize(self, X):
        X = (X - self.mean) / self.std
        return X

    def unnormalize(self, X):
        X = X * self.std + self.mean
        return X

    def dock_transformation(self, center, bid, R, t):
        def tran_func(X):
            device = center.device
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).reshape(-1, 3).to(device).float()
            X = self.centering(X, center, bid)
            X = self.normalize(X).float()
            X = X @ R + t
            X = self.unnormalize(X)
            X = self.uncentering(X, center, bid)
            return X.squeeze().cpu().numpy()
        return tran_func
