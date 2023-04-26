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


def local_orientation(coord):
    """
    Args:
        coord:  (N, n_channel, 3).
    Returns:
        local orientation, (N, 9).
    """
    coord_N = coord[:, N_INDEX]  # (N, 3)
    coord_CA = coord[:, CA_INDEX]
    coord_C = coord[:, C_INDEX]

    eps = 1e-6

    vec_u = coord_N - coord_CA  # (N, 3)
    vec_u = vec_u / (torch.norm(vec_u, dim=-1, keepdim=True) + eps)
    vec_t = coord_C - coord_CA  # (N, 3)
    vec_t = vec_t / (torch.norm(vec_t, dim=-1, keepdim=True) + eps)
    vec_n = torch.cross(vec_u, vec_t)  # (N, 3)
    vec_n = vec_n / (torch.norm(vec_n, dim=-1, keepdim=True) + eps)
    vec_v = torch.cross(vec_n, vec_u)  # (N, 3)
    vec_v = vec_v / (torch.norm(vec_v, dim=-1, keepdim=True) + eps)

    local_ori = torch.cat([vec_n, vec_u, vec_v], dim=1)  # (N, 9)

    return local_ori


def reletive_position_orientation(coord, edges):
    """
    Args:
        coord:  (N, n_channel, 3).
        edges: (2, n_edges).
    Returns:
        Inter-residue relative position, (n_edges, 3).
    """
    row, col = edges
    coord_CA_src = coord[row, CA_INDEX] # (n_edges, 3)
    coord_CA_dst = coord[col, CA_INDEX]

    coord_diff = coord_CA_src - coord_CA_dst        # (n_edges, 3)

    local_ori_src = local_orientation(coord[row])   # (n_edges, 9)
    local_ori_dst = local_orientation(coord[col])   # (n_edges, 9)

    vec_n_src, vec_u_src, vec_v_src = local_ori_src[:, :3], local_ori_src[:, 3:6], local_ori_src[:, 6:]
    vec_n_dst, vec_u_dst, vec_v_dst = local_ori_dst[:, :3], local_ori_dst[:, 3:6], local_ori_dst[:, 6:]

    local_coord_mat = torch.stack([vec_n_src, vec_u_src, vec_v_src], dim=1)    # (n_edges, 3, 3)

    rp = torch.matmul(local_coord_mat, coord_diff.unsqueeze(-1)).squeeze()     # (n_edges, 3)
    feat_q = torch.matmul(local_coord_mat, vec_n_dst.unsqueeze(-1)).squeeze()  # (n_edges, 3)
    feat_k = torch.matmul(local_coord_mat, vec_u_dst.unsqueeze(-1)).squeeze()  # (n_edges, 3)
    feat_t = torch.matmul(local_coord_mat, vec_v_dst.unsqueeze(-1)).squeeze()  # (n_edges, 3)

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
    :return: P @ R + t, R, t such that minimize RMSD(PR + t, Q)
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


def protein_surface_intersection(X, Y, sigma=25, gamma=10):
    """
    :param X: point cloud  to be referenced, (N, 3)
    :param Y: point cloud to be tested whether it is outside the protein, (M, 3)
    :param sigma, gamma: parameter
    :return: (M,)
    """
    return (gamma + sigma * torch.logsumexp(-(Y.unsqueeze(1).repeat(1, X.shape[0], 1) - X)
                                            .pow(2).sum(dim=-1) / sigma, dim=1))


def Gaussian_surface_fit(coord):
    """
    :param coord: coordinates in R3 to be fitted, (N, 3)
    :return: a := peak, bx := surface peak x-coord, y0 := surface peak y-coord,
            cx := surface peak x-std, cy := surface peak y-std
    """
    X, Y, Z = coord[:, 0], coord[:, 1], coord[:, 2]

    X = X.unsqueeze(1)  # (N, 1)
    Y = Y.unsqueeze(1)

    A = torch.cat([torch.ones_like(X), X, Y, X**2, Y**2], dim=1)    # (N, 5)
    params = torch.linalg.pinv(A.T @ A) @ A.T @ Z   # (5,)

    cx, cy = torch.sqrt(-0.5 / params[3]), torch.sqrt(-0.5 / params[4])
    x0, y0 = params[1] * cx**2, params[2] * cy**2
    a = (params[0] + 0.5 * x0 * params[1] + 0.5 * y0 * params[2]).exp()

    return (a, x0, y0, cx, cy)


def quadratic_surface_fit(coord):
    """
    :param coord: coordinates in R3 to be fitted, (N, 3)
    :return: standard quadratic coefficient, (6,)
             R (3, 3), t (3,), where XR + t fit the standard quadratic surface
    """
    device = coord.device
    scale = 1e-2

    X, Y, Z = coord[:, 0], coord[:, 1], coord[:, 2]

    X = X.unsqueeze(1)  # (N, 1)
    Y = Y.unsqueeze(1)
    Z = Z.unsqueeze(1)

    A = torch.cat([X**2, Y**2, Z**2, X*Y, Y*Z, Z*X, X, Y, Z], dim=1)    # (N, 9)
    b = scale * torch.ones_like(X).squeeze()        # (N,)
    params = torch.linalg.pinv(A.T @ A) @ A.T @ b   # (9,)

    transform_mat = torch.tensor(
        [[params[0], 0.5 * params[3], 0.5 * params[5], 0.5 * params[6]],
         [0.5 * params[3], params[1], 0.5 * params[4], 0.5 * params[7]],
         [0.5 * params[5], 0.5 * params[4], params[2], 0.5 * params[8]],
         [0.5 * params[6], 0.5 * params[7], 0.5 * params[8], -scale]], device=device
    )   # (4, 4)

    quadratic_mat = transform_mat[:3, :3]   # (3, 3), symmetric matrix
    L, Q = torch.linalg.eigh(quadratic_mat) # L: (3,), Q: (3, 3)

    # primary coefficient after rotation
    primary = Q.T @ transform_mat[:3, 3]
    non_zero = L != 0
    # translate vector
    t = torch.zeros_like(L)
    t[non_zero] = primary[non_zero] / L[non_zero]
    # const coefficient
    cons = scale + torch.sum(primary[non_zero] ** 2 / L[non_zero])
    # quadratic coefficient
    quad_params = L * scale / cons
    # primary coefficient
    prim_params = torch.zeros_like(L)
    prim_params[~non_zero] = 2 * primary[~non_zero] * scale / cons
    # standard coefficient
    std_params = torch.cat([quad_params, prim_params], dim=0)

    return std_params, Q, t


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


def modified_gram_schmidt(vecs):
    """
    :param vecs: m vectors in Rn, (n, m)
    :return: m orthogonal basis in Rn, (n, m)
    """
    n, m = vecs.shape
    assert n >= m   # otherwise orthogonal basis not exist
    q, r = torch.linalg.qr(vecs)
    return q


def recon_orthogonal_matrix(Q, v):
    """
    :param Q: any matrix with shape (n, m), n >= m
    :param v: a new basis, (n,)
    :return: a new orthogonal matrix Q' containing norm(v) in the last dimension
    """
    assert Q.shape[0] == v.shape[0] and Q.shape[0] >= Q.shape[1]
    norm_v = v / torch.norm(v)      # normalize
    norm_v = norm_v.unsqueeze(1)    # (n, 1)

    # Gram-Schmidt
    q = modified_gram_schmidt(torch.hstack([norm_v, Q[:, :-1]]))
    new_Q = torch.hstack([q[:, 1:], norm_v])

    return new_Q


def orthogonal_null_space_vec(A, init_u=None):
    """
    :param A: row-orthogonal matrix, (3, 9)
    :param init_u: initial vector, (9,), default: None
    :return: v s.t. Av=0 and v[:3], v[3:6], v[6:] constitutes an orthogonal basis of R3 space
    """
    device = A.device
    def target_derivative(A1, A2, A3, u1, u2, u3):
        partial_target2u1 = 2 * (A1.T @ A1 @ u1 + A1.T @ A2 @ u2 + A1.T @ A3 @ u3)
        partial_target2u2 = 2 * (A2.T @ A1 @ u1 + A2.T @ A2 @ u2 + A2.T @ A3 @ u3)
        partial_target2u3 = 2 * (A3.T @ A1 @ u1 + A3.T @ A2 @ u2 + A3.T @ A3 @ u3)
        return (partial_target2u1, partial_target2u2, partial_target2u3)

    def punishment_derivative(u1, u2, u3):
        partial_punish2u1 = 2 * ((torch.dot(u1, u1) - 1) * u1 + torch.dot(u1, u2) * u2 + torch.dot(u1, u3) * u3)
        partial_punish2u2 = 2 * (torch.dot(u2, u1) * u1 + (torch.dot(u2, u2) - 1) * u2 + torch.dot(u2, u3) * u3)
        partial_punish2u3 = 2 * (torch.dot(u3, u1) * u1 + torch.dot(u3, u2) * u2 + (torch.dot(u3, u3) - 1) * u3)
        return (partial_punish2u1, partial_punish2u2, partial_punish2u3)

    def derivative(A1, A2, A3, u1, u2, u3, rho=1.):
        partial_target2u1, partial_target2u2, partial_target2u3 = target_derivative(A1, A2, A3, u1, u2, u3)
        partial_punish2u1, partial_punish2u2, partial_punish2u3 = punishment_derivative(u1, u2, u3)
        return (
            partial_target2u1 + 0.5 * rho * partial_punish2u1,
            partial_target2u2 + 0.5 * rho * partial_punish2u2,
            partial_target2u3 + 0.5 * rho * partial_punish2u3
            )


    assert A.shape == (3, 9)
    # A1, A2, A3: (3, 3) matrix, under most conditions invertible
    A1 = A[:, :3]
    A2 = A[:, 3:6]
    A3 = A[:, 6:]
    # u1, u2, u3: (3,) vector
    if not init_u is None:
        u1 = init_u[:3].clone()
        u2 = init_u[3:6].clone()
        u3 = init_u[6:].clone()
    else:
        u1 = torch.rand(3).to(device)
        u2 = torch.rand(3).to(device)
        u3 = torch.rand(3).to(device)

    iteration = 50
    lr = torch.linspace(1e-1, 1e-3, steps=iteration)
    rho = 1.

    for i in range(iteration):
        deriv2u1, deriv2u2, deriv2u3 = derivative(A1, A2, A3, u1, u2, u3, rho)
        u1 = u1 - lr[i] * deriv2u1
        u2 = u2 - lr[i] * deriv2u2
        u3 = u3 - lr[i] * deriv2u3

    v = torch.hstack([u1, u2, u3])
    return v
"""
# test orthogonal null space vec
A = torch.rand(3, 9)
v = orthogonal_null_space_vec(A)
print(f"||Av||: {torch.norm(A @ v)}")
print(f"||v1||: {torch.norm(v[:3])}")
print(f"||v2||: {torch.norm(v[3:6])}")
print(f"||v3||: {torch.norm(v[6:])}")
print(f"v1 @ v2: {torch.dot(v[:3], v[3:6])}")
print(f"v2 @ v3: {torch.dot(v[3:6], v[6:])}")
print(f"v3 @ v1: {torch.dot(v[6:], v[:3])}")
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
        X = (X - self.mean) / torch.mean(self.std)
        return X

    def unnormalize(self, X):
        X = X * torch.mean(self.std) + self.mean
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
