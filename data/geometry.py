import torch
import torch.nn.functional as F
from .bio_parse import N_INDEX, CA_INDEX, C_INDEX


def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = torch.tensor(p2 - p1)
    v1 = torch.tensor(p0 - p1)
    v2 = torch.tensor(p3 - p2)
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
    coord_C = coord[:, CA_INDEX]

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
        edges: (2, n_edge).
    Returns:
        Inter-residue Phi and Psi angles, (n_edge, 2).
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
