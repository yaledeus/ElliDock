#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch

from data.dataset import test_complex_process

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom


def save_pdb(file_path, points):
    structure = Structure.Structure("null")
    model = Model.Model(0)
    structure.add(model)

    chain = Chain.Chain("A")
    model.add(chain)

    residue_num = 1
    for point in points:
        residue = Residue.Residue((' ', residue_num, ' '), "ALA", "")
        chain.add(residue)
        atom = Atom.Atom('CA', point, 0, 1, ' ', 'CA', residue_num, None, "C")
        residue.add(atom)
        residue_num += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(file_path)


def save_ply(file_path, points):
    with open(file_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % len(points))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for p in points:
            f.write("%f %f %f\n" % (p[0], p[1], p[2]))


def plot_point_cloud_mesh(ax, points, c='red'):
    tri = Delaunay(points)

    tri_verts = points[tri.simplices]
    tri_coords = tri_verts.reshape(-1, tri_verts.shape[-1])

    # tri_norms = np.cross(tri_verts[:, 1, :] - tri_verts[:, 0, :], tri_verts[:, 2, :] - tri_verts[:, 0, :])
    # tri_norms /= np.linalg.norm(tri_norms, axis=-1)[:, None]

    ax.plot_trisurf(tri_coords[:, 0], tri_coords[:, 1], tri_coords[:, 2], triangles=tri.simplices,
                    shade=True, facecolor=c)


def quadratic_surface_mesh(params, unnorm_func):
    """
    :param ax:
    :param a, b, c: ax^2 + by^2 + cz = 0
    :param R, t: x' = Rx + t
    :return:
    """
    N = 50

    a, b, c, R, t = params

    # Create meshgrid
    x_max = y_max = 1.
    x = np.linspace(-x_max, x_max, N)
    y = np.linspace(-y_max, y_max, N)
    X, Y = np.meshgrid(x, y)

    Z = -(a * X**2 + b * Y**2) / c

    undock_P = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
    dock_P = R @ undock_P + t[:, None]  # (3, *)

    # unnormalize
    undock_P = unnorm_func(undock_P.T)  # (*, 3)
    dock_P = unnorm_func(dock_P.T)      # (*, 3)

    return undock_P, dock_P


def main(args):
    # load config of the model
    config_path = os.path.join(os.path.split(args.ckpt)[0], '..', 'train_config.json')
    with open(config_path, 'r') as fin:
        config = json.load(fin)

    # model_type
    model_type = config.get('model_type', 'ExpDock')
    print(f'Model type: {model_type}')

    # load test set
    if args.dataset == 'DB5.5':
        test_path = './test_sets_pdb/db5_test_random_transformed'
    # elif args.dataset == 'DIPS':
    #     test_path = './test_sets_pdb/dips_test_random_transformed'
    elif args.dataset == 'SAbDab':
        test_path = './test_sets_pdb/sabdab_test_random_transformed'
    else:
        raise ValueError(f'model type {model_type} not implemented')

    test_pdbs = []
    with open(os.path.join(test_path, 'test.txt'), 'r') as fp:
        for item in fp.readlines():
            test_pdbs.append(item.strip())

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # create save dir
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for pdb_name in tqdm(test_pdbs):
        ligand_bound_path = os.path.join(test_path, 'complexes', pdb_name + '_l_b_COMPLEX.pdb')
        receptor_unbound_path = os.path.join(test_path, 'random_transformed', pdb_name + '_r_b.pdb')

        batch = test_complex_process(ligand_bound_path, receptor_unbound_path)
        # inference
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # docking
            re_surface, li_surface, unnorm_trans_list = model.pred_elli_surface(**batch)    # (N, 3)

        re_undock_mesh, re_dock_mesh = quadratic_surface_mesh(re_surface[0], unnorm_trans_list[0])
        li_undock_mesh, li_dock_mesh = quadratic_surface_mesh(li_surface[0], unnorm_trans_list[0])

        save_pdb(os.path.join(save_dir, f'{pdb_name}_re_mesh_ud.pdb'), re_undock_mesh)
        save_pdb(os.path.join(save_dir, f'{pdb_name}_li_mesh_ud.pdb'), li_undock_mesh)
        save_pdb(os.path.join(save_dir, f'{pdb_name}_re_mesh_d.pdb'), re_dock_mesh)
        save_pdb(os.path.join(save_dir, f'{pdb_name}_li_mesh_d.pdb'), li_dock_mesh)


def parse():
    parser = argparse.ArgumentParser(description='Docking given antibody-antigen complex')
    parser.add_argument('--dataset', type=str, required=True, default='DB5.5', choices=['DB5.5', 'SAbDab'])
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated meshes')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
