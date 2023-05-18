#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch

from data.dataset import test_complex_process
from data.bio_parse import CA_INDEX, gen_docked_pdb
from evaluate import compute_crmsd, compute_irmsd

import time


def main(args):
    # load config of the model
    # config_path = os.path.join(os.path.split(args.ckpt)[0], '..', 'train_config.json')
    # with open(config_path, 'r') as fin:
    #     config = json.load(fin)
    #
    # # model_type
    # model_type = config.get('model_type', 'ElliDock')
    model_type = 'ElliDock'
    print(f'Model type: {model_type}')

    # load test set
    if args.dataset == 'DB5.5':
        test_path = './test_sets_pdb/db5_test_random_transformed'
    elif args.dataset == 'DIPS':
        test_path = './test_sets_pdb/dips_test_random_transformed'
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

    a_crmsds, a_irmsds, u_crmsds, u_irmsds = [], [], [], []

    start = time.time()

    for pdb_name in tqdm(test_pdbs):
        ligand_bound_path = os.path.join(test_path, 'complexes', pdb_name + '_l_b_COMPLEX.pdb')
        receptor_bound_path = os.path.join(test_path, 'complexes', pdb_name + '_r_b_COMPLEX.pdb')
        receptor_unbound_path = os.path.join(test_path, 'random_transformed', pdb_name + '_r_b.pdb')

        batch = test_complex_process(ligand_bound_path, receptor_unbound_path)
        gt = test_complex_process(ligand_bound_path, receptor_bound_path)
        gt_X = gt['X'][:, CA_INDEX].numpy()
        # inference
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # docking
            dock_X, dock_trans_list = model.dock(**batch)    # (N, 3)

        Seg = batch['Seg'].cpu().numpy()
        dock_X = dock_X.cpu().numpy()
        assert dock_X.shape[0] == gt_X.shape[0], 'coordinates dimension mismatch'

        aligned_crmsd = compute_crmsd(dock_X, gt_X, aligned=False)
        aligned_irmsd = compute_irmsd(dock_X, gt_X, Seg, aligned=False)
        unaligned_crmsd = compute_crmsd(dock_X, gt_X, aligned=True)
        unaligned_irmsd = compute_irmsd(dock_X, gt_X, Seg, aligned=True)
        a_crmsds.append(aligned_crmsd)
        a_irmsds.append(aligned_irmsd)
        u_crmsds.append(unaligned_crmsd)
        u_irmsds.append(unaligned_irmsd)

        # print(f'[+] generating docked receptor pdb file: {pdb_name}')
        gen_docked_pdb(pdb_name, receptor_unbound_path, save_dir, dock_trans_list[0])

    end = time.time()
    print(f'total runtime: {end - start}')

    for name, val in zip(['CRMSD(aligned)', 'IRMSD(aligned)', 'CRMSD', 'IRMSD'],
                         [a_crmsds, a_irmsds, u_crmsds, u_irmsds]):
        print(f'{name} median: {np.median(val)}', end=' ')
        print(f'mean: {np.mean(val)}', end=' ')
        print(f'std: {np.std(val)}')


def parse():
    parser = argparse.ArgumentParser(description='Docking given antibody-antigen complex')
    parser.add_argument('--dataset', type=str, required=True, default='DB5.5', choices=['DB5.5', 'DIPS'])
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
