#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SabDabDataset, DBDataset, DIPSDataset
from data.bio_parse import CA_INDEX, gen_docked_pdb
from module.model import sample_transformation
from evaluate import compute_crmsd, compute_irmsd


def main(args):
    # load config of the model
    config_path = os.path.join(os.path.split(args.ckpt)[0], '..', 'train_config.json')
    with open(config_path, 'r') as fin:
        config = json.load(fin)

    # model_type
    model_type = config.get('model_type', 'ExpDock')
    print(f'Model type: {model_type}')

    # load test set
    if args.dataset == 'SabDab':
        Dataset = SabDabDataset
    elif args.dataset == 'DB5.5':
        Dataset = DBDataset
    elif args.dataset == 'DIPS':
        Dataset = DIPSDataset
    else:
        raise ValueError(f'model type {model_type} not implemented')

    expansion = 15

    # generate docked receptor pdb file (only for DB5.5 recently)
    if args.dataset == 'DB5.5':
        pdb_base_path = os.path.join(os.path.dirname(args.test_set), 'structures')

    test_set = Dataset(args.test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=Dataset.collate_fn)

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

    batch_iter = 0
    crmsds, irmsds = [], []

    for batch in tqdm(test_loader):
        # clone original X
        gt_X = batch['X'][:, CA_INDEX].cpu().numpy()
        # inference
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # transformation
            rot, trans = sample_transformation(batch['bid'])
            for i in range(len(rot)):
                ab_idx = torch.logical_and(batch['Seg'] == 0, batch['bid'] == i)
                batch['X'][ab_idx] = batch['X'][ab_idx] @ rot[i] + trans[i] * expansion
            # docking
            dock_X, dock_trans_list = model.dock(**batch)    # (N, 3)

        bid = batch['bid'].cpu().numpy()
        Seg = batch['Seg'].cpu().numpy()
        dock_X = dock_X.cpu().numpy()
        assert dock_X.shape[0] == gt_X.shape[0], 'coordinates dimension mismatch'

        for i in range(bid[-1] + 1):
            x = dock_X[bid == i]
            gt_x = gt_X[bid == i]
            seg = Seg[bid == i]
            crmsd = compute_crmsd(x, gt_x)
            irmsd = compute_irmsd(x, gt_x, seg)
            crmsds.append(crmsd)
            irmsds.append(irmsd)
            # DB5.5 only
            if args.dataset == 'DB5.5':
                pdb_name = test_set.data[batch_iter * args.batch_size + i].pdb_name
                # print(f'[+] generating docked receptor pdb file: {pdb_name}')
                receptor_pdb_path = os.path.join(pdb_base_path, f'{pdb_name.upper()}_r_b.pdb')
                def full_trans_func(X):
                    if isinstance(X, np.ndarray):
                        X = torch.from_numpy(X).reshape(-1, 3).to(device).float()
                    X = X @ rot[i] + trans[i] * expansion
                    return dock_trans_list[i](X)
                gen_docked_pdb(pdb_name, receptor_pdb_path, save_dir, full_trans_func)

        batch_iter += 1

    for name, val in zip(['CRMSD', 'IRMSD'], [crmsds, irmsds]):
        print(f'{name} median: {np.median(val)}', end=' ')
        print(f'{name} mean: {np.mean(val)}', end=' ')
        print(f'{name} std: {np.std(val)}')


def parse():
    parser = argparse.ArgumentParser(description='Docking given antibody-antigen complex')
    parser.add_argument('--dataset', type=str, required=True, default='SabDab', choices=['SabDab', 'DB5.5', 'DIPS'])
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
