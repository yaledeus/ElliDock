#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
import json
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DBDataset, DIPSDataset, SabDabDataset
from data.bio_parse import CA_INDEX, gen_docked_pdb
from evaluate import compute_crmsd, compute_irmsd
from module.model import sample_transformation

import time


def main(args):
    model_type = args.model_type
    print(f'Model type: {model_type}')

    # load test set
    if args.dataset == 'DB5.5':
        test_set = DBDataset(args.test_set)
        collate_fn = DBDataset.collate_fn
    elif args.dataset == 'DIPS':
        test_set = DIPSDataset(args.test_set)
        collate_fn = DIPSDataset.collate_fn
    elif args.dataset == 'SabDab':
        test_set = SabDabDataset(args.test_set)
        collate_fn = SabDabDataset.collate_fn
    else:
        raise ValueError(f'Dataset {args.dataset} not implemented')

    test_loader = DataLoader(test_set, batch_size=1,
                             num_workers=8,
                             shuffle=False,
                             collate_fn=collate_fn
                             )

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

    for batch in tqdm(test_loader):
        try:
            gt_X = batch['X'][:, CA_INDEX].numpy()
            # random transformation
            X = batch['X'].clone().to(device)
            X = model.normalizer.centering(X, batch['center'].to(device), batch['bid'].to(device))
            X = model.normalizer.normalize(X).float()
            rot, trans = sample_transformation(batch['bid'].to(device))
            receptor_idx = torch.logical_and(batch['Seg'] == 0, batch['bid'] == 0)
            X[receptor_idx] = X[receptor_idx] @ rot[0] + trans[0]
            X = model.normalizer.unnormalize(X)
            X = model.normalizer.uncentering(X, batch['center'].to(device), batch['bid'].to(device))
            batch['X'] = X.clone()
        except Exception as e:
            print(f'[!] Error: {e}')
            continue

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
        # gen_docked_pdb(pdb_name, receptor_unbound_path, save_dir, dock_trans_list[0])

    end = time.time()
    print(f'total runtime: {end - start}')

    data = {
        "model_type": model_type.upper(),
        "IRMSD": a_irmsds,
        "CRMSD": a_crmsds
    }
    data = json.dumps(data, indent=4)
    with open(os.path.join(save_dir, 'data.json'), 'w') as fp:
        fp.write(data)

    for name, val in zip(['CRMSD(aligned)', 'IRMSD(aligned)', 'CRMSD', 'IRMSD'],
                         [a_crmsds, a_irmsds, u_crmsds, u_irmsds]):
        print(f'{name} median: {np.median(val)}', end=' ')
        print(f'mean: {np.mean(val)}', end=' ')
        print(f'std: {np.std(val)}')


def parse():
    parser = argparse.ArgumentParser(description='Docking given antibody-antigen complex')
    parser.add_argument('--model_type', type=str, default='ElliDock', choices=['ElliDock'])
    parser.add_argument('--dataset', type=str, required=True, default='DB5.5', choices=['SabDab', 'DB5.5', 'DIPS'])
    parser.add_argument('--test_set', type=str, required=True, help='path to test set')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
