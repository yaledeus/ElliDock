#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SabDabDataset
from data.bio_parse import CA_INDEX
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
    if model_type == 'ExpDock':
        Dataset = SabDabDataset
    else:
        raise ValueError(f'model type {model_type} not implemented')

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

    idx = 0
    crmsds, irmsds = [], []

    for batch in tqdm(test_loader):
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # docking
            dock_X = model.dock(**batch)    # (N, 3)
            X_list = []
            bid = batch['bid']
            for i in range(bid[-1] + 1):
                X_list.append(dock_X[bid == i].cpu().numpy())

        for x in X_list:
            complex = test_set.data[idx]
            idx += 1
            gt_x = np.concatenate((complex.antibody_coord(), complex.antigen_coord()), axis=0)[:, CA_INDEX]
            seg = np.array([0 for _ in range(len(complex.antibody_seq()))] +
                           [1 for _ in range(len(complex.antigen_seq()))])
            assert x.shape[0] == gt_x.shape[0], 'docked coordinates dimension mismatch'
            crmsd = compute_crmsd(x, gt_x)
            irmsd = compute_irmsd(x, gt_x, seg)
            crmsds.append(crmsd)
            irmsds.append(irmsd)

    for name, val in zip(['RMSD(CA)'], [crmsds], [irmsds]):
        print(f'{name}: {sum(val) / len(val)}')


def parse():
    parser = argparse.ArgumentParser(description='Docking given antibody-antigen complex')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
