#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse
import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from data import SabDabDataset, DBDataset, DIPSDataset

setup_seed(SEED)

########### Import your packages below ##########
from trainer import TrainConfig


def parse():
    parser = argparse.ArgumentParser(description='training')

    # data
    parser.add_argument('--dataset', type=str, required=True, default='SabDab', choices=['SabDab', 'DB5.5', 'DIPS'])
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')

    # training related
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='patience before early stopping')
    parser.add_argument('--save_topk', type=int, default=-1,
                        help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=8)

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # model
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--model_type', type=str, required=True, choices=['ExpDock'], help='Type of model')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--n_keypoints', type=int, default=10, help='Number of keypoints')
    parser.add_argument('--att_heads', type=int, default=4, help='Number of attention heads')

    return parser.parse_args()


def main(args):
    ########### load your train / valid set ###########
    if args.dataset == 'SabDab':
        train_set = SabDabDataset(args.train_set)
        valid_set = SabDabDataset(args.valid_set)
        collate_fn = SabDabDataset.collate_fn
        mean = [-0.49545217, 0.2199743, 0.12866335]
        std = [14.85880611, 14.99745863, 17.27655463]
    elif args.dataset == 'DB5.5':
        train_set = DBDataset(args.train_set)
        valid_set = DBDataset(args.valid_set)
        collate_fn = DBDataset.collate_fn
        mean = [0.07196493, 0.06579519, 0.63590974]
        std = [14.12718586, 14.21111747, 15.96787876]
    elif args.dataset == 'DIPS':
        train_set = DIPSDataset(args.train_set)
        valid_set = DIPSDataset(args.valid_set)
        collate_fn = DIPSDataset.collate_fn
        mean = [0.05055815, -0.47526212, 0.21798682]
        std = [14.0893815, 14.69939329, 14.96120875]
    else:
        raise NotImplemented(f'model {args.model_type} not implemented')

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(args.save_dir, args.lr, args.max_epoch,
                         patience=args.patience,
                         grad_clip=args.grad_clip,
                         save_topk=args.save_topk)

    if args.model_type == 'ExpDock':
        from trainer import ExpDockTrainer as Trainer
        from module import ExpDock
        if not args.ckpt:
            model = ExpDock(args.embed_dim, args.hidden_size, k_neighbors=args.k_neighbors,
                            att_heads=args.att_heads, n_layers=args.n_layers,
                            n_keypoints=args.n_keypoints, dropout=args.dropout,
                            mean=mean, std=std)
        else:
            model = torch.load(args.ckpt, map_location='cpu')


    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle)
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)

    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = parse()
    main(args)
