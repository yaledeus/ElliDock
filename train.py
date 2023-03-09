#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse
import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

setup_seed(SEED)

########### Import your packages below ##########
from trainer import TrainConfig


def parse():
    parser = argparse.ArgumentParser(description='training')

    # TODO: add arguments
    # training related
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='patience before early stopping')
    parser.add_argument('--save_topk', type=int, default=-1,
                        help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # model
    parser.add_argument('--model_type', type=str, required=True, choices=['ExpDock'], help='Type of model')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')

    return parser.parse_args()


def main(args):
    ########### load your train / valid set ###########
    # TODO: train_set, valid_set
    train_set = None
    valid_set = None

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(args.save_dir, args.lr, args.max_epoch,
                         patience=args.patience,
                         grad_clip=args.grad_clip,
                         save_topk=args.save_topk)

    # TODO: define model and Trainer
    model = None
    Trainer = None
    collate_fn = None

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
