#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import shutil
import subprocess
from tqdm import tqdm

import numpy as np
import torch

from data.dataset import test_complex_process, BaseComplex
from data.bio_parse import CA_INDEX, gen_docked_pdb
from utils.geometry import protein_surface_intersection
from evaluate import compute_crmsd, compute_irmsd, tm_score, dockQ

from Bio import PDB

import time


def create_save_dir(args):
    # create save dir
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
    else:
        save_dir = args.save_dir

    return save_dir


def monomer2complex(monomers, save_path):
    parser = PDB.PDBParser(QUIET=True)
    comp_writer = PDB.PDBIO()
    comp_model = PDB.Model.Model('annoym')
    for mon in monomers:
        structure = parser.get_structure('annoym', mon)
        for model in structure:
            for chain in model:
                comp_model.add(chain)
    comp_writer.set_structure(comp_model)
    comp_writer.save(save_path)


def main(args):
    model_type = args.model_type
    print(f'Model type: {model_type}')

    test_desc = {}
    # load test set
    if args.dataset == 'DB5':
        test_path = './test_sets_pdb/db5_test_random_transformed'
        test_desc_path = os.path.join(test_path, 'test.json')
        with open(test_desc_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        for line in lines:
            item = json.loads(line)
            test_desc[item['pdb']] = [item['rchain'], item['lchain']]
    # elif args.dataset == 'DIPS':
    #     test_path = './test_sets_pdb/dips_test_random_transformed'
    elif args.dataset == 'SAbDab':
        test_path = './test_sets_pdb/sabdab_test_random_transformed'
        test_desc_path = os.path.join(test_path, 'test.json')
        with open(test_desc_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        for line in lines:
            item = json.loads(line)
            test_desc[item['pdb']] = [[item['heavy_chain'], item['light_chain']], item['antigen_chains']]
    else:
        raise ValueError(f'Dataset {args.dataset} not implemented')

    test_pdbs = []
    with open(os.path.join(test_path, 'test.txt'), 'r') as fp:
        for item in fp.readlines():
            test_pdbs.append(item.strip())

    save_dir = create_save_dir(args)

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    for pdb_name in tqdm(test_pdbs):

        ligand_unbound_path = os.path.join(test_path, 'random_transformed', pdb_name + '_l_b.pdb')
        receptor_unbound_path = os.path.join(test_path, 'random_transformed', pdb_name + '_r_b.pdb')

        batch = test_complex_process(ligand_unbound_path, receptor_unbound_path)
        # inference
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # docking
            dock_X, dock_trans_list = model.dock(**batch)    # (N, 3)

        pred_receptor_path = os.path.join(save_dir, f'{pdb_name}_r_d.pdb')
        gen_docked_pdb(receptor_unbound_path, pred_receptor_path, dock_trans_list[0])
        pred_complex_path = os.path.join(save_dir, f'{pdb_name}_predicted.pdb')

        monomer2complex([pred_receptor_path, ligand_unbound_path], pred_complex_path)
        os.remove(pred_receptor_path)


def parse():
    parser = argparse.ArgumentParser(description='Docking given antibody-antigen complex')
    parser.add_argument('--dataset', type=str, required=True, default='DB5', choices=['DB5', 'SAbDab'])
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
