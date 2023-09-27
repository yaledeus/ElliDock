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
    if args.model_type == 'ElliDock':
        # create save dir
        if args.save_dir is None:
            save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
        else:
            save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
    elif args.model_type == 'HDock':
        save_dir = os.path.join('./test_sets_pdb', 'hdock_results', f'{args.dataset}')
        os.makedirs(save_dir, exist_ok=True)
    elif args.model_type == 'Multimer':
        save_dir = f'./test_sets_pdb/multimer_results/{args.dataset}'
    elif args.model_type == 'DiffDock-PP':
        save_dir = f'./test_sets_pdb/diffdock_results'
    elif args.model_type == 'EquiDock':
        save_dir = f'./test_sets_pdb/equidock_results/{args.dataset}'
    else:
        raise ValueError(f'Model type {args.model_type} not implemented')
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
    elif args.dataset == 'DIPS':
        test_path = './test_sets_pdb/dips_test_random_transformed'
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

    a_crmsds, a_irmsds, u_crmsds, u_irmsds, dockqs, tmscores, intersections = [], [], [], [], [], [], []

    start = time.time()

    save_dir = create_save_dir(args)

    if args.ckpt:
        # load model
        model = torch.load(args.ckpt, map_location='cpu')
        device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
        model.to(device)
        model.eval()

    for pdb_name in tqdm(test_pdbs):

        ligand_bound_path = os.path.join(test_path, 'complexes', pdb_name + '_l_b_COMPLEX.pdb')
        receptor_bound_path = os.path.join(test_path, 'complexes', pdb_name + '_r_b_COMPLEX.pdb')
        ligand_unbound_path = os.path.join(test_path, 'random_transformed', pdb_name + '_l_b.pdb')
        receptor_unbound_path = os.path.join(test_path, 'random_transformed', pdb_name + '_r_b.pdb')

        if model_type == 'ElliDock':
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

            dock_X = dock_X.cpu().numpy()

            pred_receptor_path = os.path.join(save_dir, f'{pdb_name}_r_d.pdb')
            gen_docked_pdb(receptor_unbound_path, pred_receptor_path, dock_trans_list[0])
            pred_complex_path = os.path.join(save_dir, f'{pdb_name}_predicted.pdb')
            gt_complex_path = os.path.join(save_dir, f'{pdb_name}_gt.pdb')

            monomer2complex([receptor_bound_path, ligand_bound_path], gt_complex_path)
            monomer2complex([pred_receptor_path, ligand_bound_path], pred_complex_path)

        elif model_type == 'HDock':
            hdock_dir = os.path.abspath('./HDOCKlite-v1.1')

            shutil.copy(ligand_unbound_path, ligand_tmp_path := os.path.join(hdock_dir, pdb_name + '_l_b.pdb'))
            shutil.copy(receptor_bound_path, receptor_tmp_path := os.path.join(hdock_dir, pdb_name + '_r_b.pdb'))

            gt = test_complex_process(ligand_bound_path, receptor_bound_path)
            gt_X = gt['X'][:, CA_INDEX].numpy()
            gt_complex_path = os.path.join(save_dir, f'{pdb_name}_gt.pdb')
            pred_complex_path = os.path.join(save_dir, f'{pdb_name}_predicted.pdb')
            monomer2complex([receptor_bound_path, ligand_bound_path], gt_complex_path)
            # inference
            try:
                subprocess.run(f'cd {hdock_dir} && ./hdock {receptor_tmp_path} {ligand_tmp_path} -out Hdock.out',
                               shell=True)
                subprocess.run(f'cd {hdock_dir} && ./createpl Hdock.out top100.pdb -nmax 100 -complex -models',
                               shell=True)
                dock_X = BaseComplex.from_pdb(
                    os.path.join(hdock_dir, 'model_1.pdb'), ligand_bound_path
                ).ligand_coord()[:, CA_INDEX]
            except:
                print(f'Docking on {pdb_name} failed, skip.')
                continue
            shutil.copy(os.path.join(hdock_dir, 'model_1.pdb'), pred_complex_path)
            subprocess.run(f'cd {hdock_dir} && rm Hdock.out && rm model*.pdb', shell=True)

        elif model_type == 'Multimer':
            result_dir = f'./test_sets_pdb/multimer_results/{args.dataset}'
            gt = test_complex_process(ligand_bound_path, receptor_bound_path)
            gt_X = gt['X'][:, CA_INDEX].numpy()
            if not os.path.exists(pdb_dir := os.path.join(result_dir, f'test_{pdb_name}')):
                continue
            pred_receptor_path = os.path.join(pdb_dir, f'{pdb_name}-receptor-full.pdb')
            pred_ligand_path = os.path.join(pdb_dir, f'{pdb_name}-ligand-full.pdb')
            gt_complex_path = os.path.join(pdb_dir, f'{pdb_name}_gt.pdb')
            pred_complex_path = os.path.join(pdb_dir, f'{pdb_name}_predicted.pdb')
            monomer2complex([receptor_bound_path, ligand_bound_path], gt_complex_path)
            monomer2complex([pred_receptor_path, pred_ligand_path], pred_complex_path)
            dock_X = BaseComplex.from_pdb(
                pred_complex_path, ligand_bound_path
            ).ligand_coord()[:, CA_INDEX]

        elif model_type == 'DiffDock-PP':
            result_dir = f'./test_sets_pdb/diffdock_results'
            gt = test_complex_process(ligand_bound_path, receptor_bound_path)
            gt_X = gt['X'][:, CA_INDEX].numpy()
            if not os.path.exists(pdb_dir := os.path.join(result_dir, f'{pdb_name}')):
                continue
            pred_receptor_path = os.path.join(pdb_dir, f'{pdb_name}-receptor-full.pdb')
            pred_ligand_path = os.path.join(pdb_dir, f'{pdb_name}-ligand-full.pdb')
            gt_complex_path = os.path.join(pdb_dir, f'{pdb_name}_gt.pdb')
            pred_complex_path = os.path.join(pdb_dir, f'{pdb_name}_predicted.pdb')
            monomer2complex([receptor_bound_path, ligand_bound_path], gt_complex_path)
            monomer2complex([pred_receptor_path, pred_ligand_path], pred_complex_path)
            dock_X = BaseComplex.from_pdb(
                pred_complex_path, ligand_bound_path
            ).ligand_coord()[:, CA_INDEX]

        elif model_type == 'EquiDock':
            result_dir = f'./test_sets_pdb/equidock_results/{args.dataset}'
            save_dir = result_dir
            gt = test_complex_process(ligand_bound_path, receptor_bound_path)
            gt_X = gt['X'][:, CA_INDEX].numpy()
            pred_ligand_path = os.path.join(save_dir, f'{pdb_name}_l_b_EQUIDOCK.pdb')
            gt_complex_path = os.path.join(save_dir, f'{pdb_name}_gt.pdb')
            pred_complex_path = os.path.join(save_dir, f'{pdb_name}_predicted.pdb')
            monomer2complex([receptor_bound_path, ligand_bound_path], gt_complex_path)
            monomer2complex([receptor_bound_path, pred_ligand_path], pred_complex_path)
            dock_X = BaseComplex.from_pdb(
                pred_complex_path, ligand_bound_path
            ).ligand_coord()[:, CA_INDEX]

        else:
            raise ValueError(f'Model type {model_type} not implemented')

        Seg = gt['Seg'].numpy()
        dock_X_re, dock_X_li = torch.tensor(dock_X[Seg == 0]), torch.tensor(dock_X[Seg == 1])
        assert dock_X.shape[0] == gt_X.shape[0], 'coordinates dimension mismatch'
        aligned_crmsd = compute_crmsd(dock_X, gt_X, aligned=False)
        aligned_irmsd = compute_irmsd(dock_X, gt_X, Seg, aligned=False)
        unaligned_crmsd = compute_crmsd(dock_X, gt_X, aligned=True)
        unaligned_irmsd = compute_irmsd(dock_X, gt_X, Seg, aligned=True)
        intersection = float(protein_surface_intersection(dock_X_re, dock_X_li).relu().mean() +
            protein_surface_intersection(dock_X_li, dock_X_re).relu().mean())
        a_crmsds.append(aligned_crmsd)
        a_irmsds.append(aligned_irmsd)
        u_crmsds.append(unaligned_crmsd)
        u_irmsds.append(unaligned_irmsd)
        tmscores.append(tm_score(gt_complex_path, pred_complex_path))
        if test_desc.get(pdb_name):
            rchain_id, lchain_id = test_desc[pdb_name]
            dockqs.append(dockQ(pred_complex_path, gt_complex_path,
                          rchain_id=rchain_id, lchain_id=lchain_id))
        else:
            dockqs.append(dockQ(pred_complex_path, gt_complex_path))
        intersections.append(intersection)
        os.remove(gt_complex_path)


    end = time.time()
    print(f'total runtime: {end - start}')

    data = {
        "model_type": model_type.upper(),
        "IRMSD": a_irmsds,
        "CRMSD": a_crmsds,
        "TMscore": tmscores,
        "DockQ": dockqs,
        "intersection": intersections
    }
    data = json.dumps(data, indent=4)
    with open(os.path.join(save_dir, 'data.json'), 'w') as fp:
        fp.write(data)

    for name, val in zip(['CRMSD(aligned)', 'IRMSD(aligned)', 'TMscore', 'DockQ', 'CRMSD', 'IRMSD'],
                         [a_crmsds, a_irmsds, tmscores, dockqs, u_crmsds, u_irmsds]):
        print(f'{name} median: {np.median(val)}', end=' ')
        print(f'mean: {np.mean(val)}', end=' ')
        print(f'std: {np.std(val)}')


def parse():
    parser = argparse.ArgumentParser(description='Docking given antibody-antigen complex')
    parser.add_argument('--model_type', type=str, default='ElliDock', choices=['ElliDock', 'HDock', 'Multimer', 'DiffDock-PP', 'EquiDock'])
    parser.add_argument('--dataset', type=str, required=True, default='DB5', choices=['SAbDab', 'DB5', 'DIPS'])
    parser.add_argument('--ckpt', type=str, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
