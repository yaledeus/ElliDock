from Bio import PDB
from tqdm import tqdm
import argparse
import json
import os
import sys
sys.path.append('..')
from utils.geometry import rand_rotation_matrix
import numpy as np


def parse():
    parser = argparse.ArgumentParser(description='Docking given antibody-antigen complex')
    parser.add_argument('--test_set', type=str, required=True, help='path to test set')
    return parser.parse_args()


def ab_ag_split(file_path):
    par_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    dataset_dir = os.path.join(par_dir, 'test_sets_pdb', 'sabdab_test_random_transformed')
    complex_dir = os.path.join(dataset_dir, 'complexes')
    random_dir = os.path.join(dataset_dir, 'random_transformed')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(complex_dir, exist_ok=True)
    os.makedirs(random_dir, exist_ok=True)
    with open(file_path, 'r') as fin:
        lines = fin.read().strip().split('\n')
    pdbs = []
    for line in tqdm(lines):
        item = json.loads(line)
        pdb = item['pdb']
        pdbs.append(pdb)
        ab_complex_name, ag_complex_name = f'{pdb}_r_b_COMPLEX.pdb', f'{pdb}_l_b_COMPLEX.pdb'
        ab_random_name, ag_random_name = f'{pdb}_r_b.pdb', f'{pdb}_l_b.pdb'
        hchain, lchain, agchains = item['heavy_chain'], item['light_chain'], item['antigen_chains']
        parser = PDB.PDBParser()
        structure = parser.get_structure(pdb, item['pdb_data_path'])
        ab_writer = PDB.PDBIO()
        ag_writer = PDB.PDBIO()
        ab_model = PDB.Model.Model(f'{pdb}_r')
        ag_model = PDB.Model.Model(f'{pdb}_l')

        for model in structure:
            for chain in model:
                if chain.id in [hchain, lchain]:
                    ab_model.add(chain)
                elif chain.id in agchains:
                    ag_model.add(chain)

        ab_writer.set_structure(ab_model)
        ab_complex_path = os.path.join(complex_dir, ab_complex_name)
        ab_writer.save(ab_complex_path)

        ag_writer.set_structure(ag_model)
        ag_complex_path = os.path.join(complex_dir, ag_complex_name)
        ag_writer.save(ag_complex_path)

        q = rand_rotation_matrix().numpy()
        t = np.random.rand(3) * 15.0        # std ~ 15.0

        ab_structure = parser.get_structure(pdb, ab_complex_path)
        for model in ab_structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom_coord = atom.get_coord()
                        atom.set_coord(atom_coord @ q + t)
        io = PDB.PDBIO()
        io.set_structure(ab_structure)
        io.save(os.path.join(random_dir, ab_random_name))

        q = rand_rotation_matrix().numpy()
        t = np.random.rand(3) * 15.0  # std ~ 15.0

        ag_structure = parser.get_structure(pdb, ag_complex_path)
        for model in ag_structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom_coord = atom.get_coord()
                        atom.set_coord(atom_coord @ q + t)
        io = PDB.PDBIO()
        io.set_structure(ag_structure)
        io.save(os.path.join(random_dir, ag_random_name))


    with open(os.path.join(dataset_dir, 'test.txt'), 'w') as fp:
        for pdb in pdbs:
            fp.write(f'{pdb}\n')


if __name__ == "__main__":
    args = parse()
    ab_ag_split(args.test_set)
