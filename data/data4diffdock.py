import shutil
import os
import csv
import json
from Bio import PDB
from tqdm import tqdm


def preprocess(dataset):
    root = os.getcwd()
    src_dir = os.path.join(root, '../test_sets_pdb', f'{dataset}_test_random_transformed')
    dst_dir = os.path.join(root, '../test_sets_pdb', f'{dataset}_diffdock')
    os.makedirs(dst_dir, exist_ok=True)
    structure_dir = os.path.join(dst_dir, 'structures')
    os.makedirs(structure_dir)
    with open(os.path.join(src_dir, 'test.txt'), 'r') as fp:
        lines = fp.readlines()
    dst_csv = os.path.join(dst_dir, 'splits_test.csv')
    data = []
    for line in lines:
        data.append([line.strip(), 'test'])
        ligand_gt_src = os.path.join(src_dir, 'complexes', f'{line.strip()}_l_b_complex.pdb')
        ligand_gt_dst = os.path.join(structure_dir, f'{line.strip()}_l_b.pdb')
        receptor_gt_src = os.path.join(src_dir, 'complexes', f'{line.strip()}_r_b_complex.pdb')
        receptor_gt_dst = os.path.join(structure_dir, f'{line.strip()}_r_b.pdb')
        shutil.copy(ligand_gt_src, ligand_gt_dst)
        shutil.copy(receptor_gt_src, receptor_gt_dst)
    with open(dst_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['path', 'split'])
        writer.writerows(data)


def sabdab_for_diffdock(data_dir):
    splits = {
        "train": "train.json",
        "val": "valid.json",
        "test": "test.json"
    }
    train_set_dir = os.path.join(data_dir, 'structures_diffdock')
    os.makedirs(train_set_dir, exist_ok=True)
    csv_path = os.path.join(train_set_dir, 'data_file.csv')
    data = []
    for split in splits:
        split_path = os.path.join(data_dir, splits[split])
        with open(split_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        for line in tqdm(lines):
            item = json.loads(line)
            pdb = item['pdb']
            data.append([pdb, split])
            ab_complex_name, ag_complex_name = f'{pdb}_r_b.pdb', f'{pdb}_l_b.pdb'
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
            ab_complex_path = os.path.join(train_set_dir, ab_complex_name)
            ab_writer.save(ab_complex_path)

            ag_writer.set_structure(ag_model)
            ag_complex_path = os.path.join(train_set_dir, ag_complex_name)
            ag_writer.save(ag_complex_path)

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['path', 'split'])
        writer.writerows(data)


if __name__ == "__main__":
    preprocess('sabdab')
    # import sys
    # arg = sys.argv[1]
    # sabdab_for_diffdock(arg)
