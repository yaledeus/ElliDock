import shutil
import os
import csv
from dataset import BaseComplex
from bio_parse import AA_NAMES_1


def preprocess(dataset):
    root = os.getcwd()
    src_dir = os.path.join(root, '../test_sets_pdb', f'{dataset}_test_random_transformed')
    dst_dir = os.path.join(root, '../test_sets_pdb', f'{dataset}_multimer')
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(src_dir, 'test.txt'), 'r') as fp:
        lines = fp.readlines()
    dst_csv = os.path.join(dst_dir, 'splits_test.csv')
    data = []
    for line in lines:
        ligand_gt = os.path.join(src_dir, 'complexes', f'{line.strip()}_l_b_complex.pdb')
        receptor_gt = os.path.join(src_dir, 'complexes', f'{line.strip()}_r_b_complex.pdb')
        compl = BaseComplex.from_pdb(ligand_gt, receptor_gt)
        seq = ''.join([AA_NAMES_1[a_idx] for a_idx in compl.receptor_seq()]) + ':' + \
              ''.join([AA_NAMES_1[a_idx] for a_idx in compl.ligand_seq()])
        data.append([line.strip(), seq])
    with open(dst_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'sequence'])
        writer.writerows(data)


if __name__ == "__main__":
    preprocess('db5')
