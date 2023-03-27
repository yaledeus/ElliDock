from bio_parse import DBComplex, DIPSComplex, CA_INDEX
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def DB5_statistics(pdb_dir):
    pdb_names = []
    cplx_mean = []
    cplx_std = []
    for root, _, files in os.walk(pdb_dir):
        for file in files:
            # ignore non pdb files
            if os.path.splitext(file)[1] != '.pdb':
                continue
            pdb_name = file[:4]
            if pdb_name in pdb_names:
                continue
            pdb_names.append(pdb_name)
            try:
                complex = DBComplex.from_pdb(root, pdb_name)
            except Exception as e:
                print(f"Error during parsing pdb {pdb_name}: {e}")
                continue

            receptor_bb_coord = complex.receptor_coord()    # backbone atoms, [N_re, 4, 3]
            ligand_bb_coord = complex.ligand_coord()  # backbone atoms, [N_li, 4, 3]

            # center
            center = np.mean(ligand_bb_coord.reshape(-1, 3), axis=0)

            cplx_coord = np.concatenate((receptor_bb_coord, ligand_bb_coord), axis=0)

            # centering
            cplx_coord = cplx_coord - center
            cplx_ca_coord = cplx_coord[:, CA_INDEX]  # (N, 3)

            cplx_mean.append(np.mean(cplx_ca_coord, axis=0))
            cplx_std.append(np.std(cplx_ca_coord, axis=0))

    cplx_mean = np.asarray(cplx_mean)
    cplx_std = np.asarray(cplx_std)

    print(f"ligand mean after centering: {np.mean(cplx_mean, axis=0)}")
    print(f"ligand std after centering: {np.mean(cplx_std, axis=0)}")


def DIPS_statistics(train_set):
    cplx_mean = []
    cplx_std = []
    base_path = os.path.dirname(train_set)
    with open(train_set, 'r') as fin:
        lines = fin.read().strip().split('\n')
    for line in tqdm(lines):
        dill_name = str(line)
        dill_path = os.path.join(base_path, dill_name)
        try:
            complex = DIPSComplex.from_pdb(dill_path)
        except Exception as e:
            print(f'parse {dill_name} pdb failed: {e}, skip')
            continue

        receptor_bb_coord = complex.receptor_coord()  # backbone atoms, [N_re, 4, 3]
        ligand_bb_coord = complex.ligand_coord()  # backbone atoms, [N_li, 4, 3]

        # center
        center = np.mean(ligand_bb_coord.reshape(-1, 3), axis=0)

        cplx_coord = np.concatenate((receptor_bb_coord, ligand_bb_coord), axis=0)

        # centering
        cplx_coord = cplx_coord - center
        cplx_ca_coord = cplx_coord[:, CA_INDEX]  # (N, 3)

        cplx_mean.append(np.mean(cplx_ca_coord, axis=0))
        cplx_std.append(np.std(cplx_ca_coord, axis=0))

    cplx_mean = np.asarray(cplx_mean)
    cplx_std = np.asarray(cplx_std)

    print(f"ligand mean after centering: {np.mean(cplx_mean, axis=0)}")
    print(f"ligand std after centering: {np.mean(cplx_std, axis=0)}")



if __name__ == "__main__":
    pdb_dir = -1    # TODO: change your DB5.5 pdb directory here
    DB5_statistics(pdb_dir)

    dips_train_set = -1     # TODO: change your DIPS train set here
    cores = 28
    Parallel(n_jobs=cores)(delayed(DIPS_statistics)(dips_train_set))
