import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Polypeptide import standard_aa_names as AA_NAMES_3
from Bio.Data import IUPACData
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import os

parser = PDBParser(QUIET=True)
AA_NAMES_1 = tuple(IUPACData.protein_letters_3to1.values())
BACKBONE_ATOM = ['N', 'CA', 'C', 'O']
N_INDEX, CA_INDEX, C_INDEX, O_INDEX = 0, 1, 2, 3

### get index of the residue by 3-letter name
def aa_index_3(aa_name_3):
    return AA_NAMES_3.index(aa_name_3)

### get index of the residue by 1-letter name
def aa_index_1(aa_name_1):
    return AA_NAMES_1.index(aa_name_1)

### get 1-letter name of the residue by 3-letter name
def aa_3to1(aa_name_3):
    return AA_NAMES_1[aa_index_3(aa_name_3)]

### get 3-letter name of the residue by 1-letter name
def aa_1to3(aa_name_1):
    return AA_NAMES_3[aa_index_1(aa_name_1)]

### CDR mapping in imgt numbering method
def _cdr_mapping_imgt():
    cdr_mapping = {}    # - for non-Fv region, 0 for framework, 1/2/3 for CDR1/2/3
    for i in list(range(1, 27)) + list(range(39, 56)) + list(range(66, 105)) + list(range(118, 130)):
        cdr_mapping[i] = '0'
    for i in range(27, 39):     # cdr1
        cdr_mapping[i] = '1'
    for i in range(56, 66):     # cdr2
        cdr_mapping[i] = '2'
    for i in range(105, 118):   # cdr3
        cdr_mapping[i] = '3'
    conserved = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS'],
        # 118: ['PHE', 'TRP']
    }
    return cdr_mapping, conserved

IMGT_CDR_MAPPING, IMGT_CONSERVED = _cdr_mapping_imgt()


def base_pdb_parse(pdb_path):
    filename = os.path.basename(pdb_path)
    pdb_id = filename[:4]
    structure = parser.get_structure(pdb_id, pdb_path)

    seq = {}            # sequence
    coord = {}          # coordinates, backbone only
    len = 0             # length

    for chain in structure[0]:
        chain_name = chain.get_id()
        assert chain_name != '', 'chain name is not valid.'
        seq[chain_name] = ''
        coord[chain_name] = []
        for residue in chain:
            # print(f"residue id: {residue.get_id()}")
            if (res_name := residue.get_resname()) in AA_NAMES_3:
                seq[chain_name] += aa_3to1(res_name)
                len += 1
                backbone_coord = []
                for bb in BACKBONE_ATOM:
                    backbone_coord.append(residue[bb].get_coord())
                coord[chain_name].append(backbone_coord)
        coord[chain_name] = np.asarray(coord[chain_name]).tolist()

    return seq, coord


### parse SabDab pdb file
def sabdab_pdb_parse(pdb_path, hchain_name, lchain_name, agchain_names, numbering: str='imgt'):
    assert numbering.lower() == 'imgt', f'numbering method {numbering} not supported yet.'
    filename = os.path.basename(pdb_path)
    pdb_id, _ = os.path.splitext(filename)
    structure = parser.get_structure(pdb_id, pdb_path)
    ab_seq = {}         # antibody sequence
    cdr_pos = {}        # CDRs position
    ag_seq = {}         # antigen sequence
    ab_coord = {}       # antibody coordinates, backbone only
    ag_coord = {}       # antigen coordinates, backbone only
    ab_len = 0          # antibody length
    ag_len = 0          # antigen length
    # parse antibody and CDRs
    for chain in structure[0]:
        chain_name = chain.get_id()
        assert chain_name != '', 'chain name not valid.'
        if chain_name == hchain_name:
            c = 'H'
        elif chain_name == lchain_name:
            c = 'L'
        else:
            continue
        ab_seq[c] = ''
        ab_coord[c] = []
        res_type = ''
        for residue in chain:
            # print(f"residue id: {residue.get_id()}")
            if (res_name := residue.get_resname()) in AA_NAMES_3:
                res_idx = residue.get_id()[1]
                if res_idx in IMGT_CDR_MAPPING:
                    res_type += IMGT_CDR_MAPPING[res_idx]
                    if res_idx in IMGT_CONSERVED:
                        hit= False
                        for conserved_res in IMGT_CONSERVED[res_idx]:
                            if res_name == conserved_res:
                                hit = True
                                break
                        assert hit, f'Not {IMGT_CONSERVED[res_idx]} at {res_idx}'
                    else:
                        res_type += '-'
                ab_seq[c] += aa_3to1(res_name)
                ab_len += 1
                backbone_coord = []
                for bb in BACKBONE_ATOM:
                    backbone_coord.append(residue[bb].get_coord())
                ab_coord[c].append(backbone_coord)
        for cdr in ['1', '2', '3']:
            cdr_start, cdr_end = res_type.find(cdr), res_type.rfind(cdr)
            if cdr_start == -1:
                raise ValueError(f'cdr {cdr} not found, residue type: {res_type}')
            cdr_pos[f'CDR-{c}{cdr}'] = (cdr_start, cdr_end)
        ab_coord[c] = np.asarray(ab_coord[c]).tolist()
    if hchain_name == '':
        ab_seq['H'] = ''
        ab_coord['H'] = []
        for i in range(1, 4):
            cdr_pos[f'CDR-H{i}'] = ()
    if lchain_name == '':
        ab_seq['L'] = ''
        ab_coord['L'] = []
        for i in range(1, 4):
            cdr_pos[f'CDR-L{i}'] = ()
    # parse antigen
    for chain_name in agchain_names:
        chain = structure[0][chain_name]
        ag_seq[chain_name] = ''
        ag_coord[chain_name] = []
        for residue in chain:
            if (res_name := residue.get_resname()) in AA_NAMES_3:
                ag_seq[chain_name] += aa_3to1(res_name)
                ag_len += 1
                backbone_coord = []
                for bb in BACKBONE_ATOM:
                    backbone_coord.append(residue[bb].get_coord())
                ag_coord[chain_name].append(backbone_coord)
        ag_coord[chain_name] = np.asarray(ag_coord[chain_name]).tolist()

    assert 'H' in ab_seq and 'L' in ab_seq, 'antibody missing H/L'
    assert ab_len >= 100, 'antibody sequence length < 100'
    assert ag_len >= 20, 'antigen sequence length < 20'
    return ab_seq, ab_coord, cdr_pos, ag_seq, ag_coord


### parse DB5.5 pdb file
def DB_pdb_parse(pdb_path):
    filename = os.path.basename(pdb_path)
    assert filename[7] == 'b', f'unbound complex: {filename}'
    pdb_id, ligand = filename[:4], filename[5] == 'l'
    structure = parser.get_structure(pdb_id, pdb_path)
    pseq = {}            # sequence
    pcoord = {}          # coordinates, backbone only
    plen = 0             # length
    # parse antibody and CDRs
    for chain in structure[0]:
        chain_name = chain.get_id()
        assert chain_name != '', 'chain name not valid.'
        pseq[chain_name] = ''
        pcoord[chain_name] = []
        for residue in chain:
            # print(f"residue id: {residue.get_id()}")
            if (res_name := residue.get_resname()) in AA_NAMES_3:
                pseq[chain_name] += aa_3to1(res_name)
                plen += 1
                backbone_coord = []
                for bb in BACKBONE_ATOM:
                    backbone_coord.append(residue[bb].get_coord())
                pcoord[chain_name].append(backbone_coord)
        pcoord[chain_name] = np.asarray(pcoord[chain_name]).tolist()

    if ligand:
        assert plen >= 20, 'ligand sequence length < 20'
    else:
        assert plen >= 100 and plen <= 1000, 'receptor sequence length < 100 or > 1000'
    return pseq, pcoord


### parse DIPS pdb file
def DIPS_pdb_parse(dill_path):
    filename = os.path.basename(dill_path)
    pdb_id = filename[:4]
    x = pd.read_pickle(dill_path)
    ligand, receptor = x.df0, x.df1

    re_seq = {}
    re_coord = {}
    re_len = 0

    for residue in receptor.groupby(by=['chain', 'residue']):
        chain_name, res_idx = residue[0]
        df = residue[1]
        assert chain_name != '', 'chain name not valid.'
        res_name = df.loc[df.index[0], 'resname']
        if not res_name in AA_NAMES_3:
            continue
        if not chain_name in re_seq:
            re_seq[chain_name] = ''
            re_coord[chain_name] = []
        backbone_coord = []
        normal_res_flag = True
        for idx in range(len(BACKBONE_ATOM)):
            ridx = df.index[0] + idx
            try:
                assert df.loc[ridx, 'atom_name'] == BACKBONE_ATOM[idx], f'not {BACKBONE_ATOM[idx]} in index {idx}'
            except:
                normal_res_flag = False
                break
            backbone_coord.append([df.loc[ridx, 'x'], df.loc[ridx, 'y'], df.loc[ridx, 'z']])
        if not normal_res_flag:
            continue
        re_seq[chain_name] += aa_3to1(res_name)
        re_coord[chain_name].append(backbone_coord)
        re_len += 1

    for key in re_coord.keys():
        re_coord[key] = np.asarray(re_coord[key]).tolist()

    li_seq = {}
    li_coord = {}
    li_len = 0

    for residue in ligand.groupby(by=['chain', 'residue']):
        chain_name, res_idx = residue[0]
        df = residue[1]
        assert chain_name != '', 'chain name not valid.'
        res_name = df.loc[df.index[0], 'resname']
        if not res_name in AA_NAMES_3:
            continue
        if not chain_name in li_seq:
            li_seq[chain_name] = ''
            li_coord[chain_name] = []
        backbone_coord = []
        normal_res_flag = True
        for idx in range(len(BACKBONE_ATOM)):
            ridx = df.index[0] + idx
            try:
                assert df.loc[ridx, 'atom_name'] == BACKBONE_ATOM[idx], f'not {BACKBONE_ATOM[idx]} in index {idx}'
            except:
                normal_res_flag = False
                break
            backbone_coord.append([df.loc[ridx, 'x'], df.loc[ridx, 'y'], df.loc[ridx, 'z']])
        if not normal_res_flag:
            continue
        li_seq[chain_name] += aa_3to1(res_name)
        li_coord[chain_name].append(backbone_coord)
        li_len += 1

    for key in li_coord.keys():
        li_coord[key] = np.asarray(li_coord[key]).tolist()

    return re_seq, li_seq, re_coord, li_coord


### generate docked pdb file
def gen_docked_pdb(pdb_name, src_path, dst_path, trans_func):
    structure = parser.get_structure(pdb_name, src_path)
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coord = atom.get_coord()
                    atom.set_coord(trans_func(atom_coord))
    io = PDBIO()
    io.set_structure(structure)
    io.save(os.path.join(dst_path, f'{pdb_name}_r_d.pdb'))


class BaseComplex:
    def __init__(self, ligand_seq, receptor_seq, ligand_coord, receptor_coord):
        self.li_seq = ligand_seq
        self.re_seq = receptor_seq
        self.li_coord = ligand_coord
        self.re_coord = receptor_coord

    @classmethod
    def from_pdb(cls, ligand_path, receptor_path):
        ligand_seq, ligand_coord = base_pdb_parse(ligand_path)
        receptor_seq, receptor_coord = base_pdb_parse(receptor_path)
        return cls(ligand_seq, receptor_seq, ligand_coord, receptor_coord)

    def ligand_seq(self) -> List:
        ligand_seq = []
        for _, chain_seq in self.li_seq.items():
            for res_name_1 in chain_seq:
                ligand_seq.append(aa_index_1(res_name_1))
        return ligand_seq

    def ligand_coord(self):
        ligand_coord = np.array([])
        for _, chain_coord in self.li_coord.items():
            chain_coord = np.asarray(chain_coord)
            if not ligand_coord.shape[0]:
                ligand_coord = chain_coord
            elif ligand_coord.ndim == chain_coord.ndim:
                ligand_coord = np.concatenate((ligand_coord, chain_coord), axis=0)
            else:
                continue
        return ligand_coord

    def receptor_seq(self) -> List:
        receptor_seq = []
        for _, chain_seq in self.re_seq.items():
            for res_name_1 in chain_seq:
                receptor_seq.append(aa_index_1(res_name_1))
        return receptor_seq

    def receptor_coord(self):
        receptor_coord = np.array([])
        for _, chain_coord in self.re_coord.items():
            chain_coord = np.asarray(chain_coord)
            if not receptor_coord.shape[0]:
                receptor_coord = chain_coord
            elif receptor_coord.ndim == chain_coord.ndim:
                receptor_coord = np.concatenate((receptor_coord, chain_coord), axis=0)
            else:
                continue
        return receptor_coord

    def receptor_relative_pos(self) -> List:
        return [i for i in range(len(self.receptor_seq()))]

    def receptor_identity(self) -> List:
        receptor_id = []
        i = 0
        for _, chain_seq in self.re_seq.items():
            receptor_id.extend([i] * len(chain_seq))
            i += 1
        return receptor_id

    def ligand_relative_pos(self) -> List:
        return [i for i in range(len(self.ligand_seq()))]

    def ligand_identity(self) -> List:
        ligand_id = []
        i = 0
        for _, chain_seq in self.li_seq.items():
            ligand_id.extend([i] * len(chain_seq))
            i += 1
        return ligand_id

    # def find_keypoint(self, threshold=8.) -> np.ndarray:
    #     receptor_ca_coord = self.receptor_coord()[:, CA_INDEX]     # CA coordinates, (N, 3)
    #     ligand_ca_coord = self.ligand_coord()[:, CA_INDEX]
    #     abag_dist = cdist(receptor_ca_coord, ligand_ca_coord)
    #     ab_idx, ag_idx = np.where(abag_dist < threshold)
    #     keypoints = 0.5 * (receptor_ca_coord[ab_idx] + ligand_ca_coord[ag_idx])
    #     return keypoints


class SabDabComplex:
    def __init__(self, ab_seq, cdr_pos, ag_seq, ab_coord, ag_coord):
        self.ab_seq = ab_seq
        self.cdr_pos = cdr_pos
        self.ag_seq = ag_seq
        self.ab_coord = ab_coord
        self.ag_coord = ag_coord

    @classmethod
    def from_pdb(cls, pdb_path, hchain_name, lchain_name, agchain_names):
        ab_seq, ab_coord, cdr_pos, ag_seq, ag_coord = \
            sabdab_pdb_parse(pdb_path, hchain_name, lchain_name, agchain_names)
        return cls(ab_seq, cdr_pos, ag_seq, ab_coord, ag_coord)

    def hchain_only(self):
        return self.ab_seq['L'] == ''

    def lchain_only(self):
        return self.ab_seq['H'] == ''

    def hchain_seq(self) -> List:
        if self.lchain_only():
            return []
        else:
            h_seq = []
            for res_name_1 in self.ab_seq['H']:
                h_seq.append(aa_index_1(res_name_1))
            return h_seq

    def lchain_seq(self) -> List:
        if self.hchain_only():
            return []
        else:
            l_seq = []
            for res_name_1 in self.ab_seq['L']:
                l_seq.append(aa_index_1(res_name_1))
            return l_seq

    def antibody_seq(self) -> List:
        return self.hchain_seq() + self.lchain_seq()

    def hchain_coord(self):
        return np.asarray(self.ab_coord['H']) if not self.lchain_only() else None

    def lchain_coord(self):
        return np.asarray(self.ab_coord['L']) if not self.hchain_only() else None

    def antibody_coord(self):
        if self.hchain_only():
            return self.hchain_coord()
        elif self.lchain_only():
            return self.lchain_coord()
        else:
            return np.concatenate((self.hchain_coord(), self.lchain_coord()), axis=0)

    def antibody_relative_pos(self) -> List:
        return [i for i in range(len(self.antibody_seq()))]

    def antibody_identify(self) -> List:
        ### 0 for heavy chain, 1 for light chain
        return [0 for _ in range(len(self.hchain_seq()))] + [1 for _ in range(len(self.lchain_seq()))]

    def antigen_seq(self) -> List:
        ag_seq = []
        for _, chain_seq in self.ag_seq.items():
            for res_name_1 in chain_seq:
                ag_seq.append(aa_index_1(res_name_1))
        return ag_seq

    def antigen_coord(self):
        ag_coord = np.array([])
        for _, chain_coord in self.ag_coord.items():
            chain_coord = np.asarray(chain_coord)
            if not ag_coord.shape[0]:
                ag_coord = chain_coord
            elif ag_coord.ndim == chain_coord.ndim:
                ag_coord = np.concatenate((ag_coord, chain_coord), axis=0)
            else:
                continue
        return ag_coord

    def antigen_relative_pos(self) -> List:
        return [i for i in range(len(self.antigen_seq()))]

    def antigen_identity(self) -> List:
        ag_id = []
        i = 0
        for _, chain_seq in self.ag_seq.items():
            ag_id.extend([i] * len(chain_seq))
            i += 1
        return ag_id

    def cdr_span(self, chain, idx):
        """
        :param chain: heavy chain or light chain, choices: ('H', 'L')
        :param idx: 1/2/3
        :return: a tuple (start, end) describing cdr positions
        """
        assert chain in ('H', 'L'), f'invalid chain: {chain}'
        assert idx in (1, 2, 3), f'invalid index: {idx}'
        return self.cdr_pos[f'CDR-{chain}{idx}']

    def find_keypoint(self, threshold=8.) -> np.ndarray:
        antibody_ca_coord = self.antibody_coord()[:, CA_INDEX]     # CA coordinates, (N, 3)
        antigen_ca_coord = self.antigen_coord()[:, CA_INDEX]
        abag_dist = cdist(antibody_ca_coord, antigen_ca_coord)
        ab_idx, ag_idx = np.where(abag_dist < threshold)
        ab_keypoints = 0.5 * (antibody_ca_coord[ab_idx] + antigen_ca_coord[np.argmin(abag_dist[ab_idx], axis=1)])
        ag_keypoints = 0.5 * (antigen_ca_coord[ag_idx] + antibody_ca_coord[np.argmin(abag_dist.T[ag_idx], axis=1)])
        keypoints = np.concatenate((ab_keypoints, ag_keypoints), axis=0)
        return keypoints


class DBComplex:
    def __init__(self, pdb_name, ligand_seq, receptor_seq, ligand_coord, receptor_coord):
        self.pdb_name = pdb_name
        self.li_seq = ligand_seq
        self.re_seq = receptor_seq
        self.li_coord = ligand_coord
        self.re_coord = receptor_coord

    @classmethod
    def from_pdb(cls, base_path, pdb_name):
        ligand_bound_path = os.path.join(base_path, f'{pdb_name.upper()}_l_b.pdb')
        receptor_bound_path = os.path.join(base_path, f'{pdb_name.upper()}_r_b.pdb')
        ligand_seq, ligand_coord = DB_pdb_parse(ligand_bound_path)
        receptor_seq, receptor_coord = DB_pdb_parse(receptor_bound_path)
        return cls(pdb_name.upper(), ligand_seq, receptor_seq, ligand_coord, receptor_coord)

    def ligand_seq(self) -> List:
        ligand_seq = []
        for _, chain_seq in self.li_seq.items():
            for res_name_1 in chain_seq:
                ligand_seq.append(aa_index_1(res_name_1))
        return ligand_seq

    def ligand_coord(self):
        ligand_coord = np.array([])
        for _, chain_coord in self.li_coord.items():
            chain_coord = np.asarray(chain_coord)
            if not ligand_coord.shape[0]:
                ligand_coord = chain_coord
            elif ligand_coord.ndim == chain_coord.ndim:
                ligand_coord = np.concatenate((ligand_coord, chain_coord), axis=0)
            else:
                continue
        return ligand_coord

    def receptor_seq(self) -> List:
        receptor_seq = []
        for _, chain_seq in self.re_seq.items():
            for res_name_1 in chain_seq:
                receptor_seq.append(aa_index_1(res_name_1))
        return receptor_seq

    def receptor_coord(self):
        receptor_coord = np.array([])
        for _, chain_coord in self.re_coord.items():
            chain_coord = np.asarray(chain_coord)
            if not receptor_coord.shape[0]:
                receptor_coord = chain_coord
            elif receptor_coord.ndim == chain_coord.ndim:
                receptor_coord = np.concatenate((receptor_coord, chain_coord), axis=0)
            else:
                continue
        return receptor_coord

    def receptor_relative_pos(self) -> List:
        return [i for i in range(len(self.receptor_seq()))]

    def receptor_identity(self) -> List:
        receptor_id = []
        i = 0
        for _, chain_seq in self.re_seq.items():
            receptor_id.extend([i] * len(chain_seq))
            i += 1
        return receptor_id

    def ligand_relative_pos(self) -> List:
        return [i for i in range(len(self.ligand_seq()))]

    def ligand_identity(self) -> List:
        ligand_id = []
        i = 0
        for _, chain_seq in self.li_seq.items():
            ligand_id.extend([i] * len(chain_seq))
            i += 1
        return ligand_id

    def find_keypoint(self, threshold=8.) -> np.ndarray:
        receptor_ca_coord = self.receptor_coord()[:, CA_INDEX]     # CA coordinates, (N, 3)
        ligand_ca_coord = self.ligand_coord()[:, CA_INDEX]
        abag_dist = cdist(receptor_ca_coord, ligand_ca_coord)
        ab_idx, ag_idx = np.where(abag_dist < threshold)
        ab_keypoints = 0.5 * (receptor_ca_coord[ab_idx] + ligand_ca_coord[np.argmin(abag_dist[ab_idx], axis=1)])
        ag_keypoints = 0.5 * (ligand_ca_coord[ag_idx] + receptor_ca_coord[np.argmin(abag_dist.T[ag_idx], axis=1)])
        keypoints = np.concatenate((ab_keypoints, ag_keypoints), axis=0)
        return keypoints


class DIPSComplex:
    def __init__(self, ligand_seq, receptor_seq, ligand_coord, receptor_coord):
        self.li_seq = ligand_seq
        self.re_seq = receptor_seq
        self.li_coord = ligand_coord
        self.re_coord = receptor_coord

    @classmethod
    def from_pdb(cls, dill_path):
        receptor_seq, ligand_seq, receptor_coord, ligand_coord = DIPS_pdb_parse(dill_path)
        return cls(ligand_seq, receptor_seq, ligand_coord, receptor_coord)

    def ligand_seq(self) -> List:
        ligand_seq = []
        for _, chain_seq in self.li_seq.items():
            for res_name_1 in chain_seq:
                ligand_seq.append(aa_index_1(res_name_1))
        return ligand_seq

    def ligand_coord(self):
        ligand_coord = np.array([])
        for _, chain_coord in self.li_coord.items():
            chain_coord = np.asarray(chain_coord)
            if not ligand_coord.shape[0]:
                ligand_coord = chain_coord
            elif ligand_coord.ndim == chain_coord.ndim:
                ligand_coord = np.concatenate((ligand_coord, chain_coord), axis=0)
            else:
                continue
        return ligand_coord

    def receptor_seq(self) -> List:
        receptor_seq = []
        for _, chain_seq in self.re_seq.items():
            for res_name_1 in chain_seq:
                receptor_seq.append(aa_index_1(res_name_1))
        return receptor_seq

    def receptor_coord(self):
        receptor_coord = np.array([])
        for _, chain_coord in self.re_coord.items():
            chain_coord = np.asarray(chain_coord)
            if not receptor_coord.shape[0]:
                receptor_coord = chain_coord
            elif receptor_coord.ndim == chain_coord.ndim:
                receptor_coord = np.concatenate((receptor_coord, chain_coord), axis=0)
            else:
                continue
        return receptor_coord

    def receptor_relative_pos(self) -> List:
        return [i for i in range(len(self.receptor_seq()))]

    def receptor_identity(self) -> List:
        receptor_id = []
        i = 0
        for _, chain_seq in self.re_seq.items():
            receptor_id.extend([i] * len(chain_seq))
            i += 1
        return receptor_id

    def ligand_relative_pos(self) -> List:
        return [i for i in range(len(self.ligand_seq()))]

    def ligand_identity(self) -> List:
        ligand_id = []
        i = 0
        for _, chain_seq in self.li_seq.items():
            ligand_id.extend([i] * len(chain_seq))
            i += 1
        return ligand_id

    def find_keypoint(self, threshold=8.) -> np.ndarray:
        receptor_ca_coord = self.receptor_coord()[:, CA_INDEX]     # CA coordinates, (N, 3)
        ligand_ca_coord = self.ligand_coord()[:, CA_INDEX]
        abag_dist = cdist(receptor_ca_coord, ligand_ca_coord)
        ab_idx, ag_idx = np.where(abag_dist < threshold)
        ab_keypoints = 0.5 * (receptor_ca_coord[ab_idx] + ligand_ca_coord[np.argmin(abag_dist[ab_idx], axis=1)])
        ag_keypoints = 0.5 * (ligand_ca_coord[ag_idx] + receptor_ca_coord[np.argmin(abag_dist.T[ag_idx], axis=1)])
        keypoints = np.concatenate((ab_keypoints, ag_keypoints), axis=0)
        return keypoints


def gen_test_set_txt(test_path):
    test_pdbs = []
    suffix_len = len('_l_b_COMPLEX.pdb')
    for _, _, files in os.walk(os.path.join(test_path, 'complexes')):
        for file in files:
            pdb_name = file[:-suffix_len]
            if pdb_name not in test_pdbs:
                test_pdbs.append(pdb_name)
    with open(os.path.join(test_path, 'test.txt'), 'w') as fp:
        for pdb_name in test_pdbs:
            fp.write(f"{pdb_name}\n")


# if __name__ == "__main__":
#     gen_test_set_txt("../test_sets_pdb/db5_test_random_transformed")
#     gen_test_set_txt("../test_sets_pdb/dips_test_random_transformed")
