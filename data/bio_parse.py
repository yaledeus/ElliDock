from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import standard_aa_names as AA_NAMES_3
from Bio.Data import IUPACData
from typing import List

import numpy as np
import os

parser = PDBParser(QUIET=True)
AA_NAMES_1 = tuple(IUPACData.protein_letters_3to1.values())
BACKBONE_ATOM = ['N', 'CA', 'C', 'O']
N_INDEX, CA_INDEX, C_INDEX, O_INDEX = 0, 1, 2, 3
MAX_CHAINS = 3

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


class Complex:
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
            else:
                ag_coord = np.concatenate((ag_coord, chain_coord), axis=0)
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

    def find_keypoint(self, threshold=10) -> List:
        keypoints = []
        antibody_ca_coord = self.antibody_coord()[:, CA_INDEX]     # CA coordinates, (N, 3)
        antigen_ca_coord = self.antigen_coord()[:, CA_INDEX]
        for i in range(antibody_ca_coord.shape[0]):
            ab_ca = antibody_ca_coord[i].reshape(1, -1)     # (1, 3)
            dist = (antigen_ca_coord - ab_ca) ** 2
            dist = np.sqrt(dist.sum(axis=-1))
            valid_ag_ca_list = np.where(dist < threshold)[0]
            ab_ca = ab_ca.squeeze()     # (3,)
            for j in valid_ag_ca_list:
                ag_ca = antigen_ca_coord[j]
                keypoint_coord = 0.5 * (ab_ca + ag_ca)
                keypoints.append(keypoint_coord)
        return keypoints
