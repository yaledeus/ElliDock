from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import standard_aa_names as AA_NAMES_3
from Bio.Data import IUPACData
import numpy as np
import os

parser = PDBParser(QUIET=True)
AA_NAMES_1 = tuple(IUPACData.protein_letters_3to1.values())
BACKBONE_ATOM = ['N', 'CA', 'C', 'O']

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
                backbone_coord = []
                for bb in BACKBONE_ATOM:
                    backbone_coord.append(residue[bb].get_coord())
                ag_coord[chain_name].append(backbone_coord)
        ag_coord[chain_name] = np.asarray(ag_coord[chain_name]).tolist()

    assert 'H' in ab_seq and 'L' in ab_seq, 'blank antibody.'
    assert ag_seq != {}, 'blank antigen.'
    return ab_seq, ab_coord, cdr_pos, ag_seq, ag_coord


# pdb_path = "./12e8.pdb"
# ab_seq, ab_coord, cdr_pos, ag_seq, ag_coord = sabdab_pdb_parse(pdb_path, 'H', 'L', 'P', 'imgt')
