from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import standard_aa_names as AA_NAMES_3
from Bio.Data import IUPACData
import numpy as np
import os

parser = PDBParser()
AA_NAMES_1 = tuple(IUPACData.protein_letters_3to1.values())

### get index of the residue by 3-letter name
def aa_index(aa_name_3):
    return AA_NAMES_3.index(aa_name_3)

### get 1-letter name of the residue by 3-letter name
def aa_3to1(aa_name_3):
    return AA_NAMES_1[aa_index(aa_name_3)]

### get 3-letter name of the residue by 1-letter name
def aa_1to3(aa_name_1):
    return AA_NAMES_3[AA_NAMES_1.index(aa_name_1)]

### parse pdb file
def pdb_parse(pdb_path):
    filename = os.path.basename(pdb_path)
    pdb_id, _ = os.path.splitext(filename)
    structure = parser.get_structure(pdb_id, pdb_path)
    for model in structure:
        for chain in model:
            for residue in chain:
                print(f"residue index: {aa_index(residue.get_resname())}, Chain id: {chain.get_id()}, Model id: {model.get_id()}")


pdb_path = "./12e8.pdb"
pdb_parse(pdb_path)
