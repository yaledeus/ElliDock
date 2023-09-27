#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time
import shutil
from typing import List

from Bio.PDB import PDBIO

FILE_DIR = os.path.split(__file__)[0]
FIXEXEC = os.path.join(FILE_DIR, 'DockQ', 'scripts', 'fix_numbering.pl')
DQEXEC = os.path.join(FILE_DIR, 'DockQ', 'DockQ.py')
CACHE_DIR = os.path.join(FILE_DIR, '__dqcache__')
# create cache dir
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def dockQ(pdb_model: str, pdb_native: str, rchain_id=None, lchain_id=None):
    paths = []
    for i, src in enumerate([pdb_model, pdb_native]):
        chain_name = f'{time.time()}'  # for concurrent conflicts
        path = os.path.join(CACHE_DIR, f'{chain_name}.pdb')
        shutil.copy(src, path)
        paths.append(path)
    # os.popen(f'{FIXEXEC} {paths[0]} {paths[1]}')
    # paths[0] = f'{paths[0]}.fixed'
    if rchain_id and lchain_id:
        p = os.popen(f'{DQEXEC} {paths[0]} {paths[1]} -native_chain1 {" ".join(rchain_id)}'
                     f' -model_chain1 {" ".join(rchain_id)} -native_chain2 {" ".join(lchain_id)}'
                     f' -model_chain2 {" ".join(lchain_id)}')
    else:
        p = os.popen(f'{DQEXEC} {paths[0]} {paths[1]}')
    text = p.read()
    p.close()
    res = re.search(r'DockQ ([0-1]\.[0-9]+)', text)
    score = float(res.group(1))
    for path in paths:
        os.remove(path)
    return score
