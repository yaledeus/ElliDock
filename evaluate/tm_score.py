#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time
import shutil

from Bio.PDB import PDBIO

FILE_DIR = os.path.split(__file__)[0]
TMEXEC = os.path.join(FILE_DIR, 'TMscore')
CACHE_DIR = os.path.join(FILE_DIR, '__tmcache__')
# create cache dir
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def tm_score(pdb_path_1: str, pdb_path_2: str):
    paths = []
    for i, src in enumerate([pdb_path_1, pdb_path_2]):
        chain_name = f'{time.time()}'  # for concurrent conflicts
        path = os.path.join(CACHE_DIR, f'{chain_name}.pdb')
        shutil.copy(src, path)
        paths.append(path)
    p = os.popen(f'{TMEXEC} {paths[0]} {paths[1]}')
    text = p.read()
    p.close()
    res = re.search(r'TM-score\s*= ([0-1]\.[0-9]+)', text)
    score = float(res.group(1))
    for path in paths:
        os.remove(path)
    return score
