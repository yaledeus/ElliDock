#!/bin/zsh

python -m data.download \
    --summary ../summaries/sabdab_summary_all.tsv \
    --pdb_dir ${PDB_DIR}/all_structures/imgt \
    --fout ${PDB_DIR}/sabdab_all.json \
    --type sabdab \
    --numbering imgt \
    --pre_numbered \
    --n_cpu 4 &&
python -m data.download \
    --summary ../summaries/rabd_summary.jsonl \
    --pdb_dir ${PDB_DIR}/all_structures/imgt \
    --fout ${PDB_DIR}/rabd_all.json \
    --type rabd \
    --numbering imgt \
    --pre_numbered \
    --n_cpu 4 &&
python -m data.split \
    --data ${PDB_DIR}/sabdab_all.json \
    --out_dir ${PDB_DIR} \
    --valid_ratio 0.1 \
    --test_ratio 0 \
    --filter **1 \
    --rabd ${PDB_DIR}/rabd_all.json

