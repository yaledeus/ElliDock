#!/bin/zsh

while getopts "d:" arg
do
  case $arg in
       d)
          echo "PDB_DIR: $OPTARG"
          PDB_DIR=$OPTARG
          ;;
       ?)
          echo "unknown argument"
          exit 1
          ;;
  esac
done

if [ ! -f "${PDB_DIR}/sabdab_all.json" ]; then
  python -m download \
      --summary ../summaries/sabdab_summary_all.tsv \
      --pdb_dir ${PDB_DIR}/all_structures/imgt \
      --fout ${PDB_DIR}/sabdab_all.json \
      --type sabdab \
      --numbering imgt \
      --pre_numbered \
      --n_cpu 4
fi

if [ ! -f "${PDB_DIR}/rabd_all.json" ];then
python -m download \
    --summary ../summaries/rabd_summary.jsonl \
    --pdb_dir ${PDB_DIR}/all_structures/imgt \
    --fout ${PDB_DIR}/rabd_all.json \
    --type rabd \
    --numbering imgt \
    --pre_numbered \
    --n_cpu 4
fi

python -m split \
    --data ${PDB_DIR}/sabdab_all.json \
    --out_dir ${PDB_DIR} \
    --valid_ratio 0.1 \
    --test_ratio 0 \
    --filter 111 \
    --rabd ${PDB_DIR}/rabd_all.json

